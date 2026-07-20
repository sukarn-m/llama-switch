package main

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// Backend represents a single running llama-server process for one model.
type Backend struct {
	ID        string // canonical model config ID
	Port      int
	cmd       *exec.Cmd
	logPrefix string
	doneCh    chan struct{}
	mu        sync.Mutex
	lastReq   time.Time
}

// BackendManager manages all backend processes: lifecycle, health, and routing.
type BackendManager struct {
	cfg         *Config
	configPath  string
	vramCache   VRAMCache
	vramCacheP  string
	backends    map[string]*Backend   // keyed by canonical model ID
	loading     map[string]*sync.Cond // model ID -> condition variable for in-progress loads
	mu          sync.RWMutex
	portAlloc   portAllocator
	logger      *CondLogger
	sweeperStop chan struct{}
	sweeperOnce sync.Once
	shutdownCh  chan struct{} // closed when Stop() is called, aborts waitHealth
}

type portAllocator struct {
	used  map[int]bool
	start int
	end   int
}

func newPortAllocator(start, end int) portAllocator {
	return portAllocator{
		used:  make(map[int]bool),
		start: start,
		end:   end,
	}
}

// alloc returns a free port from the range. Caller must hold bm.mu.
func (pa *portAllocator) alloc() (int, error) {
	for p := pa.start; p <= pa.end; p++ {
		if !pa.used[p] {
			pa.used[p] = true
			return p, nil
		}
	}
	return 0, fmt.Errorf("no free ports in range %d-%d", pa.start, pa.end)
}

// release marks a port as free. Caller must hold bm.mu.
func (pa *portAllocator) release(p int) {
	delete(pa.used, p)
}

// ── Manager init ─────────────────────────────

func NewBackendManager(cfg *Config, configPath string, logger *CondLogger) *BackendManager {
	bm := &BackendManager{
		cfg:         cfg,
		configPath:  configPath,
		backends:    make(map[string]*Backend),
		loading:     make(map[string]*sync.Cond),
		portAlloc:   newPortAllocator(cfg.Server.BackendPortStart, cfg.Server.BackendPortEnd),
		logger:      logger,
		sweeperStop: make(chan struct{}),
		shutdownCh:  make(chan struct{}),
	}
	bm.vramCacheP = vramCachePath(cfg, configPath)
	bm.vramCache = LoadVRAMCache(bm.vramCacheP)
	return bm
}

func (bm *BackendManager) Start() {
	go bm.idleSweeper()
}

func (bm *BackendManager) Stop() {
	bm.sweeperOnce.Do(func() {
		close(bm.shutdownCh)
		close(bm.sweeperStop)
	})
	bm.mu.Lock()
	defer bm.mu.Unlock()
	for id, b := range bm.backends {
		bm.stopLocked(id, b)
	}
}

// ── Backend lifecycle ────────────────────────

// EnsureLoaded makes sure the model is loaded and healthy, loading or
// evicting as needed. Returns the backend or an error. The write lock is
// released during health-check polling and auto-profiling so that requests
// for other already-loaded models are not blocked.
func (bm *BackendManager) EnsureLoaded(modelID string) (*Backend, error) {
	for {
		mc := bm.cfg.FindModel(modelID)
		if mc == nil {
			return nil, fmt.Errorf("unknown model: %s", modelID)
		}
		canonicalID := mc.ID

		// Fast path: already loaded
		bm.mu.RLock()
		if b, ok := bm.backends[canonicalID]; ok {
			bm.mu.RUnlock()
			return b, nil
		}

		// Check if another goroutine is already loading this model
		if cond, ok := bm.loading[canonicalID]; ok {
			bm.mu.RUnlock()
			bm.mu.Lock()
			for bm.loading[canonicalID] == cond {
				cond.Wait()
			}
			bm.mu.Unlock()
			continue // re-check fast path
		}
		bm.mu.RUnlock()

		// Slow path: acquire write lock and load
		bm.mu.Lock()

		// Double-check after acquiring write lock
		if b, ok := bm.backends[canonicalID]; ok {
			bm.mu.Unlock()
			return b, nil
		}

		// Register as loading so concurrent requests for the same model wait
		loadCond := sync.NewCond(&bm.mu)
		bm.loading[canonicalID] = loadCond
		deleteLoading := func() {
			delete(bm.loading, canonicalID)
			loadCond.Broadcast()
		}

		// Check if we need to evict to make room (count loading entries too)
		if err := bm.ensureCapacityLocked(mc); err != nil {
			deleteLoading()
			bm.mu.Unlock()
			return nil, err
		}

		// Retry loop: if the backend fails to start (e.g. CUDA OOM because
		// VRAM estimates were stale or absent), evict an LRU backend
		// and retry, up to len(backends) times.
		var b *Backend
		var startErr error
		for attempt := 0; ; attempt++ {
			b, startErr = bm.startBackendLocked(mc)
			if startErr == nil {
				break
			}
			// If other backends are loaded, evict the LRU one and retry.
			// The error is likely VRAM-related; freeing space may fix it.
			if len(bm.backends) == 0 || attempt >= len(bm.backends) {
				deleteLoading()
				bm.mu.Unlock()
				return nil, startErr
			}
			bm.logger.Printf("load %s failed (attempt %d): %v — evicting LRU to retry",
				mc.ID, attempt+1, startErr)
			if err := bm.evictLRULocked(); err != nil {
				deleteLoading()
				bm.mu.Unlock()
				return nil, fmt.Errorf("%w (and no backends left to evict: %v)", startErr, err)
			}
			// Give GPU a moment to release VRAM after kill
			bm.mu.Unlock()
			time.Sleep(1 * time.Second)
			bm.mu.Lock()
		}

		deleteLoading()
		bm.mu.Unlock()
		return b, nil
	}
}

// ensureCapacityLocked checks model count limits and VRAM, evicting
// LRU models as needed. Counts both loaded backends and in-flight loads
// toward the MaxModels budget to prevent soft limit violations.
// Caller must hold bm.mu.
func (bm *BackendManager) ensureCapacityLocked(mc *ModelConfig) error {
	// 1. Count limit: evict if loaded + loading would exceed MaxModels
	for len(bm.backends)+len(bm.loading) > bm.cfg.Server.MaxModels {
		if err := bm.evictLRULocked(); err != nil {
			return fmt.Errorf("cannot make room for %s: %w", mc.ID, err)
		}
	}

	// 2. VRAM limit: evict LRU until estimated VRAM fits on each target GPU
	needed := vramEstimate(bm.vramCache, mc, bm.cfg.Backend.CommonArgs)
	if needed > 0 {
		for {
			stats, err := queryVRAM(bm.cfg.Backend.NvidiaSMI)
			if err != nil {
				break // can't query VRAM, proceed optimistically
			}
			headroom := 1024 // 1 GB safety margin (applied to primary GPU only)
			if vramFitsPerGPU(stats, mc, bm.vramCache, bm.cfg.Backend.CommonArgs, headroom) {
				break
			}
			if len(bm.backends) == 0 {
				break // nothing left to evict; try anyway
			}
			if err := bm.evictLRULocked(); err != nil {
				return err
			}
			// Give GPU a moment to actually release VRAM after kill
			bm.mu.Unlock()
			time.Sleep(1 * time.Second)
			bm.mu.Lock()
		}
	}

	return nil
}

// evictLRULocked unloads the least-recently-used backend.
// Caller must hold bm.mu.
func (bm *BackendManager) evictLRULocked() error {
	var oldestID string
	var oldestTime time.Time

	for id, b := range bm.backends {
		lastReq := b.lastRequest()
		if oldestID == "" || lastReq.Before(oldestTime) {
			oldestID = id
			oldestTime = lastReq
		}
	}

	if oldestID == "" {
		return fmt.Errorf("no backends to evict")
	}

	bm.stopLocked(oldestID, bm.backends[oldestID])
	return nil
}

// startBackendLocked spawns a new llama-server process for the model.
// Health-check polling and auto-profiling are done without holding the
// lock so that requests for other models are not blocked. The backend is
// added to the map only after health check passes.
// Caller must hold bm.mu, which is released and re-acquired internally.
func (bm *BackendManager) startBackendLocked(mc *ModelConfig) (*Backend, error) {
	port, err := bm.portAlloc.alloc()
	if err != nil {
		return nil, err
	}

	binPath, err := mc.ResolveBinary(&bm.cfg.Backend)
	if err != nil {
		bm.portAlloc.release(port)
		return nil, fmt.Errorf("backend binary: %w", err)
	}

	args := mc.BuildArgs(bm.cfg.Backend.CommonArgs, port)
	cmd := exec.Command(binPath, args...)
	cmd.Env = mc.BuildModelEnv(&bm.cfg.Backend)

	if wd := expand(bm.cfg.Backend.Workdir); wd != "" {
		if abs, err := filepath.Abs(wd); err == nil {
			cmd.Dir = abs
		}
	}

	// Capture stdout+stderr for prefix logging
	pr, pw := io.Pipe()
	cmd.Stdout = pw
	cmd.Stderr = pw

	b := &Backend{
		ID:        mc.ID,
		Port:      port,
		cmd:       cmd,
		logPrefix: fmt.Sprintf("[%s]", mc.ID),
		doneCh:    make(chan struct{}),
	}
	b.setLastRequest(time.Now())

	// Start the process
	if err := cmd.Start(); err != nil {
		bm.portAlloc.release(port)
		pr.Close()
		pw.Close()
		return nil, fmt.Errorf("failed to start %s: %w", mc.ID, err)
	}

	// Pipe stdout/stderr to our logger with prefix
	go bm.logPipe(pr, b.logPrefix)

	// Wait for process exit in background, then reap zombie if crashed
	go func() {
		if err := cmd.Wait(); err != nil {
			bm.logger.Printf("%s process exited: %v", b.logPrefix, err)
		}
		pw.Close()
		close(b.doneCh)

		// Reaper: if the backend is still in the map (not being actively
		// stopped), it crashed unexpectedly. Remove it so future requests
		// trigger a fresh load instead of proxying to a dead process.
		bm.mu.Lock()
		if _, ok := bm.backends[mc.ID]; ok && bm.backends[mc.ID] == b {
			delete(bm.backends, mc.ID)
			bm.portAlloc.release(b.Port)
			bm.logger.Printf("%s removed dead backend (unexpected exit)", b.logPrefix)
		}
		bm.mu.Unlock()
	}()

	// --- Health check: release lock during polling (CONC-1 fix) ---
	timeout := time.Duration(bm.cfg.Server.HealthTimeoutSeconds) * time.Second
	bm.mu.Unlock()

	err = bm.waitHealth(b.doneCh, bm.shutdownCh, port, mc.HealthEndpoint(), timeout)

	bm.mu.Lock()

	if err != nil {
		bm.stopLocked(mc.ID, b)
		return nil, fmt.Errorf("health check failed for %s: %w", mc.ID, err)
	}

	// Add to map only after health check passes
	bm.backends[mc.ID] = b

	// --- Auto-profile: release lock during the 3s settle (CONC-2 fix) ---
	// Only profile when this is the sole backend, so the delta isn't
	// corrupted by a concurrent model load (BUG-R2-3 fix).
	// Skip if a valid (non-stale) cache entry already exists.
	snap := mc.Snapshot(bm.cfg.Backend.CommonArgs)
	if entry, ok := bm.vramCache[mc.ID]; !ok || !entry.Config.Equal(snap) {
		if len(bm.backends) == 1 {
			wasStale := ok
			bm.mu.Unlock()

			before, _ := queryVRAM(bm.cfg.Backend.NvidiaSMI)
			time.Sleep(3 * time.Second)
			after, err := queryVRAM(bm.cfg.Backend.NvidiaSMI)

			var profiledVram int
			var profiledGPUVRAM []int
			var profiledValid bool
			if err == nil && after.Used > 0 {
				used := after.Used
				if before != nil && before.Used > 0 {
					delta := after.Used - before.Used
					if delta > 0 {
						used = delta
					}
				}
				profiledGPUVRAM = computePerGPUDelta(before, after)
				if verr := validateVRAMMeasurement(mc, used); verr != nil {
					bm.logger.Printf("%s VRAM measurement rejected: %v (not caching)", b.logPrefix, verr)
				} else {
					profiledVram = used
					profiledValid = true
				}
			}

			bm.mu.Lock()

			if profiledValid {
				bm.vramCache[mc.ID] = CacheEntry{Vram: profiledVram, GPUVRAM: profiledGPUVRAM, Config: snap}
				// Copy cache and save outside the lock (MINOR-5 fix)
				cacheCopy := make(VRAMCache, len(bm.vramCache))
				for k, v := range bm.vramCache {
					cacheCopy[k] = v
				}
				bm.mu.Unlock()
				if saveErr := SaveVRAMCache(bm.vramCacheP, cacheCopy); saveErr != nil {
					bm.logger.Printf("%s VRAM profiled: %d MB (auto) — cache save failed: %v", b.logPrefix, profiledVram, saveErr)
				} else if wasStale {
					bm.logger.Printf("%s VRAM re-profiled (config changed): %d MB (auto)", b.logPrefix, profiledVram)
				} else {
					bm.logger.Printf("%s VRAM profiled: %d MB (auto)", b.logPrefix, profiledVram)
				}
				bm.mu.Lock()
			}
		}
	}

	return b, nil
}

// stopLocked kills a backend process and removes it from the map.
// Caller must hold bm.mu.
func (bm *BackendManager) stopLocked(id string, b *Backend) {
	if b == nil {
		return
	}
	if b.cmd != nil && b.cmd.Process != nil {
		_ = b.cmd.Process.Signal(os.Interrupt)
		select {
		case <-b.doneCh:
		case <-time.After(5 * time.Second):
			_ = b.cmd.Process.Kill()
		}
	}
	bm.portAlloc.release(b.Port)
	delete(bm.backends, id)
}

// StopModel unloads a specific model by ID. Resolves the canonical ID
// from any accepted name (id, alias, or display name).
func (bm *BackendManager) StopModel(id string) error {
	mc := bm.cfg.FindModel(id)
	if mc == nil {
		return fmt.Errorf("model %s is not loaded", id)
	}
	canonicalID := mc.ID

	bm.mu.Lock()
	defer bm.mu.Unlock()
	b, ok := bm.backends[canonicalID]
	if !ok {
		return fmt.Errorf("model %s is not loaded", id)
	}
	bm.stopLocked(canonicalID, b)
	return nil
}

// IsLoaded checks whether a model is currently loaded, resolving the
// canonical ID from any accepted name.
func (bm *BackendManager) IsLoaded(id string) bool {
	mc := bm.cfg.FindModel(id)
	if mc == nil {
		return false
	}
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	_, ok := bm.backends[mc.ID]
	return ok
}

// ── Health check ─────────────────────────────

// waitHealth polls the backend's health endpoint until it responds 200,
// the timeout expires, or shutdownCh is closed. Does not require bm.mu.
func (bm *BackendManager) waitHealth(doneCh, shutdownCh chan struct{}, port int, healthPath string, timeout time.Duration) error {
	url := fmt.Sprintf("http://127.0.0.1:%d%s", port, healthPath)
	client := &http.Client{Timeout: 5 * time.Second}
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		select {
		case <-doneCh:
			return fmt.Errorf("process exited during startup")
		case <-shutdownCh:
			return fmt.Errorf("shutdown requested")
		default:
		}

		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return nil
			}
		}
		// Sleep in small increments so shutdown is responsive
		select {
		case <-shutdownCh:
			return fmt.Errorf("shutdown requested")
		case <-time.After(2 * time.Second):
		}
	}
	return fmt.Errorf("timeout after %v", timeout)
}

// ── Idle sweeper ─────────────────────────────

func (bm *BackendManager) idleSweeper() {
	interval := time.Duration(bm.cfg.Server.SweepIntervalSeconds) * time.Second
	idleLimit := time.Duration(bm.cfg.Server.IdleTimeoutMinutes) * time.Minute
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			bm.sweepIdle(idleLimit)
		case <-bm.sweeperStop:
			return
		}
	}
}

func (bm *BackendManager) sweepIdle(idleLimit time.Duration) {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	now := time.Now()
	for id, b := range bm.backends {
		if now.Sub(b.lastRequest()) > idleLimit {
			bm.stopLocked(id, b)
		}
	}
}

// ── Backend accessors ────────────────────────

// LoadedModels returns IDs of all currently loaded backends.
func (bm *BackendManager) LoadedModels() []string {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	ids := make([]string, 0, len(bm.backends))
	for id := range bm.backends {
		ids = append(ids, id)
	}
	return ids
}

// LoadedModelsSet returns a set of loaded model IDs for fast lookup.
func (bm *BackendManager) LoadedModelsSet() map[string]bool {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	set := make(map[string]bool, len(bm.backends))
	for id := range bm.backends {
		set[id] = true
	}
	return set
}

// BaseURL returns the backend's base URL for proxying.
func (b *Backend) BaseURL() string {
	return fmt.Sprintf("http://127.0.0.1:%d", b.Port)
}

// ── Backend: last-request tracking ───────────

func (b *Backend) lastRequest() time.Time {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.lastReq
}

func (b *Backend) setLastRequest(t time.Time) {
	b.mu.Lock()
	b.lastReq = t
	b.mu.Unlock()
}

func (b *Backend) Touch() {
	b.setLastRequest(time.Now())
}

// ── Logging helpers ──────────────────────────

func (bm *BackendManager) logPipe(r io.ReadCloser, prefix string) {
	defer r.Close()
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)
	for scanner.Scan() {
		bm.logger.Printf("%s %s", prefix, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		bm.logger.Printf("%s log reader error: %v", prefix, err)
	}
}
