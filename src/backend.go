package main

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
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
	inFlight  int64 // atomic; number of active proxied requests
}

// InFlightAdd increments the in-flight request counter.
func (b *Backend) InFlightAdd() { atomic.AddInt64(&b.inFlight, 1) }

// InFlightDone decrements the in-flight request counter.
func (b *Backend) InFlightDone() { atomic.AddInt64(&b.inFlight, -1) }

// InFlight returns the current in-flight request count.
func (b *Backend) InFlight() int64 { return atomic.LoadInt64(&b.inFlight) }

// BackendManager manages all backend processes: lifecycle, health, and routing.
type BackendManager struct {
	cfg             *Config
	configPath      string
	vramCache       VRAMCache
	vramCacheP      string
	backends        map[string]*Backend   // keyed by canonical model ID
	loading         map[string]*sync.Cond // model ID -> condition variable for in-progress loads
	mu              sync.RWMutex
	portAlloc       portAllocator
	logger          *CondLogger
	sweeperStop     chan struct{}
	sweeperOnce     sync.Once
	shutdownCh      chan struct{} // closed when Stop() is called, aborts waitHealth
	capCh           chan struct{} // closed+recreated to signal capacity queue waiters
	queueDepth      int64         // atomic; number of goroutines waiting in the capacity queue
	draining        atomic.Bool   // when true, reject new load requests
	profiling       atomic.Bool   // when true, profiling is in progress (prevents concurrent profiling)
	skipAutoProfile atomic.Bool   // when true, startBackendLocked skips auto-profiling
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

// alloc returns a free port from the range. It probes each candidate by
// attempting a TCP bind; ports already in use by other processes (e.g. a
// running llama-switch service) are skipped. Caller must hold bm.mu.
func (pa *portAllocator) alloc() (int, error) {
	for p := pa.start; p <= pa.end; p++ {
		if pa.used[p] {
			continue
		}
		ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", p))
		if err != nil {
			continue // port in use by another process
		}
		ln.Close()
		pa.used[p] = true
		return p, nil
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
	bm.capCh = make(chan struct{})
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

// ── Capacity queue ──────────────────────────

// signalCapacityLocked closes and recreates the capacity signal channel.
// This wakes all goroutines waiting in the capacity queue so they can
// re-check whether capacity is now available.
// Caller must hold bm.mu.
func (bm *BackendManager) signalCapacityLocked() {
	close(bm.capCh)
	bm.capCh = make(chan struct{})
}

// SignalCapacity wakes goroutines waiting in the capacity queue.
// Called after a request completes (InFlightDone) or after a backend
// is stopped. Safe to call without holding bm.mu.
func (bm *BackendManager) SignalCapacity() {
	bm.mu.Lock()
	bm.signalCapacityLocked()
	bm.mu.Unlock()
}

// ── Drain mode ───────────────────────────────

// StartDraining enters drain mode. New backend loads are rejected.
// Existing in-flight requests are allowed to complete.
func (bm *BackendManager) StartDraining() { bm.draining.Store(true) }

// StopDraining exits drain mode. Normal operation resumes.
func (bm *BackendManager) StopDraining() { bm.draining.Store(false) }

// IsDraining returns whether the manager is in drain mode.
func (bm *BackendManager) IsDraining() bool { return bm.draining.Load() }

// ── Idle waiting ────────────────────────────

// WaitIdle blocks until all backends have zero in-flight requests,
// or the timeout expires. Returns nil if idle, error on timeout.
func (bm *BackendManager) WaitIdle(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		bm.mu.RLock()
		busy := false
		count := len(bm.backends)
		for _, b := range bm.backends {
			if b.InFlight() > 0 {
				busy = true
				break
			}
		}
		bm.mu.RUnlock()

		if !busy {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out waiting for %d backends to become idle", count)
		}
		<-ticker.C
	}
}

// ── Bulk eviction ────────────────────────────

// evictAllLocked stops all backends unconditionally. Caller must hold bm.mu.
// Use only after WaitIdle confirms no active requests.
func (bm *BackendManager) evictAllLocked() {
	for id, b := range bm.backends {
		bm.stopLocked(id, b)
	}
}

// ── Profiling ────────────────────────────────

// ProfileProgress is a callback for profiling progress updates.
type ProfileProgress func(format string, args ...any)

// ProfileResult holds the outcome of profiling a single model.
type ProfileResult struct {
	ModelID string
	Vram    int
	GPUVRAM []int
	Err     error
	Cached  bool
}

// ProfileModels profiles each model one at a time. It drains active
// requests, evicts all backends, then loads/measures/unloads each model.
// The progress callback is called for each step. If a model has a valid
// cache entry, it is skipped (Cached=true) unless force is true.
func (bm *BackendManager) ProfileModels(force bool, progress ProfileProgress) ([]ProfileResult, error) {
	results := make([]ProfileResult, 0, len(bm.cfg.Models))

	// Prevent concurrent profiling
	if !bm.profiling.CompareAndSwap(false, true) {
		return nil, fmt.Errorf("profiling already in progress")
	}
	defer bm.profiling.Store(false)

	// Enter drain mode: reject new requests
	bm.StartDraining()
	defer bm.StopDraining()

	// Wait for active requests to finish
	progress("Draining active requests...")
	drainTimeout := time.Duration(bm.cfg.Server.ProfileDrainSeconds) * time.Second
	if err := bm.WaitIdle(drainTimeout); err != nil {
		progress("Drain timeout — force-evicting (%v)", err)
	}

	// Evict all backends
	bm.mu.Lock()
	bm.evictAllLocked()
	bm.mu.Unlock()
	progress("All backends evicted. Starting profiling...")

	// Copy cache under lock to avoid data race with startBackendLocked
	bm.mu.RLock()
	cache := make(VRAMCache, len(bm.vramCache))
	for k, v := range bm.vramCache {
		cache[k] = v
	}
	bm.mu.RUnlock()

	// Profile each model
	for i := range bm.cfg.Models {
		mc := &bm.cfg.Models[i]

		snap := mc.Snapshot(bm.cfg.Backend.CommonArgs)
		if !force {
			if entry, ok := cache[mc.ID]; ok && entry.Vram > 0 && entry.Config.Equal(snap) {
				progress("[%s] CACHE HIT — %d MB (delete entry to re-profile)", mc.ID, entry.Vram)
				results = append(results, ProfileResult{
					ModelID: mc.ID, Vram: entry.Vram, GPUVRAM: entry.GPUVRAM, Cached: true,
				})
				continue
			}
		}

		progress("[%s] profiling...", mc.ID)

		used, gpuVRAM, err := bm.profileSingleModel(mc)
		if err != nil {
			progress("[%s] FAILED: %v", mc.ID, err)
			results = append(results, ProfileResult{ModelID: mc.ID, Err: err})
			continue
		}

		cache[mc.ID] = CacheEntry{Vram: used, GPUVRAM: gpuVRAM, Config: snap}

		// Write cache back under lock
		bm.mu.Lock()
		bm.vramCache = cache
		bm.mu.Unlock()
		_ = SaveVRAMCache(bm.vramCacheP, cache)

		gpuStr := ""
		if len(gpuVRAM) > 0 {
			parts := make([]string, len(gpuVRAM))
			for gi, v := range gpuVRAM {
				parts[gi] = fmt.Sprintf("GPU%d:%dMB", gi, v)
			}
			gpuStr = " (" + strings.Join(parts, ", ") + ")"
		}
		progress("[%s] %d MB%s — saved", mc.ID, used, gpuStr)
		results = append(results, ProfileResult{
			ModelID: mc.ID, Vram: used, GPUVRAM: gpuVRAM,
		})
	}

	progress("Done. VRAM estimates saved to %s", bm.vramCacheP)
	return results, nil
}

// profileSingleModel loads one model, measures VRAM, then unloads.
func (bm *BackendManager) profileSingleModel(mc *ModelConfig) (int, []int, error) {
	before, err := queryVRAM(bm.cfg.Backend.NvidiaSMI)
	if err != nil {
		return 0, nil, fmt.Errorf("baseline VRAM query: %w", err)
	}
	baselineUsed := before.Used

	// Suppress auto-profiling during explicit measurement
	bm.skipAutoProfile.Store(true)
	defer bm.skipAutoProfile.Store(false)
	bm.mu.Lock()
	_, err = bm.startBackendLocked(mc)
	bm.mu.Unlock()
	if err != nil {
		return 0, nil, fmt.Errorf("load failed: %w", err)
	}

	time.Sleep(3 * time.Second)

	after, err := queryVRAM(bm.cfg.Backend.NvidiaSMI)
	if err != nil {
		_ = bm.StopModel(mc.ID)
		return 0, nil, fmt.Errorf("post-load VRAM query: %w", err)
	}

	used := after.Used - baselineUsed
	if used < 0 {
		used = after.Used
	}
	gpuVRAM := computePerGPUDelta(before, after)

	_ = bm.StopModel(mc.ID)

	if verr := validateVRAMMeasurement(mc, used); verr != nil {
		return 0, nil, verr
	}

	return used, gpuVRAM, nil
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
			b.InFlightAdd()
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
			b.InFlightAdd()
			bm.mu.Unlock()
			return b, nil
		}

		// Reject new loads during drain mode (profiling, maintenance)
		if bm.IsDraining() {
			bm.mu.Unlock()
			return nil, fmt.Errorf("service is draining (profiling or maintenance in progress)")
		}

		// Wait for capacity: if we can't evict right now (all backends
		// in-flight), block on capCh until a request completes and frees
		// capacity. Already-loaded-model requests hit the fast path
		// above and never reach here.
		//
		// IMPORTANT: the loading map registration happens AFTER this
		// wait, not before. If we registered first, len(loading) would
		// count against MaxModels in ensureCapacityLocked, causing a
		// deadlock when two goroutines wait for different models
		// simultaneously (both count each other's loading entry).
		queueTimeout := time.Duration(bm.cfg.Server.QueueTimeoutSeconds) * time.Second
		deadline := time.Now().Add(queueTimeout)

		for {
			// Check queue depth limit
			currentDepth := atomic.LoadInt64(&bm.queueDepth)
			if int(currentDepth) >= bm.cfg.Server.QueueMaxDepth {
				bm.mu.Unlock()
				return nil, fmt.Errorf("queue full (%d pending requests)", currentDepth)
			}

			err := bm.ensureCapacityLocked(mc)
			if err == nil {
				break // capacity available
			}

			if time.Now().After(deadline) {
				bm.mu.Unlock()
				return nil, fmt.Errorf("queue timeout after %v waiting for capacity: %w", queueTimeout, err)
			}

			// Increment queue depth, snapshot channel, release lock, wait
			atomic.AddInt64(&bm.queueDepth, 1)
			capCh := bm.capCh
			bm.mu.Unlock()

			select {
			case <-capCh:
			case <-time.After(time.Until(deadline)):
			}

			atomic.AddInt64(&bm.queueDepth, -1)
			bm.mu.Lock()

			// Re-check fast path: another goroutine may have loaded
			// this model while we were waiting for capacity.
			if b, ok := bm.backends[canonicalID]; ok {
				b.InFlightAdd()
				bm.mu.Unlock()
				return b, nil
			}

			// Re-check drain mode after waking
			if bm.IsDraining() {
				bm.mu.Unlock()
				return nil, fmt.Errorf("service is draining (profiling or maintenance in progress)")
			}
		}

		// Now register as loading so concurrent requests for the same
		// model wait on loadCond instead of trying to load it again.
		loadCond := sync.NewCond(&bm.mu)
		bm.loading[canonicalID] = loadCond
		deleteLoading := func() {
			delete(bm.loading, canonicalID)
			loadCond.Broadcast()
			bm.signalCapacityLocked()
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
		b.InFlightAdd() // protect against eviction until handleProxy's defer takes over
		bm.mu.Unlock()
		return b, nil
	}
}

// ensureCapacityLocked checks model count limits and VRAM, evicting
// LRU models as needed. Counts both loaded backends and in-flight loads
// toward the MaxModels budget to prevent soft limit violations.
// Caller must hold bm.mu.
func (bm *BackendManager) ensureCapacityLocked(mc *ModelConfig) error {
	// 1. Count limit: evict if loaded + loading would exceed MaxModels.
	// Note: the caller hasn't registered in loading yet (it's done after
	// capacity is secured), so we add 1 to account for the model being loaded.
	for len(bm.backends)+len(bm.loading)+1 > bm.cfg.Server.MaxModels {
		if err := bm.evictLRULocked(); err != nil {
			return fmt.Errorf("cannot make room for %s: %w", mc.ID, err)
		}
	}

	// 2. VRAM limit: evict LRU until estimated VRAM fits on each target GPU.
	// Account for VRAM that will be consumed by models currently loading
	// (they haven't hit nvidia-smi yet but will shortly).
	needed := vramEstimate(bm.vramCache, mc, bm.cfg.Backend.CommonArgs)
	if needed > 0 {
		// Compute pending VRAM from loading models so we don't admit
		// two big models simultaneously.
		pendingPerGPU := make([]int, 0)
		for loadID := range bm.loading {
			lmc := bm.cfg.FindModel(loadID)
			if lmc == nil {
				continue
			}
			pg := vramEstimatePerGPU(bm.vramCache, lmc, bm.cfg.Backend.CommonArgs)
			if pg == nil {
				// No per-GPU data; use aggregate as fallback
				agg := vramEstimate(bm.vramCache, lmc, bm.cfg.Backend.CommonArgs)
				if agg > 0 {
					if len(pendingPerGPU) == 0 {
						pendingPerGPU = []int{0}
					}
					pendingPerGPU[0] += agg
				}
				continue
			}
			for len(pendingPerGPU) < len(pg) {
				pendingPerGPU = append(pendingPerGPU, 0)
			}
			for i, v := range pg {
				pendingPerGPU[i] += v
			}
		}
		for {
			stats, err := queryVRAM(bm.cfg.Backend.NvidiaSMI)
			if err != nil {
				break // can't query VRAM, proceed optimistically
			}
			// Adjust free VRAM by subtracting pending loads
			if len(pendingPerGPU) > 0 {
				adj := *stats
				adj.GPUs = make([]GPUInfo, len(stats.GPUs))
				for i, g := range stats.GPUs {
					pending := 0
					if i < len(pendingPerGPU) {
						pending = pendingPerGPU[i]
					}
					used := g.Used + pending
					if used > g.Total {
						used = g.Total
					}
					adj.GPUs[i] = GPUInfo{Total: g.Total, Used: used, Free: g.Total - used}
				}
				adj.Used = 0
				adj.Free = 0
				for _, g := range adj.GPUs {
					adj.Used += g.Used
					adj.Free += g.Free
				}
				stats = &adj
			}
			headroom := 1024 // 1 GB safety margin (applied to primary GPU only)
			if vramFitsPerGPU(stats, mc, bm.vramCache, bm.cfg.Backend.CommonArgs, headroom) {
				break
			}
			if len(bm.backends) == 0 && len(bm.loading) == 0 {
				break // nothing left to evict and nothing loading; try anyway
			}
			if len(bm.backends) == 0 {
				// Can't evict, but a model is loading — its VRAM isn't
				// reflected in nvidia-smi yet. Return an error so the
				// caller blocks in the capacity queue until it finishes.
				return fmt.Errorf("waiting for %d loading model(s) to free VRAM", len(bm.loading))
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

// evictLRULocked unloads the least-recently-used backend that has no
// active requests. If all backends are busy, returns an error.
// Caller must hold bm.mu.
func (bm *BackendManager) evictLRULocked() error {
	var oldestID string
	var oldestTime time.Time

	for id, b := range bm.backends {
		if b.InFlight() == 0 {
			lastReq := b.lastRequest()
			if oldestID == "" || lastReq.Before(oldestTime) {
				oldestID = id
				oldestTime = lastReq
			}
		}
	}

	if oldestID == "" {
		return fmt.Errorf("no evictable backends (all have active requests)")
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
			bm.signalCapacityLocked()
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
	// Skip during explicit profiling (skipAutoProfile flag).
	snap := mc.Snapshot(bm.cfg.Backend.CommonArgs)
	if !bm.skipAutoProfile.Load() {
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
	bm.signalCapacityLocked()
}

// StopModel unloads a specific model by ID. Resolves the canonical ID
// from any accepted name (id or display name). Returns an error if the
// backend has active in-flight requests.
func (bm *BackendManager) StopModel(id string) error {
	mc := bm.cfg.FindModel(id)
	if mc == nil {
		return fmt.Errorf("unknown model: %s", id)
	}
	canonicalID := mc.ID

	bm.mu.Lock()
	defer bm.mu.Unlock()
	b, ok := bm.backends[canonicalID]
	if !ok {
		return fmt.Errorf("model %s is not loaded", id)
	}
	if b.InFlight() > 0 {
		return fmt.Errorf("model %s has %d active request(s), cannot unload", id, b.InFlight())
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
	if bm.IsDraining() {
		return
	}
	bm.mu.Lock()
	defer bm.mu.Unlock()

	now := time.Now()
	for id, b := range bm.backends {
		if b.InFlight() == 0 {
			if now.Sub(b.lastRequest()) > idleLimit {
				bm.stopLocked(id, b)
			}
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
