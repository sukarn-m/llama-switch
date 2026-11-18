package main

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// Helpers for the VRAM settle-window tests. A backend's VRAM only becomes
// visible to nvidia-smi some seconds after its health check passes (see
// vramSettleWindow); these tests pin the admission and eviction behaviour
// that keeps the window from overcommitting the GPU.

// settleTestConfig builds a Config pointing at a fake nvidia-smi.
func settleTestConfig(nvidiaSMI string, maxModels int) *Config {
	return &Config{
		Server: ServerConfig{
			MaxModels:           maxModels,
			BackendPortStart:    18201,
			BackendPortEnd:      18299,
			QueueMaxDepth:       64,
			QueueTimeoutSeconds: 5,
		},
		Backend: BackendConfig{NvidiaSMI: nvidiaSMI},
		Models: []ModelConfig{
			{ID: "a", Name: "a", Path: "/dev/null", Devices: []string{"CUDA0"}},
			{ID: "b", Name: "b", Path: "/dev/null", Devices: []string{"CUDA0"}},
			{ID: "c", Name: "c", Path: "/dev/null", Devices: []string{"CUDA0"}},
		},
	}
}

// settleTestSMI writes a fake nvidia-smi reporting one 24000 MB GPU whose
// used MB can be changed at any time by rewriting usedFile.
func settleTestSMI(t *testing.T, used int) (smi string, usedFile string) {
	t.Helper()
	dir := t.TempDir()
	usedFile = dir + "/used"
	if err := os.WriteFile(usedFile, []byte(fmt.Sprintf("%d", used)), 0o644); err != nil {
		t.Fatal(err)
	}
	smi = dir + "/fake-smi"
	script := "#!/bin/sh\nu=$(cat " + usedFile + ")\necho \"24000, $u, $((24000-u))\"\n"
	if err := os.WriteFile(smi, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return smi, usedFile
}

// settleTestCache gives model id a profiled VRAM estimate of 15000 MB,
// all of it on GPU 0.
func settleTestCache(bm *BackendManager, cfg *Config, id string) {
	mc := cfg.FindModel(id)
	bm.vramCache[id] = CacheEntry{Vram: 15000, GPUVRAM: []int{15000}, Config: mc.Snapshot(cfg.Backend.CommonArgs)}
}

func TestPendingVRAMPerGPU(t *testing.T) {
	cfg := settleTestConfig("", 3)
	bm := NewBackendManager(cfg, "", NewLogger())
	for _, id := range []string{"a", "b", "c"} {
		settleTestCache(bm, cfg, id)
	}

	bm.mu.Lock()
	defer bm.mu.Unlock()

	// A freshly started backend is inside the settle window: its VRAM is
	// spoken for but may not be visible to nvidia-smi yet.
	a := &Backend{ID: "a", Port: 18201, startedAt: time.Now()}
	bm.backends["a"] = a
	if p := bm.pendingVRAMPerGPU(""); len(p) != 1 || p[0] != 15000 {
		t.Fatalf("young backend should count as pending 15000 on GPU 0, got %v", p)
	}

	// Past the window, nvidia-smi reflects it: nothing pending.
	a.startedAt = time.Now().Add(-time.Minute)
	if p := bm.pendingVRAMPerGPU(""); len(p) != 0 {
		t.Fatalf("settled backend must not count as pending, got %v", p)
	}

	// Reserved but not yet running: nvidia-smi cannot see it either.
	bm.reserved["b"] = true
	if p := bm.pendingVRAMPerGPU(""); len(p) != 1 || p[0] != 15000 {
		t.Fatalf("reserved-not-running model should count as pending 15000, got %v", p)
	}

	// Reserved AND young-in-map must be counted once, not twice: the
	// reserved loop skips ids already in the map, the young loop picks
	// them up regardless of reservation state.
	delete(bm.reserved, "b")
	bm.reserved["a"] = true
	a.startedAt = time.Now()
	if p := bm.pendingVRAMPerGPU(""); len(p) != 1 || p[0] != 15000 {
		t.Fatalf("reserved+young backend must be counted once, got %v", p)
	}
	delete(bm.reserved, "a")

	// excludeID is skipped: b is both reserved and young, but excluded,
	// so only a's VRAM remains.
	bm.reserved["b"] = true
	if p := bm.pendingVRAMPerGPU("b"); len(p) != 1 || p[0] != 15000 {
		t.Fatalf("excludeID must be skipped, got %v", p)
	}
}

// TestEvictSkipsYoungBackends pins the eviction guard: a settled backend
// is evictable even if the young one holds the older lastRequest.
func TestEvictSkipsYoungBackends(t *testing.T) {
	cfg := settleTestConfig("", 2)
	bm := NewBackendManager(cfg, "", NewLogger())

	bm.mu.Lock()
	young := &Backend{ID: "young", Port: 18201, startedAt: time.Now()}
	young.setLastRequest(time.Now().Add(-time.Hour)) // oldest request, but young backend
	old := &Backend{ID: "old", Port: 18202, startedAt: time.Now().Add(-time.Minute)}
	old.setLastRequest(time.Now())
	bm.backends["young"] = young
	bm.backends["old"] = old
	err := bm.evictLRULocked("test")
	_, youngLeft := bm.backends["young"]
	_, oldLeft := bm.backends["old"]
	bm.mu.Unlock()

	if err != nil {
		t.Fatalf("expected successful eviction, got: %v", err)
	}
	if !youngLeft || oldLeft {
		t.Fatal("eviction must pick the settled backend, not the young one")
	}
}

// TestEvictAllYoungErrors: if every backend is inside the settle window,
// nothing may be evicted — the freshly started backends are about to serve
// the requests that triggered their loads.
func TestEvictAllYoungErrors(t *testing.T) {
	cfg := settleTestConfig("", 2)
	bm := NewBackendManager(cfg, "", NewLogger())

	bm.mu.Lock()
	bm.backends["x"] = &Backend{ID: "x", Port: 18201, startedAt: time.Now()}
	bm.backends["y"] = &Backend{ID: "y", Port: 18202, startedAt: time.Now()}
	err := bm.evictLRULocked("test")
	n := len(bm.backends)
	bm.mu.Unlock()

	if err == nil {
		t.Fatal("expected error when all backends are young")
	}
	if n != 2 {
		t.Fatalf("young backends must survive eviction attempts, %d left", n)
	}
}

// TestAdmissionDefersToUnsettledBackend reproduces the production race
// (2026-11-13): nemotron became healthy, its 18 GB were not yet visible to
// nvidia-smi, and lfm3 was admitted next to it. The loser's llama-server
// silently split itself between GPU and CPU and ran at a fraction of its
// normal speed. Here backend "a" (15000 MB) is young and nvidia-smi does
// not yet show its VRAM (2000 MB used). Admitting "b" (15000 MB + 1 GB
// headroom) must fail while "a" is unsettled, and succeed — evicting "a" —
// once "a" has settled and the GPU is genuinely full.
func TestAdmissionDefersToUnsettledBackend(t *testing.T) {
	smi, usedFile := settleTestSMI(t, 2000)
	cfg := settleTestConfig(smi, 3)
	bm := NewBackendManager(cfg, "", NewLogger())
	for _, id := range []string{"a", "b"} {
		settleTestCache(bm, cfg, id)
	}

	a := &Backend{ID: "a", Port: 18201, startedAt: time.Now()}
	a.setLastRequest(time.Now())
	bm.mu.Lock()
	bm.backends["a"] = a
	bm.mu.Unlock()

	// nvidia-smi reports 22000 MB free — enough for b on its own. Without
	// the settle window, b would be admitted and overcommit the GPU.
	bm.mu.Lock()
	err := bm.ensureCapacityLocked(cfg.FindModel("b"))
	_, aStill := bm.backends["a"]
	bm.mu.Unlock()

	if err == nil {
		t.Fatal("b must not be admitted while a's VRAM is unsettled (overcommit race)")
	}
	if !aStill {
		t.Fatal("young backend a must not be evicted to admit b")
	}

	// Age a past the settle window and make the GPU genuinely full, so b
	// can only fit by evicting a.
	if err := os.WriteFile(usedFile, []byte("20000"), 0o644); err != nil {
		t.Fatal(err)
	}
	bm.mu.Lock()
	a.startedAt = time.Now().Add(-time.Minute)
	err = bm.ensureCapacityLocked(cfg.FindModel("b"))
	_, aStill = bm.backends["a"]
	bm.mu.Unlock()

	if err != nil {
		t.Fatalf("expected b admitted once a settled, got: %v", err)
	}
	if aStill {
		t.Fatal("settled backend a should have been evicted for b")
	}
}

// A request for another model that arrives while the only loaded backend is
// still inside its settle window cannot evict it yet. Nothing signals the
// capacity queue when the window ends, so the waiter must wake up on its own
// at that moment instead of sleeping until the queue timeout (observed
// 2026-11-14: a model switch 6 s after a load stalled for the full 600 s).
func TestCapacityQueueWakesWhenBackendSettles(t *testing.T) {
	cfg := settleTestConfig("", 1)
	cfg.Server.QueueTimeoutSeconds = 30
	bm := NewBackendManager(cfg, "", NewLogger())

	a := &Backend{ID: "a", Port: 18201, startedAt: time.Now().Add(-vramSettleWindow + time.Second)}
	a.setLastRequest(time.Now())
	bm.mu.Lock()
	bm.backends["a"] = a
	bm.mu.Unlock()

	start := time.Now()
	done := make(chan error, 1)
	go func() {
		_, err := bm.EnsureLoaded("b")
		done <- err
	}()

	select {
	case err := <-done:
		if err != nil && strings.Contains(err.Error(), "queue timeout") {
			t.Fatalf("waiter timed out instead of waking at settle: %v", err)
		}
		if elapsed := time.Since(start); elapsed > 5*time.Second {
			t.Fatalf("waiter took %v; should wake ~1 s after a settles", elapsed)
		}
		bm.mu.RLock()
		_, aStill := bm.backends["a"]
		bm.mu.RUnlock()
		if aStill {
			t.Fatal("settled backend a should have been evicted for b")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("EnsureLoaded still waiting 10 s after a settled")
	}
}
