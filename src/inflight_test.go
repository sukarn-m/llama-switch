package main

import (
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBackendInFlightTracking(t *testing.T) {
	b := &Backend{ID: "test"}

	if b.InFlight() != 0 {
		t.Fatalf("expected 0 in-flight, got %d", b.InFlight())
	}

	// Use a barrier to ensure all goroutines are simultaneously in-flight
	barrier := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			b.InFlightAdd()
			defer b.InFlightDone()
			<-barrier // wait until all goroutines have incremented
		}()
	}

	// Wait until all 5 goroutines have incremented
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if b.InFlight() == 5 {
			break
		}
		time.Sleep(2 * time.Millisecond)
	}
	if count := b.InFlight(); count != 5 {
		t.Fatalf("expected 5 in-flight, got %d", count)
	}

	// Release all goroutines
	close(barrier)
	wg.Wait()
	if b.InFlight() != 0 {
		t.Fatalf("expected 0 in-flight after completion, got %d", b.InFlight())
	}
}

func TestEvictSkipsInFlightBackends(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{MaxModels: 2, BackendPortStart: 18201, BackendPortEnd: 18299,
			QueueMaxDepth: 64, QueueTimeoutSeconds: 5},
		Models: []ModelConfig{
			{ID: "a", Name: "a", Path: "/dev/null"},
			{ID: "b", Name: "b", Path: "/dev/null"},
		},
	}
	bm := NewBackendManager(cfg, "", NewLogger())

	bm.mu.Lock()
	bmA := &Backend{ID: "a", Port: 18201}
	bmB := &Backend{ID: "b", Port: 18202}
	bmA.setLastRequest(time.Now().Add(-10 * time.Minute))
	bmB.setLastRequest(time.Now())
	bm.backends["a"] = bmA
	bm.backends["b"] = bmB

	bmA.InFlightAdd()
	bmB.InFlightAdd()

	err := bm.evictLRULocked()
	if err == nil {
		t.Fatal("expected error when all backends are in-flight")
	}

	// Release A (oldest), now eviction should pick A
	bmA.InFlightDone()
	bmB.InFlightDone()
	err = bm.evictLRULocked()
	if err != nil {
		t.Fatalf("expected successful eviction, got: %v", err)
	}

	_, aExists := bm.backends["a"]
	bm.mu.Unlock()
	if aExists {
		t.Error("backend A should have been evicted")
	}
}

func TestSweepIdleSkipsInFlight(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{MaxModels: 1, BackendPortStart: 18201, BackendPortEnd: 18299,
			QueueMaxDepth: 64, QueueTimeoutSeconds: 5},
	}
	bm := NewBackendManager(cfg, "", NewLogger())

	b := &Backend{ID: "old", Port: 18201}
	b.setLastRequest(time.Now().Add(-2 * time.Hour))
	bm.mu.Lock()
	bm.backends["old"] = b
	bm.mu.Unlock()

	b.InFlightAdd()
	bm.sweepIdle(time.Minute)
	bm.mu.RLock()
	_, exists := bm.backends["old"]
	bm.mu.RUnlock()
	if !exists {
		t.Fatal("in-flight backend should not be swept")
	}

	b.InFlightDone()
	bm.sweepIdle(time.Minute)
	bm.mu.RLock()
	_, exists = bm.backends["old"]
	bm.mu.RUnlock()
	if exists {
		t.Fatal("idle backend should be swept")
	}
}

func TestStopModelRefusesInFlight(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{MaxModels: 1, BackendPortStart: 18201, BackendPortEnd: 18299,
			QueueMaxDepth: 64, QueueTimeoutSeconds: 5},
		Models: []ModelConfig{{ID: "a", Name: "a", Path: "/dev/null"}},
	}
	bm := NewBackendManager(cfg, "", NewLogger())

	b := &Backend{ID: "a", Port: 18201}
	bm.mu.Lock()
	bm.backends["a"] = b
	bm.mu.Unlock()

	b.InFlightAdd()

	err := bm.StopModel("a")
	if err == nil {
		t.Fatal("expected error when stopping in-flight backend")
	}

	b.InFlightDone()
	err = bm.StopModel("a")
	if err != nil {
		t.Fatalf("expected success after in-flight completes, got: %v", err)
	}
}

func TestCapacityQueueBlocksAndProceeds(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			MaxModels:           1,
			BackendPortStart:    18201,
			BackendPortEnd:      18299,
			QueueMaxDepth:       64,
			QueueTimeoutSeconds: 5,
		},
		Models: []ModelConfig{
			{ID: "a", Name: "a", Path: "/dev/null"},
			{ID: "b", Name: "b", Path: "/dev/null"},
		},
	}
	bm := NewBackendManager(cfg, "", NewLogger())

	bmA := &Backend{ID: "a", Port: 18201}
	bmA.setLastRequest(time.Now())
	bmA.InFlightAdd()
	bm.mu.Lock()
	bm.backends["a"] = bmA
	bm.mu.Unlock()

	done := make(chan error, 1)
	go func() {
		_, err := bm.EnsureLoaded("b")
		done <- err
	}()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if atomic.LoadInt64(&bm.queueDepth) == 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if depth := atomic.LoadInt64(&bm.queueDepth); depth != 1 {
		t.Fatalf("expected queue depth 1, got %d", depth)
	}

	bmA.InFlightDone()
	bm.SignalCapacity()

	select {
	case err := <-done:
		if err == nil {
			return
		}
		if strings.Contains(err.Error(), "queue timeout") {
			t.Fatal("request should have unblocked, not timed out")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("EnsureLoaded did not unblock after capacity freed")
	}
}

func TestCapacityQueueTimeout(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			MaxModels:           1,
			BackendPortStart:    18201,
			BackendPortEnd:      18299,
			QueueMaxDepth:       64,
			QueueTimeoutSeconds: 1,
		},
		Models: []ModelConfig{
			{ID: "a", Name: "a", Path: "/dev/null"},
			{ID: "b", Name: "b", Path: "/dev/null"},
		},
	}
	bm := NewBackendManager(cfg, "", NewLogger())

	bmA := &Backend{ID: "a", Port: 18201}
	bmA.setLastRequest(time.Now())
	bmA.InFlightAdd()
	bm.mu.Lock()
	bm.backends["a"] = bmA
	bm.mu.Unlock()

	start := time.Now()
	_, err := bm.EnsureLoaded("b")
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected timeout error")
	}
	if elapsed < 900*time.Millisecond || elapsed > 2*time.Second {
		t.Fatalf("expected ~1s timeout, got %v", elapsed)
	}
	if !strings.Contains(err.Error(), "queue timeout") {
		t.Fatalf("expected queue timeout error, got: %v", err)
	}
}

func TestCapacityQueueMaxDepth(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			MaxModels:           1,
			BackendPortStart:    18201,
			BackendPortEnd:      18299,
			QueueMaxDepth:       2,
			QueueTimeoutSeconds: 10,
		},
		Models: []ModelConfig{
			{ID: "a", Name: "a", Path: "/dev/null"},
			{ID: "b", Name: "b", Path: "/dev/null"},
			{ID: "c", Name: "c", Path: "/dev/null"},
			{ID: "d", Name: "d", Path: "/dev/null"},
		},
	}
	bm := NewBackendManager(cfg, "", NewLogger())

	bmA := &Backend{ID: "a", Port: 18201}
	bmA.setLastRequest(time.Now())
	bmA.InFlightAdd()
	bm.mu.Lock()
	bm.backends["a"] = bmA
	bm.mu.Unlock()

	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		modelID := string(rune('b' + i))
		go func(mid string) {
			defer wg.Done()
			bm.EnsureLoaded(mid)
		}(modelID)
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if atomic.LoadInt64(&bm.queueDepth) == 2 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	_, err := bm.EnsureLoaded("d")
	if err == nil {
		t.Fatal("expected queue full error")
	}
	if !strings.Contains(err.Error(), "queue full") {
		t.Fatalf("expected queue full error, got: %v", err)
	}

	bmA.InFlightDone()
	bm.SignalCapacity()
	wg.Wait()
}

func TestAlreadyLoadedBypassesQueue(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			MaxModels:           1,
			BackendPortStart:    18201,
			BackendPortEnd:      18299,
			QueueMaxDepth:       1,
			QueueTimeoutSeconds: 10,
		},
		Models: []ModelConfig{
			{ID: "a", Name: "a", Path: "/dev/null"},
			{ID: "b", Name: "b", Path: "/dev/null"},
		},
	}
	bm := NewBackendManager(cfg, "", NewLogger())

	bmA := &Backend{ID: "a", Port: 18201}
	bmA.setLastRequest(time.Now())
	bm.mu.Lock()
	bm.backends["a"] = bmA
	bm.mu.Unlock()

	start := time.Now()
	b, err := bm.EnsureLoaded("a")
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("expected success, got: %v", err)
	}
	if b == nil {
		t.Fatal("expected non-nil backend")
	}
	if elapsed > 10*time.Millisecond {
		t.Fatalf("expected instant return, took %v", elapsed)
	}
}

func TestDrainMode(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{MaxModels: 1, BackendPortStart: 18201, BackendPortEnd: 18299,
			QueueMaxDepth: 64, QueueTimeoutSeconds: 5},
		Models: []ModelConfig{{ID: "a", Name: "a", Path: "/dev/null"}},
	}
	bm := NewBackendManager(cfg, "", NewLogger())

	bm.StartDraining()
	if !bm.IsDraining() {
		t.Fatal("expected draining to be true")
	}

	_, err := bm.EnsureLoaded("a")
	if err == nil {
		t.Fatal("expected EnsureLoaded to fail during drain")
	}

	bm.StopDraining()
	if bm.IsDraining() {
		t.Fatal("expected draining to be false after StopDraining")
	}
}

func TestWaitIdle(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{MaxModels: 1, BackendPortStart: 18201, BackendPortEnd: 18299,
			QueueMaxDepth: 64, QueueTimeoutSeconds: 5},
	}
	bm := NewBackendManager(cfg, "", NewLogger())

	b := &Backend{ID: "a", Port: 18201}
	bm.mu.Lock()
	bm.backends["a"] = b
	bm.mu.Unlock()

	b.InFlightAdd()

	err := bm.WaitIdle(100 * time.Millisecond)
	if err == nil {
		t.Fatal("expected timeout when requests are in-flight")
	}

	go func() {
		time.Sleep(50 * time.Millisecond)
		b.InFlightDone()
	}()

	err = bm.WaitIdle(2 * time.Second)
	if err != nil {
		t.Fatalf("expected success, got: %v", err)
	}
}

func TestEvictAll(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{MaxModels: 5, BackendPortStart: 18201, BackendPortEnd: 18299,
			QueueMaxDepth: 64, QueueTimeoutSeconds: 5},
	}
	bm := NewBackendManager(cfg, "", NewLogger())

	bm.mu.Lock()
	bm.backends["a"] = &Backend{ID: "a", Port: 18201}
	bm.backends["b"] = &Backend{ID: "b", Port: 18202}
	bm.backends["c"] = &Backend{ID: "c", Port: 18203}
	bm.evictAllLocked()
	count := len(bm.backends)
	bm.mu.Unlock()

	if count != 0 {
		t.Fatalf("expected 0 backends after evictAllLocked, got %d", count)
	}
}
