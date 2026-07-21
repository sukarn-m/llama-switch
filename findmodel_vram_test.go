package main

import (
	"testing"
)

// ── FindModel tests ──────────────────────────────────────────────────

func TestFindModelExactID(t *testing.T) {
	cfg := &Config{
		Models: []ModelConfig{
			{ID: "gpt-4o", Name: "GPT-4o"},
			{ID: "llama3-70b", Name: "Llama 3 70B"},
		},
	}
	m := cfg.FindModel("gpt-4o")
	if m == nil {
		t.Fatal("expected to find model by exact ID 'gpt-4o'")
	}
	if m.ID != "gpt-4o" {
		t.Errorf("got ID %q, want %q", m.ID, "gpt-4o")
	}
}

func TestFindModelCaseInsensitiveName(t *testing.T) {
	cfg := &Config{
		Models: []ModelConfig{
			{ID: "gpt-4o", Name: "GPT-4o"},
			{ID: "llama3-70b", Name: "Llama 3 70B"},
		},
	}
	// Various casings of the display name should all match
	for _, query := range []string{"gpt-4o", "GPT-4O", "gpt-4O"} {
		m := cfg.FindModel(query)
		if m == nil {
			t.Fatalf("expected case-insensitive match for %q", query)
		}
		if m.ID != "gpt-4o" {
			t.Errorf("for query %q got ID %q, want %q", query, m.ID, "gpt-4o")
		}
	}
}

func TestFindModelNoMatchReturnsNil(t *testing.T) {
	cfg := &Config{
		Models: []ModelConfig{
			{ID: "gpt-4o", Name: "GPT-4o"},
		},
	}
	if m := cfg.FindModel("nonexistent-model"); m != nil {
		t.Errorf("expected nil for nonexistent model, got %+v", m)
	}
}

func TestFindModelEmptyString(t *testing.T) {
	cfg := &Config{
		Models: []ModelConfig{
			{ID: "gpt-4o", Name: "GPT-4o"},
		},
	}
	if m := cfg.FindModel(""); m != nil {
		t.Errorf("expected nil for empty string, got %+v", m)
	}
}

func TestFindModelIDAndNameBothMatch(t *testing.T) {
	// When a model's ID equals its own Name, both paths should find it.
	cfg := &Config{
		Models: []ModelConfig{
			{ID: "model-a", Name: "model-a"},
		},
	}
	// Match by ID
	if m := cfg.FindModel("model-a"); m == nil || m.ID != "model-a" {
		t.Errorf("expected to find model-a by ID")
	}
}

func TestFindModelEmptyModels(t *testing.T) {
	cfg := &Config{Models: []ModelConfig{}}
	if m := cfg.FindModel("anything"); m != nil {
		t.Errorf("expected nil for empty models list, got %+v", m)
	}
}

func TestFindModelWithEmptyNameField(t *testing.T) {
	// A model with an empty Name should only match by ID, not by empty string.
	cfg := &Config{
		Models: []ModelConfig{
			{ID: "no-name-model", Name: ""},
		},
	}
	// Empty string should NOT match a model with empty Name
	if m := cfg.FindModel(""); m != nil {
		t.Errorf("expected nil for empty query, got %+v", m)
	}
	// But ID should still work
	if m := cfg.FindModel("no-name-model"); m == nil {
		t.Error("expected to find model by ID even with empty Name")
	}
}

// ── VRAM admission control tests ─────────────────────────────────────

func TestVRAMEstimateCacheHit(t *testing.T) {
	mc := &ModelConfig{ID: "model-1", Path: "/models/m1.gguf"}
	snap := mc.Snapshot(nil)
	cache := VRAMCache{
		"model-1": {Vram: 4096, Config: snap},
	}
	if got := vramEstimate(cache, mc, nil); got != 4096 {
		t.Errorf("vramEstimate: got %d, want 4096", got)
	}
}

func TestVRAMEstimateCacheMiss(t *testing.T) {
	mc := &ModelConfig{ID: "model-1", Path: "/models/m1.gguf"}
	cache := VRAMCache{}
	if got := vramEstimate(cache, mc, nil); got != 0 {
		t.Errorf("vramEstimate on miss: got %d, want 0", got)
	}
}

func TestVRAMEstimateStaleConfig(t *testing.T) {
	mc := &ModelConfig{ID: "model-1", Path: "/models/m1.gguf"}
	// Cached snapshot has different path
	staleSnap := ModelConfigSnapshot{Path: "/models/old.gguf"}
	cache := VRAMCache{
		"model-1": {Vram: 4096, Config: staleSnap},
	}
	if got := vramEstimate(cache, mc, nil); got != 0 {
		t.Errorf("vramEstimate with stale config: got %d, want 0", got)
	}
}

func TestVRAMEstimatePerGPUHit(t *testing.T) {
	mc := &ModelConfig{ID: "model-1", Path: "/models/m1.gguf"}
	snap := mc.Snapshot(nil)
	cache := VRAMCache{
		"model-1": {Vram: 8192, GPUVRAM: []int{4096, 4096}, Config: snap},
	}
	got := vramEstimatePerGPU(cache, mc, nil)
	if got == nil {
		t.Fatal("expected non-nil per-GPU slice")
	}
	if len(got) != 2 || got[0] != 4096 || got[1] != 4096 {
		t.Errorf("got %v, want [4096, 4096]", got)
	}
}

func TestVRAMEstimatePerGPUMissing(t *testing.T) {
	// Cache entry without GPUVRAM should return nil
	mc := &ModelConfig{ID: "model-1", Path: "/models/m1.gguf"}
	snap := mc.Snapshot(nil)
	cache := VRAMCache{
		"model-1": {Vram: 8192, Config: snap}, // no GPUVRAM
	}
	if got := vramEstimatePerGPU(cache, mc, nil); got != nil {
		t.Errorf("expected nil for entry without GPUVRAM, got %v", got)
	}
}

func TestVRAMEstimatePerGPUStale(t *testing.T) {
	mc := &ModelConfig{ID: "model-1", Path: "/models/m1.gguf"}
	staleSnap := ModelConfigSnapshot{Path: "/different.gguf"}
	cache := VRAMCache{
		"model-1": {Vram: 8192, GPUVRAM: []int{4096, 4096}, Config: staleSnap},
	}
	if got := vramEstimatePerGPU(cache, mc, nil); got != nil {
		t.Errorf("expected nil for stale cache, got %v", got)
	}
}

func TestVRAMFitsPerGPUWithNoPerGPUData(t *testing.T) {
	// Falls back to aggregate check
	mc := &ModelConfig{ID: "model-1", Path: "/models/m1.gguf"}
	snap := mc.Snapshot(nil)
	cache := VRAMCache{
		"model-1": {Vram: 4096, Config: snap}, // no GPUVRAM → aggregate fallback
	}
	stats := &VRAMStats{
		Total: 16384, Used: 8192, Free: 8192,
		GPUs: []GPUInfo{{Total: 16384, Used: 8192, Free: 8192}},
	}
	if !vramFitsPerGPU(stats, mc, cache, nil, 1024) {
		t.Error("expected model to fit (free 8192 > 4096+1024)")
	}
}

func TestVRAMFitsPerGPUWithNoPerGPUDataTooSmall(t *testing.T) {
	mc := &ModelConfig{ID: "model-1", Path: "/models/m1.gguf"}
	snap := mc.Snapshot(nil)
	cache := VRAMCache{
		"model-1": {Vram: 8192, Config: snap},
	}
	stats := &VRAMStats{
		Total: 16384, Used: 8192, Free: 8192,
		GPUs: []GPUInfo{{Total: 16384, Used: 8192, Free: 8192}},
	}
	// 8192 < 8192 + 1024 headroom
	if vramFitsPerGPU(stats, mc, cache, nil, 1024) {
		t.Error("expected model NOT to fit (free 8192 < 8192+1024)")
	}
}

func TestVRAMFitsPerGPUUnknownVRAM(t *testing.T) {
	// No cache entry at all → proceed optimistically
	mc := &ModelConfig{ID: "model-1", Path: "/models/m1.gguf"}
	cache := VRAMCache{}
	stats := &VRAMStats{
		GPUs: []GPUInfo{{Total: 16384, Used: 0, Free: 16384}},
	}
	if !vramFitsPerGPU(stats, mc, cache, nil, 1024) {
		t.Error("expected true for unknown VRAM (optimistic)")
	}
}

func TestVRAMFitsPerGPUSingleGPU(t *testing.T) {
	mc := &ModelConfig{
		ID:      "model-1",
		Path:    "/models/m1.gguf",
		Devices: []string{"CUDA0"},
	}
	snap := mc.Snapshot(nil)
	cache := VRAMCache{
		"model-1": {Vram: 4096, GPUVRAM: []int{4096}, Config: snap},
	}
	stats := &VRAMStats{
		GPUs: []GPUInfo{{Total: 16384, Used: 4096, Free: 12288}},
	}
	// needed 4096 + headroom 1024 = 5120, free = 12288 → fits
	if !vramFitsPerGPU(stats, mc, cache, nil, 1024) {
		t.Error("expected model to fit on single GPU")
	}
}

func TestVRAMFitsPerGPUSingleGPUNotEnough(t *testing.T) {
	mc := &ModelConfig{
		ID:      "model-1",
		Path:    "/models/m1.gguf",
		Devices: []string{"CUDA0"},
	}
	snap := mc.Snapshot(nil)
	cache := VRAMCache{
		"model-1": {Vram: 4096, GPUVRAM: []int{4096}, Config: snap},
	}
	stats := &VRAMStats{
		GPUs: []GPUInfo{{Total: 16384, Used: 12288, Free: 4096}},
	}
	// needed 4096 + headroom 1024 = 5120, free = 4096 → doesn't fit
	if vramFitsPerGPU(stats, mc, cache, nil, 1024) {
		t.Error("expected model NOT to fit (free 4096 < 4096+1024)")
	}
}

func TestVRAMFitsPerGPUMultiGPU(t *testing.T) {
	mc := &ModelConfig{
		ID:      "model-1",
		Path:    "/models/m1.gguf",
		Devices: []string{"CUDA0", "CUDA1"},
	}
	snap := mc.Snapshot(nil)
	cache := VRAMCache{
		"model-1": {Vram: 8192, GPUVRAM: []int{4096, 4096}, Config: snap},
	}
	stats := &VRAMStats{
		GPUs: []GPUInfo{
			{Total: 16384, Used: 8192, Free: 8192},
			{Total: 16384, Used: 8192, Free: 8192},
		},
	}
	// GPU0: needs 4096+1024 headroom = 5120, free 8192 → ok
	// GPU1: needs 4096, no headroom, free 8192 → ok
	if !vramFitsPerGPU(stats, mc, cache, nil, 1024) {
		t.Error("expected model to fit across two GPUs")
	}
}

func TestVRAMFitsPerGPUMultiGPUSecondFails(t *testing.T) {
	mc := &ModelConfig{
		ID:      "model-1",
		Path:    "/models/m1.gguf",
		Devices: []string{"CUDA0", "CUDA1"},
	}
	snap := mc.Snapshot(nil)
	cache := VRAMCache{
		"model-1": {Vram: 8192, GPUVRAM: []int{4096, 4096}, Config: snap},
	}
	stats := &VRAMStats{
		GPUs: []GPUInfo{
			{Total: 16384, Used: 8192, Free: 8192},  // GPU0 OK
			{Total: 16384, Used: 14336, Free: 2048}, // GPU1 not enough (2048 < 4096)
		},
	}
	if vramFitsPerGPU(stats, mc, cache, nil, 1024) {
		t.Error("expected model NOT to fit (GPU1 has insufficient free VRAM)")
	}
}

func TestVRAMFitsPerGPUHeadroomOnlyOnPrimary(t *testing.T) {
	// Headroom should only apply to GPU index 0 (primary/display GPU)
	mc := &ModelConfig{
		ID:      "model-1",
		Path:    "/models/m1.gguf",
		Devices: []string{"CUDA0", "CUDA1"},
	}
	snap := mc.Snapshot(nil)
	cache := VRAMCache{
		"model-1": {Vram: 8192, GPUVRAM: []int{4096, 4096}, Config: snap},
	}
	// GPU1 has exactly 4096 free — should fit since no headroom on secondary
	stats := &VRAMStats{
		GPUs: []GPUInfo{
			{Total: 16384, Used: 4096, Free: 12288}, // GPU0: plenty
			{Total: 16384, Used: 12288, Free: 4096}, // GPU1: exactly needed, no headroom
		},
	}
	if !vramFitsPerGPU(stats, mc, cache, nil, 1024) {
		t.Error("expected model to fit (GPU1 free == needed, no headroom on secondary)")
	}
}

func TestVRAMFitsPerGPUUnmappedDevice(t *testing.T) {
	// A device name that can't be mapped to a GPU index should be skipped
	mc := &ModelConfig{
		ID:      "model-1",
		Path:    "/models/m1.gguf",
		Devices: []string{"INVALID"},
	}
	snap := mc.Snapshot(nil)
	cache := VRAMCache{
		"model-1": {Vram: 4096, GPUVRAM: []int{4096}, Config: snap},
	}
	stats := &VRAMStats{
		GPUs: []GPUInfo{{Total: 16384, Used: 0, Free: 16384}},
	}
	// Should return true since no valid device to check against
	if !vramFitsPerGPU(stats, mc, cache, nil, 1024) {
		t.Error("expected true when device can't be mapped (skipped)")
	}
}

// ── computePerGPUDelta tests ────────────────────────────────────────

func TestComputePerGPUDeltaWithBefore(t *testing.T) {
	before := &VRAMStats{
		GPUs: []GPUInfo{{Used: 1000}, {Used: 2000}},
	}
	after := &VRAMStats{
		GPUs: []GPUInfo{{Used: 5000}, {Used: 2000}}, // GPU1 unchanged
	}
	got := computePerGPUDelta(before, after)
	want := []int{4000, 2000} // GPU0: delta 4000, GPU1: delta 0 → fallback to used
	if len(got) != 2 || got[0] != 4000 || got[1] != 2000 {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestComputePerGPUDeltaNilBefore(t *testing.T) {
	after := &VRAMStats{
		GPUs: []GPUInfo{{Used: 5000}, {Used: 3000}},
	}
	got := computePerGPUDelta(nil, after)
	if len(got) != 2 || got[0] != 5000 || got[1] != 3000 {
		t.Errorf("got %v, want [5000, 3000]", got)
	}
}

func TestComputePerGPUDeltaNilAfter(t *testing.T) {
	before := &VRAMStats{
		GPUs: []GPUInfo{{Used: 1000}},
	}
	if got := computePerGPUDelta(before, nil); got != nil {
		t.Errorf("expected nil for nil after, got %v", got)
	}
}

func TestComputePerGPUDeltaEmptyAfter(t *testing.T) {
	after := &VRAMStats{GPUs: []GPUInfo{}}
	if got := computePerGPUDelta(nil, after); got != nil {
		t.Errorf("expected nil for empty after.GPUs, got %v", got)
	}
}

func TestComputePerGPUDeltaNegativeDelta(t *testing.T) {
	// If after.Used < before.Used (VRAM freed), delta is negative,
	// which means fallback to after.Used
	before := &VRAMStats{
		GPUs: []GPUInfo{{Used: 8000}},
	}
	after := &VRAMStats{
		GPUs: []GPUInfo{{Used: 3000}},
	}
	got := computePerGPUDelta(before, after)
	if len(got) != 1 || got[0] != 3000 {
		t.Errorf("got %v, want [3000] (fallback to used when delta negative)", got)
	}
}

func TestComputePerGPUDeltaFewerBeforeGPUs(t *testing.T) {
	// before has fewer GPUs than after
	before := &VRAMStats{
		GPUs: []GPUInfo{{Used: 1000}}, // 1 GPU
	}
	after := &VRAMStats{
		GPUs: []GPUInfo{{Used: 5000}, {Used: 6000}}, // 2 GPUs
	}
	got := computePerGPUDelta(before, after)
	// GPU0: delta 4000, GPU1: no before → after.Used
	if len(got) != 2 || got[0] != 4000 || got[1] != 6000 {
		t.Errorf("got %v, want [4000, 6000]", got)
	}
}

// ── deviceNameToIndex tests ─────────────────────────────────────────

func TestDeviceNameToIndex(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  int
	}{
		{"CUDA0", "CUDA0", 0},
		{"CUDA1", "CUDA1", 1},
		{"cuda0 case-insensitive", "cuda0", 0},
		{"CUDA12", "CUDA12", 12},
		{"invalid prefix", "GPU0", -1},
		{"too short", "CU", -1},
		{"no number", "CUDA", -1},
		{"negative", "CUDA-1", -1},
		{"empty", "", -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := deviceNameToIndex(tt.input); got != tt.want {
				t.Errorf("deviceNameToIndex(%q) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}
