package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// GPUInfo holds per-GPU VRAM figures (in MB).
type GPUInfo struct {
	Total int
	Used  int
	Free  int
}

// VRAMStats holds aggregate VRAM across all GPUs.
type VRAMStats struct {
	Total int
	Used  int
	Free  int
	GPUs  []GPUInfo
}

// queryVRAM runs nvidia-smi and returns aggregate VRAM stats.
func queryVRAM(smiPath string) (*VRAMStats, error) {
	out, err := exec.Command(smiPath,
		"--query-gpu=memory.total,memory.used,memory.free",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi: %w", err)
	}

	stats := &VRAMStats{}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) != 3 {
			return nil, fmt.Errorf("nvidia-smi: unexpected line format: %q", line)
		}
		total, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			return nil, fmt.Errorf("nvidia-smi: failed to parse total: %w", err)
		}
		used, err := strconv.Atoi(strings.TrimSpace(parts[1]))
		if err != nil {
			return nil, fmt.Errorf("nvidia-smi: failed to parse used: %w", err)
		}
		free, err := strconv.Atoi(strings.TrimSpace(parts[2]))
		if err != nil {
			return nil, fmt.Errorf("nvidia-smi: failed to parse free: %w", err)
		}
		stats.Total += total
		stats.Used += used
		stats.Free += free
		stats.GPUs = append(stats.GPUs, GPUInfo{Total: total, Used: used, Free: free})
	}

	if len(stats.GPUs) == 0 {
		return nil, fmt.Errorf("no GPUs found in nvidia-smi output")
	}
	return stats, nil
}

// ── VRAM cache (JSON sidecar) ────────────────

// ModelConfigSnapshot captures the subset of ModelConfig fields that
// affect VRAM usage. It is stored alongside each cached VRAM measurement
// so the code can detect when a config change has made a cached value
// stale. Backend-level common_args are also included since they affect
// batch size, flash attention, and cache settings.
type ModelConfigSnapshot struct {
	Path           string   `json:"path"`
	Mmproj         string   `json:"mmproj"`
	ContextSize    int      `json:"context_size"`
	Parallel       int      `json:"parallel"`
	Devices        []string `json:"devices"`
	TensorSplit    string   `json:"tensor_split"`
	CtxCheckpoints int      `json:"ctx_checkpoints"`
	ExtraArgs      []string `json:"extra_args"`
	CommonArgs     []string `json:"common_args"`
}

// CacheEntry stores a measured VRAM value together with the config
// snapshot that was in effect when the measurement was taken.
type CacheEntry struct {
	Vram   int                 `json:"vram"`
	Config ModelConfigSnapshot `json:"config"`
}

// VRAMCache maps model IDs to their profiled cache entries.
type VRAMCache map[string]CacheEntry

func vramCachePath(cfg *Config, configPath string) string {
	dir := filepath.Dir(expand(configPath))
	return filepath.Join(dir, "vram-cache.json")
}

// LoadVRAMCache reads the VRAM cache from disk. It gracefully handles
// the legacy flat format (map[string]int) by returning an empty cache
// so that all models are re-profiled.
func LoadVRAMCache(path string) VRAMCache {
	data, err := os.ReadFile(path)
	if err != nil {
		return VRAMCache{}
	}
	var cache VRAMCache
	if err := json.Unmarshal(data, &cache); err != nil {
		// Legacy format was map[string]int, or corrupt. Force re-profiling.
		return VRAMCache{}
	}
	return cache
}

func SaveVRAMCache(path string, cache VRAMCache) error {
	data, err := json.MarshalIndent(cache, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0644)
}

// vramEstimate returns the estimated VRAM (MB) for a model from the
// profiled cache. Returns 0 if the model hasn't been profiled or the
// cached entry is stale (config snapshot doesn't match current config).
func vramEstimate(cache VRAMCache, mc *ModelConfig, commonArgs []string) int {
	if entry, ok := cache[mc.ID]; ok && entry.Vram > 0 {
		if entry.Config.Equal(mc.Snapshot(commonArgs)) {
			return entry.Vram
		}
	}
	return 0
}

// ── Snapshot helpers ─────────────────────────

// Snapshot extracts the VRAM-affecting fields from a ModelConfig into
// a ModelConfigSnapshot suitable for cache invalidation. Includes
// backend-level common_args since they affect VRAM too.
func (m *ModelConfig) Snapshot(commonArgs []string) ModelConfigSnapshot {
	return ModelConfigSnapshot{
		Path:           m.Path,
		Mmproj:         m.Mmproj,
		ContextSize:    m.ContextSize,
		Parallel:       m.Parallel,
		Devices:        m.Devices,
		TensorSplit:    m.TensorSplit,
		CtxCheckpoints: m.CtxCheckpoints,
		ExtraArgs:      m.ExtraArgs,
		CommonArgs:     commonArgs,
	}
}

// Equal reports whether two snapshots describe the same set of
// VRAM-affecting configuration. Slices are compared element-wise
// (nil and empty slices are treated as equal).
func (s ModelConfigSnapshot) Equal(other ModelConfigSnapshot) bool {
	if s.Path != other.Path ||
		s.Mmproj != other.Mmproj ||
		s.ContextSize != other.ContextSize ||
		s.Parallel != other.Parallel ||
		s.TensorSplit != other.TensorSplit ||
		s.CtxCheckpoints != other.CtxCheckpoints {
		return false
	}
	if !sliceEqual(s.Devices, other.Devices) {
		return false
	}
	if !sliceEqual(s.ExtraArgs, other.ExtraArgs) {
		return false
	}
	if !sliceEqual(s.CommonArgs, other.CommonArgs) {
		return false
	}
	return true
}

// sliceEqual compares two string slices, treating nil and empty as equal.
func sliceEqual(a, b []string) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
