package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
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
	// Per-model runtime override fields (affect VRAM when set)
	Binary string   `json:"binary,omitempty"`
	Args   []string `json:"args,omitempty"`
	Env    []string `json:"env,omitempty"`
}

// CacheEntry stores a measured VRAM value together with the config
// snapshot that was in effect when the measurement was taken.
// GPUVRAM holds per-GPU VRAM (MB), indexed by nvidia-smi GPU index.
// It is empty for legacy cache entries (pre-dating per-GPU profiling).
type CacheEntry struct {
	Vram    int                 `json:"vram"`
	GPUVRAM []int               `json:"gpu_vram,omitempty"`
	Config  ModelConfigSnapshot `json:"config"`
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

// vramEstimate returns the estimated total VRAM (MB) for a model from the
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

// vramEstimatePerGPU returns the estimated per-GPU VRAM (MB) for a model.
// The returned slice is indexed by nvidia-smi GPU index. Entries for GPUs
// the model doesn't use are 0. Returns nil if the model hasn't been
// profiled, the cache is stale, or the cache entry predates per-GPU
// profiling (no GPUVRAM data).
func vramEstimatePerGPU(cache VRAMCache, mc *ModelConfig, commonArgs []string) []int {
	if entry, ok := cache[mc.ID]; ok && entry.Vram > 0 {
		if entry.Config.Equal(mc.Snapshot(commonArgs)) && len(entry.GPUVRAM) > 0 {
			return entry.GPUVRAM
		}
	}
	return nil
}

// vramFitsPerGPU checks whether each GPU the model targets has enough
// free VRAM for its profiled share. The headroom accounts for
// OS/display-server/other-app VRAM usage and is applied only to the
// primary GPU (CUDA0); secondary GPUs have no display output and thus
// no such overhead. The profiled measurement already includes KV cache
// and compute buffers (it is taken after the model is healthy).
// If per-GPU profile data is unavailable, falls back to the aggregate check.
func vramFitsPerGPU(stats *VRAMStats, mc *ModelConfig, cache VRAMCache, commonArgs []string, headroom int) bool {
	perGPU := vramEstimatePerGPU(cache, mc, commonArgs)
	if perGPU == nil {
		// No per-GPU data: fall back to aggregate check
		needed := vramEstimate(cache, mc, commonArgs)
		if needed == 0 {
			return true // unknown VRAM, proceed optimistically
		}
		return stats.Free >= needed+headroom
	}

	// Map device names (CUDA0, CUDA1, ...) to nvidia-smi GPU indices.
	// nvidia-smi lists GPUs in index order; CUDA0 = index 0, etc.
	for _, devName := range mc.Devices {
		gpuIdx := deviceNameToIndex(devName)
		if gpuIdx < 0 || gpuIdx >= len(stats.GPUs) {
			continue // can't map, skip this GPU
		}
		needed := 0
		if gpuIdx < len(perGPU) {
			needed = perGPU[gpuIdx]
		}
		if needed == 0 {
			continue
		}
		// Headroom applies only to the primary GPU (index 0):
		// secondary GPUs have no display output, so no OS overhead.
		required := needed
		if gpuIdx == 0 {
			required += headroom
		}
		if stats.GPUs[gpuIdx].Free < required {
			return false
		}
	}
	return true
}

// deviceNameToIndex converts a CUDA device name like "CUDA0" to the
// nvidia-smi GPU index 0. Returns -1 if the name can't be parsed.
func deviceNameToIndex(name string) int {
	if len(name) < 5 || !strings.EqualFold(name[:4], "CUDA") {
		return -1
	}
	n, err := strconv.Atoi(name[4:])
	if err != nil || n < 0 {
		return -1
	}
	return n
}

// computePerGPUDelta returns the per-GPU VRAM delta (in MB) between
// two VRAMStats snapshots. The result is indexed by nvidia-smi GPU
// index. If before is nil, after.Used per GPU is returned directly.
func computePerGPUDelta(before, after *VRAMStats) []int {
	if after == nil || len(after.GPUs) == 0 {
		return nil
	}
	result := make([]int, len(after.GPUs))
	for i, gpu := range after.GPUs {
		if before != nil && i < len(before.GPUs) {
			delta := gpu.Used - before.GPUs[i].Used
			if delta > 0 {
				result[i] = delta
			} else {
				result[i] = gpu.Used
			}
		} else {
			result[i] = gpu.Used
		}
	}
	return result
}

// ── Snapshot helpers ─────────────────────────

// modelFileSizeMB returns the size of the model's .gguf file in MB,
// or 0 if the file can't be stat'd.
func modelFileSizeMB(mc *ModelConfig) int {
	path := expand(mc.Path)
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return int(info.Size() / (1024 * 1024))
}

// validateVRAMMeasurement checks whether a profiled VRAM figure is
// plausible. The measured VRAM must be at least as large as the model
// file (weights alone exceed that), plus a small floor to catch
// near-zero measurements from corrupted deltas. Returns nil if the
// measurement is plausible, or an error explaining why it's rejected.
//
// For models using custom binaries (non-llama-server backends where
// Path may be empty or point to a HuggingFace model ID rather than a
// local file), the file-size check is skipped — only the minimum floor
// applies.
func validateVRAMMeasurement(mc *ModelConfig, measuredMB int) error {
	const minPlausible = 256 // 256 MB floor for any real model

	if measuredMB < minPlausible {
		return fmt.Errorf("measured %d MB is below %d MB floor — likely a corrupted measurement", measuredMB, minPlausible)
	}

	// Only validate against file size for llama-server models (local .gguf file).
	// Custom-binary models may not have a local Path.
	if mc.Binary == "" && mc.Path != "" {
		fileMB := modelFileSizeMB(mc)
		if fileMB > 0 && measuredMB < fileMB {
			return fmt.Errorf("measured %d MB is less than model file size %d MB — model may not have fully loaded", measuredMB, fileMB)
		}
	}

	return nil
}

// Snapshot extracts the VRAM-affecting fields from a ModelConfig into
// a ModelConfigSnapshot suitable for cache invalidation. Includes
// backend-level common_args since they affect VRAM too.
func (m *ModelConfig) Snapshot(commonArgs []string) ModelConfigSnapshot {
	// Sort env keys for deterministic comparison
	envKeys := make([]string, 0, len(m.Env))
	for k := range m.Env {
		envKeys = append(envKeys, k+"="+expand(m.Env[k]))
	}
	sort.Strings(envKeys)

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
		Binary:         m.Binary,
		Args:           m.Args,
		Env:            envKeys,
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
		s.CtxCheckpoints != other.CtxCheckpoints ||
		s.Binary != other.Binary {
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
	if !sliceEqual(s.Args, other.Args) {
		return false
	}
	if !sliceEqual(s.Env, other.Env) {
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
