package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"
)

const usageText = `llama-switch — GPU model multiplexer for llama-server

Usage:
  llama-switch <command> [flags]

Commands:
  serve [config]        Start the proxy server (default).
                        Config path defaults to ./config.yaml next to the binary.

  profile [config]      Profile VRAM usage for each model.
                        Loads each model once (unloading others), measures
                        VRAM, saves results to vram-cache.json.

  models [config]       List configured models and current VRAM estimates.

  status [config]       Show loaded models and VRAM status (requires a running server).

  help                  Show this help message.

Environment:
  LLAMA_SWITCH_CONFIG   Path to config file (overrides default).
`

func main() {
	if len(os.Args) < 2 {
		serve("")
		return
	}

	cmd := os.Args[1]
	args := os.Args[2:]

	switch cmd {
	case "serve":
		serve(parseConfigArg(args))
	case "profile":
		profile(parseConfigArg(args))
	case "models":
		listModels(parseConfigArg(args))
	case "status":
		status(parseConfigArg(args))
	case "help", "-h", "--help":
		fmt.Print(usageText)
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", cmd)
		fmt.Print(usageText)
		os.Exit(1)
	}
}

// parseConfigArg extracts the config path from positional args, falling
// back to LLAMA_SWITCH_CONFIG env var or ./config.yaml.
func parseConfigArg(args []string) string {
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		return args[0]
	}
	if p := os.Getenv("LLAMA_SWITCH_CONFIG"); p != "" {
		return p
	}
	return "config.yaml"
}

// ── serve ───────────────────────────────────

func serve(configPath string) {
	if configPath == "" {
		configPath = parseConfigArg(nil)
	}

	cfg, err := LoadConfig(configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(1)
	}

	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "config validation error: %v\n", err)
		os.Exit(1)
	}

	logger := NewLogger()
	bm := NewBackendManager(cfg, configPath, logger)
	bm.Start()

	ps := NewProxyServer(cfg, bm)

	// Graceful shutdown: signal handler initiates shutdown, main
	// goroutine waits for ListenAndServe to return after draining.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		logger.Printf("received signal %v, shutting down...", sig)
		bm.Stop()
		ps.Shutdown()
	}()

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	logger.Printf("llama-switch listening on %s", addr)
	logger.Printf("  %d models configured, max %d concurrent",
		len(cfg.Models), cfg.Server.MaxModels)

	if err := ps.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		logger.Printf("server error: %v", err)
		os.Exit(1)
	}
}

// ── profile ─────────────────────────────────

// profile loads each model one at a time, measures VRAM, and saves the
// results to vram-cache.json. Already-profiled models are skipped.
func profile(configPath string) {
	cfg, err := LoadConfig(configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(1)
	}

	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "config validation error: %v\n", err)
		os.Exit(1)
	}

	cachePath := vramCachePath(cfg, configPath)
	cache := LoadVRAMCache(cachePath)

	if _, err := cfg.Backend.ResolveBinary(); err != nil {
		fmt.Fprintf(os.Stderr, "backend binary not found: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Profiling VRAM for %d models (one at a time)\n\n", len(cfg.Models))

	logger := NewLogger()

	for i := range cfg.Models {
		mc := &cfg.Models[i]

		snap := mc.Snapshot(cfg.Backend.CommonArgs)
		if entry, ok := cache[mc.ID]; ok && entry.Vram > 0 && entry.Config.Equal(snap) {
			fmt.Printf("[%s] CACHE HIT — vram-cache.json says %d MB (delete entry to re-profile)\n", mc.ID, entry.Vram)
			continue
		} else if ok && !entry.Config.Equal(snap) {
			fmt.Printf("[%s] config changed since last profile, re-profiling...\n", mc.ID)
		}

		fmt.Printf("[%s] profiling...\n", mc.ID)

		used, gpuVRAM, err := profileSingle(cfg, configPath, mc, logger)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[%s] FAILED: %v\n", mc.ID, err)
			continue
		}

		cache[mc.ID] = CacheEntry{Vram: used, GPUVRAM: gpuVRAM, Config: mc.Snapshot(cfg.Backend.CommonArgs)}
		_ = SaveVRAMCache(cachePath, cache)

		// Build per-GPU display string
		gpuStr := ""
		if len(gpuVRAM) > 0 {
			parts := make([]string, len(gpuVRAM))
			for i, v := range gpuVRAM {
				parts[i] = fmt.Sprintf("GPU%d:%dMB", i, v)
			}
			gpuStr = " (" + strings.Join(parts, ", ") + ")"
		}

		fmt.Printf("[%s] %d MB%s — saved to vram-cache.json\n", mc.ID, used, gpuStr)
		fmt.Println()
	}

	fmt.Println("Done. VRAM estimates saved to", cachePath)
}

// profileSingle loads one model, waits for health, measures VRAM, then unloads.
func profileSingle(cfg *Config, configPath string, mc *ModelConfig, logger *CondLogger) (int, []int, error) {
	bm := NewBackendManager(cfg, configPath, logger)

	before, err := queryVRAM(cfg.Backend.NvidiaSMI)
	if err != nil {
		return 0, nil, fmt.Errorf("baseline VRAM query: %w", err)
	}
	baselineUsed := before.Used

	bm.mu.Lock()
	_, err = bm.startBackendLocked(mc)
	bm.mu.Unlock()
	if err != nil {
		return 0, nil, fmt.Errorf("load failed: %w", err)
	}

	time.Sleep(3 * time.Second)

	after, err := queryVRAM(cfg.Backend.NvidiaSMI)
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

	if err := validateVRAMMeasurement(mc, used); err != nil {
		return 0, nil, fmt.Errorf("VRAM measurement rejected: %w", err)
	}

	return used, gpuVRAM, nil
}

// ── models ──────────────────────────────────

func listModels(configPath string) {
	cfg, err := LoadConfig(configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(1)
	}

	cachePath := vramCachePath(cfg, configPath)
	cache := LoadVRAMCache(cachePath)

	ids := make([]string, len(cfg.Models))
	for i := range cfg.Models {
		ids[i] = cfg.Models[i].ID
	}
	sort.Strings(ids)

	fmt.Printf("%-16s  %-30s  %8s  %-20s  %s\n", "ID", "MODEL", "VRAM_MB", "DEVICES", "PER-GPU (MB)")
	fmt.Println(strings.Repeat("-", 100))

	for _, id := range ids {
		mc := cfg.FindModel(id)
		entry := cache[id]
		vramStr := "???"
		perGPUStr := ""
		if entry.Vram > 0 {
			snap := mc.Snapshot(cfg.Backend.CommonArgs)
			if entry.Config.Equal(snap) {
				vramStr = fmt.Sprintf("%d", entry.Vram)
				if len(entry.GPUVRAM) > 0 {
					parts := make([]string, 0, len(entry.GPUVRAM))
					for i, v := range entry.GPUVRAM {
						if v > 0 {
							parts = append(parts, fmt.Sprintf("GPU%d:%d", i, v))
						}
					}
					perGPUStr = strings.Join(parts, ", ")
				}
			} else {
				vramStr = fmt.Sprintf("%d (stale)", entry.Vram)
				perGPUStr = "(stale)"
			}
		}
		devices := strings.Join(mc.Devices, ",")
		fmt.Printf("%-16s  %-30s  %8s  %-20s  %s\n", mc.ID, mc.Model, vramStr, devices, perGPUStr)
	}
}

// ── status ──────────────────────────────────

func status(configPath string) {
	cfg, err := LoadConfig(configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(1)
	}

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	url := fmt.Sprintf("http://%s/v1/loaded", addr)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid server address: %v\n", err)
		os.Exit(1)
	}
	client := &http.Client{Timeout: 5 * time.Second}

	resp, err := client.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot reach llama-switch at %s: %v\n", addr, err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("Loaded models at %s:\n%s\n", addr, string(body))
}
