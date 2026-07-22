package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
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
                        Config path defaults to ./config/config.yaml next to
                        the binary (or ../config/ relative to the binary).

  profile [config]      Profile VRAM usage for each model.
                        Loads each model once (unloading others), measures
                        VRAM, saves results to vram-cache.json.

  models [config]       List configured models and current VRAM estimates.

  status [config]       Show loaded models and VRAM status (requires a running server).

  help                  Show this help message.

Environment:
  LLAMA_SWITCH_CONFIG   Path to config file (overrides default).

Config path resolution order:
  1. Positional argument
  2. $LLAMA_SWITCH_CONFIG
  3. config.yaml next to the binary (./ and ../config/ relative to it)
  4. ~/.config/llama-switch/config.yaml
  5. ./config/config.yaml (current directory)
`

// Version is the current llama-switch version. Bump on feature releases.
const Version = "0.3.1"

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
	case "version", "--version", "-v":
		fmt.Printf("llama-switch %s\n", Version)
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", cmd)
		fmt.Print(usageText)
		os.Exit(1)
	}
}

// parseConfigArg extracts the config path from positional args, falling
// back to LLAMA_SWITCH_CONFIG env var, the binary's directory, or
// ~/.config/llama-switch/config.yaml.
func parseConfigArg(args []string) string {
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		return args[0]
	}
	if p := os.Getenv("LLAMA_SWITCH_CONFIG"); p != "" {
		return p
	}
	// Check relative to the binary: same dir (./) and ../config/
	if exe, err := os.Executable(); err == nil {
		binDir := filepath.Dir(exe)
		for _, rel := range []string{".", filepath.Join("..", "config")} {
			p := filepath.Join(binDir, rel, "config.yaml")
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	// Check ~/.config/llama-switch/config.yaml
	if home, err := os.UserHomeDir(); err == nil {
		p := filepath.Join(home, ".config", "llama-switch", "config.yaml")
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return filepath.Join("config", "config.yaml")
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
	logger.Printf("llama-switch v%s listening on %s", Version, addr)
	logger.Printf("  %d models configured, max %d concurrent",
		len(cfg.Models), cfg.Server.MaxModels)

	if err := ps.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		logger.Printf("server error: %v", err)
		os.Exit(1)
	}
}

// ── profile ─────────────────────────────────

var flagForcedProfile bool

// profile loads each model one at a time, measures VRAM, and saves the
// results to vram-cache.json. Already-profiled models are skipped.
// If a running llama-switch service is detected, delegates to it via SSE.
func profile(configPath string) {
	// Check for --force flag in remaining args
	for i, arg := range os.Args {
		if arg == "--force" || arg == "-force" {
			flagForcedProfile = true
			os.Args = append(os.Args[:i], os.Args[i+1:]...)
			break
		}
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

	if _, err := cfg.Backend.ResolveBinary(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: backend binary not found: %v (models with per-model binaries will still work)\n", err)
	}

	// If a running service is detected, delegate to it
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	if isServiceRunning(addr) {
		runRemoteProfile(addr)
		return
	}

	// Local profiling
	logger := NewLogger()
	bm := NewBackendManager(cfg, configPath, logger)

	progress := func(format string, args ...any) {
		fmt.Printf(format+"\n", args...)
	}

	_, err = bm.ProfileModels(flagForcedProfile, progress)
	if err != nil {
		fmt.Fprintf(os.Stderr, "profiling error: %v\n", err)
		os.Exit(1)
	}
}

// isServiceRunning checks if a llama-switch service is listening on addr.
func isServiceRunning(addr string) bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://%s/health", addr))
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == 200
}

// runRemoteProfile sends a profiling request to a running service and
// streams SSE progress to stdout.
func runRemoteProfile(addr string) {
	fmt.Printf("Detected running llama-switch at %s — requesting remote profile...\n\n", addr)

	url := fmt.Sprintf("http://%s/admin/profile", addr)
	if flagForcedProfile {
		url += "?force=true"
	}

	resp, err := http.Post(url, "application/json", bytes.NewReader(nil))
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to request remote profile: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		fmt.Fprintf(os.Stderr, "remote profile failed: %s: %s\n", resp.Status, string(body))
		os.Exit(1)
	}

	// Stream SSE events to stdout
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			fmt.Println(strings.TrimPrefix(line, "data: "))
		}
	}
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

	fmt.Printf("%-30s  %8s  %8s  %-20s  %s\n", "MODEL", "CONTEXT", "VRAM_MB", "DEVICES", "PER-GPU (MB)")
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
		parallel := mc.Parallel
		if parallel == 0 {
			parallel = 1
		}
		ctxStr := fmt.Sprintf("%d", mc.ContextSize/parallel)
		fmt.Printf("%-30s  %8s  %8s  %-20s  %s\n", mc.Name, ctxStr, vramStr, devices, perGPUStr)
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
