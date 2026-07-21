package main

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestBuildArgsRawMode tests that when Args is set, BuildArgs uses the raw
// args with {port} substitution instead of llama-server flags.
func TestBuildArgsRawMode(t *testing.T) {
	mc := &ModelConfig{
		ID: "test-ocr",
		Args: []string{
			"serve",
			"--port", "{port}",
			"--model", "datalab-to/chandra-ocr-2",
		},
	}

	args := mc.BuildArgs(nil, 8301)

	want := []string{"serve", "--port", "8301", "--model", "datalab-to/chandra-ocr-2"}
	if len(args) != len(want) {
		t.Fatalf("expected %d args, got %d: %v", len(want), len(args), args)
	}
	for i, a := range args {
		if a != want[i] {
			t.Errorf("arg[%d]: want %q, got %q", i, want[i], a)
		}
	}
}

// TestBuildArgsLlamaServerMode tests that when Args is NOT set, BuildArgs
// produces llama-server-style flags.
func TestBuildArgsLlamaServerMode(t *testing.T) {
	mc := &ModelConfig{
		ID:          "test-model",
		Path:        "/models/test.gguf",
		Name:        "test-model",
		ContextSize: 4096,
		Parallel:    2,
		Devices:     []string{"CUDA0"},
	}

	common := []string{"--flash-attn", "on"}
	args := mc.BuildArgs(common, 8301)

	// Should contain -m, --alias, -c, --parallel, --device, --host, --port
	found := map[string]bool{}
	for _, a := range args {
		found[a] = true
	}
	for _, expected := range []string{"-m", "--alias", "-c", "--parallel", "--device", "--host", "--port", "--flash-attn"} {
		if !found[expected] {
			t.Errorf("expected flag %q in args, not found: %v", expected, args)
		}
	}
}

// TestResolveBinaryOverride tests that per-model Binary overrides backend binary.
func TestResolveBinaryOverride(t *testing.T) {
	mc := &ModelConfig{
		ID:     "test-ocr",
		Binary: "/usr/bin/python3",
	}
	bc := &BackendConfig{Binary: "/nonexistent/llama-server"}

	bin, err := mc.ResolveBinary(bc)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if bin != "/usr/bin/python3" {
		t.Errorf("expected /usr/bin/python3, got %s", bin)
	}
}

// TestResolveBinaryFallback tests that backend binary is used when model has no override.
func TestResolveBinaryFallback(t *testing.T) {
	mc := &ModelConfig{ID: "test"}
	bc := &BackendConfig{Binary: "/usr/bin/env"}

	bin, err := mc.ResolveBinary(bc)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// /usr/bin/env should be found via PATH or directly
	if bin == "" {
		t.Error("expected non-empty binary path")
	}
}

// TestHealthEndpointDefault tests default health path.
func TestHealthEndpointDefault(t *testing.T) {
	mc := &ModelConfig{ID: "test"}
	if mc.HealthEndpoint() != "/health" {
		t.Errorf("expected /health, got %s", mc.HealthEndpoint())
	}
}

// TestHealthEndpointOverride tests custom health path.
func TestHealthEndpointOverride(t *testing.T) {
	mc := &ModelConfig{ID: "test", HealthPath: "/v1/health"}
	if mc.HealthEndpoint() != "/v1/health" {
		t.Errorf("expected /v1/health, got %s", mc.HealthEndpoint())
	}
}

// TestBuildModelEnv tests that per-model env overrides backend env.
func TestBuildModelEnv(t *testing.T) {
	mc := &ModelConfig{
		ID: "test",
		Env: map[string]string{
			"CUDA_VISIBLE_DEVICES": "0",
			"MODEL_SPECIFIC":       "yes",
		},
	}
	bc := &BackendConfig{
		Env: map[string]string{
			"CUDA_VISIBLE_DEVICES": "0,1", // should be overridden by model
			"BACKEND_VAR":          "backend",
		},
	}

	env := mc.BuildModelEnv(bc)
	envMap := make(map[string]string)
	for _, e := range env {
		parts := strings.SplitN(e, "=", 2)
		if len(parts) == 2 {
			envMap[parts[0]] = parts[1]
		}
	}

	if envMap["CUDA_VISIBLE_DEVICES"] != "0" {
		t.Errorf("expected CUDA_VISIBLE_DEVICES=0 (model override), got %s", envMap["CUDA_VISIBLE_DEVICES"])
	}
	if envMap["MODEL_SPECIFIC"] != "yes" {
		t.Errorf("expected MODEL_SPECIFIC=yes, got %s", envMap["MODEL_SPECIFIC"])
	}
	if envMap["BACKEND_VAR"] != "backend" {
		t.Errorf("expected BACKEND_VAR=backend, got %s", envMap["BACKEND_VAR"])
	}
}

// TestSnapshotEquality tests that snapshots detect config changes in new fields.
func TestSnapshotEquality(t *testing.T) {
	mc1 := &ModelConfig{
		ID:     "test",
		Path:   "/models/test.gguf",
		Binary: "/usr/bin/python3",
		Args:   []string{"serve", "--port", "{port}"},
		Env:    map[string]string{"FOO": "bar"},
	}
	mc2 := &ModelConfig{
		ID:     "test",
		Path:   "/models/test.gguf",
		Binary: "/usr/bin/python3",
		Args:   []string{"serve", "--port", "{port}"},
		Env:    map[string]string{"FOO": "bar"},
	}
	mc3 := &ModelConfig{
		ID:     "test",
		Path:   "/models/test.gguf",
		Binary: "/usr/bin/vllm", // different binary
		Args:   []string{"serve", "--port", "{port}"},
		Env:    map[string]string{"FOO": "bar"},
	}

	s1 := mc1.Snapshot(nil)
	s2 := mc2.Snapshot(nil)
	s3 := mc3.Snapshot(nil)

	if !s1.Equal(s2) {
		t.Error("identical configs should have equal snapshots")
	}
	if s1.Equal(s3) {
		t.Error("different binary should produce unequal snapshots")
	}
}

// TestConfigValidationCustomBinary tests that a model with custom binary
// doesn't require Path.
func TestConfigValidationCustomBinary(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			BackendPortStart: 8201,
			BackendPortEnd:   8299,
			MaxModels:        1,
		},
		Backend: BackendConfig{Binary: "/nonexistent/llama-server"},
		Models: []ModelConfig{
			{
				ID:     "ocr-model",
				Name:   "ocr-model",
				Binary: "python3",
				Args:   []string{"-m", "surya.server", "--port", "{port}"},
			},
		},
	}

	if err := cfg.Validate(); err != nil {
		t.Fatalf("validation should pass for custom-binary model without path: %v", err)
	}
}

// TestConfigValidationLlamaServerRequiresPath tests that llama-server models
// still require Path.
func TestConfigValidationLlamaServerRequiresPath(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			BackendPortStart: 8201,
			BackendPortEnd:   8299,
			MaxModels:        1,
		},
		Backend: BackendConfig{Binary: "/nonexistent/llama-server"},
		Models: []ModelConfig{
			{
				ID:   "llama-model",
				Name: "llama-model",
				// no Path, no Binary, no Args — should fail
			},
		},
	}

	if err := cfg.Validate(); err == nil {
		t.Fatal("validation should fail for llama-server model without path")
	}
}

// ── Integration test: mock backend ───────────────────────────────────────

// mockBackend is a minimal HTTP server that mimics a non-llama-server
// backend. It responds to a custom health endpoint and echoes requests.
func mockBackend(t *testing.T, healthPath string) (*httptest.Server, int) {
	mux := http.NewServeMux()
	mux.HandleFunc(healthPath, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"model":  "mock-ocr",
			"echo":   r.URL.Path,
			"method": r.Method,
		})
	})
	srv := httptest.NewServer(mux)
	return srv, srv.Listener.Addr().(*net.TCPAddr).Port
}

// TestEndToEndCustomBackend tests the full lifecycle with a mock backend
// that uses a custom binary (a shell script that starts a Python HTTP server).
func TestEndToEndCustomBackend(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	// Create a mock backend script
	tmpDir := t.TempDir()
	scriptPath := filepath.Join(tmpDir, "mock-backend.sh")
	mockScript := `#!/bin/bash
# Mock backend that starts a simple HTTP server on the given port
PORT="$1"
HEALTH_PATH="$2"

python3 -c "
import http.server, json, sys, os
port = int(sys.argv[1])
health = sys.argv[2] or '/health'
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == health:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{\"status\":\"ok\"}')
        elif self.path == '/v1/models':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'object':'list','data':[{'id':'mock-ocr','status':'loaded'}]}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'choices':[{'message':{'content':'mock response'}}]}).encode())
    def log_message(self, format, *args):
        pass
http.server.HTTPServer(('127.0.0.1', port), Handler).serve_forever()
" "$PORT" "$HEALTH_PATH"
`
	if err := os.WriteFile(scriptPath, []byte(mockScript), 0755); err != nil {
		t.Fatalf("failed to write mock script: %v", err)
	}

	// Create a temp config
	configPath := filepath.Join(tmpDir, "config.yaml")
	configContent := fmt.Sprintf(`
server:
  host: "127.0.0.1"
  port: 18080
  backend_port_start: 18201
  backend_port_end: 18299
  max_models: 1
  idle_timeout_minutes: 60
  health_timeout_seconds: 30
  sweep_interval_seconds: 300

backend:
  binary: "/nonexistent/llama-server"
  nvidia_smi: "nvidia-smi"

models:
  - id: mock-ocr
    name: mock-ocr
    binary: %q
    args:
      - "{port}"
      - "/health"
    health_path: "/health"
`, scriptPath)

	if err := os.WriteFile(configPath, []byte(configContent), 0644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}

	// Load and validate config
	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig failed: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate failed: %v", err)
	}

	// Verify the model config
	mc := cfg.FindModel("mock-ocr")
	if mc == nil {
		t.Fatal("model not found")
	}
	if mc.Binary != scriptPath {
		t.Errorf("expected binary %s, got %s", scriptPath, mc.Binary)
	}
	if len(mc.Args) != 2 {
		t.Fatalf("expected 2 args, got %d", len(mc.Args))
	}

	// Verify BuildArgs produces the right args
	args := mc.BuildArgs(nil, 18201)
	if args[0] != "18201" {
		t.Errorf("expected first arg '18201', got %s", args[0])
	}
	if args[1] != "/health" {
		t.Errorf("expected second arg '/health', got %s", args[1])
	}

	// Start llama-switch
	logger := NewLogger()
	bm := NewBackendManager(cfg, configPath, logger)
	bm.Start()
	defer bm.Stop()

	ps := NewProxyServer(cfg, bm)

	go func() {
		if err := ps.ListenAndServe(); err != nil {
			// expected on shutdown
		}
	}()
	defer ps.Shutdown()

	// Wait for proxy to start
	time.Sleep(500 * time.Millisecond)

	// Load the model
	loadReq, _ := json.Marshal(map[string]string{"model": "mock-ocr"})
	resp, err := http.Post("http://127.0.0.1:18080/models/load", "application/json", strings.NewReader(string(loadReq)))
	if err != nil {
		// Check if python3 is available
		if _, perr := exec.LookPath("python3"); perr != nil {
			t.Skip("python3 not available, skipping integration test")
		}
		t.Fatalf("failed to load model: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body := make([]byte, 1024)
		n, _ := resp.Body.Read(body)
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, string(body[:n]))
	}

	// Send a test request through the proxy
	testReq, _ := json.Marshal(map[string]any{
		"model":    "mock-ocr",
		"messages": []map[string]string{{"role": "user", "content": "test"}},
	})
	resp2, err := http.Post("http://127.0.0.1:18080/v1/chat/completions", "application/json", strings.NewReader(string(testReq)))
	if err != nil {
		t.Fatalf("failed to send request: %v", err)
	}
	defer resp2.Body.Close()

	if resp2.StatusCode != http.StatusOK {
		body := make([]byte, 1024)
		n, _ := resp2.Body.Read(body)
		t.Fatalf("expected 200, got %d: %s", resp2.StatusCode, string(body[:n]))
	}

	// Verify response
	var result map[string]any
	if err := json.NewDecoder(resp2.Body).Decode(&result); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	choices, ok := result["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatal("expected choices in response")
	}
}
