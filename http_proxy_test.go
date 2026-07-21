package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// ── Helper: build a ProxyServer with a mock BackendManager ──────────

// newTestProxyServer creates a ProxyServer backed by a real BackendManager
// whose backends map has been pre-populated with mock backends pointing
// at the given httptest.Server ports.
func newTestProxyServer(t *testing.T, models []ModelConfig, backends map[string]*Backend) *ProxyServer {
	t.Helper()
	cfg := &Config{
		Server: ServerConfig{
			Host:             "127.0.0.1",
			Port:             0, // not used in tests
			BackendPortStart: 8201,
			BackendPortEnd:   8299,
			MaxModels:        2,
		},
		Models: models,
	}
	bm := &BackendManager{
		cfg:         cfg,
		backends:    backends,
		loading:     make(map[string]*sync.Cond),
		portAlloc:   newPortAllocator(8201, 8299),
		logger:      NewLogger(),
		sweeperStop: make(chan struct{}),
		shutdownCh:  make(chan struct{}),
		vramCache:   VRAMCache{},
	}
	bm.capCh = make(chan struct{})
	return NewProxyServer(cfg, bm)
}

// ── handleV1Models tests ─────────────────────────────────────────────

func TestHandleV1Models(t *testing.T) {
	models := []ModelConfig{
		{ID: "model-a", Name: "Model A", Path: "/a.gguf"},
		{ID: "model-b", Name: "Model B", Path: "/b.gguf"},
	}
	// model-a is loaded, model-b is not
	backends := map[string]*Backend{
		"model-a": {ID: "model-a", Port: 9001},
	}
	ps := newTestProxyServer(t, models, backends)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()
	ps.handleV1Models(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var result map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	data, ok := result["data"].([]any)
	if !ok {
		t.Fatal("expected 'data' to be a slice")
	}
	if len(data) != 2 {
		t.Fatalf("expected 2 models, got %d", len(data))
	}

	// Check that model-a is "loaded" and model-b is "unloaded"
	for _, entry := range data {
		m := entry.(map[string]any)
		id := m["id"].(string)
		status := m["status"].(map[string]any)
		statusVal := status["value"].(string)
		switch id {
		case "Model A":
			if statusVal != "loaded" {
				t.Errorf("model-a: expected status 'loaded', got %q", statusVal)
			}
		case "Model B":
			if statusVal != "unloaded" {
				t.Errorf("model-b: expected status 'unloaded', got %q", statusVal)
			}
		}
	}
}

func TestHandleV1ModelsEmpty(t *testing.T) {
	ps := newTestProxyServer(t, nil, nil)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()
	ps.handleV1Models(w, req)

	resp := w.Result()
	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)
	data, _ := result["data"].([]any)
	if len(data) != 0 {
		t.Errorf("expected empty data array, got %d items", len(data))
	}
}

// ── handleModels tests ────────────────────────────────────────────────

func TestHandleModels(t *testing.T) {
	models := []ModelConfig{
		{ID: "model-a", Name: "Model A", Path: "/a.gguf"},
		{ID: "model-b", Name: "Model B", Path: "/b.gguf"},
	}
	backends := map[string]*Backend{
		"model-a": {ID: "model-a", Port: 9001},
	}
	ps := newTestProxyServer(t, models, backends)

	req := httptest.NewRequest("GET", "/models", nil)
	w := httptest.NewRecorder()
	ps.handleModels(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)

	data, ok := result["data"].([]any)
	if !ok || len(data) != 2 {
		t.Fatalf("expected 2 models in data, got %v", result["data"])
	}

	// Verify entries have id, path, and status fields
	for _, entry := range data {
		m := entry.(map[string]any)
		if _, ok := m["id"]; !ok {
			t.Error("missing 'id' field in model entry")
		}
		if _, ok := m["path"]; !ok {
			t.Error("missing 'path' field in model entry")
		}
		status, ok := m["status"].(map[string]any)
		if !ok {
			t.Fatal("missing or invalid 'status' field")
		}
		if status["value"] == "" {
			t.Error("missing status value")
		}
	}
}

// ── handleLoaded tests ───────────────────────────────────────────────

func TestHandleLoaded(t *testing.T) {
	models := []ModelConfig{
		{ID: "model-a", Name: "Model A", Path: "/a.gguf"},
	}
	backends := map[string]*Backend{
		"model-a": {ID: "model-a", Port: 9001},
	}
	ps := newTestProxyServer(t, models, backends)

	req := httptest.NewRequest("GET", "/v1/loaded", nil)
	w := httptest.NewRecorder()
	ps.handleLoaded(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)

	data, ok := result["data"].([]any)
	if !ok || len(data) != 1 {
		t.Fatalf("expected 1 loaded model, got %d", len(data))
	}

	entry := data[0].(map[string]any)
	// handleLoaded uses Name as display name when available
	if entry["id"] != "Model A" {
		t.Errorf("expected display name 'Model A', got %v", entry["id"])
	}
	if entry["status"] != "loaded" {
		t.Errorf("expected status 'loaded', got %v", entry["status"])
	}
}

func TestHandleLoadedEmpty(t *testing.T) {
	ps := newTestProxyServer(t, nil, nil)

	req := httptest.NewRequest("GET", "/v1/loaded", nil)
	w := httptest.NewRecorder()
	ps.handleLoaded(w, req)

	resp := w.Result()
	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)
	data, _ := result["data"].([]any)
	if len(data) != 0 {
		t.Errorf("expected empty data array, got %d items", len(data))
	}
}

// ── handleModelLoad tests ────────────────────────────────────────────

func TestHandleModelLoadUnknownModel(t *testing.T) {
	models := []ModelConfig{
		{ID: "model-a", Name: "Model A", Path: "/a.gguf"},
	}
	ps := newTestProxyServer(t, models, nil)

	body := `{"model":"nonexistent"}`
	req := httptest.NewRequest("POST", "/models/load", strings.NewReader(body))
	w := httptest.NewRecorder()
	ps.handleModelLoad(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", resp.StatusCode)
	}

	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)
	if result["success"] != false {
		t.Error("expected success=false")
	}
	if !strings.Contains(result["error"].(string), "unknown model") {
		t.Errorf("expected 'unknown model' in error, got %v", result["error"])
	}
}

func TestHandleModelLoadMissingModelField(t *testing.T) {
	ps := newTestProxyServer(t, nil, nil)

	body := `{}`
	req := httptest.NewRequest("POST", "/models/load", strings.NewReader(body))
	w := httptest.NewRecorder()
	ps.handleModelLoad(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", resp.StatusCode)
	}
}

func TestHandleModelLoadInvalidJSON(t *testing.T) {
	ps := newTestProxyServer(t, nil, nil)

	body := `not json`
	req := httptest.NewRequest("POST", "/models/load", strings.NewReader(body))
	w := httptest.NewRecorder()
	ps.handleModelLoad(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", resp.StatusCode)
	}
}

func TestHandleModelLoadEmptyBody(t *testing.T) {
	ps := newTestProxyServer(t, nil, nil)

	req := httptest.NewRequest("POST", "/models/load", strings.NewReader(""))
	w := httptest.NewRecorder()
	ps.handleModelLoad(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", resp.StatusCode)
	}
}

// ── handleModelUnload tests ─────────────────────────────────────────

func TestHandleModelUnloadUnknownModel(t *testing.T) {
	models := []ModelConfig{
		{ID: "model-a", Name: "Model A", Path: "/a.gguf"},
	}
	ps := newTestProxyServer(t, models, nil)

	body := `{"model":"nonexistent"}`
	req := httptest.NewRequest("POST", "/models/unload", strings.NewReader(body))
	w := httptest.NewRecorder()
	ps.handleModelUnload(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", resp.StatusCode)
	}
}

func TestHandleModelUnloadNotLoaded(t *testing.T) {
	// Model exists in config but is not loaded → should return success
	// with a note
	models := []ModelConfig{
		{ID: "model-a", Name: "Model A", Path: "/a.gguf"},
	}
	ps := newTestProxyServer(t, models, nil)

	body := `{"model":"model-a"}`
	req := httptest.NewRequest("POST", "/models/unload", strings.NewReader(body))
	w := httptest.NewRecorder()
	ps.handleModelUnload(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}

	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)
	if result["success"] != true {
		t.Error("expected success=true")
	}
	if result["note"] == nil {
		t.Error("expected a 'note' field for not-loaded model")
	}
}

func TestHandleModelUnloadMissingModelField(t *testing.T) {
	ps := newTestProxyServer(t, nil, nil)

	body := `{}`
	req := httptest.NewRequest("POST", "/models/unload", strings.NewReader(body))
	w := httptest.NewRecorder()
	ps.handleModelUnload(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", resp.StatusCode)
	}
}

// ── handleHealth test ────────────────────────────────────────────────

func TestHandleHealth(t *testing.T) {
	ps := newTestProxyServer(t, nil, nil)

	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	ps.handleHealth(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)
	if result["status"] != "ok" {
		t.Errorf("expected status 'ok', got %v", result["status"])
	}
}

// ── writeJSONError test ──────────────────────────────────────────────

func TestWriteJSONError(t *testing.T) {
	w := httptest.NewRecorder()
	writeJSONError(w, http.StatusBadRequest, "test error message")

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected Content-Type application/json, got %q", ct)
	}
	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)
	if result["success"] != false {
		t.Error("expected success=false")
	}
	if result["error"] != "test error message" {
		t.Errorf("expected error message, got %v", result["error"])
	}
}

// ── extractModel test ────────────────────────────────────────────────

func TestExtractModel(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		want    string
		wantErr bool
	}{
		{"valid model field", `{"model":"gpt-4o","messages":[]}`, "gpt-4o", false},
		{"missing model field", `{"messages":[]}`, "", true},
		{"empty model field", `{"model":""}`, "", true},
		{"invalid JSON", `not json`, "", true},
		{"empty body", ``, "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := extractModel([]byte(tt.body))
			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

// ── isHopByHop test ──────────────────────────────────────────────────

func TestIsHopByHop(t *testing.T) {
	hopByHop := []string{"Connection", "Keep-Alive", "Proxy-Authenticate",
		"Proxy-Authorization", "TE", "Trailers", "Transfer-Encoding", "Upgrade"}
	for _, h := range hopByHop {
		if !isHopByHop(h) {
			t.Errorf("expected %q to be hop-by-hop", h)
		}
	}
	notHopByHop := []string{"Content-Type", "Authorization", "Accept", "X-Custom-Header"}
	for _, h := range notHopByHop {
		if isHopByHop(h) {
			t.Errorf("expected %q to NOT be hop-by-hop", h)
		}
	}
	// Case-insensitive
	if !isHopByHop("connection") {
		t.Error("expected case-insensitive match for 'connection'")
	}
}

// ── Routing test via handle() ────────────────────────────────────────

func TestHandleRouting(t *testing.T) {
	ps := newTestProxyServer(t, nil, nil)

	tests := []struct {
		path   string
		method string
		// We just check the handler doesn't panic and returns some JSON
		expectStatus int
	}{
		{"/health", "GET", http.StatusOK},
		{"/", "GET", http.StatusOK},
		{"/v1/models", "GET", http.StatusOK},
		{"/models", "GET", http.StatusOK},
		{"/v1/loaded", "GET", http.StatusOK},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.path, nil)
			w := httptest.NewRecorder()
			ps.handle(w, req)
			resp := w.Result()
			if resp.StatusCode != tt.expectStatus {
				t.Errorf("path %s: expected %d, got %d", tt.path, tt.expectStatus, resp.StatusCode)
			}
			// All these endpoints should return JSON
			ct := resp.Header.Get("Content-Type")
			if !strings.HasPrefix(ct, "application/json") {
				t.Errorf("path %s: expected JSON content type, got %q", tt.path, ct)
			}
		})
	}
}

// ── forceNonStream tests ─────────────────────────────────────────────

func TestForceNonStreamTrueToFalse(t *testing.T) {
	input := `{"model":"gpt-4o","stream":true,"messages":[]}`
	output := forceNonStream([]byte(input))

	var result map[string]any
	if err := json.Unmarshal(output, &result); err != nil {
		t.Fatalf("failed to parse output: %v", err)
	}
	if result["stream"] != false {
		t.Errorf("expected stream=false, got %v", result["stream"])
	}
	// Other fields should be preserved
	if result["model"] != "gpt-4o" {
		t.Errorf("expected model field preserved, got %v", result["model"])
	}
}

func TestForceNonStreamNoStreamField(t *testing.T) {
	// If stream is not present, it should be added as false
	input := `{"model":"gpt-4o","messages":[]}`
	output := forceNonStream([]byte(input))

	var result map[string]any
	json.Unmarshal(output, &result)
	if result["stream"] != false {
		t.Errorf("expected stream=false added, got %v", result["stream"])
	}
}

func TestForceNonStreamAlreadyFalse(t *testing.T) {
	input := `{"model":"gpt-4o","stream":false}`
	output := forceNonStream([]byte(input))

	var result map[string]any
	json.Unmarshal(output, &result)
	if result["stream"] != false {
		t.Errorf("expected stream=false, got %v", result["stream"])
	}
}

func TestForceNonStreamInvalidJSON(t *testing.T) {
	// Invalid JSON should be returned unchanged
	input := []byte(`not json at all`)
	output := forceNonStream(input)
	if string(output) != string(input) {
		t.Errorf("expected invalid JSON to be returned unchanged")
	}
}

func TestForceNonStreamPreservesMessages(t *testing.T) {
	input := `{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hello"}]}`
	output := forceNonStream([]byte(input))

	var result map[string]any
	json.Unmarshal(output, &result)
	messages, ok := result["messages"].([]any)
	if !ok || len(messages) != 1 {
		t.Fatalf("expected messages to be preserved, got %v", result["messages"])
	}
}

// ── forward() streaming proxy tests ─────────────────────────────────
//
// These tests verify the proxy's streaming behavior by setting up a mock
// backend HTTP server and calling forward() directly with a pre-created
// Backend pointing at it.

func TestForwardNonStreamingResponse(t *testing.T) {
	// Mock backend returns a simple JSON response (non-streaming)
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"choices":[{"message":{"content":"hello"}}]}`))
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backend := &Backend{ID: "test-model", Port: port}

	ps := newTestProxyServer(t, nil, nil)

	bodyBytes := []byte(`{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	req.Host = "test-host"
	w := httptest.NewRecorder()

	ps.forward(w, req, bodyBytes, backend)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	respBody, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(respBody), "hello") {
		t.Errorf("expected response to contain 'hello', got %s", string(respBody))
	}
}

func TestForwardStreamingResponse(t *testing.T) {
	// Mock backend returns SSE chunks
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		// Write multiple SSE chunks
		chunks := []string{
			`data: {"choices":[{"delta":{"content":"Hello"}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":" World"}}]}` + "\n\n",
			`data: [DONE]` + "\n\n",
		}
		for _, chunk := range chunks {
			w.Write([]byte(chunk))
			if flusher != nil {
				flusher.Flush()
			}
		}
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backend := &Backend{ID: "test-model", Port: port}

	ps := newTestProxyServer(t, nil, nil)

	bodyBytes := []byte(`{"model":"test-model","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	w := httptest.NewRecorder()

	ps.forward(w, req, bodyBytes, backend)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	respBody, _ := io.ReadAll(resp.Body)
	bodyStr := string(respBody)
	// Should contain all SSE chunks
	if !strings.Contains(bodyStr, "Hello") {
		t.Errorf("expected 'Hello' in response, got %s", bodyStr)
	}
	if !strings.Contains(bodyStr, " World") {
		t.Errorf("expected ' World' in response, got %s", bodyStr)
	}
	if !strings.Contains(bodyStr, "[DONE]") {
		t.Errorf("expected [DONE] in response, got %s", bodyStr)
	}
}

func TestForwardPreservesHeaders(t *testing.T) {
	var receivedAuth string
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backend := &Backend{ID: "test-model", Port: port}
	ps := newTestProxyServer(t, nil, nil)

	bodyBytes := []byte(`{"model":"test-model"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	req.Header.Set("Authorization", "Bearer sk-test-token")
	w := httptest.NewRecorder()

	ps.forward(w, req, bodyBytes, backend)

	if receivedAuth != "Bearer sk-test-token" {
		t.Errorf("expected Authorization header forwarded, got %q", receivedAuth)
	}
}

func TestForwardForwardsQueryParams(t *testing.T) {
	var receivedPath string
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path + "?" + r.URL.RawQuery
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backend := &Backend{ID: "test-model", Port: port}
	ps := newTestProxyServer(t, nil, nil)

	bodyBytes := []byte(`{"model":"test-model"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions?foo=bar&baz=1", strings.NewReader(string(bodyBytes)))
	w := httptest.NewRecorder()

	ps.forward(w, req, bodyBytes, backend)

	if !strings.Contains(receivedPath, "foo=bar") || !strings.Contains(receivedPath, "baz=1") {
		t.Errorf("expected query params forwarded, got %s", receivedPath)
	}
}

func TestForwardXForwardedFor(t *testing.T) {
	var receivedXFF string
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedXFF = r.Header.Get("X-Forwarded-For")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backend := &Backend{ID: "test-model", Port: port}
	ps := newTestProxyServer(t, nil, nil)

	bodyBytes := []byte(`{"model":"test-model"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	req.Header.Set("X-Forwarded-For", "10.0.0.1")
	w := httptest.NewRecorder()

	ps.forward(w, req, bodyBytes, backend)

	// Should append, not overwrite
	if !strings.Contains(receivedXFF, "10.0.0.1") {
		t.Errorf("expected original XFF preserved, got %q", receivedXFF)
	}
	// Should also contain the remote addr from the test request
	if !strings.Contains(receivedXFF, ",") {
		t.Errorf("expected appended XFF (comma-separated), got %q", receivedXFF)
	}
}

func TestForwardBackendError(t *testing.T) {
	// Backend returns an error status code
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte(`{"error":"model overloaded"}`))
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backend := &Backend{ID: "test-model", Port: port}
	ps := newTestProxyServer(t, nil, nil)

	bodyBytes := []byte(`{"model":"test-model"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	w := httptest.NewRecorder()

	ps.forward(w, req, bodyBytes, backend)

	resp := w.Result()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("expected 503, got %d", resp.StatusCode)
	}
}

func TestForwardBackendUnreachable(t *testing.T) {
	// Point at a port that's definitely not listening
	backend := &Backend{ID: "test-model", Port: 1} // port 1: privileged, unlikely to respond
	ps := newTestProxyServer(t, nil, nil)

	bodyBytes := []byte(`{"model":"test-model"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	w := httptest.NewRecorder()

	ps.forward(w, req, bodyBytes, backend)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("expected 502 Bad Gateway for unreachable backend, got %d", resp.StatusCode)
	}
}

func TestForwardCopiesResponseHeaders(t *testing.T) {
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Custom-Header", "custom-value")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backend := &Backend{ID: "test-model", Port: port}
	ps := newTestProxyServer(t, nil, nil)

	bodyBytes := []byte(`{"model":"test-model"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	w := httptest.NewRecorder()

	ps.forward(w, req, bodyBytes, backend)

	resp := w.Result()
	if resp.Header.Get("X-Custom-Header") != "custom-value" {
		t.Errorf("expected custom header forwarded, got %q", resp.Header.Get("X-Custom-Header"))
	}
}

func TestForwardStripsHopByHopHeaders(t *testing.T) {
	// The proxy should not forward hop-by-hop headers from the response
	var receivedConn string
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedConn = r.Header.Get("Connection")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backend := &Backend{ID: "test-model", Port: port}
	ps := newTestProxyServer(t, nil, nil)

	bodyBytes := []byte(`{"model":"test-model"}`)
	// Add a Connection header to the request
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	req.Header.Set("Connection", "keep-alive")
	w := httptest.NewRecorder()

	ps.forward(w, req, bodyBytes, backend)

	// Connection header should NOT be forwarded to backend
	if receivedConn != "" {
		t.Errorf("expected Connection header stripped, got %q", receivedConn)
	}
}

// ── forwardWithHandler() tests ───────────────────────────────────────

// noopHandler is a test handler that doesn't modify the response.
type noopHandler struct{}

func (h *noopHandler) MatchesModel(modelID string) bool { return true }
func (h *noopHandler) ProcessResponse(resp *http.Response, isStream bool) ([]byte, error) {
	return []byte(`{"processed":true}`), nil
}

func TestForwardWithHandlerForcesNonStream(t *testing.T) {
	var receivedStream any
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		receivedStream = req["stream"]
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"choices":[{"message":{"content":"hi"}}]}`))
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backend := &Backend{ID: "test-model", Port: port}
	ps := newTestProxyServer(t, nil, nil)

	// Request with stream:true should be forced to stream:false
	bodyBytes := []byte(`{"model":"test-model","stream":true,"messages":[]}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	w := httptest.NewRecorder()

	handler := &noopHandler{}
	ps.forwardWithHandler(w, req, bodyBytes, backend, handler)

	if receivedStream != false {
		t.Errorf("expected stream to be forced to false in backend request, got %v", receivedStream)
	}
}

func TestForwardWithHandlerReturnsProcessedResponse(t *testing.T) {
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"choices":[{"message":{"content":"original"}}]}`))
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backend := &Backend{ID: "test-model", Port: port}
	ps := newTestProxyServer(t, nil, nil)

	bodyBytes := []byte(`{"model":"test-model","stream":true}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	w := httptest.NewRecorder()

	handler := &noopHandler{}
	ps.forwardWithHandler(w, req, bodyBytes, backend, handler)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	respBody, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(respBody), `"processed":true`) {
		t.Errorf("expected processed response from handler, got %s", string(respBody))
	}
}

// erroringHandler returns an error to test the fallback path.
type erroringHandler struct{}

func (h *erroringHandler) MatchesModel(modelID string) bool { return true }
func (h *erroringHandler) ProcessResponse(resp *http.Response, isStream bool) ([]byte, error) {
	return nil, fmt.Errorf("processing error")
}

func TestForwardWithHandlerFallbackOnError(t *testing.T) {
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"choices":[{"message":{"content":"fallback"}}]}`))
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backend := &Backend{ID: "test-model", Port: port}
	ps := newTestProxyServer(t, nil, nil)

	bodyBytes := []byte(`{"model":"test-model","stream":true}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	w := httptest.NewRecorder()

	handler := &erroringHandler{}
	ps.forwardWithHandler(w, req, bodyBytes, backend, handler)

	// On handler error, should fall back to raw forwarding
	resp := w.Result()
	respBody, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(respBody), "fallback") {
		t.Errorf("expected fallback response on handler error, got %s", string(respBody))
	}
}

// ── Backend.Touch / lastRequest / BaseURL tests ─────────────────────

func TestBackendTouchAndLastRequest(t *testing.T) {
	b := &Backend{ID: "test", Port: 8080}
	// Set last request to a known time
	past := time.Now().Add(-1 * time.Hour)
	b.setLastRequest(past)
	if got := b.lastRequest(); !got.Equal(past) {
		t.Errorf("expected lastRequest %v, got %v", past, got)
	}
	// Touch should update to ~now
	b.Touch()
	if got := b.lastRequest(); !got.After(past) {
		t.Errorf("expected Touch to update lastRequest to after %v, got %v", past, got)
	}
}

func TestBackendBaseURL(t *testing.T) {
	b := &Backend{ID: "test", Port: 8301}
	expected := "http://127.0.0.1:8301"
	if got := b.BaseURL(); got != expected {
		t.Errorf("expected %q, got %q", expected, got)
	}
}

// net.TCPAddr is a helper type alias to extract port from httptest server.
// We use net.TCPAddr via type assertion.
