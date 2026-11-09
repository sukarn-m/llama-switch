package main

import (
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ── Origin allow-list ────────────────────────────────────────────────

func TestIsAllowedOrigin(t *testing.T) {
	cfg := &Config{Server: ServerConfig{
		CORSAllowOrigins: []string{"https://app.example", "http://localhost:3000"},
	}}

	tests := []struct {
		origin string
		want   bool
	}{
		{"https://app.example", true},
		{"HTTPS://APP.EXAMPLE", true},
		{"http://localhost:3000", true},
		{"https://app.example.evil.com", false},  // no prefix matching
		{"https://evil.com", false},
		{"", false},
	}
	for _, tc := range tests {
		if got := cfg.isAllowedOrigin(tc.origin); got != tc.want {
			t.Errorf("isAllowedOrigin(%q) = %v, want %v", tc.origin, got, tc.want)
		}
	}
}

func TestIsAllowedOriginWildcard(t *testing.T) {
	cfg := &Config{Server: ServerConfig{CORSAllowOrigins: []string{"*"}}}
	if !cfg.isAllowedOrigin("https://anything.example") {
		t.Error("expected wildcard to allow any origin")
	}
	if !cfg.allowsAnyOrigin() {
		t.Error("expected allowsAnyOrigin to be true")
	}
}

func TestIsAllowedOriginNone(t *testing.T) {
	cfg := &Config{Server: ServerConfig{CORSAllowOrigins: []string{"none"}}}
	if cfg.isAllowedOrigin("https://app.example") {
		t.Error("expected 'none' to allow no origin")
	}
}

func TestIsAllowedOriginLocalhostSpecialValue(t *testing.T) {
	cfg := &Config{Server: ServerConfig{CORSAllowOrigins: []string{"localhost"}}}

	allowed := []string{
		"http://localhost",
		"http://localhost:3000",
		"https://localhost:8443",
		"http://127.0.0.1:8080",
		"http://[::1]:8080",
	}
	for _, o := range allowed {
		if !cfg.isAllowedOrigin(o) {
			t.Errorf("expected localhost origin %q to be allowed", o)
		}
	}

	denied := []string{
		"https://app.example",
		"http://localhost.evil.com",
		"http://notlocalhost",
	}
	for _, o := range denied {
		if cfg.isAllowedOrigin(o) {
			t.Errorf("expected origin %q to be denied", o)
		}
	}
}

func TestEmptyOriginListAllowsNothing(t *testing.T) {
	cfg := &Config{}
	if cfg.isAllowedOrigin("https://app.example") {
		t.Error("expected empty allow-list to deny all origins")
	}
}

// ── Secret expansion ─────────────────────────────────────────────────

func TestExpandSecretResolvesEnvVar(t *testing.T) {
	t.Setenv("LS_TEST_SECRET", "s3cr3t")
	if got := expandSecret("$LS_TEST_SECRET"); got != "s3cr3t" {
		t.Errorf("expected $VAR to resolve, got %q", got)
	}
	if got := expandSecret("${LS_TEST_SECRET}"); got != "s3cr3t" {
		t.Errorf("expected ${VAR} to resolve, got %q", got)
	}
	if got := expandSecret("prefix-$LS_TEST_SECRET"); got != "prefix-s3cr3t" {
		t.Errorf("expected embedded $VAR to resolve, got %q", got)
	}
}

func TestExpandSecretLeavesColonAndTildeAlone(t *testing.T) {
	// expand() treats ":" as a path-list separator and "~" as a home
	// directory; a secret must survive both untouched.
	in := "a:b~c"
	if got := expandSecret(in); got != in {
		t.Errorf("expected secret %q to be unchanged, got %q", in, got)
	}
}

// ── Forward-path allow-list ──────────────────────────────────────────

func TestForwardPathAllowed(t *testing.T) {
	cfg := &Config{}

	allowed := []string{
		"/v1/chat/completions",
		"/chat/completions",
		"/v1/completions",
		"/v1/embeddings",
		"/v1/messages",
		"/v1/responses",
		"/v1/rerank",
		"/v1/audio/transcriptions",
		"/tokenize",
		"/detokenize",
		"/apply-template",
	}
	for _, p := range allowed {
		if !cfg.forwardPathAllowed(p) {
			t.Errorf("expected %q to be forwardable", p)
		}
	}

	// llama.cpp's administrative surface must not be reachable.
	denied := []string{
		"/props",
		"/slots",
		"/slots/0",
		"/lora-adapters",
		"/cors-proxy",
		"/tools",
		"/metrics",
		"/models",
		"/models/load",
		"/admin/profile",
		"/",
	}
	for _, p := range denied {
		if cfg.forwardPathAllowed(p) {
			t.Errorf("expected %q to be blocked", p)
		}
	}
}

func TestForwardPathsOverride(t *testing.T) {
	cfg := &Config{Server: ServerConfig{ForwardPaths: []string{"/custom"}}}
	if !cfg.forwardPathAllowed("/custom") {
		t.Error("expected custom path to be allowed")
	}
	if cfg.forwardPathAllowed("/v1/chat/completions") {
		t.Error("expected default paths to be replaced by the override")
	}
}

// ── CORS headers ─────────────────────────────────────────────────────

func TestCORSAllowedOriginGetsHeader(t *testing.T) {
	cfg := &Config{Server: ServerConfig{
		CORSAllowOrigins: []string{"https://app.example"},
	}}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Origin", "https://app.example")
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "https://app.example" {
		t.Errorf("expected ACAO to be the allowed origin, got %q", got)
	}
	if got := w.Header().Get("Vary"); !strings.Contains(got, "Origin") {
		t.Errorf("expected Vary to contain Origin, got %q", got)
	}
}

func TestCORSDisallowedOriginGetsNoHeader(t *testing.T) {
	cfg := &Config{Server: ServerConfig{
		CORSAllowOrigins: []string{"https://app.example"},
	}}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Origin", "https://evil.example")
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Errorf("expected no ACAO header for disallowed origin, got %q", got)
	}
}

func TestCORSPreflightIsAnsweredNotForwarded(t *testing.T) {
	cfg := &Config{Server: ServerConfig{
		CORSAllowOrigins:     []string{"https://app.example"},
		CORSPreflightMaxAge: 600,
	}}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("OPTIONS", "/v1/chat/completions", nil)
	req.Header.Set("Origin", "https://app.example")
	req.Header.Set("Access-Control-Request-Method", "POST")
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if w.Code != http.StatusNoContent {
		t.Errorf("expected 204 for preflight, got %d", w.Code)
	}
	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "https://app.example" {
		t.Errorf("expected ACAO on preflight, got %q", got)
	}
	if got := w.Header().Get("Access-Control-Allow-Methods"); !strings.Contains(got, "POST") {
		t.Errorf("expected POST in Allow-Methods, got %q", got)
	}
	if got := w.Header().Get("Access-Control-Max-Age"); got != "600" {
		t.Errorf("expected Max-Age 600, got %q", got)
	}
}

func TestCORSPreflightFromDisallowedOriginIsRejected(t *testing.T) {
	cfg := &Config{Server: ServerConfig{
		CORSAllowOrigins: []string{"https://app.example"},
	}}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("OPTIONS", "/v1/chat/completions", nil)
	req.Header.Set("Origin", "https://evil.example")
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("expected 403 for preflight from disallowed origin, got %d", w.Code)
	}
	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Errorf("expected no ACAO header on rejected preflight, got %q", got)
	}
}

func TestCORSWildcardWithCredentialsEchoesOrigin(t *testing.T) {
	cfg := &Config{Server: ServerConfig{
		CORSAllowOrigins:     []string{"*"},
		CORSAllowCredentials: true,
	}}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Origin", "https://app.example")
	w := httptest.NewRecorder()

	ps.handle(w, req)

	// "*" is invalid alongside credentials, so the origin must be echoed.
	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "https://app.example" {
		t.Errorf("expected echoed origin, got %q", got)
	}
	if got := w.Header().Get("Access-Control-Allow-Credentials"); got != "true" {
		t.Errorf("expected Allow-Credentials true, got %q", got)
	}
}

func TestCORSWildcardWithoutCredentialsUsesStar(t *testing.T) {
	cfg := &Config{Server: ServerConfig{CORSAllowOrigins: []string{"*"}}}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Origin", "https://app.example")
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("expected ACAO *, got %q", got)
	}
	if got := w.Header().Get("Access-Control-Allow-Credentials"); got != "" {
		t.Errorf("expected no Allow-Credentials, got %q", got)
	}
}

// ── Origin check (CSRF) ──────────────────────────────────────────────

func TestStateChangeFromDisallowedOriginIsRejected(t *testing.T) {
	cfg := &Config{Server: ServerConfig{
		CORSAllowOrigins: []string{"https://app.example"},
	}}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("POST", "/models/unload", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Origin", "https://evil.example")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("expected 403 for cross-origin state change, got %d", w.Code)
	}
}

func TestStateChangeFromAllowedOriginPasses(t *testing.T) {
	cfg := &Config{Server: ServerConfig{
		CORSAllowOrigins: []string{"https://app.example"},
	}}
	models := []ModelConfig{{ID: "model-a", Name: "Model A", Path: "/a.gguf"}}
	ps := newTestProxyServer(t, models, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("POST", "/models/unload", strings.NewReader(`{"model":"model-a"}`))
	req.Header.Set("Origin", "https://app.example")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if w.Code == http.StatusForbidden {
		t.Error("expected allowed origin to pass the origin check")
	}
}

func TestStateChangeWithoutOriginPasses(t *testing.T) {
	// Server-to-server clients (curl, Hermes, opencode) send no Origin.
	models := []ModelConfig{{ID: "model-a", Name: "Model A", Path: "/a.gguf"}}
	ps := newTestProxyServer(t, models, nil)
	ps.cfg = &Config{Server: ServerConfig{CORSAllowOrigins: []string{"https://app.example"}}}

	req := httptest.NewRequest("POST", "/models/unload", strings.NewReader(`{"model":"model-a"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if w.Code == http.StatusForbidden {
		t.Error("expected origin-less request to pass")
	}
}

func TestGetFromDisallowedOriginIsNotRejected(t *testing.T) {
	// GETs are safe (no side effects); the browser blocks the response
	// because no ACAO header is returned.
	cfg := &Config{Server: ServerConfig{
		CORSAllowOrigins: []string{"https://app.example"},
	}}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Origin", "https://evil.example")
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if w.Code == http.StatusForbidden {
		t.Error("expected GET from disallowed origin not to be rejected outright")
	}
}

// ── API key authentication ───────────────────────────────────────────

func TestAPIKeyRequiredWhenConfigured(t *testing.T) {
	cfg := &Config{Server: ServerConfig{APIKey: "secret123"}}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 without API key, got %d", w.Code)
	}
	if got := w.Header().Get("WWW-Authenticate"); got == "" {
		t.Error("expected WWW-Authenticate header")
	}
}

func TestAPIKeyAcceptedAsBearer(t *testing.T) {
	cfg := &Config{Server: ServerConfig{APIKey: "secret123"}}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer secret123")
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if w.Code == http.StatusUnauthorized {
		t.Errorf("expected bearer token to authenticate, got %d", w.Code)
	}
}

func TestAPIKeyAcceptedAsXApiKey(t *testing.T) {
	cfg := &Config{Server: ServerConfig{APIKey: "secret123"}}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("X-Api-Key", "secret123")
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if w.Code == http.StatusUnauthorized {
		t.Errorf("expected X-Api-Key to authenticate, got %d", w.Code)
	}
}

func TestAPIKeyWrongValueRejected(t *testing.T) {
	cfg := &Config{Server: ServerConfig{APIKey: "secret123"}}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer wrong")
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for wrong key, got %d", w.Code)
	}
}

func TestHealthIsExemptFromAPIKey(t *testing.T) {
	cfg := &Config{Server: ServerConfig{APIKey: "secret123"}}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = cfg

	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected /health to be reachable without a key, got %d", w.Code)
	}
}

func TestNoAPIKeyConfiguredAllowsAll(t *testing.T) {
	ps := newTestProxyServer(t, nil, nil)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if w.Code == http.StatusUnauthorized {
		t.Error("expected no auth when no key is configured")
	}
}

// ── Backend CORS headers are stripped ─────────────────────────────────

func TestBackendCORSHeadersAreStripped(t *testing.T) {
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// A backend started with --cors-origins * would emit these.
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		w.Header().Set("Content-Type", "application/json")
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
	for _, h := range []string{
		"Access-Control-Allow-Origin",
		"Access-Control-Allow-Credentials",
		"Access-Control-Allow-Methods",
	} {
		if got := resp.Header.Get(h); got != "" {
			t.Errorf("expected backend %s to be stripped, got %q", h, got)
		}
	}
}

// ── Backend authentication ───────────────────────────────────────────

func TestBackendAPIKeyIsSentToBackend(t *testing.T) {
	var receivedAuth string
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backend := &Backend{ID: "test-model", Port: port, apiKey: "backend-key"}
	ps := newTestProxyServer(t, nil, nil)

	bodyBytes := []byte(`{"model":"test-model"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	req.Header.Set("Authorization", "Bearer client-key")
	w := httptest.NewRecorder()

	ps.forward(w, req, bodyBytes, backend)

	if receivedAuth != "Bearer backend-key" {
		t.Errorf("expected backend key to be sent, got %q", receivedAuth)
	}
}

func TestClientAPIKeyIsNotForwardedToBackend(t *testing.T) {
	var receivedAuth, receivedXKey string
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		receivedXKey = r.Header.Get("X-Api-Key")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backend := &Backend{ID: "test-model", Port: port}
	ps := newTestProxyServer(t, nil, nil)
	ps.cfg = &Config{Server: ServerConfig{APIKey: "proxy-key"}}

	bodyBytes := []byte(`{"model":"test-model"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	req.Header.Set("Authorization", "Bearer proxy-key")
	req.Header.Set("X-Api-Key", "proxy-key")
	w := httptest.NewRecorder()

	ps.forward(w, req, bodyBytes, backend)

	if receivedAuth != "" {
		t.Errorf("expected client Authorization not to reach backend, got %q", receivedAuth)
	}
	if receivedXKey != "" {
		t.Errorf("expected client X-Api-Key not to reach backend, got %q", receivedXKey)
	}
}

func TestNoKeysConfiguredLeavesClientHeadersIntact(t *testing.T) {
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
	req.Header.Set("Authorization", "Bearer client-token")
	w := httptest.NewRecorder()

	ps.forward(w, req, bodyBytes, backend)

	if receivedAuth != "Bearer client-token" {
		t.Errorf("expected client Authorization to be forwarded unchanged, got %q", receivedAuth)
	}
}

// ── Blocked paths ────────────────────────────────────────────────────

func TestBlockedPathReturnsForbidden(t *testing.T) {
	ps := newTestProxyServer(t, nil, nil)

	req := httptest.NewRequest("POST", "/props", strings.NewReader(`{"model":"m"}`))
	w := httptest.NewRecorder()

	ps.handle(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("expected 403 for /props, got %d", w.Code)
	}
}

func TestBlockedPathDoesNotReachBackend(t *testing.T) {
	called := false
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	defer backendSrv.Close()

	port := backendSrv.Listener.Addr().(*net.TCPAddr).Port
	backends := map[string]*Backend{
		"test-model": {ID: "test-model", Port: port},
	}
	models := []ModelConfig{{ID: "test-model", Name: "Test Model", Path: "/t.gguf"}}
	ps := newTestProxyServer(t, models, backends)

	bodyBytes := []byte(`{"model":"test-model"}`)
	req := httptest.NewRequest("POST", "/lora-adapters", strings.NewReader(string(bodyBytes)))
	w := httptest.NewRecorder()

	ps.handleProxy(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("expected 403 for /lora-adapters, got %d", w.Code)
	}
	if called {
		t.Error("backend was called for a blocked path")
	}
}
