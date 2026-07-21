package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// ProxyServer is the HTTP server that clients connect to.
type ProxyServer struct {
	cfg    *Config
	bm     *BackendManager
	srv    *http.Server
	client *http.Client // shared client for backend requests
}

func NewProxyServer(cfg *Config, bm *BackendManager) *ProxyServer {
	ps := &ProxyServer{
		cfg: cfg,
		bm:  bm,
		client: &http.Client{
			Timeout: 0, // no timeout — streaming and long inference
		},
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", ps.handle)
	ps.srv = &http.Server{
		Handler:           mux,
		ReadTimeout:       0,                // streaming bodies can be long-lived
		WriteTimeout:      0,                // streaming responses can be long-lived
		ReadHeaderTimeout: 10 * time.Second, // prevents slowloris
	}
	return ps
}

func (ps *ProxyServer) ListenAndServe() error {
	addr := fmt.Sprintf("%s:%d", ps.cfg.Server.Host, ps.cfg.Server.Port)
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	return ps.srv.Serve(ln)
}

func (ps *ProxyServer) Shutdown() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ps.srv.Shutdown(ctx)
}

// ── Request handling ─────────────────────────

func (ps *ProxyServer) handle(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/" || r.URL.Path == "/health" {
		ps.handleHealth(w, r)
		return
	}
	if r.URL.Path == "/v1/models" {
		ps.handleV1Models(w, r)
		return
	}
	if r.URL.Path == "/models" && r.Method == "GET" {
		ps.handleModels(w, r)
		return
	}
	if r.URL.Path == "/models/load" && r.Method == "POST" {
		ps.handleModelLoad(w, r)
		return
	}
	if r.URL.Path == "/models/unload" && r.Method == "POST" {
		ps.handleModelUnload(w, r)
		return
	}
	if r.URL.Path == "/v1/loaded" {
		ps.handleLoaded(w, r)
		return
	}
	if r.URL.Path == "/admin/profile" && r.Method == "POST" {
		ps.handleAdminProfile(w, r)
		return
	}
	ps.handleProxy(w, r)
}

func (ps *ProxyServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"status": "ok"})
}

// handleLoaded returns just the loaded models (legacy endpoint).
func (ps *ProxyServer) handleLoaded(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	loaded := ps.bm.LoadedModels()
	data := make([]map[string]any, 0, len(loaded))
	for _, id := range loaded {
		displayName := id
		if mc := ps.cfg.FindModel(id); mc != nil {
			displayName = mc.Name
		}
		data = append(data, map[string]any{
			"id":     displayName,
			"status": "loaded",
		})
	}
	json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   data,
	})
}

func (ps *ProxyServer) handleV1Models(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	loaded := ps.bm.LoadedModelsSet()
	models := make([]map[string]any, 0, len(ps.cfg.Models))
	for _, m := range ps.cfg.Models {
		statusVal := "unloaded"
		if loaded[m.ID] {
			statusVal = "loaded"
		}
		models = append(models, map[string]any{
			"id":       m.Name,
			"object":   "model",
			"created":  0,
			"owned_by": "llama-switch",
			"status":   map[string]any{"value": statusVal},
		})
	}
	json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   models,
	})
}

func (ps *ProxyServer) handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	loaded := ps.bm.LoadedModelsSet()
	data := make([]map[string]any, 0, len(ps.cfg.Models))
	for _, m := range ps.cfg.Models {
		statusVal := "unloaded"
		if loaded[m.ID] {
			statusVal = "loaded"
		}
		entry := map[string]any{
			"id":     m.Name,
			"path":   expand(m.Path),
			"status": map[string]any{"value": statusVal},
		}
		data = append(data, entry)
	}
	json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   data,
	})
}

func (ps *ProxyServer) handleModelLoad(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1 MB
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil || req.Model == "" {
		writeJSONError(w, http.StatusBadRequest, "missing or invalid 'model' field")
		return
	}

	if ps.cfg.FindModel(req.Model) == nil {
		writeJSONError(w, http.StatusBadRequest, "unknown model: "+req.Model)
		return
	}

	if ps.bm.IsDraining() {
		w.Header().Set("Retry-After", "60")
		writeJSONError(w, http.StatusServiceUnavailable,
			"service is profiling models, please retry later")
		return
	}

	_, err = ps.bm.EnsureLoaded(req.Model)
	if err != nil {
		w.Header().Set("Retry-After", "30")
		writeJSONError(w, http.StatusServiceUnavailable,
			fmt.Sprintf("failed to load model %s: %v", req.Model, err))
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"success": true})
}

func (ps *ProxyServer) handleModelUnload(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1 MB
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil || req.Model == "" {
		writeJSONError(w, http.StatusBadRequest, "missing or invalid 'model' field")
		return
	}

	if ps.cfg.FindModel(req.Model) == nil {
		writeJSONError(w, http.StatusBadRequest, "unknown model: "+req.Model)
		return
	}

	// Check if loaded first, so we distinguish "not loaded" from real errors
	if !ps.bm.IsLoaded(req.Model) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"success": true, "note": "model was not loaded"})
		return
	}

	err = ps.bm.StopModel(req.Model)
	if err != nil {
		if strings.Contains(err.Error(), "active request(s)") {
			writeJSONError(w, http.StatusConflict,
				fmt.Sprintf("failed to unload model %s: %v", req.Model, err))
		} else {
			writeJSONError(w, http.StatusInternalServerError,
				fmt.Sprintf("failed to unload model %s: %v", req.Model, err))
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"success": true})
}

func writeJSONError(w http.ResponseWriter, code int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{
		"success": false,
		"error":   message,
	})
}

// handleAdminProfile triggers VRAM profiling on the running service.
// It drains active requests, evicts all backends, profiles each model
// one at a time, and streams progress as SSE events. The service is
// unavailable for proxied requests during profiling.
func (ps *ProxyServer) handleAdminProfile(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSONError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	force := r.URL.Query().Get("force") == "true"

	progress := func(format string, args ...any) {
		fmt.Fprintf(w, "data: "+format+"\n\n", args...)
		flusher.Flush()
	}

	progress("Profiling VRAM for %d models (one at a time)", len(ps.cfg.Models))

	_, err := ps.bm.ProfileModels(force, progress)
	if err != nil {
		progress("Profiling error: %v", err)
		return
	}
}

// ── Core proxy logic ─────────────────────────

func (ps *ProxyServer) handleProxy(w http.ResponseWriter, r *http.Request) {
	// Limit request body size (SEC-1)
	r.Body = http.MaxBytesReader(w, r.Body, 100<<20) // 100 MB

	var bodyBytes []byte
	var err error
	if r.Body != nil {
		bodyBytes, err = io.ReadAll(r.Body)
		if err != nil {
			var mbe *http.MaxBytesError
			if errors.As(err, &mbe) {
				http.Error(w, "request body too large (100MB limit)", http.StatusRequestEntityTooLarge)
			} else {
				http.Error(w, "failed to read request body", http.StatusBadRequest)
			}
			return
		}
		r.Body.Close()
	}

	modelID, err := extractModel(bodyBytes)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("could not determine model: %v", err))
		return
	}

	mc := ps.cfg.FindModel(modelID)
	if mc == nil {
		writeJSONError(w, http.StatusBadRequest,
			fmt.Sprintf("unknown model: %s. Use GET /v1/models to list available models.", modelID))
		return
	}

	// During profiling/maintenance, reject proxied requests
	if ps.bm.IsDraining() {
		w.Header().Set("Retry-After", "60")
		writeJSONError(w, http.StatusServiceUnavailable,
			"service is profiling models, please retry later")
		return
	}

	backend, err := ps.bm.EnsureLoaded(modelID)
	if err != nil {
		w.Header().Set("Retry-After", "30")
		writeJSONError(w, http.StatusServiceUnavailable,
			fmt.Sprintf("failed to load model %s: %v", modelID, err))
		return
	}

	backend.Touch()
	backend.InFlightAdd()
	defer func() {
		backend.InFlightDone()
		ps.bm.SignalCapacity()
	}()

	// Check if this model has a registered handler for response post-processing
	handler := NewModelHandler(modelID)
	if handler != nil {
		ps.forwardWithHandler(w, r, bodyBytes, backend, handler)
		return
	}

	ps.forward(w, r, bodyBytes, backend)
}

// forward proxies the request to the backend. Always uses the streaming
// path (flush after each write) so SSE works regardless of how the client
// formatted the JSON body. (HTTP-1 fix)
func (ps *ProxyServer) forward(w http.ResponseWriter, r *http.Request, bodyBytes []byte, backend *Backend) {
	target := backend.BaseURL() + r.URL.Path
	if r.URL.RawQuery != "" {
		target += "?" + r.URL.RawQuery
	}

	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, target, bytes.NewReader(bodyBytes))
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to create request: %v", err), http.StatusInternalServerError)
		return
	}

	// Copy headers (except hop-by-hop)
	for key, values := range r.Header {
		if isHopByHop(key) {
			continue
		}
		for _, v := range values {
			proxyReq.Header.Add(key, v)
		}
	}
	// Forward the original Host (HTTP-2 fix: use proxyReq.Host, not Header.Set)
	proxyReq.Host = r.Host

	// Add forwarding headers (HTTP-3)
	// Append to X-Forwarded-For instead of overwriting (BUG-R2-5)
	if prior := r.Header.Get("X-Forwarded-For"); prior != "" {
		proxyReq.Header.Set("X-Forwarded-For", prior+", "+r.RemoteAddr)
	} else {
		proxyReq.Header.Set("X-Forwarded-For", r.RemoteAddr)
	}
	proxyReq.Header.Set("X-Forwarded-Proto", "http")

	// Use shared client (HTTP-4 fix)
	resp, err := ps.client.Do(proxyReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("backend request failed: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Copy response headers
	for key, values := range resp.Header {
		if isHopByHop(key) {
			continue
		}
		for _, v := range values {
			w.Header().Add(key, v)
		}
	}

	w.WriteHeader(resp.StatusCode)

	// Always flush after each write so SSE streaming works (HTTP-1 fix).
	// The overhead for non-streaming responses is negligible.
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return // client disconnected
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			break
		}
	}
}

// forwardWithHandler proxies the request to the backend, then passes
// the response through the model handler for post-processing. The handler
// reads the full response, transforms it, and returns a modified JSON body.
// Streaming responses are converted to non-streaming (the handler needs
// the complete output to process it correctly).
func (ps *ProxyServer) forwardWithHandler(w http.ResponseWriter, r *http.Request, bodyBytes []byte, backend *Backend, handler ModelHandler) {
	target := backend.BaseURL() + r.URL.Path
	if r.URL.RawQuery != "" {
		target += "?" + r.URL.RawQuery
	}

	// Force non-streaming by removing "stream":true from the request body.
	// The handler needs the complete response to process it.
	modifiedBody := forceNonStream(bodyBytes)

	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, target, bytes.NewReader(modifiedBody))
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to create request: %v", err), http.StatusInternalServerError)
		return
	}

	// Copy headers (except hop-by-hop)
	for key, values := range r.Header {
		if isHopByHop(key) {
			continue
		}
		for _, v := range values {
			proxyReq.Header.Add(key, v)
		}
	}
	proxyReq.Host = r.Host
	if prior := r.Header.Get("X-Forwarded-For"); prior != "" {
		proxyReq.Header.Set("X-Forwarded-For", prior+", "+r.RemoteAddr)
	} else {
		proxyReq.Header.Set("X-Forwarded-For", r.RemoteAddr)
	}
	proxyReq.Header.Set("X-Forwarded-Proto", "http")

	resp, err := ps.client.Do(proxyReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("backend request failed: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Process the response through the handler
	processed, err := handler.ProcessResponse(resp, false)
	if err != nil {
		// On error, fall back to raw forwarding
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		raw, _ := io.ReadAll(resp.Body)
		w.Write(raw)
		return
	}

	// Return the processed response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(processed)
}

// forceNonStream sets "stream":false in the JSON body so the backend
// returns a single JSON response instead of SSE chunks.
func forceNonStream(body []byte) []byte {
	var data map[string]any
	if err := json.Unmarshal(body, &data); err != nil {
		return body
	}
	data["stream"] = false
	out, err := json.Marshal(data)
	if err != nil {
		return body
	}
	return out
}

// ── Helpers ──────────────────────────────────

func extractModel(body []byte) (string, error) {
	var partial struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &partial); err != nil {
		return "", fmt.Errorf("invalid JSON body: %w", err)
	}
	if partial.Model == "" {
		return "", fmt.Errorf("missing or empty 'model' field in request body")
	}
	return partial.Model, nil
}

func isHopByHop(header string) bool {
	switch strings.ToLower(header) {
	case "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
		"te", "trailers", "transfer-encoding", "upgrade":
		return true
	}
	return false
}
