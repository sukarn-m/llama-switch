package main

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// ── Security policy ──────────────────────────
//
// The proxy is the only client of llama-server, so it is the natural
// place to enforce the security policy that used to be delegated to
// llama-server's own (much looser) defaults:
//
//   - CORS is decided here, not inherited from the backend. The backend's
//     Access-Control-* response headers are stripped before they reach
//     the client, so a backend started with "--cors-origins *" cannot
//     widen the proxy's policy.
//   - State-changing requests that carry an Origin header from a browser
//     are rejected unless that origin is allow-listed. This is the CSRF
//     defence for /models/load, /models/unload and /admin/profile.
//   - Only known inference endpoints are forwarded to the backend.
//     llama.cpp's administrative surface (/props, /slots, /lora-adapters,
//     /cors-proxy, /tools, /metrics) is not reachable through the proxy.
//   - An optional client API key protects every endpoint but /health.

// defaultForwardPaths is the set of backend endpoints the proxy will
// forward requests to. Everything else is rejected with 403.
//
// The list covers the OpenAI-compatible surface plus the llama.cpp
// aliases clients commonly use. It deliberately excludes llama.cpp's
// administrative endpoints: POST /props mutates the server, POST /slots
// mutates a slot, POST /lora-adapters loads an arbitrary file from disk,
// /cors-proxy is an open forward proxy, /tools and /mcp expose the
// server's tool runtime, and /props discloses the on-disk model path.
var defaultForwardPaths = []string{
	"/v1/chat/completions",
	"/chat/completions",
	"/v1/completions",
	"/completions",
	"/completion",
	"/v1/embeddings",
	"/embeddings",
	"/embedding",
	"/v1/rerank",
	"/rerank",
	"/reranking",
	"/v1/reranking",
	"/v1/responses",
	"/responses",
	"/v1/messages",
	"/v1/audio/transcriptions",
	"/audio/transcriptions",
	"/tts",
	"/infill",
	"/tokenize",
	"/detokenize",
	"/apply-template",
	"/v1/chat/completions/control",
	"/v1/chat/completions/input_tokens",
	"/chat/completions/input_tokens",
	"/v1/responses/input_tokens",
	"/responses/input_tokens",
	"/v1/messages/count_tokens",
}

// corsAllowMethods are the methods the proxy answers preflights for.
const corsAllowMethods = "GET, POST, OPTIONS"

// corsAllowHeaders are the request headers a browser client may send.
const corsAllowHeaders = "Authorization, Content-Type, X-Api-Key"

// ── Origin allow-list ─────────────────────────

// isAllowedOrigin reports whether the given Origin header value is
// permitted by server.cors_allow_origins.
//
// Entries are matched exactly (case-insensitively) against the full
// origin, so "http://evil.example" cannot match a configured
// "http://app.example" by prefix. Two special entries are recognised:
//
//   "*"       — any origin is allowed.
//   "none"    — no browser origin is allowed at all.
//   "localhost" — any localhost origin (http/https, host localhost,
//                 127.0.0.1 or [::1], any port). This mirrors llama-server's
//                 own --cors-origins=localhost semantics.
func (c *Config) isAllowedOrigin(origin string) bool {
	if origin == "" {
		return false
	}
	for _, allowed := range c.Server.CORSAllowOrigins {
		allowed = strings.TrimSpace(allowed)
		if allowed == "" {
			continue
		}
		switch allowed {
		case "*":
			return true
		case "none":
			return false
		case "localhost":
			if isLocalhostOrigin(origin) {
				return true
			}
			continue
		}
		if strings.EqualFold(allowed, origin) {
			return true
		}
	}
	return false
}

// isLocalhostOrigin reports whether the origin's host is localhost,
// 127.0.0.1 or [::1]. Any port and either scheme are accepted.
func isLocalhostOrigin(origin string) bool {
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	host := u.Hostname()
	switch strings.ToLower(host) {
	case "localhost", "127.0.0.1", "::1":
		return true
	}
	return false
}

// allowsAnyOrigin reports whether the allow-list contains the "*" entry.
func (c *Config) allowsAnyOrigin() bool {
	for _, allowed := range c.Server.CORSAllowOrigins {
		if strings.TrimSpace(allowed) == "*" {
			return true
		}
	}
	return false
}

// ── Forward-path allow-list ───────────────────

// forwardPathAllowed reports whether the proxy will forward this request
// path to a backend. server.forward_paths overrides the built-in list.
func (c *Config) forwardPathAllowed(path string) bool {
	allowed := defaultForwardPaths
	if len(c.Server.ForwardPaths) > 0 {
		allowed = c.Server.ForwardPaths
	}
	for _, p := range allowed {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if p == path {
			return true
		}
	}
	return false
}

// ── Middleware ────────────────────────────────

// generateKey returns a random 256-bit key, hex encoded. Used for the
// backend API key when none is configured: it only needs to be
// unguessable by other local processes, not memorable.
func generateKey() string {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		// crypto/rand does not fail on any supported platform; fall back
		// to a timestamp-derived value rather than an empty key, which
		// would leave the backend unauthenticated.
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

// applyCORS sets the CORS response headers for a request. It returns true
// when the request was an OPTIONS preflight that has been fully answered
// and must not be forwarded upstream.
//
// The allow-list is consulted once per request: an allowed origin gets
// Access-Control-Allow-Origin plus Vary: Origin, a disallowed origin gets
// no CORS headers at all (the browser then blocks the response), and a
// preflight from a disallowed origin is refused outright.
func (ps *ProxyServer) applyCORS(w http.ResponseWriter, r *http.Request) bool {
	origin := r.Header.Get("Origin")
	allowed := ps.cfg.isAllowedOrigin(origin)

	// Always advertise that the response varies by Origin: a cached
	// response for one origin must not be replayed for another.
	w.Header().Add("Vary", "Origin")

	if r.Method != http.MethodOptions {
		if allowed {
			ps.setCORSOrigin(w, origin)
		}
		return false
	}

	// Preflight: answer it here, never forward it to a backend.
	w.Header().Set("Access-Control-Allow-Methods", corsAllowMethods)
	w.Header().Set("Access-Control-Allow-Headers", corsAllowHeaders)
	w.Header().Set("Access-Control-Max-Age", strconv.Itoa(ps.cfg.Server.CORSPreflightMaxAge))

	if allowed {
		ps.setCORSOrigin(w, origin)
		w.WriteHeader(http.StatusNoContent)
		return true
	}

	if origin != "" {
		writeJSONError(w, http.StatusForbidden,
			fmt.Sprintf("origin %q is not allowed; add it to server.cors_allow_origins", origin))
		return true
	}

	// OPTIONS without an Origin header: not a browser preflight.
	w.WriteHeader(http.StatusNoContent)
	return true
}

// setCORSOrigin writes the Access-Control-Allow-Origin (and, when
// configured, Access-Control-Allow-Credentials) response headers.
//
// A wildcard allow-list is served as "*" unless credentials are enabled,
// in which case the concrete origin is echoed back instead — "*" is
// invalid alongside credentials, so echoing is the only form the browser
// will honour.
func (ps *ProxyServer) setCORSOrigin(w http.ResponseWriter, origin string) {
	if ps.cfg.allowsAnyOrigin() && !ps.cfg.Server.CORSAllowCredentials {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		return
	}
	w.Header().Set("Access-Control-Allow-Origin", origin)
	if ps.cfg.Server.CORSAllowCredentials {
		w.Header().Set("Access-Control-Allow-Credentials", "true")
	}
}

// checkOrigin rejects state-changing requests that a browser made on
// behalf of a disallowed origin. Browsers attach Origin to every
// cross-origin state-changing request, so this blocks cross-site request
// forgery against /models/load, /models/unload, /admin/profile and the
// proxied inference endpoints, while leaving origin-less
// (server-to-server) clients untouched.
func (ps *ProxyServer) checkOrigin(w http.ResponseWriter, r *http.Request) bool {
	switch r.Method {
	case http.MethodGet, http.MethodHead, http.MethodOptions:
		return true
	}
	origin := r.Header.Get("Origin")
	if origin == "" {
		return true
	}
	if ps.cfg.isAllowedOrigin(origin) {
		return true
	}
	writeJSONError(w, http.StatusForbidden,
		fmt.Sprintf("origin %q is not allowed; add it to server.cors_allow_origins", origin))
	return false
}

// authenticate enforces the optional client API key (server.api_key).
// Every endpoint except the health probes requires it, so that liveness
// checks and the CLI's service detection keep working.
//
// The key may be supplied as "Authorization: Bearer <key>" or as
// "X-Api-Key: <key>"; the comparison is constant-time.
func (ps *ProxyServer) authenticate(w http.ResponseWriter, r *http.Request) bool {
	key := ps.cfg.Server.APIKey
	if key == "" {
		return true
	}
	switch r.URL.Path {
	case "/", "/health":
		return true
	}
	if tokenMatches(r, key) {
		return true
	}
	w.Header().Set("WWW-Authenticate", `Bearer realm="llama-switch"`)
	writeJSONError(w, http.StatusUnauthorized, "missing or invalid API key")
	return false
}

// tokenMatches extracts the presented credential from the request and
// compares it against the expected key in constant time.
func tokenMatches(r *http.Request, expected string) bool {
	presented := ""
	if h := r.Header.Get("Authorization"); h != "" {
		presented = strings.TrimSpace(h)
		if len(presented) >= 7 && strings.EqualFold(presented[:7], "Bearer ") {
			presented = strings.TrimSpace(presented[7:])
		}
	} else if h := r.Header.Get("X-Api-Key"); h != "" {
		presented = strings.TrimSpace(h)
	}
	if presented == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(presented), []byte(expected)) == 1
}

// isBackendCORSHeader reports whether a response header is one of the
// CORS headers llama.cpp adds. The proxy strips these so that the
// backend's own --cors-origins setting can never widen or contradict the
// proxy's policy.
func isBackendCORSHeader(header string) bool {
	return strings.HasPrefix(strings.ToLower(header), "access-control-")
}

// prepareBackendRequest prepares a request that is about to be sent to a
// backend.
//
// It authenticates the proxy to the backend and keeps the proxy's own
// client credential out of the backend. When the backend requires an API
// key, the client's Authorization header is replaced with the backend's
// key: the client's credential belongs to the proxy and llama-server
// would otherwise reject it as an invalid backend key. When the proxy
// itself enforces an API key but the backend has none, the client's
// credential is dropped rather than forwarded. With neither configured,
// the client's headers are left untouched.
//
// The Origin header is always dropped. CORS is decided by the proxy, and
// the backend is a server-to-server hop: forwarding the browser's origin
// would only let the backend's own --cors-origins setting interfere with
// the proxy's policy.
func (ps *ProxyServer) prepareBackendRequest(req *http.Request, backend *Backend) {
	if backend != nil && backend.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+backend.apiKey)
		req.Header.Del("X-Api-Key")
	} else if ps.cfg.Server.APIKey != "" {
		req.Header.Del("Authorization")
		req.Header.Del("X-Api-Key")
	}
	req.Header.Del("Origin")
}
