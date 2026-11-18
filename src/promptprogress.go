package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// ── Prompt-progress keepalive ────────────────
//
// llama-server sends nothing on a streamed request while it processes the
// prompt, apart from a bare ":" SSE comment every 30 s. On a slow model a
// long prompt takes minutes (a 29k-token prompt on a 27B dense model at
// ~185 tok/s is ~155 s), and a client that waits a fixed time for the
// first data event — sai-portal's 60 s first-event gate, for one — treats
// the backend as wedged, cancels, and fails over to another model, which
// makes llama-switch evict this one.
//
// For a streamed chat request the client did not ask for progress on,
// llama-switch adds "return_progress": true to the body. llama-server then
// sends a progress-only chunk after every prompt batch (2048 tokens by
// default). The proxy rewrites those chunks:
//
//   - the first becomes llama-server's ordinary opening chunk
//     (delta {"role":"assistant","content":null}) with the prompt_progress
//     member removed, so the client sees a standard data event within a
//     second of the request reaching the backend;
//   - later ones become SSE comments
//     (": prompt progress processed=4096 total=29000 cache=0"), which
//     every SSE client ignores but which keep the connection busy. The
//     values are llama-server's own, printed as-is.
//
// A client that sends return_progress itself gets llama-server's chunks
// unchanged. Models with raw `args` (possibly not llama-server) and the
// server.stream_prompt_progress: false setting turn the feature off.

// promptProgressPaths are the forwarded paths whose streamed responses are
// chat completion chunks llama-server can report prompt progress on.
var promptProgressPaths = map[string]bool{
	"/v1/chat/completions": true,
	"/chat/completions":    true,
}

// streamPromptProgress reports whether the prompt-progress keepalive is on
// (server.stream_prompt_progress; unset means on).
func (s *ServerConfig) streamPromptProgress() bool {
	return s.StreamPromptProgress == nil || *s.StreamPromptProgress
}

// preparePromptProgress returns the body to forward and, when the
// keepalive applies to this request, the rewriter streamSSE runs the
// response through. A nil rewriter means the request and its response
// pass through unchanged.
func (ps *ProxyServer) preparePromptProgress(path string, backend *Backend, body []byte) ([]byte, *progressRewriter) {
	if !promptProgressPaths[path] || !ps.cfg.Server.streamPromptProgress() {
		return body, nil
	}
	mc := ps.cfg.FindModel(backend.ID)
	if mc == nil || len(mc.Args) > 0 {
		return body, nil
	}
	out, ok := injectReturnProgress(body)
	if !ok {
		return body, nil
	}
	return out, &progressRewriter{}
}

// injectReturnProgress adds a top-level "return_progress": true to a
// streamed request body. It returns ok=false, leaving the body alone, when
// the body is not a JSON object, is not "stream": true, or already carries
// return_progress (the client's own choice). The member is spliced in
// after the opening brace so every other byte reaches llama-server as the
// client sent it.
func injectReturnProgress(body []byte) ([]byte, bool) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return body, false
	}
	if string(bytes.TrimSpace(top["stream"])) != "true" {
		return body, false
	}
	if _, set := top["return_progress"]; set {
		return body, false
	}
	open := bytes.IndexByte(body, '{') // the first non-space byte, since the body is a JSON object
	member := `"return_progress":true`
	if len(top) > 0 {
		member += ","
	}
	out := make([]byte, 0, len(body)+len(member))
	out = append(out, body[:open+1]...)
	out = append(out, member...)
	out = append(out, body[open+1:]...)
	return out, true
}

// progressRewriter rewrites llama-server's prompt-progress chunks in one
// streamed response. Not safe for concurrent use; one per response.
type progressRewriter struct {
	sentFirst bool // the first progress chunk went out as the opening chunk
	done      bool // a non-progress data event arrived; pass everything through
}

// rewrite takes one line of the SSE stream (as read, newline included)
// and returns the text to send to the client in its place.
func (p *progressRewriter) rewrite(line string) string {
	if p.done || !strings.HasPrefix(line, "data:") {
		return line
	}
	pp, rest, ok := splitProgressChunk(line)
	if !ok {
		// Generation has started (or the chunk is something this code does
		// not recognise): from here on the stream is llama-server's own.
		p.done = true
		return line
	}
	if !p.sentFirst {
		p.sentFirst = true
		return "data: " + string(rest) + "\n"
	}
	return fmt.Sprintf(": prompt progress processed=%d total=%d cache=%d\n", pp.Processed, pp.Total, pp.Cache)
}

// promptProgress is llama-server's prompt_progress member.
type promptProgress struct {
	Total     int `json:"total"`
	Cache     int `json:"cache"`
	Processed int `json:"processed"`
	TimeMS    int `json:"time_ms"`
}

// splitProgressChunk reports whether an SSE line is a progress-only chunk:
// a data event with a non-null prompt_progress member, no usage, no finish
// reason, and deltas that carry at most a role and null values. It returns
// the progress and the chunk re-encoded without prompt_progress.
//
// Live shape (llama-server b12347, 2026-11-18):
//
//	data: {"choices":[{"finish_reason":null,"index":0,"delta":{"role":"assistant","content":null}}],
//	  "created":…,"id":"chatcmpl-…","model":"…","system_fingerprint":"…",
//	  "object":"chat.completion.chunk","prompt_progress":{"total":136702,"cache":0,"processed":2048,"time_ms":529}}
func splitProgressChunk(line string) (promptProgress, []byte, bool) {
	var pp promptProgress
	if !strings.Contains(line, `"prompt_progress"`) {
		return pp, nil, false
	}
	payload := strings.TrimSpace(strings.TrimPrefix(strings.TrimRight(line, "\r\n"), "data:"))
	var top map[string]json.RawMessage
	if err := json.Unmarshal([]byte(payload), &top); err != nil {
		return pp, nil, false
	}
	if !jsonPresent(top["prompt_progress"]) || jsonPresent(top["usage"]) {
		return pp, nil, false
	}
	if err := json.Unmarshal(top["prompt_progress"], &pp); err != nil {
		return pp, nil, false
	}
	var choices []struct {
		FinishReason json.RawMessage            `json:"finish_reason"`
		Delta        map[string]json.RawMessage `json:"delta"`
	}
	if raw, ok := top["choices"]; ok {
		if err := json.Unmarshal(raw, &choices); err != nil {
			return pp, nil, false
		}
	}
	for _, c := range choices {
		if jsonPresent(c.FinishReason) {
			return pp, nil, false
		}
		for k, v := range c.Delta {
			if k != "role" && jsonPresent(v) {
				return pp, nil, false
			}
		}
	}
	delete(top, "prompt_progress")
	rest, err := json.Marshal(top)
	if err != nil {
		return pp, nil, false
	}
	return pp, rest, true
}

// jsonPresent reports whether a raw JSON member is present and not null.
func jsonPresent(raw json.RawMessage) bool {
	v := bytes.TrimSpace(raw)
	return len(v) > 0 && string(v) != "null"
}
