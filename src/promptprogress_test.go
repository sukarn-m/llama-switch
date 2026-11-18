package main

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// progressLine is a progress-only chunk in the shape llama-server b12347
// sends with return_progress (captured 2026-11-18, ids shortened).
func progressLine(processed int) string {
	return `data: {"choices":[{"finish_reason":null,"index":0,"delta":{"role":"assistant","content":null}}],"created":1795016397,"id":"chatcmpl-x","model":"m","system_fingerprint":"b12347","object":"chat.completion.chunk","prompt_progress":{"total":6144,"cache":0,"processed":` + itoa(processed) + `,"time_ms":10}}` + "\n"
}

func itoa(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}

const contentLine = `data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"Hi"}}],"object":"chat.completion.chunk"}` + "\n"

func TestInjectReturnProgress(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
		ok   bool
	}{
		{"streamed", `{"model":"m","stream":true}`, `{"return_progress":true,"model":"m","stream":true}`, true},
		{"leading space", ` {"stream":true}`, ` {"return_progress":true,"stream":true}`, true},
		{"not streamed", `{"model":"m","stream":false}`, `{"model":"m","stream":false}`, false},
		{"stream unset", `{"model":"m"}`, `{"model":"m"}`, false},
		{"client set it", `{"stream":true,"return_progress":false}`, `{"stream":true,"return_progress":false}`, false},
		{"not json", `nope`, `nope`, false},
		{"array", `[1]`, `[1]`, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := injectReturnProgress([]byte(tt.in))
			if string(got) != tt.want || ok != tt.ok {
				t.Fatalf("injectReturnProgress(%s) = %s, %v; want %s, %v", tt.in, got, ok, tt.want, tt.ok)
			}
			if ok && !json.Valid(got) {
				t.Fatalf("output is not valid JSON: %s", got)
			}
		})
	}
}

func TestSplitProgressChunk(t *testing.T) {
	pp, rest, ok := splitProgressChunk(progressLine(2048))
	if !ok {
		t.Fatal("live progress chunk not recognised")
	}
	if pp.Processed != 2048 || pp.Total != 6144 {
		t.Errorf("progress = %+v, want processed 2048 of 6144", pp)
	}
	if strings.Contains(string(rest), "prompt_progress") || !strings.Contains(string(rest), `"role":"assistant"`) {
		t.Errorf("rest = %s, want the chunk without prompt_progress", rest)
	}

	notProgress := map[string]string{
		"content":       `data: {"choices":[{"delta":{"content":"Hi"}}],"prompt_progress":{"total":1,"processed":1}}` + "\n",
		"reasoning":     `data: {"choices":[{"delta":{"reasoning_content":"x"}}],"prompt_progress":{"total":1,"processed":1}}` + "\n",
		"finish reason": `data: {"choices":[{"finish_reason":"stop","delta":{}}],"prompt_progress":{"total":1,"processed":1}}` + "\n",
		"usage":         `data: {"choices":[],"usage":{"prompt_tokens":1},"prompt_progress":{"total":1,"processed":1}}` + "\n",
		"null progress": `data: {"choices":[{"delta":{"role":"assistant"}}],"prompt_progress":null}` + "\n",
		"no progress":   `data: {"choices":[{"delta":{"role":"assistant"}}]}` + "\n",
		"bad json":      `data: {"prompt_progress":` + "\n",
	}
	for name, line := range notProgress {
		if _, _, ok := splitProgressChunk(line); ok {
			t.Errorf("%s: recognised as progress-only: %s", name, line)
		}
	}
}

// TestStreamSSEPromptProgress feeds a llama-server-shaped stream through
// streamSSE: the first progress chunk becomes the opening data event, the
// rest become comments, and everything from the first token on passes
// through untouched (including a later chunk mentioning prompt_progress).
func TestStreamSSEPromptProgress(t *testing.T) {
	in := progressLine(0) + "\n" +
		progressLine(2048) + "\n" +
		":\n\n" +
		progressLine(4096) + "\n" +
		contentLine + "\n" +
		progressLine(6144) + "\n" +
		"data: [DONE]\n\n"
	w := httptest.NewRecorder()
	streamSSE(w, strings.NewReader(in), w, &progressRewriter{})
	out := w.Body.String()

	events := strings.Split(strings.TrimSuffix(out, "\n\n"), "\n\n")
	if len(events) != 7 {
		t.Fatalf("got %d events, want 7:\n%s", len(events), out)
	}
	if !strings.HasPrefix(events[0], "data: ") || strings.Contains(events[0], "prompt_progress") || !strings.Contains(events[0], `"role":"assistant"`) {
		t.Errorf("event 0 = %q, want the opening chunk without prompt_progress", events[0])
	}
	if events[1] != ": prompt progress processed=2048 total=6144 cache=0" {
		t.Errorf("event 1 = %q, want a progress comment", events[1])
	}
	if events[2] != ":" {
		t.Errorf("event 2 = %q, want llama-server's own ping kept", events[2])
	}
	if events[3] != ": prompt progress processed=4096 total=6144 cache=0" {
		t.Errorf("event 3 = %q, want a progress comment", events[3])
	}
	if events[4]+"\n" != contentLine {
		t.Errorf("event 4 = %q, want the content chunk unchanged", events[4])
	}
	if events[5]+"\n" != progressLine(6144) {
		t.Errorf("event 5 = %q, want chunks after the first token passed through", events[5])
	}
	if events[6] != "data: [DONE]" {
		t.Errorf("event 6 = %q, want [DONE]", events[6])
	}
}

func TestStreamSSENilRewriterPassesThrough(t *testing.T) {
	in := progressLine(0) + "\n" + contentLine + "\n"
	w := httptest.NewRecorder()
	streamSSE(w, strings.NewReader(in), w, nil)
	if w.Body.String() != in {
		t.Fatalf("stream changed without a rewriter:\n%s", w.Body.String())
	}
}

// TestForwardPromptProgress runs forward() end to end against a mock
// backend: the request gains return_progress only when it applies, and
// the response is rewritten only when it was added.
func TestForwardPromptProgress(t *testing.T) {
	off := false
	tests := []struct {
		name      string
		path      string
		body      string
		model     ModelConfig
		serverOff bool
		wantSent  bool // backend saw return_progress:true
		wantRewr  bool // response progress chunks rewritten
	}{
		{"streamed chat", "/v1/chat/completions", `{"model":"m","stream":true}`, ModelConfig{ID: "m"}, false, true, true},
		{"unprefixed path", "/chat/completions", `{"model":"m","stream":true}`, ModelConfig{ID: "m"}, false, true, true},
		{"client asked", "/v1/chat/completions", `{"model":"m","stream":true,"return_progress":true}`, ModelConfig{ID: "m"}, false, true, false},
		{"not streamed", "/v1/chat/completions", `{"model":"m"}`, ModelConfig{ID: "m"}, false, false, false},
		{"completions path", "/v1/completions", `{"model":"m","stream":true}`, ModelConfig{ID: "m"}, false, false, false},
		{"raw args model", "/v1/chat/completions", `{"model":"m","stream":true}`, ModelConfig{ID: "m", Args: []string{"--port", "{port}"}}, false, false, false},
		{"disabled", "/v1/chat/completions", `{"model":"m","stream":true}`, ModelConfig{ID: "m"}, true, false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var sent map[string]json.RawMessage
			backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(b, &sent)
				w.Header().Set("Content-Type", "text/event-stream")
				io.WriteString(w, progressLine(0)+"\n"+progressLine(2048)+"\n"+contentLine+"\n"+"data: [DONE]\n\n")
			}))
			defer backendSrv.Close()

			ps := newTestProxyServer(t, []ModelConfig{tt.model}, nil)
			if tt.serverOff {
				ps.cfg.Server.StreamPromptProgress = &off
			}
			backend := &Backend{ID: "m", Port: backendSrv.Listener.Addr().(*net.TCPAddr).Port}
			req := httptest.NewRequest("POST", tt.path, strings.NewReader(tt.body))
			w := httptest.NewRecorder()
			ps.forward(w, req, []byte(tt.body), backend)

			if got := string(sent["return_progress"]) == "true"; got != tt.wantSent {
				t.Errorf("backend saw return_progress=%s, want true=%v", sent["return_progress"], tt.wantSent)
			}
			out := w.Body.String()
			rewritten := strings.Contains(out, ": prompt progress processed=2048 total=6144 cache=0") && strings.Count(out, "prompt_progress") == 0
			if rewritten != tt.wantRewr {
				t.Errorf("response rewritten=%v, want %v:\n%s", rewritten, tt.wantRewr, out)
			}
			if !strings.Contains(out, contentLine) {
				t.Errorf("content chunk missing from response:\n%s", out)
			}
		})
	}
}
