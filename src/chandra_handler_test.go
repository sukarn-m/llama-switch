package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChandraHandlerMatchesModel(t *testing.T) {
	h := &ChandraHandler{}
	tests := []struct {
		modelID string
		want    bool
	}{
		{"chandra-ocr-2", true},
		{"Chandra-OCR-2", true},
		{"chandra", true},
		{"surya-ocr", false},
		{"gemma-4-12b", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := h.MatchesModel(tt.modelID); got != tt.want {
			t.Errorf("MatchesModel(%q) = %v, want %v", tt.modelID, got, tt.want)
		}
	}
}

func TestStripThinking(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			"pure HTML",
			`<div data-bbox="1 2 3 4" data-label="Text"><p>Hello</p></div>`,
			`<div data-bbox="1 2 3 4" data-label="Text"><p>Hello</p></div>`,
		},
		{
			"thinking + HTML",
			`The user wants me to extract text. I will do OCR.

<div data-bbox="1 2 3 4" data-label="Text"><p>Hello</p></div>`,
			`<div data-bbox="1 2 3 4" data-label="Text"><p>Hello</p></div>`,
		},
		{
			"no HTML",
			`Just some text without any HTML`,
			`Just some text without any HTML`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripThinking(tt.input)
			if got != tt.want {
				t.Errorf("stripThinking() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestExtractHTML(t *testing.T) {
	h := &ChandraHandler{}

	// Content field has HTML
	msg1 := map[string]any{
		"content":           `<div data-label="Text"><p>Hello</p></div>`,
		"reasoning_content": "Some thinking text",
	}
	if got := h.extractHTML(msg1); !strings.Contains(got, "Hello") {
		t.Errorf("extractHTML from content: got %q, expected to contain 'Hello'", got)
	}

	// Only reasoning_content has HTML
	msg2 := map[string]any{
		"content":           "",
		"reasoning_content": `<div data-label="Text"><p>World</p></div>`,
	}
	if got := h.extractHTML(msg2); !strings.Contains(got, "World") {
		t.Errorf("extractHTML from reasoning: got %q, expected to contain 'World'", got)
	}

	// No HTML in either field
	msg3 := map[string]any{
		"content":           "Just text",
		"reasoning_content": "More thinking",
	}
	if got := h.extractHTML(msg3); got != "" {
		t.Errorf("extractHTML with no HTML: got %q, expected empty", got)
	}
}

func TestHTMLToMarkdown(t *testing.T) {
	html := `<div data-bbox="0 0 100 50" data-label="Section-Header"><p>GOVERNMENT OF INDIA<br/>MINISTRY OF FINANCE</p></div>
<div data-bbox="0 50 100 100" data-label="Text"><p>Hello World</p></div>
<div data-bbox="0 100 100 120" data-label="Page-Header"><p>Page 1</p></div>`

	md := htmlToMarkdown(html)

	// Should contain the header text
	if !strings.Contains(md, "GOVERNMENT OF INDIA") {
		t.Errorf("expected 'GOVERNMENT OF INDIA' in markdown, got: %s", md)
	}
	if !strings.Contains(md, "Hello World") {
		t.Errorf("expected 'Hello World' in markdown, got: %s", md)
	}
	// Page-Header should be stripped
	if strings.Contains(md, "Page 1") {
		t.Errorf("Page-Header should be stripped, but found 'Page 1' in: %s", md)
	}
}

func TestHTMLToMarkdownTable(t *testing.T) {
	html := `<div data-label="Form"><table border="1"><tr><td>PAN:<br/>ABIPV8161E</td><td>A.Y:<br/>2020-21</td></tr></table></div>`

	md := htmlToMarkdown(html)

	if !strings.Contains(md, "ABIPV8161E") {
		t.Errorf("expected PAN in markdown, got: %s", md)
	}
	if !strings.Contains(md, "2020-21") {
		t.Errorf("expected A.Y. in markdown, got: %s", md)
	}
	// Should have pipe-separated table
	if !strings.Contains(md, "|") {
		t.Errorf("expected table with pipes, got: %s", md)
	}
}

func TestHTMLToMarkdownImage(t *testing.T) {
	html := `<div data-label="Image"><img alt="Emblem of India"/></div>
<div data-label="Text"><p>Hello</p></div>`

	md := htmlToMarkdown(html)

	if !strings.Contains(md, "Emblem of India") {
		t.Errorf("expected image description in markdown, got: %s", md)
	}
}

func TestFixDevanagariConjuncts(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"कय", "क्रय"},
		{"केता", "क्रेता"},
		{"केती", "क्रेती"},
		{"विकय", "विक्रय"},
		// Should not double-fix already correct conjuncts
		{"क्रय", "क्रय"},
		{"विक्रय", "विक्रय"},
		// Mixed text
		{"कय विकय क्रेता", "क्रय विक्रय क्रेता"},
	}
	for _, tt := range tests {
		got := fixDevanagariConjuncts(tt.input)
		if got != tt.want {
			t.Errorf("fixDevanagariConjuncts(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestProcessNonStream(t *testing.T) {
	h := &ChandraHandler{}

	// Simulate a Chandra response with HTML in content
	response := map[string]any{
		"id":     "test",
		"object": "chat.completion",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":              "assistant",
					"content":           `<div data-label="Text"><p>Hello World</p></div>`,
					"reasoning_content": "Thinking about the document...",
				},
				"finish_reason": "stop",
			},
		},
	}
	body, _ := json.Marshal(response)

	processed, err := h.processNonStream(body)
	if err != nil {
		t.Fatalf("processNonStream error: %v", err)
	}

	var result map[string]any
	if err := json.Unmarshal(processed, &result); err != nil {
		t.Fatalf("failed to unmarshal processed response: %v", err)
	}

	choices := result["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)

	// content should be markdown, not HTML
	content := msg["content"].(string)
	if strings.Contains(content, "<div") {
		t.Errorf("content should be markdown, not HTML: %s", content)
	}
	if !strings.Contains(content, "Hello World") {
		t.Errorf("content should contain 'Hello World': %s", content)
	}

	// reasoning_content should be removed
	if _, ok := msg["reasoning_content"]; ok {
		t.Errorf("reasoning_content should be removed")
	}
}

func TestProcessNonStreamWithThinking(t *testing.T) {
	h := &ChandraHandler{}

	// Simulate a response where HTML is in reasoning_content (with thinking)
	response := map[string]any{
		"id":     "test",
		"object": "chat.completion",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "",
					"reasoning_content": `The user wants me to extract text.

<div data-label="Text"><p>Hello from reasoning</p></div>`,
				},
				"finish_reason": "stop",
			},
		},
	}
	body, _ := json.Marshal(response)

	processed, err := h.processNonStream(body)
	if err != nil {
		t.Fatalf("processNonStream error: %v", err)
	}

	var result map[string]any
	json.Unmarshal(processed, &result)

	choices := result["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	content := msg["content"].(string)

	if !strings.Contains(content, "Hello from reasoning") {
		t.Errorf("content should contain extracted text: %s", content)
	}
	if strings.Contains(content, "The user wants") {
		t.Errorf("thinking text should be stripped: %s", content)
	}
}

func TestStripThinkingDivVariations(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			"div with space (standard)",
			`Thinking\n<div data-label="Text"><p>Hello</p></div>`,
			`<div data-label="Text"><p>Hello</p></div>`,
		},
		{
			"div with newline",
			"Thinking\n<div\ndata-label=\"Text\"><p>Hello</p></div>",
			"<div\ndata-label=\"Text\"><p>Hello</p></div>",
		},
		{
			"div with tab",
			"Thinking\n<div\tdata-label=\"Text\"><p>Hello</p></div>",
			"<div\tdata-label=\"Text\"><p>Hello</p></div>",
		},
		{
			"div with no attributes",
			`Thinking\n<div><p>Hello</p></div>`,
			`<div><p>Hello</p></div>`,
		},
		{
			"no div at all",
			`Just thinking text`,
			`Just thinking text`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripThinking(tt.input)
			if got != tt.want {
				t.Errorf("stripThinking() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestHasHTMLDivVariations(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{`<div data-label="Text">`, true},
		{"<div\ndata-label=\"Text\">", true},
		{"<div\tdata-label=\"Text\">", true},
		{`<div>`, true},
		{`<div`, false},
		{`no html here`, false},
		{`<span>not a div</span>`, false},
	}
	for _, tt := range tests {
		if got := hasHTML(tt.input); got != tt.want {
			t.Errorf("hasHTML(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

func TestExtractHTMLDivWithNewline(t *testing.T) {
	h := &ChandraHandler{}

	msg := map[string]any{
		"content": "Thinking text\n<div\ndata-label=\"Text\"><p>Hello</p></div>",
	}
	got := h.extractHTML(msg)
	if !strings.Contains(got, "Hello") {
		t.Errorf("extractHTML with <div\\n: got %q, expected to contain 'Hello'", got)
	}
	if strings.Contains(got, "Thinking text") {
		t.Errorf("extractHTML with <div\\n: thinking text should be stripped, got %q", got)
	}
}

func TestProcessNonStreamDivWithNewline(t *testing.T) {
	h := &ChandraHandler{}

	response := map[string]any{
		"id":     "test",
		"object": "chat.completion",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "Thinking\n<div\ndata-label=\"Text\"><p>Hello World</p></div>",
				},
				"finish_reason": "stop",
			},
		},
	}
	body, _ := json.Marshal(response)

	processed, err := h.processNonStream(body)
	if err != nil {
		t.Fatalf("processNonStream error: %v", err)
	}

	var result map[string]any
	json.Unmarshal(processed, &result)

	choices := result["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	content := msg["content"].(string)

	if strings.Contains(content, "<div") {
		t.Errorf("content should be markdown, not HTML: %s", content)
	}
	if !strings.Contains(content, "Hello World") {
		t.Errorf("content should contain 'Hello World': %s", content)
	}
	if strings.Contains(content, "Thinking") {
		t.Errorf("thinking text should be stripped: %s", content)
	}
}

func TestProcessNonStreamRawHTMLNotPassedThrough(t *testing.T) {
	h := &ChandraHandler{}

	rawHTML := `<div data-bbox="0 0 100 50" data-label="Section-Header"><p>RELIEVING LETTER</p></div>
<div data-bbox="0 50 100 100" data-label="Text"><p>Dear Employee,</p></div>`

	response := map[string]any{
		"id":     "test",
		"object": "chat.completion",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":              "assistant",
					"content":           rawHTML,
					"reasoning_content": "I need to extract the text.",
				},
				"finish_reason": "stop",
			},
		},
	}
	body, _ := json.Marshal(response)

	processed, err := h.processNonStream(body)
	if err != nil {
		t.Fatalf("processNonStream error: %v", err)
	}

	var result map[string]any
	json.Unmarshal(processed, &result)

	choices := result["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	content := msg["content"].(string)

	if strings.Contains(content, "data-bbox") {
		t.Errorf("raw bbox attributes should not be in output: %s", content)
	}
	if strings.Contains(content, "<div") {
		t.Errorf("HTML should be converted to markdown: %s", content)
	}
	if !strings.Contains(content, "RELIEVING LETTER") {
		t.Errorf("content should contain extracted text: %s", content)
	}
	if _, ok := msg["reasoning_content"]; ok {
		t.Errorf("reasoning_content should be removed")
	}
}

func TestProcessNonStreamContentNullReasoningHTML(t *testing.T) {
	h := &ChandraHandler{}

	response := map[string]any{
		"id":     "test",
		"object": "chat.completion",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":              "assistant",
					"content":           nil,
					"reasoning_content": "Let me analyze.\n<div\ndata-label=\"Text\"><p>From Reasoning</p></div>",
				},
				"finish_reason": "stop",
			},
		},
	}
	body, _ := json.Marshal(response)

	processed, err := h.processNonStream(body)
	if err != nil {
		t.Fatalf("processNonStream error: %v", err)
	}

	var result map[string]any
	json.Unmarshal(processed, &result)

	choices := result["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	content := msg["content"].(string)

	if !strings.Contains(content, "From Reasoning") {
		t.Errorf("content should contain text from reasoning_content: %s", content)
	}
	if strings.Contains(content, "Let me analyze") {
		t.Errorf("thinking text should be stripped: %s", content)
	}
	if _, ok := msg["reasoning_content"]; ok {
		t.Errorf("reasoning_content should be removed")
	}
}

func TestProcessNonStreamFallbackStripsTags(t *testing.T) {
	h := &ChandraHandler{}

	response := map[string]any{
		"id":     "test",
		"object": "chat.completion",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "<span>Raw</span> <b>text</b> with <unknown>tags</unknown>",
				},
				"finish_reason": "stop",
			},
		},
	}
	body, _ := json.Marshal(response)

	processed, err := h.processNonStream(body)
	if err != nil {
		t.Fatalf("processNonStream error: %v", err)
	}

	var result map[string]any
	json.Unmarshal(processed, &result)

	choices := result["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	content := msg["content"].(string)

	if strings.Contains(content, "<") {
		t.Errorf("HTML tags should be stripped in fallback: %s", content)
	}
	if !strings.Contains(content, "Raw") || !strings.Contains(content, "text") {
		t.Errorf("text content should be preserved: %s", content)
	}
}

func TestProcessNonStreamNoHTMLSkipsCleanly(t *testing.T) {
	h := &ChandraHandler{}

	response := map[string]any{
		"id":     "test",
		"object": "chat.completion",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "Just plain text with no HTML tags at all",
				},
				"finish_reason": "stop",
			},
		},
	}
	body, _ := json.Marshal(response)

	processed, err := h.processNonStream(body)
	if err != nil {
		t.Fatalf("processNonStream error: %v", err)
	}

	var result map[string]any
	json.Unmarshal(processed, &result)

	choices := result["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	content := msg["content"].(string)

	if content != "Just plain text with no HTML tags at all" {
		t.Errorf("plain text should be left unchanged: %s", content)
	}
}

// ── Integration test with a mock backend ─────────────────────────────

func TestEndToEndChandraHandler(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	// Mock backend that returns a Chandra-style response
	mockResp := map[string]any{
		"choices": []any{
			map[string]any{
				"message": map[string]any{
					"role":    "assistant",
					"content": `<div data-label="Section-Header"><p>क्रय विक्रय</p></div><div data-label="Text"><p>कय केता विकय</p></div>`,
				},
				"finish_reason": "stop",
			},
		},
	}
	mockBody, _ := json.Marshal(mockResp)

	// Create a mock backend server
	backendSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify the request was forced to non-streaming
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		if stream, ok := req["stream"]; ok {
			if stream == true {
				t.Error("expected stream=false in request to backend")
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.Write(mockBody)
	}))
	defer backendSrv.Close()

	// Directly test the handler with a real HTTP response
	resp, err := http.Get(backendSrv.URL)
	if err != nil {
		t.Fatalf("failed to call mock backend: %v", err)
	}
	defer resp.Body.Close()

	h := &ChandraHandler{}
	processed, err := h.ProcessResponse(resp, false)
	if err != nil {
		t.Fatalf("ProcessResponse error: %v", err)
	}

	var result map[string]any
	if err := json.Unmarshal(processed, &result); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	choices := result["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	content := msg["content"].(string)

	// The conjunct fix should have been applied
	if !strings.Contains(content, "क्रय") {
		t.Errorf("expected क्रय in output: %s", content)
	}
	if !strings.Contains(content, "विक्रय") {
		t.Errorf("expected विक्रय in output: %s", content)
	}
	if strings.Contains(content, "कय") {
		t.Errorf("broken conjunct कय should be fixed: %s", content)
	}
	if strings.Contains(content, "<div") {
		t.Errorf("HTML should be converted to markdown: %s", content)
	}
}
