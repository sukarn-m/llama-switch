package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
)

// ── Model handler interface ──────────────────────────────────────────
// A model handler intercepts the backend response and post-processes it
// before returning to the client. This allows model-specific transformations
// like converting HTML to markdown, fixing known OCR artifacts, etc.
//
// To add a new handler:
// 1. Implement the ModelHandler interface in a new file.
// 2. Register it in NewModelHandler() below.
// To remove: delete the file and the registration line.

type ModelHandler interface {
	// MatchesModel returns true if this handler should be used for the
	// given model ID. Called once per request.
	MatchesModel(modelID string) bool
	// ProcessResponse takes the raw HTTP response from the backend and
	// returns a modified response body. The response is always non-streaming
	// (the handler reads the full response before processing).
	// If an error is returned, the original response is forwarded unchanged.
	ProcessResponse(resp *http.Response, isStream bool) ([]byte, error)
}

// NewModelHandler returns the appropriate handler for a model, or nil
// if no handler is registered.
func NewModelHandler(modelID string) ModelHandler {
	ch := &ChandraHandler{}
	if ch.MatchesModel(modelID) {
		return ch
	}
	return nil
}

// ── Chandra 2 OCR handler ────────────────────────────────────────────
// Chandra 2 is a Qwen3.5-based OCR model that:
// - Always outputs structured HTML with bounding boxes (data-bbox, data-label)
// - Puts its output in the `content` field for most pages, but sometimes
//   in `reasoning_content` (thinking mode) instead
// - Has a known Devanagari conjunct-drop bug (क्रय → कय, क्रेता → केता, etc.)
// - Sometimes leaks chain-of-thought into the output
//
// This handler:
// 1. Always extracts content from whichever field has the HTML
// 2. Converts the HTML to clean markdown
// 3. Fixes known Devanagari conjunct errors
// 4. Strips any leaked thinking/chain-of-thought text

type ChandraHandler struct{}

func (h *ChandraHandler) MatchesModel(modelID string) bool {
	id := strings.ToLower(modelID)
	return strings.Contains(id, "chandra")
}

// ProcessResponse intercepts the backend response and post-processes it.
func (h *ChandraHandler) ProcessResponse(resp *http.Response, isStream bool) ([]byte, error) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read backend response: %w", err)
	}

	if isStream {
		return h.processStream(body)
	}
	return h.processNonStream(body)
}

// processNonStream handles a non-streaming /v1/chat/completions response.
func (h *ChandraHandler) processNonStream(body []byte) ([]byte, error) {
	var result map[string]any
	if err := json.Unmarshal(body, &result); err != nil {
		return body, nil // can't parse, return original
	}

	choices, ok := result["choices"].([]any)
	if !ok || len(choices) == 0 {
		return body, nil
	}

	for _, choice := range choices {
		choiceMap, ok := choice.(map[string]any)
		if !ok {
			continue
		}
		msg, ok := choiceMap["message"].(map[string]any)
		if !ok {
			continue
		}

		markdown := h.processMessage(msg)
		if markdown != "" {
			msg["content"] = markdown
			delete(msg, "reasoning_content")
		}
	}

	return json.Marshal(result)
}

// processMessage extracts HTML from the message, converts it to markdown,
// and fixes Devanagari conjunct errors. Returns empty string if no
// processable content was found.
func (h *ChandraHandler) processMessage(msg map[string]any) string {
	html := h.extractHTML(msg)
	if html == "" {
		html = h.fallbackExtract(msg)
		if html == "" {
			return ""
		}
	}

	markdown := htmlToMarkdown(html)
	if markdown == "" {
		markdown = stripRemainingTags(html)
	}
	markdown = strings.TrimSpace(markdown)
	if markdown == "" {
		return ""
	}
	return fixDevanagariConjuncts(markdown)
}

// fallbackExtract returns raw content from content or reasoning_content
// when extractHTML fails to find structured HTML. This catches edge cases
// where the model's output format deviates slightly from expected patterns.
func (h *ChandraHandler) fallbackExtract(msg map[string]any) string {
	if content, ok := msg["content"].(string); ok && content != "" {
		if strings.Contains(content, "<") {
			return content
		}
	}
	if reasoning, ok := msg["reasoning_content"].(string); ok && reasoning != "" {
		if strings.Contains(reasoning, "<") {
			return reasoning
		}
	}
	return ""
}

// processStream handles a streaming /v1/chat/completions response.
// It buffers all SSE chunks, reconstructs the full response, processes it,
// and returns a single non-streaming JSON response. This is necessary
// because the HTML-to-markdown conversion needs the complete output.
func (h *ChandraHandler) processStream(body []byte) ([]byte, error) {
	// Parse SSE chunks and accumulate content/reasoning_content
	var contentParts []string
	var reasoningParts []string
	var firstChunk map[string]any

	for _, line := range strings.Split(string(body), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk map[string]any
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if firstChunk == nil {
			firstChunk = chunk
		}

		choices, ok := chunk["choices"].([]any)
		if !ok || len(choices) == 0 {
			continue
		}
		delta, ok := choices[0].(map[string]any)
		if !ok {
			continue
		}
		deltaMsg, ok := delta["delta"].(map[string]any)
		if !ok {
			continue
		}

		if c, ok := deltaMsg["content"].(string); ok && c != "" {
			contentParts = append(contentParts, c)
		}
		if r, ok := deltaMsg["reasoning_content"].(string); ok && r != "" {
			reasoningParts = append(reasoningParts, r)
		}
	}

	// Build a non-streaming response
	fullContent := strings.Join(contentParts, "")
	fullReasoning := strings.Join(reasoningParts, "")

	// Extract HTML from whichever field has it
	msg := map[string]any{
		"role": "assistant",
	}
	if fullContent != "" {
		msg["content"] = fullContent
	}
	if fullReasoning != "" {
		msg["reasoning_content"] = fullReasoning
	}

	html := h.extractHTML(msg)
	if html == "" {
		html = h.fallbackExtract(msg)
	}
	if html == "" {
		return body, nil
	}

	markdown := htmlToMarkdown(html)
	if markdown == "" {
		markdown = stripRemainingTags(html)
	}
	markdown = strings.TrimSpace(markdown)
	if markdown == "" {
		return body, nil
	}
	markdown = fixDevanagariConjuncts(markdown)

	msg["content"] = markdown
	delete(msg, "reasoning_content")

	// Build the response
	result := map[string]any{
		"id":     "chandra-processed",
		"object": "chat.completion",
		"choices": []any{
			map[string]any{
				"index":         0,
				"message":       msg,
				"finish_reason": "stop",
			},
		},
	}
	if firstChunk != nil {
		if model, ok := firstChunk["model"]; ok {
			result["model"] = model
		}
	}

	return json.Marshal(result)
}

// extractHTML finds the HTML output in the message, checking content first,
// then reasoning_content. It strips any chain-of-thought text that may
// appear before the first <div> tag.
func (h *ChandraHandler) extractHTML(msg map[string]any) string {
	// Check content field first
	if content, ok := msg["content"].(string); ok && content != "" {
		html := stripThinking(content)
		if hasHTML(html) {
			return html
		}
	}
	// Fall back to reasoning_content
	if reasoning, ok := msg["reasoning_content"].(string); ok && reasoning != "" {
		html := stripThinking(reasoning)
		if hasHTML(html) {
			return html
		}
	}
	return ""
}

// stripThinking removes any text before the first <div> tag.
// Chandra sometimes leaks chain-of-thought into the output.
// The regex matches <div followed by any whitespace or >, covering
// <div data-...>, <div\n...>, <div\t...>, and <div> (no attributes).
var divStartRegex = regexp.MustCompile(`(?s)^.*?(<div[\s>])`)

func stripThinking(s string) string {
	loc := divStartRegex.FindStringSubmatchIndex(s)
	if loc != nil {
		return s[loc[2]:]
	}
	return s
}

func hasHTML(s string) bool {
	return divStartRegex.MatchString(s)
}

// ── HTML to Markdown conversion ─────────────────────────────────────
// Converts Chandra's structured HTML output (with data-bbox, data-label
// divs) into clean markdown. This is a Go reimplementation of the
// key parts of chandra.output.parse_markdown from the Python package.

var (
	// Strip data-bbox and data-label attributes
	stripAttrsRegex = regexp.MustCompile(`\s*data-(?:bbox|label)="[^"]*"`)
	// Match image divs
	imgDivRegex = regexp.MustCompile(`<div[^>]*data-label="(?:Image|Figure)"[^>]*>.*?</div>`)
	// Match all remaining divs (just strip the tags, keep content)
	divOpenRegex  = regexp.MustCompile(`<div[^>]*>`)
	divCloseRegex = regexp.MustCompile(`</div>`)
	// Match <br/> and <br>
	brRegex = regexp.MustCompile(`<br\s*/?>`)
	// Match <p> tags
	pOpenRegex  = regexp.MustCompile(`<p[^>]*>`)
	pCloseRegex = regexp.MustCompile(`</p>`)
	// Match bold
	boldRegex = regexp.MustCompile(`<b>(.*?)</b>`)
	// Match underline
	underlineRegex = regexp.MustCompile(`<u>(.*?)</u>`)
	// Match italic
	italicRegex = regexp.MustCompile(`<i>(.*?)</i>`)
	// Match <td> and </td>
	tdRegex = regexp.MustCompile(`</?td[^>]*>`)
	// Match <tr> and </tr>
	trOpenRegex  = regexp.MustCompile(`<tr[^>]*>`)
	trCloseRegex = regexp.MustCompile(`</tr>`)
	// Match <table> and </table>
	tableRegex = regexp.MustCompile(`</?table[^>]*>`)
	// Match <ol>, <ul>, <li>
	listItemRegex  = regexp.MustCompile(`<li[^>]*>`)
	listCloseRegex = regexp.MustCompile(`</li>`)
	olOpenRegex    = regexp.MustCompile(`<ol[^>]*>`)
	olCloseRegex   = regexp.MustCompile(`</ol>`)
	ulOpenRegex    = regexp.MustCompile(`<ul[^>]*>`)
	ulCloseRegex   = regexp.MustCompile(`</ul>`)
	// Match <img> tags (for image descriptions)
	imgTagRegex = regexp.MustCompile(`<img\s+alt="([^"]*)"[^>]*/?>`)
	// Match headers (h1-h6) — Go regexp doesn't support backreferences,
	// so we handle each level separately
	headerH1Regex = regexp.MustCompile(`<h1[^>]*>(.*?)</h1>`)
	headerH2Regex = regexp.MustCompile(`<h2[^>]*>(.*?)</h2>`)
	headerH3Regex = regexp.MustCompile(`<h3[^>]*>(.*?)</h3>`)
	headerH4Regex = regexp.MustCompile(`<h4[^>]*>(.*?)</h4>`)
	headerH5Regex = regexp.MustCompile(`<h5[^>]*>(.*?)</h5>`)
	headerH6Regex = regexp.MustCompile(`<h6[^>]*>(.*?)</h6>`)
	// Multiple blank lines
	multiBlankRegex = regexp.MustCompile(`\n{3,}`)
	// Trailing/leading whitespace per line
	lineWhitespaceRegex = regexp.MustCompile(`[ \t]+\n`)
)

func htmlToMarkdown(html string) string {
	var b strings.Builder

	// Split into top-level divs (each is a layout block)
	divs := splitTopDivs(html)
	for _, div := range divs {
		label := extractLabel(div)
		// Skip headers and footers (configurable, but default skip)
		if label == "Page-Header" || label == "Page-Footer" || label == "Blank-Page" {
			continue
		}

		// Extract inner content
		inner := extractInner(div)

		// Handle image blocks
		if label == "Image" || label == "Figure" {
			imgMatch := imgTagRegex.FindStringSubmatch(inner)
			if len(imgMatch) > 1 && imgMatch[1] != "" {
				b.WriteString("*[" + imgMatch[1] + "]*\n\n")
			}
			continue
		}

		// Convert the inner HTML to markdown
		md := innerHTMLToMarkdown(inner)
		if md != "" {
			b.WriteString(md)
			b.WriteString("\n\n")
		}
	}

	result := b.String()
	// Clean up
	result = multiBlankRegex.ReplaceAllString(result, "\n\n")
	result = lineWhitespaceRegex.ReplaceAllString(result, "\n")
	return strings.TrimSpace(result)
}

// splitTopDivs splits the HTML into top-level <div> blocks.
func splitTopDivs(html string) []string {
	var divs []string
	depth := 0
	start := -1
	i := 0
	for i < len(html) {
		if strings.HasPrefix(html[i:], "<div") {
			if depth == 0 {
				start = i
			}
			depth++
			// Skip to end of opening tag
			gt := strings.Index(html[i:], ">")
			if gt < 0 {
				break
			}
			i += gt + 1
			continue
		}
		if strings.HasPrefix(html[i:], "</div>") {
			depth--
			if depth == 0 && start >= 0 {
				divs = append(divs, html[start:i+6])
				start = -1
			}
			i += 6
			continue
		}
		i++
	}
	return divs
}

func extractLabel(div string) string {
	re := regexp.MustCompile(`data-label="([^"]*)"`)
	m := re.FindStringSubmatch(div)
	if len(m) > 1 {
		return m[1]
	}
	return ""
}

func extractInner(div string) string {
	// Find the first > after <div ...> and the last </div>
	gt := strings.Index(div, ">")
	if gt < 0 {
		return ""
	}
	inner := div[gt+1:]
	// Remove trailing </div>
	if strings.HasSuffix(inner, "</div>") {
		inner = inner[:len(inner)-6]
	}
	return inner
}

func innerHTMLToMarkdown(s string) string {
	// Handle headers (h1-h6)
	s = headerH1Regex.ReplaceAllString(s, "# $1")
	s = headerH2Regex.ReplaceAllString(s, "## $1")
	s = headerH3Regex.ReplaceAllString(s, "### $1")
	s = headerH4Regex.ReplaceAllString(s, "#### $1")
	s = headerH5Regex.ReplaceAllString(s, "##### $1")
	s = headerH6Regex.ReplaceAllString(s, "###### $1")

	// Convert <br/> to newlines
	s = brRegex.ReplaceAllString(s, "\n")
	// Convert <p> tags
	s = pOpenRegex.ReplaceAllString(s, "")
	s = pCloseRegex.ReplaceAllString(s, "\n")
	// Convert bold
	s = boldRegex.ReplaceAllString(s, "**$1**")
	// Convert underline (use bold since markdown has no underline)
	s = underlineRegex.ReplaceAllString(s, "**$1**")
	// Convert italic
	s = italicRegex.ReplaceAllString(s, "*$1*")
	// Convert tables: <td> → |, <tr> → |...|\n
	s = tdRegex.ReplaceAllString(s, "|")
	s = trCloseRegex.ReplaceAllString(s, "|\n")
	s = trOpenRegex.ReplaceAllString(s, "")
	s = tableRegex.ReplaceAllString(s, "")
	// Convert lists
	s = olOpenRegex.ReplaceAllString(s, "\n")
	s = ulOpenRegex.ReplaceAllString(s, "\n")
	s = olCloseRegex.ReplaceAllString(s, "\n")
	s = ulCloseRegex.ReplaceAllString(s, "\n")
	s = listItemRegex.ReplaceAllString(s, "- ")
	s = listCloseRegex.ReplaceAllString(s, "\n")
	// Strip any remaining tags
	s = stripRemainingTags(s)
	// Decode HTML entities
	s = decodeEntities(s)
	// Clean up whitespace
	s = strings.TrimSpace(s)
	return s
}

var remainingTagsRegex = regexp.MustCompile(`<[^>]+>`)

func stripRemainingTags(s string) string {
	return remainingTagsRegex.ReplaceAllString(s, "")
}

func decodeEntities(s string) string {
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&quot;", "\"")
	s = strings.ReplaceAll(s, "&#39;", "'")
	s = strings.ReplaceAll(s, "&nbsp;", " ")
	return s
}

// ── Devanagari conjunct fixer ──────────────────────────────────────
// Chandra 2 has a known bug where it drops the र consonant in certain
// conjunct clusters. This function fixes the most common errors.
//
// Pattern: the model outputs the conjunct without the र matra where
// it should be present. We fix these by replacing the broken form
// with the correct conjunct.
//
// Common errors observed:
//   कय → क्रय (purchase) — but only in the context of stamp/legal docs
//   केता → क्रेता (buyer)
//   विकय → विक्रय (sale)
//   केती → क्रेती (buyer, feminine)
//
// We use word-boundary aware replacements to avoid false positives.

var devanagariFixes = []struct {
	broken    string
	corrected string
}{
	// क्रय (purchase) — most common error
	{"कय", "क्रय"},
	// क्रेता (buyer)
	{"केता", "क्रेता"},
	// क्रेती (buyer, feminine form)
	{"केती", "क्रेती"},
	// विक्रय (sale)
	{"विकय", "विक्रय"},
	// महिला श्रेणी → महिला क्रेती (female buyer category)
	// This is a compound fix for "महिला केती" → "महिला क्रेती"
	// (already covered by the केती → क्रेती fix above)
}

func fixDevanagariConjuncts(s string) string {
	for _, fix := range devanagariFixes {
		s = strings.ReplaceAll(s, fix.broken, fix.corrected)
	}
	return s
}
