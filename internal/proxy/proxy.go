package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/balancer"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/config"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/health"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/prefixtrie"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/protocol"
)

// Server is the LLM proxy server.
type Server struct {
	Config   *config.Config
	Client   *http.Client
	Checker  *health.Checker
	Balancer *balancer.Balancer
}

// NewServer creates a new proxy server with health checking and load balancing.
func NewServer(cfg *config.Config) *Server {
	var backends []health.Backend
	for _, b := range cfg.Backends {
		backends = append(backends, health.Backend{Name: b.Name, URL: b.URL})
	}

	checker := health.NewChecker(backends, cfg.HealthInterval(), cfg.HealthTimeout())
	checker.Start()

	trie := prefixtrie.New(prefixtrie.DefaultChunkSize)

	return &Server{
		Config:   cfg,
		Client:   &http.Client{Timeout: 5 * time.Minute},
		Checker:  checker,
		Balancer: balancer.NewBalancer(checker, trie),
	}
}

// Close stops background health checks.
func (s *Server) Close() {
	if s.Checker != nil {
		s.Checker.Stop()
	}
}

// Handler returns the HTTP handler for the proxy.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// OpenAI-compatible endpoints
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/models", s.handleModels)

	// Anthropic-compatible endpoint
	mux.HandleFunc("/v1/messages", s.handleAnthropicMessages)

	// Health
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("/health/backends", s.handleBackendHealth)

	return mux
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	models := s.Config.AdvertisedModels()
	var entries []protocol.ModelEntry
	for _, m := range models {
		entries = append(entries, protocol.ModelEntry{
			ID:          m.Name,
			Object:      "model",
			Created:     time.Now().Unix(),
			OwnedBy:     "femtollm",
			MaxModelLen: m.MaxContext,
		})
	}

	resp := protocol.ModelsResponse{Object: "list", Data: entries}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	var req protocol.ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	// Parse model@backend preference
	model, preferredBackend := config.ParseModelBackend(req.Model)
	req.Model = model

	backends := s.Config.FindBackends(model)
	if preferredBackend != "" {
		backends = filterByName(backends, preferredBackend)
	}
	if len(backends) == 0 {
		if preferredBackend != "" {
			writeError(w, http.StatusNotFound, fmt.Sprintf("no backend named %q for model %q", preferredBackend, model))
		} else {
			writeError(w, http.StatusNotFound, fmt.Sprintf("no backend for model %q", model))
		}
		return
	}

	// Extract prompt for prefix-cache routing
	prompt := extractPrompt(&req)

	// Select backend using load + prefix-cache scoring
	backends = s.Balancer.SelectWithPrompt(backends, model, prompt)

	// Try backends in order (fallback)
	var lastErr error
	for _, backend := range backends {
		err := s.forwardOpenAI(w, r, body, &req, &backend)
		if err == nil {
			// Record successful route for prefix trie
			s.Balancer.RecordRoute(prompt, backend.Name)
			return
		}
		lastErr = err
		log.Printf("Backend %s failed for model %s: %v", backend.Name, req.Model, err)
	}

	writeError(w, http.StatusBadGateway, fmt.Sprintf("all backends failed: %v", lastErr))
}

func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	var req protocol.AnthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	if req.Stream {
		writeError(w, http.StatusNotImplemented, "anthropic streaming not yet supported, use non-streaming")
		return
	}

	// Parse model@backend preference
	model, preferredBackend := config.ParseModelBackend(req.Model)
	req.Model = model

	backends := s.Config.FindBackends(model)
	if preferredBackend != "" {
		backends = filterByName(backends, preferredBackend)
	}
	if len(backends) == 0 {
		if preferredBackend != "" {
			writeError(w, http.StatusNotFound, fmt.Sprintf("no backend named %q for model %q", preferredBackend, model))
		} else {
			writeError(w, http.StatusNotFound, fmt.Sprintf("no backend for model %q", model))
		}
		return
	}

	// Extract prompt for prefix-cache routing
	prompt := extractAnthropicPrompt(&req)

	// Select backend using load + prefix-cache scoring
	backends = s.Balancer.SelectWithPrompt(backends, model, prompt)

	// Convert to OpenAI format, send to backend, convert response back
	openaiReq := protocol.AnthropicToOpenAI(&req)

	var lastErr error
	for _, backend := range backends {
		openaiReq.Model = backend.TargetModel(model)

		reqBody, _ := json.Marshal(openaiReq)
		url := strings.TrimRight(backend.URL, "/") + "/v1/chat/completions"

		backendReq, _ := http.NewRequestWithContext(r.Context(), http.MethodPost, url, bytes.NewReader(reqBody))
		backendReq.Header.Set("Content-Type", "application/json")
		if backend.APIKey != "" {
			backendReq.Header.Set("Authorization", "Bearer "+backend.APIKey)
		}

		resp, err := s.Client.Do(backendReq)
		if err != nil {
			lastErr = fmt.Errorf("backend %s: %w", backend.Name, err)
			log.Printf("Backend %s failed: %v", backend.Name, err)
			continue
		}

		respBody, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = fmt.Errorf("backend %s: read response: %w", backend.Name, err)
			continue
		}

		if resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("backend %s: status %d", backend.Name, resp.StatusCode)
			log.Printf("Backend %s returned %d, trying next", backend.Name, resp.StatusCode)
			continue
		}

		if resp.StatusCode != http.StatusOK {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(resp.StatusCode)
			_, _ = w.Write(respBody)
			return
		}

		var openaiResp protocol.ChatResponse
		if err := json.Unmarshal(respBody, &openaiResp); err != nil {
			writeError(w, http.StatusBadGateway, "invalid backend response")
			return
		}

		anthropicResp := protocol.OpenAIToAnthropicResponse(&openaiResp)
		s.Balancer.RecordRoute(prompt, backend.Name)
		writeJSON(w, http.StatusOK, anthropicResp)
		return
	}

	writeError(w, http.StatusBadGateway, fmt.Sprintf("all backends failed: %v", lastErr))
}

// forwardOpenAI proxies a request to an OpenAI-compatible backend.
// For streaming requests, it pipes the SSE response directly.
func (s *Server) forwardOpenAI(w http.ResponseWriter, r *http.Request, body []byte, req *protocol.ChatRequest, backend *config.Backend) error {
	// Rewrite model name and inject chat_template_kwargs
	var modified map[string]json.RawMessage
	if err := json.Unmarshal(body, &modified); err != nil {
		return fmt.Errorf("unmarshal: %w", err)
	}
	targetModel := backend.TargetModel(req.Model)
	modified["model"], _ = json.Marshal(targetModel)

	// Inject chat_template_kwargs if configured and not already present in the request
	if len(backend.ChatTemplateKwargs) > 0 {
		if _, exists := modified["chat_template_kwargs"]; !exists {
			modified["chat_template_kwargs"], _ = json.Marshal(backend.ChatTemplateKwargs)
		}
	}

	reqBody, _ := json.Marshal(modified)

	url := strings.TrimRight(backend.URL, "/") + "/v1/chat/completions"
	backendReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	backendReq.Header.Set("Content-Type", "application/json")
	if backend.APIKey != "" {
		backendReq.Header.Set("Authorization", "Bearer "+backend.APIKey)
	}

	// Use a client without timeout for streaming
	client := s.Client
	if req.Stream {
		client = &http.Client{}
	}

	resp, err := client.Do(backendReq)
	if err != nil {
		return fmt.Errorf("request: %w", err)
	}

	if resp.StatusCode >= 500 {
		resp.Body.Close()
		return fmt.Errorf("status %d", resp.StatusCode)
	}

	// Copy response headers
	for k, vals := range resp.Header {
		for _, v := range vals {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)

	// For streaming SSE responses, flush after each read to prevent buffering.
	// Without explicit Flush(), Go's ResponseWriter accumulates small SSE chunks
	// in its internal buffer, causing multi-second pauses between token batches.
	if req.Stream {
		if flusher, ok := w.(http.Flusher); ok {
			buf := make([]byte, 4096)
			for {
				n, readErr := resp.Body.Read(buf)
				if n > 0 {
					_, _ = w.Write(buf[:n])
					flusher.Flush()
				}
				if readErr != nil {
					break
				}
			}
		} else {
			_, _ = io.Copy(w, resp.Body)
		}
	} else {
		_, _ = io.Copy(w, resp.Body)
	}

	resp.Body.Close()
	return nil
}

// extractPrompt concatenates message contents from a ChatRequest for prefix matching.
func extractPrompt(req *protocol.ChatRequest) string {
	var b strings.Builder
	for _, msg := range req.Messages {
		// Content can be a string or array of content parts
		var s string
		if err := json.Unmarshal(msg.Content, &s); err == nil {
			b.WriteString(s)
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// extractAnthropicPrompt concatenates message contents from an AnthropicRequest.
func extractAnthropicPrompt(req *protocol.AnthropicRequest) string {
	var b strings.Builder
	// Include system prompt as prefix
	if len(req.System) > 0 {
		var s string
		if err := json.Unmarshal(req.System, &s); err == nil {
			b.WriteString(s)
			b.WriteByte('\n')
		}
	}
	for _, msg := range req.Messages {
		var s string
		if err := json.Unmarshal(msg.Content, &s); err == nil {
			b.WriteString(s)
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    "proxy_error",
		},
	})
}

func filterByName(backends []config.Backend, name string) []config.Backend {
	var out []config.Backend
	for _, b := range backends {
		if b.Name == name {
			out = append(out, b)
		}
	}
	return out
}

func (s *Server) handleBackendHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	type entry struct {
		Name            string  `json:"name"`
		URL             string  `json:"url"`
		Alive           bool    `json:"alive"`
		Preferred       bool    `json:"preferred,omitempty"`
		LastCheck       string  `json:"last_check,omitempty"`
		LastError       string  `json:"last_error,omitempty"`
		KVCacheUsage    float64 `json:"kv_cache_usage"`
		RequestsRunning int     `json:"requests_running"`
		RequestsWaiting int     `json:"requests_waiting"`
	}

	statuses := s.Checker.Statuses()
	var entries []entry
	for _, b := range s.Config.Backends {
		e := entry{Name: b.Name, URL: b.URL, Preferred: b.Preferred}
		if st, ok := statuses[b.Name]; ok {
			e.Alive = st.Alive
			if !st.LastCheck.IsZero() {
				e.LastCheck = st.LastCheck.Format(time.RFC3339)
			}
			e.LastError = st.LastError
			e.KVCacheUsage = st.KVCacheUsage
			e.RequestsRunning = st.RequestsRunning
			e.RequestsWaiting = st.RequestsWaiting
		}
		entries = append(entries, e)
	}

	resp := map[string]any{"backends": entries}
	if ts := s.Balancer.TrieStats(); ts != nil {
		resp["prefix_trie"] = ts
	}
	writeJSON(w, http.StatusOK, resp)
}
