package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/config"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/protocol"
)

// newTestBackend creates a mock LLM backend server.
func newTestBackend(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	return httptest.NewServer(handler)
}

// chatCompletionHandler returns a handler that echoes the model and a fixed response.
func chatCompletionHandler(responseText string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]json.RawMessage
		_ = json.Unmarshal(body, &req)

		var model string
		_ = json.Unmarshal(req["model"], &model)

		stop := "stop"
		resp := protocol.ChatResponse{
			ID:    "test-123",
			Model: model,
			Choices: []protocol.ChatChoice{
				{
					Message:      protocol.ChatMessage{Role: "assistant", Content: mustMarshal(responseText)},
					FinishReason: &stop,
				},
			},
			Usage: &protocol.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}
}

func mustMarshal(v any) json.RawMessage {
	data, _ := json.Marshal(v)
	return data
}

func TestHealthEndpoint(t *testing.T) {
	cfg := &config.Config{Backends: []config.Backend{}}
	srv := NewServer(cfg)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestModelsEndpoint(t *testing.T) {
	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "qwen", Model: "Qwen/Qwen3"},
			{Name: "gpt", Model: "gpt-4"},
		},
	}
	srv := NewServer(cfg)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp protocol.ModelsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(resp.Data) != 2 {
		t.Errorf("expected 2 models, got %d", len(resp.Data))
	}
}

func TestModelsEndpoint_MaxContext(t *testing.T) {
	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "gemma", Model: "gemma-4", MaxContext: 106496},
			{Name: "qwen", Model: "qwen-3"},
		},
	}
	srv := NewServer(cfg)
	defer srv.Close()

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp protocol.ModelsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(resp.Data) != 2 {
		t.Fatalf("expected 2 models, got %d", len(resp.Data))
	}

	// First model should have max_model_len set
	if resp.Data[0].MaxModelLen != 106496 {
		t.Errorf("expected max_model_len 106496 for gemma, got %d", resp.Data[0].MaxModelLen)
	}

	// Second model should have max_model_len omitted (0)
	if resp.Data[1].MaxModelLen != 0 {
		t.Errorf("expected max_model_len 0 (omitted) for qwen, got %d", resp.Data[1].MaxModelLen)
	}

	// Verify JSON: max_model_len should be absent for qwen
	raw := w.Body.String()
	// gemma entry should contain max_model_len
	if !strings.Contains(raw, `"max_model_len":106496`) {
		t.Errorf("expected max_model_len in JSON for gemma, got: %s", raw)
	}
}

func TestChatCompletions_InjectsChatTemplateKwargs(t *testing.T) {
	var receivedBody []byte
	backend := newTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
		receivedBody, _ = io.ReadAll(r.Body)
		chatCompletionHandler("ok")(w, r)
	})
	defer backend.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{
				Name: "vllm", Pattern: ".*", URL: backend.URL, Model: "gemma",
				ChatTemplateKwargs: map[string]any{"enable_thinking": true},
			},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	defer srv.Close()

	body := `{"model":"gemma","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify chat_template_kwargs was injected
	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(receivedBody, &parsed); err != nil {
		t.Fatalf("parse forwarded body: %v", err)
	}
	kwargs, ok := parsed["chat_template_kwargs"]
	if !ok {
		t.Fatal("expected chat_template_kwargs to be injected")
	}
	var kwargsMap map[string]any
	if err := json.Unmarshal(kwargs, &kwargsMap); err != nil {
		t.Fatalf("parse kwargs: %v", err)
	}
	if kwargsMap["enable_thinking"] != true {
		t.Errorf("expected enable_thinking=true, got %v", kwargsMap["enable_thinking"])
	}
}

func TestChatCompletions_DoesNotOverrideExistingKwargs(t *testing.T) {
	var receivedBody []byte
	backend := newTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
		receivedBody, _ = io.ReadAll(r.Body)
		chatCompletionHandler("ok")(w, r)
	})
	defer backend.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{
				Name: "vllm", Pattern: ".*", URL: backend.URL, Model: "gemma",
				ChatTemplateKwargs: map[string]any{"enable_thinking": true},
			},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	defer srv.Close()

	// Request already has chat_template_kwargs — proxy should NOT override
	body := `{"model":"gemma","messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false}}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var parsed map[string]json.RawMessage
	json.Unmarshal(receivedBody, &parsed)
	var kwargsMap map[string]any
	json.Unmarshal(parsed["chat_template_kwargs"], &kwargsMap)
	if kwargsMap["enable_thinking"] != false {
		t.Errorf("expected client's enable_thinking=false to be preserved, got %v", kwargsMap["enable_thinking"])
	}
}

func TestChatCompletions_RoutesToBackend(t *testing.T) {
	backend := newTestBackend(t, chatCompletionHandler("Hello from vLLM"))
	defer backend.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "vllm", Pattern: ".*", URL: backend.URL, Model: "Qwen/Qwen3"},
		},
	}
	// Pre-compile
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	body := `{"model":"qwen3","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp protocol.ChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	// Model should be rewritten to backend model
	if resp.Model != "Qwen/Qwen3" {
		t.Errorf("expected model Qwen/Qwen3, got %s", resp.Model)
	}
}

func TestChatCompletions_ModelRouting(t *testing.T) {
	backendA := newTestBackend(t, chatCompletionHandler("from A"))
	defer backendA.Close()
	backendB := newTestBackend(t, chatCompletionHandler("from B"))
	defer backendB.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "qwen", Pattern: `(?i)qwen`, URL: backendA.URL, Model: "Qwen/Qwen3"},
			{Name: "fallback", Pattern: `.*`, URL: backendB.URL, Model: "default-model"},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)

	// Request for qwen -> goes to backendA
	body := `{"model":"qwen3-coder","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	var resp protocol.ChatResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Model != "Qwen/Qwen3" {
		t.Errorf("expected Qwen/Qwen3 for qwen request, got %s", resp.Model)
	}

	// Request for unknown model -> goes to fallback (backendB)
	body = `{"model":"llama-3","messages":[{"role":"user","content":"hi"}]}`
	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w = httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Model != "default-model" {
		t.Errorf("expected default-model for fallback, got %s", resp.Model)
	}
}

func TestChatCompletions_DirectBackendRoute(t *testing.T) {
	a := newTestBackend(t, chatCompletionHandler("from A"))
	defer a.Close()
	b := newTestBackend(t, chatCompletionHandler("from B"))
	defer b.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "alpha", Pattern: `(?i)qwen`, URL: a.URL, Model: "alpha-model"},
			{Name: "beta", Pattern: `(?i)qwen`, URL: b.URL, Model: "beta-model"},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}
	srv := NewServer(cfg)

	// Direct route /beta/... must hit beta even though both match the model regex.
	body := `{"model":"qwen-test","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/beta/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("direct /beta route: expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp protocol.ChatResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Model != "beta-model" {
		t.Errorf("direct /beta route: expected beta-model, got %s", resp.Model)
	}

	// Direct route also bypasses model-pattern check — requesting an unrelated model
	// via /alpha/... still lands on alpha.
	body = `{"model":"llama-something","messages":[{"role":"user","content":"hi"}]}`
	req = httptest.NewRequest(http.MethodPost, "/alpha/v1/chat/completions", strings.NewReader(body))
	w = httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("direct /alpha for non-matching model: expected 200, got %d: %s", w.Code, w.Body.String())
	}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Model != "alpha-model" {
		t.Errorf("direct /alpha route: expected alpha-model, got %s", resp.Model)
	}

	// Default (non-pinned) /v1/... still works and goes through model-regex matching.
	body = `{"model":"qwen-test","messages":[{"role":"user","content":"hi"}]}`
	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w = httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("default route: expected 200, got %d", w.Code)
	}
}

func TestChatCompletions_DirectRoute_UnknownBackend(t *testing.T) {
	a := newTestBackend(t, chatCompletionHandler("from A"))
	defer a.Close()
	cfg := &config.Config{
		Backends: []config.Backend{{Name: "alpha", Pattern: `.*`, URL: a.URL, Model: "alpha-model"}},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}
	srv := NewServer(cfg)

	// /ghost/... — no such backend or host. The mux has no route registered for /ghost,
	// so this falls through to the default mux 404. Verify it doesn't accidentally land
	// on the only backend.
	body := `{"model":"x","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/ghost/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code == http.StatusOK {
		t.Errorf("expected non-200 for unknown direct route, got 200")
	}
}

func TestChatCompletions_FallbackOnError(t *testing.T) {
	// First backend returns 500
	failingBackend := newTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"down"}`))
	})
	defer failingBackend.Close()

	// Second backend works
	workingBackend := newTestBackend(t, chatCompletionHandler("fallback worked"))
	defer workingBackend.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "primary", Pattern: `.*`, URL: failingBackend.URL},
			{Name: "fallback", Pattern: `.*`, URL: workingBackend.URL},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	body := `{"model":"any","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 after fallback, got %d: %s", w.Code, w.Body.String())
	}
}

// TestChatCompletions_DiscoveryRoundRobin verifies that when two backends
// advertise the same model id via their /v1/models endpoints (the design
// used by the gemma-4-fleet alias on spark-07 + thor-04), femtollm
// discovers both and round-robins requests across them.
func TestChatCompletions_DiscoveryRoundRobin(t *testing.T) {
	const sharedModelID = "gemma-4-fleet"

	// Each backend tracks how many chat completions it received and
	// includes its name in /v1/models so we can confirm discovery picked
	// it up. /v1/models returns the shared id so femtollm should treat
	// the two as interchangeable.
	makeBackend := func(name string, hits *int32) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/v1/models":
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"data":[{"id":"` + sharedModelID + `"}]}`))
			case "/v1/chat/completions":
				atomic.AddInt32(hits, 1)
				body, _ := io.ReadAll(r.Body)
				var req map[string]json.RawMessage
				_ = json.Unmarshal(body, &req)
				var requestedModel string
				_ = json.Unmarshal(req["model"], &requestedModel)
				stop := "stop"
				resp := protocol.ChatResponse{
					ID:    "resp-" + name,
					Model: requestedModel,
					Choices: []protocol.ChatChoice{{
						Message:      protocol.ChatMessage{Role: "assistant", Content: mustMarshal("from " + name)},
						FinishReason: &stop,
					}},
					Usage: &protocol.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(resp)
			default:
				w.WriteHeader(http.StatusNotFound)
			}
		}))
	}

	var hitsA, hitsB int32
	backendA := makeBackend("A", &hitsA)
	defer backendA.Close()
	backendB := makeBackend("B", &hitsB)
	defer backendB.Close()

	// No Pattern, no Model — pure discovery routing. Both backends advertise
	// `gemma-4-fleet` so any request for that id should hit either of them.
	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "fleet-a", URL: backendA.URL},
			{Name: "fleet-b", URL: backendB.URL},
		},
	}
	srv := NewServer(cfg)
	defer srv.Close()

	const total = 12
	for i := 0; i < total; i++ {
		body := `{"model":"` + sharedModelID + `","messages":[{"role":"user","content":"hi"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("request %d: expected 200, got %d: %s", i, w.Code, w.Body.String())
		}
	}

	totalHits := atomic.LoadInt32(&hitsA) + atomic.LoadInt32(&hitsB)
	if int(totalHits) != total {
		t.Fatalf("expected %d total hits across both backends, got %d (A=%d B=%d)",
			total, totalHits, hitsA, hitsB)
	}

	// Each backend should get at least 1/3 of the traffic — load is
	// identical (both at zero KV usage) so round-robin tie-break should
	// distribute nearly evenly. Allow generous slack for ordering noise.
	minPerBackend := int32(total / 3)
	if hitsA < minPerBackend || hitsB < minPerBackend {
		t.Errorf("round-robin imbalance: A=%d B=%d (each should be >= %d)",
			hitsA, hitsB, minPerBackend)
	}
}

// TestChatCompletions_DiscoveryMultiModel verifies that a backend can
// expose multiple model ids via /v1/models (vLLM --served-model-name alias
// list) and that femtollm routes each id to the backend that advertises it.
func TestChatCompletions_DiscoveryMultiModel(t *testing.T) {
	gemmaBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"gemma-real"},{"id":"gemma-4-fleet"}]}`))
			return
		}
		chatCompletionHandler("gemma response")(w, r)
	}))
	defer gemmaBackend.Close()

	qwenBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"qwen-real"}]}`))
			return
		}
		chatCompletionHandler("qwen response")(w, r)
	}))
	defer qwenBackend.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "gemma", URL: gemmaBackend.URL},
			{Name: "qwen", URL: qwenBackend.URL},
		},
	}
	srv := NewServer(cfg)
	defer srv.Close()

	// Both gemma-real and gemma-4-fleet should land on the gemma backend.
	// qwen-real should land on the qwen backend.
	for _, model := range []string{"gemma-real", "gemma-4-fleet", "qwen-real"} {
		body := `{"model":"` + model + `","messages":[{"role":"user","content":"hi"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("model %s: expected 200, got %d: %s", model, w.Code, w.Body.String())
		}
	}

	// Asking for an unknown id should 404 — neither backend advertises it.
	body := `{"model":"not-served-anywhere","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusNotFound {
		t.Errorf("unknown model should 404, got %d", w.Code)
	}
}

func TestChatCompletions_NoBackend(t *testing.T) {
	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "qwen", Pattern: `qwen`, URL: "http://unused"},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

func TestAnthropicMessages_ConvertsToOpenAI(t *testing.T) {
	backend := newTestBackend(t, chatCompletionHandler("Converted response"))
	defer backend.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "vllm", Pattern: ".*", URL: backend.URL, Model: "Qwen/Qwen3"},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	body := `{
		"model": "claude-3-sonnet",
		"max_tokens": 1024,
		"system": "You are helpful",
		"messages": [{"role": "user", "content": "Hello"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp protocol.AnthropicResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resp.Type != "message" {
		t.Errorf("expected type message, got %s", resp.Type)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("expected stop_reason end_turn, got %s", resp.StopReason)
	}
	if len(resp.Content) == 0 || resp.Content[0].Text != "Converted response" {
		t.Errorf("expected converted response text, got %v", resp.Content)
	}
}

func TestChatCompletions_RoundRobin(t *testing.T) {
	var countA, countB atomic.Int32

	backendA := newTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			countA.Add(1)
		}
		chatCompletionHandler("from A")(w, r)
	})
	defer backendA.Close()

	backendB := newTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			countB.Add(1)
		}
		chatCompletionHandler("from B")(w, r)
	})
	defer backendB.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "a", Pattern: `.*`, URL: backendA.URL},
			{Name: "b", Pattern: `.*`, URL: backendB.URL},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	defer srv.Close()

	for i := 0; i < 10; i++ {
		body := `{"model":"any","messages":[{"role":"user","content":"hi"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("request %d: expected 200, got %d", i, w.Code)
		}
	}

	a, b := countA.Load(), countB.Load()
	if a != 5 || b != 5 {
		t.Errorf("expected 5/5 distribution, got a=%d b=%d", a, b)
	}
}

func TestChatCompletions_BackendPreference(t *testing.T) {
	backendA := newTestBackend(t, chatCompletionHandler("from A"))
	defer backendA.Close()
	backendB := newTestBackend(t, chatCompletionHandler("from B"))
	defer backendB.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "alpha", Pattern: `.*`, URL: backendA.URL, Model: "model-a"},
			{Name: "beta", Pattern: `.*`, URL: backendB.URL, Model: "model-b"},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	defer srv.Close()

	// Pin to "beta" backend
	body := `{"model":"anything@beta","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp protocol.ChatResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Model != "model-b" {
		t.Errorf("expected model-b (beta backend), got %s", resp.Model)
	}
}

func TestChatCompletions_BackendPreference_NotFound(t *testing.T) {
	backend := newTestBackend(t, chatCompletionHandler("hello"))
	defer backend.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "real", Pattern: `.*`, URL: backend.URL},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	defer srv.Close()

	body := `{"model":"any@nonexistent","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404 for unknown backend, got %d", w.Code)
	}
}

func TestBackendHealthEndpoint(t *testing.T) {
	backend := newTestBackend(t, chatCompletionHandler("hi"))
	defer backend.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "test-be", Pattern: `.*`, URL: backend.URL},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	defer srv.Close()

	req := httptest.NewRequest(http.MethodGet, "/health/backends", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp struct {
		Backends []struct {
			Name  string `json:"name"`
			Alive bool   `json:"alive"`
		} `json:"backends"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(resp.Backends) != 1 {
		t.Fatalf("expected 1 backend, got %d", len(resp.Backends))
	}
	if !resp.Backends[0].Alive {
		t.Error("expected backend to be alive")
	}
}

func TestChatCompletions_StreamingFlush(t *testing.T) {
	// Simulate an SSE streaming backend that sends chunks with small delays
	sseChunks := []string{
		"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
		"data: [DONE]\n\n",
	}

	backend := newTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("backend ResponseWriter does not support Flusher")
			return
		}
		for _, chunk := range sseChunks {
			_, _ = w.Write([]byte(chunk))
			flusher.Flush()
		}
	})
	defer backend.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "sse", Pattern: ".*", URL: backend.URL, Model: "test-model"},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	defer srv.Close()

	body := `{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify all SSE chunks were forwarded
	result := w.Body.String()
	for _, chunk := range sseChunks {
		if !strings.Contains(result, strings.TrimSpace(chunk)) {
			t.Errorf("expected chunk %q in response, got: %s", chunk, result)
		}
	}

	// Verify Content-Type is SSE
	ct := w.Header().Get("Content-Type")
	if ct != "text/event-stream" {
		t.Errorf("expected Content-Type text/event-stream, got %s", ct)
	}
}

func TestAnthropicMessages_FallbackOnError(t *testing.T) {
	failing := newTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})
	defer failing.Close()
	working := newTestBackend(t, chatCompletionHandler("ok"))
	defer working.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "bad", Pattern: ".*", URL: failing.URL},
			{Name: "good", Pattern: ".*", URL: working.URL},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	body := `{"model":"x","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 after fallback, got %d", w.Code)
	}
}
