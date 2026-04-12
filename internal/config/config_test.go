package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	data := `{
		"listen": ":9090",
		"backends": [
			{"name": "vllm-local", "pattern": ".*qwen.*", "url": "http://localhost:8000", "model": "Qwen/Qwen3-Coder"},
			{"name": "fallback", "pattern": ".*", "url": "http://localhost:8001"}
		]
	}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Listen != ":9090" {
		t.Errorf("expected listen :9090, got %s", cfg.Listen)
	}
	if len(cfg.Backends) != 2 {
		t.Fatalf("expected 2 backends, got %d", len(cfg.Backends))
	}
}

func TestLoad_DefaultListen(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	data := `{"backends": [{"name": "b", "pattern": ".*", "url": "http://x"}]}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Listen != ":8080" {
		t.Errorf("expected default listen :8080, got %s", cfg.Listen)
	}
}

func TestLoad_InvalidPattern(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	data := `{"backends": [{"name": "bad", "pattern": "[invalid", "url": "http://x"}]}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for invalid regex")
	}
}

func TestBackend_Match(t *testing.T) {
	b := Backend{Pattern: `(?i)qwen.*coder`}
	if !b.Match("Qwen3-Coder-Next") {
		t.Error("expected match for Qwen3-Coder-Next")
	}
	if b.Match("gpt-4") {
		t.Error("expected no match for gpt-4")
	}
}

func TestBackend_TargetModel(t *testing.T) {
	b := Backend{Model: "override-model"}
	if got := b.TargetModel("original"); got != "override-model" {
		t.Errorf("expected override-model, got %s", got)
	}
	b2 := Backend{}
	if got := b2.TargetModel("original"); got != "original" {
		t.Errorf("expected original, got %s", got)
	}
}

func TestFindBackends(t *testing.T) {
	cfg := &Config{
		Backends: []Backend{
			{Name: "qwen", Pattern: `(?i)qwen`, URL: "http://a"},
			{Name: "gpt", Pattern: `(?i)gpt`, URL: "http://b"},
			{Name: "fallback", Pattern: `.*`, URL: "http://c"},
		},
	}
	// Pre-compile
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	matches := cfg.FindBackends("qwen3-coder")
	if len(matches) != 2 { // qwen + fallback
		t.Errorf("expected 2 matches for qwen3-coder, got %d", len(matches))
	}

	matches = cfg.FindBackends("gpt-4")
	if len(matches) != 2 { // gpt + fallback
		t.Errorf("expected 2 matches for gpt-4, got %d", len(matches))
	}

	matches = cfg.FindBackends("llama-3")
	if len(matches) != 1 { // fallback only
		t.Errorf("expected 1 match for llama-3, got %d", len(matches))
	}
}

func TestParseModelBackend(t *testing.T) {
	tests := []struct {
		raw     string
		model   string
		backend string
	}{
		{"gemma4", "gemma4", ""},
		{"gemma4@vllm-1", "gemma4", "vllm-1"},
		{"qwen3-coder@qwen-backend", "qwen3-coder", "qwen-backend"},
		{"model@", "model", ""},
		{"@backend", "", "backend"},
		{"a@b@c", "a", "b@c"},
	}
	for _, tt := range tests {
		model, backend := ParseModelBackend(tt.raw)
		if model != tt.model || backend != tt.backend {
			t.Errorf("ParseModelBackend(%q) = (%q, %q), want (%q, %q)",
				tt.raw, model, backend, tt.model, tt.backend)
		}
	}
}

func TestHealthInterval_Default(t *testing.T) {
	cfg := &Config{}
	if got := cfg.HealthInterval(); got != 30*1e9 {
		t.Errorf("expected 30s default, got %v", got)
	}
}

func TestHealthInterval_Custom(t *testing.T) {
	cfg := &Config{HealthCheckInterval: "10s"}
	if got := cfg.HealthInterval(); got != 10*1e9 {
		t.Errorf("expected 10s, got %v", got)
	}
}

func TestAdvertisedModels(t *testing.T) {
	cfg := &Config{
		Backends: []Backend{
			{Name: "qwen", Model: "Qwen/Qwen3"},
			{Name: "gpt", Model: "gpt-4"},
			{Name: "fallback", Model: "Qwen/Qwen3"}, // duplicate
		},
	}
	models := cfg.AdvertisedModels()
	if len(models) != 2 {
		t.Errorf("expected 2 unique models, got %d", len(models))
	}
}

func TestAdvertisedModels_MaxContext(t *testing.T) {
	cfg := &Config{
		Backends: []Backend{
			{Name: "gemma", Model: "gemma-4", MaxContext: 106496},
			{Name: "qwen", Model: "qwen-3"},
		},
	}
	models := cfg.AdvertisedModels()
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	if models[0].MaxContext != 106496 {
		t.Errorf("expected MaxContext 106496, got %d", models[0].MaxContext)
	}
	if models[1].MaxContext != 0 {
		t.Errorf("expected MaxContext 0 (unset), got %d", models[1].MaxContext)
	}
}
