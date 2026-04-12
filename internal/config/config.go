package config

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"
)

// Backend defines an LLM backend with routing rules.
type Backend struct {
	Name       string `json:"name"`
	Pattern    string `json:"pattern"`              // regex to match model names
	URL        string `json:"url"`                  // backend base URL (e.g. http://localhost:8000)
	Model      string `json:"model,omitempty"`       // override model name sent to backend
	APIKey     string `json:"api_key,omitempty"`     // API key for the backend
	Protocol   string `json:"protocol,omitempty"`    // "openai" (default), "anthropic"
	Priority   int    `json:"priority,omitempty"`    // lower = tried first for fallback groups
	MaxContext int    `json:"max_context,omitempty"` // override max context length reported to clients (0 = use backend default)

	compiled *regexp.Regexp
}

// Match returns true if the model name matches this backend's pattern.
func (b *Backend) Match(model string) bool {
	if b.compiled == nil {
		b.compiled = regexp.MustCompile(b.Pattern)
	}
	return b.compiled.MatchString(model)
}

// TargetModel returns the model name to use when forwarding to this backend.
func (b *Backend) TargetModel(requestModel string) string {
	if b.Model != "" {
		return b.Model
	}
	return requestModel
}

// Config holds the proxy configuration.
type Config struct {
	Listen               string    `json:"listen"`                          // address to listen on (default ":8080")
	Backends             []Backend `json:"backends"`                        // ordered list of backends
	HealthCheckInterval  string    `json:"health_check_interval,omitempty"` // e.g. "30s" (default "30s")
	HealthCheckTimeout   string    `json:"health_check_timeout,omitempty"`  // e.g. "5s" (default "5s")
}

// HealthInterval returns the parsed health check interval, or the default (30s).
func (c *Config) HealthInterval() time.Duration {
	if c.HealthCheckInterval != "" {
		if d, err := time.ParseDuration(c.HealthCheckInterval); err == nil {
			return d
		}
	}
	return 30 * time.Second
}

// HealthTimeout returns the parsed health check timeout, or the default (5s).
func (c *Config) HealthTimeout() time.Duration {
	if c.HealthCheckTimeout != "" {
		if d, err := time.ParseDuration(c.HealthCheckTimeout); err == nil {
			return d
		}
	}
	return 5 * time.Second
}

// ParseModelBackend splits "model@backend-name" into (model, backendName).
// If no "@" is present, backendName is empty.
func ParseModelBackend(raw string) (model, backendName string) {
	idx := strings.IndexByte(raw, '@')
	if idx < 0 {
		return raw, ""
	}
	return raw[:idx], raw[idx+1:]
}

// Load reads configuration from a JSON file.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	if cfg.Listen == "" {
		cfg.Listen = ":8080"
	}
	// Pre-compile regexes
	for i := range cfg.Backends {
		re, err := regexp.Compile(cfg.Backends[i].Pattern)
		if err != nil {
			return nil, fmt.Errorf("invalid pattern %q for backend %q: %w", cfg.Backends[i].Pattern, cfg.Backends[i].Name, err)
		}
		cfg.Backends[i].compiled = re
	}
	return &cfg, nil
}

// FindBackends returns all backends matching the given model, ordered by priority.
func (c *Config) FindBackends(model string) []Backend {
	var matches []Backend
	for _, b := range c.Backends {
		if b.Match(model) {
			matches = append(matches, b)
		}
	}
	return matches
}

// AdvertisedModel holds a model name and optional context limit for the /v1/models response.
type AdvertisedModel struct {
	Name       string
	MaxContext int // 0 means not set (omit from response)
}

// AdvertisedModels returns the deduplicated list of models from all backends.
func (c *Config) AdvertisedModels() []AdvertisedModel {
	seen := map[string]bool{}
	var models []AdvertisedModel
	for _, b := range c.Backends {
		name := b.Model
		if name == "" {
			name = b.Name
		}
		if !seen[name] {
			seen[name] = true
			models = append(models, AdvertisedModel{
				Name:       name,
				MaxContext: b.MaxContext,
			})
		}
	}
	return models
}
