package health

import (
	"bufio"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Status tracks the health and load metrics of a single backend.
type Status struct {
	Alive     bool
	LastCheck time.Time
	LastError string
	// Metrics from vLLM /metrics endpoint (updated each health check)
	KVCacheUsage    float64 // 0.0–1.0, fraction of KV-cache in use
	RequestsRunning int     // number of requests currently being processed
	RequestsWaiting int     // number of requests queued
}

// Load returns a score representing how loaded this backend is (lower = less loaded).
// Returns -1 if the backend is not alive.
func (s *Status) Load() float64 {
	if !s.Alive {
		return -1
	}
	// KV-cache usage is the primary signal: a backend near capacity can't accept
	// long-context requests. Running+waiting requests indicate queuing pressure.
	return s.KVCacheUsage + 0.01*float64(s.RequestsRunning+s.RequestsWaiting)
}

// Checker periodically probes backends and tracks health + load metrics.
type Checker struct {
	mu       sync.RWMutex
	status   map[string]*Status // keyed by backend name
	backends []Backend
	client   *http.Client
	interval time.Duration
	stopCh   chan struct{}
}

// Backend is the minimal info needed for health checks.
type Backend struct {
	Name string
	URL  string
}

// NewChecker creates a health checker. Call Start() to begin background probing.
func NewChecker(backends []Backend, interval time.Duration, timeout time.Duration) *Checker {
	status := make(map[string]*Status, len(backends))
	for _, b := range backends {
		status[b.Name] = &Status{Alive: true} // fail-open: assume alive until first check
	}
	return &Checker{
		status:   status,
		backends: backends,
		client:   &http.Client{Timeout: timeout},
		interval: interval,
		stopCh:   make(chan struct{}),
	}
}

// Start runs an initial synchronous check, then launches background probing.
func (c *Checker) Start() {
	c.CheckNow()
	go c.loop()
}

// Stop signals the background goroutine to stop.
func (c *Checker) Stop() {
	close(c.stopCh)
}

// IsAlive returns whether the named backend is healthy.
// Returns true for unknown backends (fail-open).
func (c *Checker) IsAlive(name string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	s, ok := c.status[name]
	if !ok {
		return true
	}
	return s.Alive
}

// GetStatus returns the full status for a backend (nil if unknown).
func (c *Checker) GetStatus(name string) *Status {
	c.mu.RLock()
	defer c.mu.RUnlock()
	s, ok := c.status[name]
	if !ok {
		return nil
	}
	copy := *s
	return &copy
}

// Statuses returns a snapshot of all backend health statuses.
func (c *Checker) Statuses() map[string]Status {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make(map[string]Status, len(c.status))
	for k, v := range c.status {
		out[k] = *v
	}
	return out
}

// CheckNow runs a synchronous health check of all backends.
func (c *Checker) CheckNow() {
	var wg sync.WaitGroup
	for i := range c.backends {
		wg.Add(1)
		go func(b *Backend) {
			defer wg.Done()
			c.checkOne(b)
		}(&c.backends[i])
	}
	wg.Wait()
}

func (c *Checker) loop() {
	ticker := time.NewTicker(c.interval)
	defer ticker.Stop()
	for {
		select {
		case <-c.stopCh:
			return
		case <-ticker.C:
			c.CheckNow()
		}
	}
}

func (c *Checker) checkOne(b *Backend) {
	// Health check via /v1/models
	url := strings.TrimRight(b.URL, "/") + "/v1/models"
	resp, err := c.client.Get(url)

	c.mu.Lock()
	defer c.mu.Unlock()

	s := c.status[b.Name]
	if s == nil {
		s = &Status{}
		c.status[b.Name] = s
	}
	s.LastCheck = time.Now()

	if err != nil {
		wasAlive := s.Alive
		s.Alive = false
		s.LastError = err.Error()
		if wasAlive {
			log.Printf("health: backend %s is DOWN: %v", b.Name, err)
		}
		return
	}
	resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 400 {
		wasAlive := s.Alive
		s.Alive = true
		s.LastError = ""
		if !wasAlive {
			log.Printf("health: backend %s is UP", b.Name)
		}
	} else {
		wasAlive := s.Alive
		s.Alive = false
		s.LastError = resp.Status
		if wasAlive {
			log.Printf("health: backend %s is DOWN: %s", b.Name, resp.Status)
		}
		return
	}

	// Scrape load metrics from /metrics (best-effort, don't fail health on this)
	c.mu.Unlock() // release lock during metrics fetch
	metrics := c.scrapeMetrics(b)
	c.mu.Lock() // re-acquire
	s.KVCacheUsage = metrics.kvCache
	s.RequestsRunning = metrics.running
	s.RequestsWaiting = metrics.waiting
}

type metricsResult struct {
	kvCache float64
	running int
	waiting int
}

// scrapeMetrics fetches vLLM Prometheus metrics. Returns zero values on failure.
func (c *Checker) scrapeMetrics(b *Backend) metricsResult {
	url := strings.TrimRight(b.URL, "/") + "/metrics"
	resp, err := c.client.Get(url)
	if err != nil {
		return metricsResult{}
	}
	defer resp.Body.Close()

	return parsePrometheusMetrics(resp.Body)
}

// parsePrometheusMetrics extracts vLLM metrics from Prometheus text format.
func parsePrometheusMetrics(r io.Reader) metricsResult {
	var m metricsResult
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "#") {
			continue
		}
		// Lines look like: vllm:kv_cache_usage_perc{engine="0",...} 0.42
		if strings.HasPrefix(line, "vllm:kv_cache_usage_perc") {
			if v, ok := parsePrometheusValue(line); ok {
				m.kvCache = v
			}
		} else if strings.HasPrefix(line, "vllm:num_requests_running") {
			if v, ok := parsePrometheusValue(line); ok {
				m.running = int(v)
			}
		} else if strings.HasPrefix(line, "vllm:num_requests_waiting") {
			if v, ok := parsePrometheusValue(line); ok {
				m.waiting = int(v)
			}
		}
	}
	return m
}

// parsePrometheusValue extracts the float value from a Prometheus metric line.
// Format: metric_name{labels...} value
func parsePrometheusValue(line string) (float64, bool) {
	// Value is the last space-separated field
	idx := strings.LastIndexByte(line, ' ')
	if idx < 0 {
		return 0, false
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(line[idx+1:]), 64)
	if err != nil {
		return 0, false
	}
	return v, true
}
