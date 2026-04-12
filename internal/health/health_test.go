package health

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestChecker_HealthyBackend(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewChecker([]Backend{{Name: "ok", URL: srv.URL}}, time.Hour, 5*time.Second)
	c.CheckNow()

	if !c.IsAlive("ok") {
		t.Error("expected backend to be alive")
	}
}

func TestChecker_UnhealthyBackend(t *testing.T) {
	// Use a closed server to simulate unreachable backend
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	srv.Close()

	c := NewChecker([]Backend{{Name: "down", URL: url}}, time.Hour, 1*time.Second)
	c.CheckNow()

	if c.IsAlive("down") {
		t.Error("expected backend to be dead")
	}
}

func TestChecker_500IsUnhealthy(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	c := NewChecker([]Backend{{Name: "err", URL: srv.URL}}, time.Hour, 5*time.Second)
	c.CheckNow()

	if c.IsAlive("err") {
		t.Error("expected 500 backend to be unhealthy")
	}
}

func TestChecker_Recovery(t *testing.T) {
	alive := true
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if alive {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer srv.Close()

	c := NewChecker([]Backend{{Name: "b", URL: srv.URL}}, time.Hour, 5*time.Second)

	// Initially alive
	c.CheckNow()
	if !c.IsAlive("b") {
		t.Fatal("expected alive initially")
	}

	// Goes down
	alive = false
	c.CheckNow()
	if c.IsAlive("b") {
		t.Fatal("expected dead after failure")
	}

	// Comes back
	alive = true
	c.CheckNow()
	if !c.IsAlive("b") {
		t.Fatal("expected alive after recovery")
	}
}

func TestChecker_UnknownBackendIsAlive(t *testing.T) {
	c := NewChecker(nil, time.Hour, 5*time.Second)
	if !c.IsAlive("nonexistent") {
		t.Error("expected unknown backend to be treated as alive (fail-open)")
	}
}

func TestChecker_Statuses(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewChecker([]Backend{{Name: "a", URL: srv.URL}}, time.Hour, 5*time.Second)
	c.CheckNow()

	statuses := c.Statuses()
	s, ok := statuses["a"]
	if !ok {
		t.Fatal("expected status for backend 'a'")
	}
	if !s.Alive {
		t.Error("expected alive")
	}
	if s.LastCheck.IsZero() {
		t.Error("expected LastCheck to be set")
	}
}

func TestChecker_StopGraceful(t *testing.T) {
	c := NewChecker(nil, 10*time.Millisecond, 5*time.Second)
	c.Start()
	c.Stop()
	// Should not panic or hang
}

func TestChecker_ScrapesMetrics(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			w.Write([]byte(`# HELP vllm:kv_cache_usage_perc KV-cache usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="gemma"} 0.42
# HELP vllm:num_requests_running Running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="gemma"} 3.0
# HELP vllm:num_requests_waiting Waiting requests
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="gemma"} 1.0
`))
			return
		}
		// /v1/models health check
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewChecker([]Backend{{Name: "vllm", URL: srv.URL}}, time.Hour, 5*time.Second)
	c.CheckNow()

	st := c.GetStatus("vllm")
	if st == nil {
		t.Fatal("expected status")
	}
	if !st.Alive {
		t.Fatal("expected alive")
	}
	if st.KVCacheUsage != 0.42 {
		t.Errorf("expected KVCacheUsage 0.42, got %f", st.KVCacheUsage)
	}
	if st.RequestsRunning != 3 {
		t.Errorf("expected RequestsRunning 3, got %d", st.RequestsRunning)
	}
	if st.RequestsWaiting != 1 {
		t.Errorf("expected RequestsWaiting 1, got %d", st.RequestsWaiting)
	}
}

func TestChecker_MetricsUnavailable_StillHealthy(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewChecker([]Backend{{Name: "b", URL: srv.URL}}, time.Hour, 5*time.Second)
	c.CheckNow()

	if !c.IsAlive("b") {
		t.Error("backend should be alive even if /metrics fails")
	}
	st := c.GetStatus("b")
	if st.KVCacheUsage != 0 {
		t.Errorf("expected zero KVCacheUsage when metrics unavailable, got %f", st.KVCacheUsage)
	}
}

func TestParsePrometheusMetrics(t *testing.T) {
	input := `# HELP vllm:kv_cache_usage_perc KV-cache usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0"} 0.75
# HELP vllm:num_requests_running Running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0"} 5.0
# HELP vllm:num_requests_waiting Waiting
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0"} 2.0
`
	m := parsePrometheusMetrics(strings.NewReader(input))
	if m.kvCache != 0.75 {
		t.Errorf("expected kvCache 0.75, got %f", m.kvCache)
	}
	if m.running != 5 {
		t.Errorf("expected running 5, got %d", m.running)
	}
	if m.waiting != 2 {
		t.Errorf("expected waiting 2, got %d", m.waiting)
	}
}

func TestStatusLoad(t *testing.T) {
	tests := []struct {
		name string
		s    Status
		want float64
	}{
		{"dead", Status{Alive: false}, -1},
		{"idle", Status{Alive: true}, 0},
		{"loaded", Status{Alive: true, KVCacheUsage: 0.5, RequestsRunning: 10}, 0.6},
		{"full cache", Status{Alive: true, KVCacheUsage: 0.95, RequestsRunning: 2, RequestsWaiting: 3}, 1.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.s.Load()
			if got != tt.want {
				t.Errorf("Load() = %f, want %f", got, tt.want)
			}
		})
	}
}
