package balancer

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/config"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/health"
)

func backends(names ...string) []config.Backend {
	var out []config.Backend
	for _, n := range names {
		out = append(out, config.Backend{Name: n, URL: "http://" + n})
	}
	return out
}

// vllmServer creates a test server that responds to /v1/models and /metrics.
func vllmServer(t *testing.T, kvCache float64, running, waiting int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			fmt.Fprintf(w, "vllm:kv_cache_usage_perc{engine=\"0\"} %f\n", kvCache)
			fmt.Fprintf(w, "vllm:num_requests_running{engine=\"0\"} %d.0\n", running)
			fmt.Fprintf(w, "vllm:num_requests_waiting{engine=\"0\"} %d.0\n", waiting)
			return
		}
		w.WriteHeader(http.StatusOK) // /v1/models
	}))
}

func TestSelect_RoundRobin(t *testing.T) {
	b := NewBalancer(nil) // nil checker = all healthy, no metrics
	bs := backends("a", "b", "c")

	counts := map[string]int{}
	for i := 0; i < 9; i++ {
		selected := b.Select(bs, "model")
		counts[selected[0].Name]++
	}

	for _, name := range []string{"a", "b", "c"} {
		if counts[name] != 3 {
			t.Errorf("expected backend %s to start first 3 times, got %d", name, counts[name])
		}
	}
}

func TestSelect_FiltersUnhealthy(t *testing.T) {
	alive := vllmServer(t, 0, 0, 0)
	defer alive.Close()
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := dead.URL
	dead.Close()

	checker := health.NewChecker([]health.Backend{
		{Name: "alive", URL: alive.URL},
		{Name: "dead", URL: deadURL},
	}, time.Hour, 1*time.Second)
	checker.CheckNow()

	b := NewBalancer(checker)
	bs := []config.Backend{
		{Name: "alive", URL: alive.URL},
		{Name: "dead", URL: deadURL},
	}

	selected := b.Select(bs, "model")
	if len(selected) != 1 {
		t.Fatalf("expected 1 healthy backend, got %d", len(selected))
	}
	if selected[0].Name != "alive" {
		t.Errorf("expected alive backend, got %s", selected[0].Name)
	}
}

func TestSelect_AllDead_FailOpen(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := srv.URL
	srv.Close()

	checker := health.NewChecker([]health.Backend{
		{Name: "a", URL: deadURL},
		{Name: "b", URL: deadURL},
	}, time.Hour, 1*time.Second)
	checker.CheckNow()

	b := NewBalancer(checker)
	bs := backends("a", "b")

	selected := b.Select(bs, "model")
	if len(selected) != 2 {
		t.Fatalf("expected all 2 backends on fail-open, got %d", len(selected))
	}
}

func TestSelect_SingleBackend(t *testing.T) {
	b := NewBalancer(nil)
	bs := backends("only")

	selected := b.Select(bs, "model")
	if len(selected) != 1 || selected[0].Name != "only" {
		t.Errorf("single backend should pass through unchanged")
	}
}

func TestSelect_Empty(t *testing.T) {
	b := NewBalancer(nil)
	selected := b.Select(nil, "model")
	if len(selected) != 0 {
		t.Error("expected empty result for empty input")
	}
}

func TestSelect_PreferredAlwaysFirst(t *testing.T) {
	b := NewBalancer(nil)
	bs := []config.Backend{
		{Name: "regular-a", URL: "http://a"},
		{Name: "preferred", URL: "http://p", Preferred: true},
		{Name: "regular-b", URL: "http://b"},
	}

	for i := 0; i < 10; i++ {
		selected := b.Select(bs, "model")
		if selected[0].Name != "preferred" {
			t.Fatalf("request %d: expected preferred first, got %s", i, selected[0].Name)
		}
	}
}

func TestSelect_PreferredDown_FallsBackToRoundRobin(t *testing.T) {
	aServer := vllmServer(t, 0, 0, 0)
	defer aServer.Close()
	bServer := vllmServer(t, 0, 0, 0)
	defer bServer.Close()
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := dead.URL
	dead.Close()

	checker := health.NewChecker([]health.Backend{
		{Name: "preferred", URL: deadURL},
		{Name: "fallback-a", URL: aServer.URL},
		{Name: "fallback-b", URL: bServer.URL},
	}, time.Hour, 1*time.Second)
	checker.CheckNow()

	b := NewBalancer(checker)
	bs := []config.Backend{
		{Name: "preferred", URL: deadURL, Preferred: true},
		{Name: "fallback-a", URL: aServer.URL},
		{Name: "fallback-b", URL: bServer.URL},
	}

	counts := map[string]int{}
	for i := 0; i < 10; i++ {
		selected := b.Select(bs, "model")
		if len(selected) != 2 {
			t.Fatalf("expected 2 healthy backends, got %d", len(selected))
		}
		counts[selected[0].Name]++
	}

	if counts["preferred"] > 0 {
		t.Errorf("dead preferred should not appear, got %d times first", counts["preferred"])
	}
	if counts["fallback-a"] != 5 || counts["fallback-b"] != 5 {
		t.Errorf("expected 5/5 round-robin among fallbacks, got a=%d b=%d", counts["fallback-a"], counts["fallback-b"])
	}
}

// TestSelect_KVCacheAware verifies that a preferred backend with high KV-cache
// usage loses its priority, and the less-loaded backend is chosen instead.
func TestSelect_KVCacheAware(t *testing.T) {
	// Preferred backend: 95% KV-cache (above threshold)
	spark := vllmServer(t, 0.95, 5, 2)
	defer spark.Close()
	// Fallback backend: 20% KV-cache (plenty of room)
	thor := vllmServer(t, 0.20, 1, 0)
	defer thor.Close()

	checker := health.NewChecker([]health.Backend{
		{Name: "spark", URL: spark.URL},
		{Name: "thor", URL: thor.URL},
	}, time.Hour, 5*time.Second)
	checker.CheckNow()

	b := NewBalancer(checker)
	bs := []config.Backend{
		{Name: "spark", URL: spark.URL, Preferred: true},
		{Name: "thor", URL: thor.URL},
	}

	// All requests should go to thor because spark's KV-cache is above threshold
	for i := 0; i < 10; i++ {
		selected := b.Select(bs, "model")
		if selected[0].Name != "thor" {
			t.Fatalf("request %d: expected thor (less loaded), got %s", i, selected[0].Name)
		}
	}
}

// TestSelect_PreferredWinsWhenIdle verifies that a preferred backend with low
// KV-cache usage is chosen over a non-preferred backend.
func TestSelect_PreferredWinsWhenIdle(t *testing.T) {
	// Preferred backend: idle
	spark := vllmServer(t, 0.10, 0, 0)
	defer spark.Close()
	// Fallback: also idle
	thor := vllmServer(t, 0.05, 0, 0)
	defer thor.Close()

	checker := health.NewChecker([]health.Backend{
		{Name: "spark", URL: spark.URL},
		{Name: "thor", URL: thor.URL},
	}, time.Hour, 5*time.Second)
	checker.CheckNow()

	b := NewBalancer(checker)
	bs := []config.Backend{
		{Name: "spark", URL: spark.URL, Preferred: true},
		{Name: "thor", URL: thor.URL},
	}

	// Preferred should always win when idle (preferred bonus outweighs small load diff)
	for i := 0; i < 10; i++ {
		selected := b.Select(bs, "model")
		if selected[0].Name != "spark" {
			t.Fatalf("request %d: expected preferred spark, got %s", i, selected[0].Name)
		}
	}
}

// TestSelect_RoutesToLeastLoadedWithoutPreference verifies that when no backend
// is preferred, the least loaded one is chosen.
func TestSelect_RoutesToLeastLoadedWithoutPreference(t *testing.T) {
	heavy := vllmServer(t, 0.80, 10, 5)
	defer heavy.Close()
	light := vllmServer(t, 0.10, 1, 0)
	defer light.Close()

	checker := health.NewChecker([]health.Backend{
		{Name: "heavy", URL: heavy.URL},
		{Name: "light", URL: light.URL},
	}, time.Hour, 5*time.Second)
	checker.CheckNow()

	b := NewBalancer(checker)
	bs := []config.Backend{
		{Name: "heavy", URL: heavy.URL},
		{Name: "light", URL: light.URL},
	}

	for i := 0; i < 10; i++ {
		selected := b.Select(bs, "model")
		if selected[0].Name != "light" {
			t.Fatalf("request %d: expected light (less loaded), got %s", i, selected[0].Name)
		}
	}
}

// TestSelect_BothEquallyLoaded_RoundRobins verifies that backends with similar
// load get round-robined.
func TestSelect_BothEquallyLoaded_RoundRobins(t *testing.T) {
	a := vllmServer(t, 0.50, 3, 0)
	defer a.Close()
	b2 := vllmServer(t, 0.50, 3, 0)
	defer b2.Close()

	checker := health.NewChecker([]health.Backend{
		{Name: "a", URL: a.URL},
		{Name: "b", URL: b2.URL},
	}, time.Hour, 5*time.Second)
	checker.CheckNow()

	bal := NewBalancer(checker)
	bs := []config.Backend{
		{Name: "a", URL: a.URL},
		{Name: "b", URL: b2.URL},
	}

	counts := map[string]int{}
	for i := 0; i < 10; i++ {
		selected := bal.Select(bs, "model")
		counts[selected[0].Name]++
	}

	if counts["a"] != 5 || counts["b"] != 5 {
		t.Errorf("expected 5/5 round-robin for equal load, got a=%d b=%d", counts["a"], counts["b"])
	}
}
