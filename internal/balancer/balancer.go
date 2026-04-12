package balancer

import (
	"sort"
	"sync"

	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/config"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/health"
)

// KVCacheThreshold is the KV-cache usage level above which a backend is considered
// overloaded and non-preferred backends should be tried instead.
const KVCacheThreshold = 0.9

// Balancer selects backends using load-aware routing with preferred-first semantics.
type Balancer struct {
	mu      sync.Mutex
	counter map[string]uint64
	checker *health.Checker
}

// NewBalancer creates a balancer. If checker is nil, all backends are treated as healthy with zero load.
func NewBalancer(checker *health.Checker) *Balancer {
	return &Balancer{
		counter: make(map[string]uint64),
		checker: checker,
	}
}

// Select returns backends ordered by load-aware priority:
//  1. Filter to healthy backends.
//  2. Score each backend: preferred + low KV-cache + few running requests wins.
//  3. Sort by score (lowest first). Ties broken by round-robin among equal scores.
//  4. If all are unhealthy, returns the original list (fail-open).
func (b *Balancer) Select(backends []config.Backend, model string) []config.Backend {
	if len(backends) <= 1 {
		return backends
	}

	type scored struct {
		backend config.Backend
		score   float64
		order   int // for stable tie-breaking
	}

	var healthy []scored
	for i, be := range backends {
		if b.checker != nil && !b.checker.IsAlive(be.Name) {
			continue
		}
		s := b.score(be)
		healthy = append(healthy, scored{backend: be, score: s, order: i})
	}

	if len(healthy) == 0 {
		return backends // fail-open
	}

	// Round-robin counter for tie-breaking
	b.mu.Lock()
	rr := b.counter[model]
	b.counter[model]++
	b.mu.Unlock()

	sort.SliceStable(healthy, func(i, j int) bool {
		si, sj := healthy[i].score, healthy[j].score
		// If scores differ meaningfully, prefer lower
		if diff := si - sj; diff < -0.05 || diff > 0.05 {
			return si < sj
		}
		// Tie: round-robin among equal-scored backends
		ri := (uint64(healthy[i].order) + rr) % uint64(len(healthy))
		rj := (uint64(healthy[j].order) + rr) % uint64(len(healthy))
		return ri < rj
	})

	result := make([]config.Backend, len(healthy))
	for i, h := range healthy {
		result[i] = h.backend
	}
	return result
}

// score computes a routing score for a backend (lower = better).
//
// Scoring:
//   - Preferred backends get a -1.0 bonus, UNLESS their KV-cache is above threshold.
//   - KV-cache usage (0.0–1.0) is the primary load signal.
//   - Running+waiting requests add a small penalty (0.01 per request).
func (b *Balancer) score(be config.Backend) float64 {
	s := float64(0)

	if b.checker != nil {
		st := b.checker.GetStatus(be.Name)
		if st != nil {
			s += st.KVCacheUsage
			s += 0.01 * float64(st.RequestsRunning+st.RequestsWaiting)

			// Preferred bonus only applies when KV-cache is not saturated
			if be.Preferred && st.KVCacheUsage < KVCacheThreshold {
				s -= 1.0
			}
			return s
		}
	}

	// No metrics available — use preferred flag as only signal
	if be.Preferred {
		s -= 1.0
	}
	return s
}
