package balancer

import (
	"sync"

	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/config"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/health"
)

// Balancer selects backends using preferred-first routing with round-robin fallback.
type Balancer struct {
	mu      sync.Mutex
	counter map[string]uint64
	checker *health.Checker
}

// NewBalancer creates a balancer. If checker is nil, all backends are treated as healthy.
func NewBalancer(checker *health.Checker) *Balancer {
	return &Balancer{
		counter: make(map[string]uint64),
		checker: checker,
	}
}

// Select returns backends ordered for a request:
//  1. Healthy preferred backends come first (in config order, no rotation).
//  2. Remaining healthy backends follow, rotated via round-robin.
//  3. If all are unhealthy, returns the original list (fail-open).
func (b *Balancer) Select(backends []config.Backend, model string) []config.Backend {
	if len(backends) <= 1 {
		return backends
	}

	// Split into preferred and regular, filtering to healthy
	var preferred, regular []config.Backend
	for _, be := range backends {
		if b.checker != nil && !b.checker.IsAlive(be.Name) {
			continue
		}
		if be.Preferred {
			preferred = append(preferred, be)
		} else {
			regular = append(regular, be)
		}
	}

	// Fail-open: if all are dead, return original list
	if len(preferred) == 0 && len(regular) == 0 {
		return backends
	}

	// Round-robin only among regular (non-preferred) backends
	if len(regular) > 1 {
		b.mu.Lock()
		idx := b.counter[model] % uint64(len(regular))
		b.counter[model]++
		b.mu.Unlock()

		rotated := make([]config.Backend, len(regular))
		for i := range regular {
			rotated[i] = regular[(int(idx)+i)%len(regular)]
		}
		regular = rotated
	}

	// Preferred first, then round-robin regular as fallback
	return append(preferred, regular...)
}
