package proxy

import (
	"context"
	"net/http"
	"net/url"
)

// Direct per-backend endpoints.
//
// In addition to the default protocol routes (which match by model name and
// load-balance), each backend gets pinned routes:
//
//   /<backend-name>/v1/chat/completions
//   /<backend-name>/v1/messages
//   /<backend-name>/v1/models
//
// And, if the backend URL's hostname is unique across the config, the same
// routes under /<host>/... — so callers can address a backend by either its
// configured name or its host (e.g. /spark-05/v1/chat/completions).

type pinnedBackendCtxKey struct{}

// PinnedBackendName returns the backend name pinned via a direct route, or "".
func PinnedBackendName(r *http.Request) string {
	if v, ok := r.Context().Value(pinnedBackendCtxKey{}).(string); ok {
		return v
	}
	return ""
}

// withPinnedBackend wraps next so the request carries a pinned backend name.
func withPinnedBackend(name string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		r = r.WithContext(context.WithValue(r.Context(), pinnedBackendCtxKey{}, name))
		next(w, r)
	}
}

// hostFromURL returns the hostname (without port) parsed from a backend URL,
// or "" if the URL doesn't parse.
func hostFromURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return u.Hostname()
}
