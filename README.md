# femtollm

Minimal LLM proxy router with protocol conversion and fallback support. Routes requests to vLLM and other OpenAI-compatible backends.

## Features

- **Discovery-based routing** — femtollm scrapes each backend's `/v1/models` on every health check, then routes incoming requests to backends that actually advertise the requested model. No model names in config required.
- **Pattern routing (legacy / optional)** — regex `pattern` in config still works as an explicit override, useful for aliasing names that aren't in a backend's advertised list.
- **Protocol conversion** — Anthropic Messages API to OpenAI Chat Completions
- **Fallback** — multiple backends per model, tries in order on 5xx errors
- **Load-aware routing** — scrapes vLLM `/metrics` for KV-cache usage and queue depth
- **Prefix-cache routing** — HashTrie tracks prompt prefixes per backend for KV-cache affinity
- **Preferred backends** — pin traffic to a primary backend, fall back when overloaded
- **Chat template injection** — optionally inject `chat_template_kwargs` (e.g., thinking mode)
- **Streaming** — SSE passthrough for OpenAI streaming responses
- **Zero dependencies** — pure Go, single static binary

## Usage

```bash
# Build
go build ./cmd/femtollm

# Run
cp config.example.json config.json
# Edit config.json with your backends
./femtollm -config config.json
```

## Configuration

The minimal config now just lists the upstream URLs. femtollm probes each
one's `/v1/models` and routes incoming requests by the model IDs each backend
reports. The `model` and `pattern` fields are kept for backward compatibility
but are optional.

```json
{
  "listen": ":8880",
  "backends": [
    { "name": "gemma4-spark", "url": "http://spark-07:8000", "preferred": true },
    { "name": "gemma4-thor",  "url": "http://thor-04:8000" }
  ]
}
```

When the upstream serves a model under multiple aliases via vLLM
`--served-model-name`, **all** the aliases are picked up and become routable —
no extra config needed.

### Backend options

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique backend identifier |
| `pattern` | string | Regex to match requested model names |
| `url` | string | Backend base URL |
| `model` | string | Override model name sent to backend (optional) |
| `api_key` | string | Bearer token for the backend (optional) |
| `max_context` | int | Override `max_model_len` in `/v1/models` response (optional, 0 = use backend default) |
| `preferred` | bool | Always try first when healthy; skips round-robin (optional) |
| `chat_template_kwargs` | object | Injected into requests as `chat_template_kwargs` (optional, see below) |

### Chat template kwargs (thinking mode)

When `chat_template_kwargs` is set on a backend, femtollm injects it into every
forwarded request body — unless the client already provides its own. This enables
proxy-level control of vLLM chat template features like Gemma 4 thinking mode.

```json
"chat_template_kwargs": {"enable_thinking": true}
```

Three modes:

| Config | Behavior |
|--------|----------|
| Omitted (default) | Transparent — caller decides |
| `{"enable_thinking": true}` | Force thinking on for all requests |
| `{"enable_thinking": false}` | Force thinking off for all requests |

Client-provided `chat_template_kwargs` always takes precedence (never overwritten).

Requires vLLM to be launched with `--reasoning-parser gemma4` to properly separate
thinking tokens into `message.reasoning` instead of leaking them into `message.content`.

## Endpoints

### Default (load-balanced)

| Endpoint | Protocol | Description |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | Chat completions; model-pattern match → load-aware backend pick |
| `POST /v1/messages` | Anthropic | Messages API (converted to OpenAI internally) |
| `GET /v1/models` | OpenAI | List of all advertised models (with optional `max_model_len`) |
| `GET /health` | — | Liveness probe |
| `GET /health/backends` | — | Per-backend health, KV-cache metrics, prefix-trie stats |

### Direct per-backend routes

Every backend is also exposed under its own URL prefix, **and** under its URL
hostname when that hostname is unique across the config. These bypass the
model-pattern match and the load balancer — they always hit that one backend:

| Endpoint | Description |
|---|---|
| `POST /<backend>/v1/chat/completions` | Pinned to backend `<backend>` |
| `POST /<backend>/v1/messages` | Pinned, Anthropic protocol |
| `GET /<backend>/v1/models` | Reports only that backend's model |
| `POST /<host>/v1/chat/completions` | Same, addressed by URL hostname (e.g. `/spark-05/...`) |

Use the direct routes to:

- Pin a long task to one box for prefix-cache locality without putting `@backend`
  in the model name.
- Health-test a specific node end-to-end (`curl /spark-05/v1/models`).
- Drain a node by routing all traffic away from its prefix while you debug it.

The default `/v1/...` endpoints continue to round-robin / load-balance across
the matched set.

## Deploy with stevedore

```bash
stevedore repo add femtollm git@github.com:jonnyzzz/jonnyzzz-femtollm.git --branch main
stevedore deploy sync femtollm && stevedore deploy up femtollm
```

Place `config.json` in the stevedore data volume (`${STEVEDORE_DATA}/config.json`).
