# llama-switch

A GPU model multiplexer for [llama-server](https://github.com/ggml-org/llama.cpp). It acts as an OpenAI-compatible reverse proxy that dynamically loads and unloads GGUF models based on demand, available VRAM, and recent usage.

No model is loaded until the first request for it arrives. When VRAM is tight, the least-recently-used model is evicted to make room. Each model runs as a separate `llama-server` process on a loopback port; clients only ever talk to the proxy.

> **Config:** llama-switch is configured via `config/config.yaml` (gitignored). Copy `config/config.example.yaml` as a starting point and configure for your own hardware and models.

## How it works

```
Client ──HTTP──▶ llama-switch (:8080)
                    │
                    ├─ GET  /v1/models       → model list with loaded/unloaded status
                    ├─ GET  /models          → llama.cpp router-style model list
                    ├─ POST /models/load     → explicitly load a model
                    ├─ POST /models/unload   → eject a model from VRAM
                    └─ POST /v1/chat/*        → proxy to the right backend
                         │
                    parse "model" from JSON body
                    spawn or reuse llama-server on 127.0.0.1:8201-8299
                    evict LRU models if VRAM or max_models limit hit
```

Key behaviours:

- **On-demand loading** — no GPU memory used until a model is requested
- **LRU eviction** — when a new model won't fit, the least-recently-used one is unloaded
- **Idle timeout** — models unload automatically after 60 minutes (configurable) with no requests
- **Per-GPU VRAM admission control** — profiling captures per-GPU VRAM usage; admission checks each GPU individually (see [VRAM & scheduling](#vram--scheduling))
- **Per-model runtime override** — models can use custom binaries (vLLM, Python servers) with their own args, env, and health endpoints (see [Per-model runtime override](#per-model-runtime-override))
- **Model handlers** — pluggable response post-processing for models that need it (e.g. Chandra 2 OCR: HTML-to-markdown conversion, Devanagari conjunct fixes) (see [Model handlers](#model-handlers))
- **Stdout prefixing** — each backend's log output is prefixed with `[model-id]`
- **OpenAI-compatible** — works with Open WebUI, opencode, agent frameworks, and anything that speaks the OpenAI Chat Completions API

## Getting started

### Requirements

- Linux with NVIDIA GPUs and `nvidia-smi`
- `llama-server` binary (CUDA build)
- Go 1.26+ (to build)

### Build

```bash
mkdir -p bin && go build -o bin/llama-switch ./src
```

### Configure

Copy the template and edit for your machine:

```bash
cp config/config.example.yaml config/config.yaml
```

Key settings:

| Field | Description |
|---|---|
| `server.host` / `server.port` | Proxy bind address |
| `server.max_models` | Max simultaneously loaded models |
| `server.idle_timeout_minutes` | Auto-unload after this many minutes idle |
| `server.queue_max_depth` | Max requests queued when eviction blocked (default 64) |
| `server.queue_timeout_seconds` | Max wait in capacity queue, seconds (default 600) |
| `server.profile_drain_seconds` | Drain timeout before profiling, seconds (default 60) |
| `backend.binary` | Path to `llama-server` |
| `backend.env.LD_LIBRARY_PATH` | CUDA library paths |
| `models[].id` | Short routing ID (used in logs and internal tracking) |
| `models[].name` | Display name (what clients see in API responses) |
| `models[].path` | Path to `.gguf` file |
| `models[].devices` | Which CUDA devices to use |

After configuring, profile each model's VRAM usage once so admission control has data to work with:

```bash
./bin/llama-switch profile
```

### Run

```bash
# Start the proxy
./bin/llama-switch serve

# List configured models and VRAM estimates (includes PER-GPU column)
./bin/llama-switch models
```

The `model` field in API requests accepts the display name or the short ID. The proxy loads the model on first request (blocking up to `health_timeout_seconds`), then forwards the request.

### Run as a systemd service

A user service unit is included. Install:

```bash
cp config/llama-switch.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-switch
```

Tail logs:

```bash
journalctl --user -u llama-switch -f
```

## Using the API

### Chat completions

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Ornith-1.0-9B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

### Explicit load / unload

```bash
# Load without sending a completion
curl -X POST http://localhost:8080/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "Ornith-1.0-9B"}'

# Free VRAM (Open WebUI "Eject" button calls this)
curl -X POST http://localhost:8080/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "Ornith-1.0-9B"}'
```

### Inspect state

```bash
# Query loaded models on a running server
curl http://localhost:8080/v1/loaded
```

## VRAM & scheduling

llama-switch profiles VRAM usage by loading each model, measuring the delta via `nvidia-smi`, and caching the result in `config/vram-cache.json`. This cache drives admission control: when a model is requested, the proxy checks whether enough VRAM is free before loading it (evicting LRU models if necessary).

**Per-GPU admission control.** Profiling captures VRAM usage per GPU (not just aggregate). The cache file has a `gpu_vram` field — an array of MB values indexed by `nvidia-smi` GPU index. When deciding whether a model fits, `ensureCapacityLocked` checks each GPU the model targets individually: every GPU must have enough free VRAM for the model's profiled share on that GPU.

**Headroom.** A 1024 MB (1 GB) safety margin is applied to the **primary GPU only** (CUDA0 / index 0) to account for OS and display-server VRAM overhead. Secondary GPUs have no display output, so no headroom is added for them.

**Validation.** Profiled VRAM measurements are validated before caching. A measurement is rejected if it is below 256 MB or below the model's `.gguf` file size — both indicate a corrupted or incomplete measurement. This applies to both the `profile` command and auto-profiling during `serve`.

**Fallback.** If per-GPU data is unavailable (e.g. legacy cache entries without `gpu_vram`), admission control falls back to the aggregate VRAM check.

### Profiling commands

```bash
./bin/llama-switch profile              # profile all unprofiled models
./bin/llama-switch profile --force      # force re-profiling (skip cache)
```

### Remote profiling

If `llama-switch serve` is already running, the `profile` command automatically detects it and delegates profiling to the running service:

```bash
./bin/llama-switch profile              # auto-detects running service
./bin/llama-switch profile --force      # force re-profiling (skip cache)
```

The service drains active requests (up to `profile_drain_seconds`, default 60s), evicts all backends, profiles each model one at a time, and streams progress via SSE. New requests during profiling get `503` with `Retry-After: 60`.

## Per-model runtime override

Models can use a different binary than `backend.binary`, with their own arguments, environment, and health endpoint. This allows running non-llama-server backends (vLLM, Python servers, etc.) alongside regular GGUF models, managed through the same VRAM management, LRU eviction, and idle sweeping infrastructure.

Optional `ModelConfig` fields:

| Field | YAML | Purpose | Default |
|-------|------|---------|---------|
| `Binary` | `binary` | Per-model binary path | `backend.binary` |
| `Args` | `args` | Raw args with `{port}` placeholder, bypasses llama-server flags | `BuildArgs()` output |
| `HealthPath` | `health_path` | Custom health check endpoint | `/health` |
| `Env` | `env` | Per-model env vars, merged on top of `backend.env` | `backend.env` only |

Example — a Python OCR server:

```yaml
models:
  - id: custom-ocr
    name: custom-ocr
    binary: "python3"
    args:
      - "-m"
      - "ocr_server"
      - "--port"
      - "{port}"
    health_path: "/health"
    env:
      CUDA_VISIBLE_DEVICES: "0"
```

Existing llama-server models are unchanged — no new fields are required.

## Model handlers

Some models produce output that needs post-processing before it's useful to clients (e.g. a VLM that returns structured HTML instead of plain text). Model handlers intercept the backend response and transform it transparently.

A handler implements the `ModelHandler` interface:

```go
type ModelHandler interface {
    MatchesModel(modelID string) bool
    ProcessResponse(resp *http.Response, isStream bool) ([]byte, error)
}
```

When a request targets a model with a registered handler, the proxy forces non-streaming, reads the full response, runs `ProcessResponse()`, and returns the transformed JSON. Models without handlers use the normal streaming proxy path.

### Chandra 2 OCR handler

The included Chandra handler (`src/chandra_handler.go`) post-processes responses from the [Chandra 2 OCR](https://huggingface.co/datalab-to/chandra-ocr-2) model. Chandra is a Qwen3.5-based VLM that outputs structured HTML with bounding boxes. The handler:

1. Extracts HTML from `content` or `reasoning_content` (handles Qwen3.5 thinking mode)
2. Strips chain-of-thought leakage
3. Converts HTML to clean markdown (tables, lists, bold, headers, images)
4. Fixes known Devanagari conjunct-drop errors (कय→क्रय, केता→क्रेता, etc.)
5. Strips page headers/footers

Clients send a standard chat completions request with an `image_url` and receive clean markdown in the `content` field — no knowledge of the model's HTML format is required.

**To remove the handler:** delete `src/chandra_handler.go` and `src/chandra_handler_test.go`, remove the registration line in `NewModelHandler()`. The generic handler infrastructure in `src/proxy.go` stays for future use.

## Project structure

| Path | Purpose |
|---|---|
| `src/main.go` | CLI entry point (`serve`, `profile`, `models`, `status`) |
| `src/config.go` | Config types, YAML loading, path expansion, argument building, per-model binary/env resolution |
| `src/backend.go` | Backend process lifecycle, port allocation, health checks, LRU eviction, idle sweeper |
| `src/proxy.go` | HTTP proxy server, model routing, streaming support, load/unload endpoints, model handler hook |
| `src/chandra_handler.go` | Model handler for Chandra 2 OCR (HTML→markdown, Devanagari fixes, thinking stripping) |
| `src/vram.go` | `nvidia-smi` querying, VRAM cache, per-GPU admission control |
| `src/logger.go` | Thin stdout logger |
| `src/*_test.go` | Unit and integration tests |
| `config/config.example.yaml` | Configuration template |
| `config/llama-switch.service` | systemd user unit |
| `bin/` | Compiled binary (built by `go build -o bin/llama-switch ./src`) |

## License

[MIT](LICENSE.md)
