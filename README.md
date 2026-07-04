# llama-switch

A GPU model multiplexer for [llama-server](https://github.com/ggml-org/llama.cpp). It acts as an OpenAI-compatible reverse proxy that dynamically loads and unloads GGUF models based on demand, available VRAM, and recent usage.

No model is loaded until the first request for it arrives. When VRAM is tight, the least-recently-used model is evicted to make room. Each model runs as a separate `llama-server` process on a loopback port; clients only ever talk to the proxy.

**This project is configured via `config.yaml` (gitignored). Copy `config.example.yaml` as a starting point and configure for your own hardware and models.**

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
- **Idle timeout** — models unload automatically after 60 minutes with no requests
- **Auto-profiling** — VRAM usage is measured on first load and cached for future admission decisions
- **Stdout prefixing** — each backend's log output is prefixed with `[model-id]`
- **OpenAI-compatible** — works with Open WebUI, opencode, agent frameworks, and anything that speaks the OpenAI Chat Completions API

## Requirements

- Linux with NVIDIA GPUs and `nvidia-smi`
- `llama-server` binary (CUDA build)
- Go 1.26+ (to build)

## Build

```bash
go build -o llama-switch .
```

## Configure

Copy the template and edit for your machine:

```bash
cp config.example.yaml config.yaml
```

Key settings:

| Field | Description |
|---|---|
| `server.host` / `server.port` | Proxy bind address |
| `server.max_models` | Max simultaneously loaded models |
| `server.idle_timeout_minutes` | Auto-unload after this many minutes idle |
| `backend.binary` | Path to `llama-server` |
| `backend.env.LD_LIBRARY_PATH` | CUDA library paths |
| `models[].id` | Short routing ID (used in logs and internal tracking) |
| `models[].model` | Display name (what clients see in API responses) |
| `models[].path` | Path to `.gguf` file |
| `models[].devices` | Which CUDA devices to use |

## Usage

```bash
# Start the proxy
./llama-switch serve

# List configured models and VRAM estimates
./llama-switch models

# Profile VRAM for all unprofiled models (loads each one at a time)
./llama-switch profile

# Query loaded models on a running server
curl http://localhost:8080/v1/loaded
```

### Sending requests

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Ornith-1.0-9B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

The `model` field accepts the display name, the short ID, or the alias. The proxy loads the model on first request (blocking up to `health_timeout_seconds`), then forwards the request.

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

## systemd

A user service unit is included. Install:

```bash
cp llama-switch.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-switch
```

Logs:

```bash
journalctl --user -u llama-switch -f
```

## Project structure

| File | Purpose |
|---|---|
| `main.go` | CLI entry point (`serve`, `profile`, `models`, `status`) |
| `config.go` | Config types, YAML loading, path expansion, argument building |
| `backend.go` | Backend process lifecycle, port allocation, health checks, LRU eviction, idle sweeper |
| `proxy.go` | HTTP proxy server, model routing, streaming support, load/unload endpoints |
| `vram.go` | `nvidia-smi` querying, VRAM cache |
| `logger.go` | Thin stdout logger |
| `config.example.yaml` | Configuration template |

## License

[MIT](LICENSE.md)
