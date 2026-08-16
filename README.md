# llm-gateway

**English** | [简体中文](README.zh-CN.md)

A backend-agnostic **LLM lifecycle gateway**: on a single device with a single active model, it manages **any OpenAI-compatible inference backend** — SGLang / vLLM / llama.cpp / OpenVINO / ...

> Start and switch models on demand driven by the OpenAI API `model` field; route directly when the backend is running with the matching model; start only when VRAM/RAM is sufficient; refuse switches while the model is busy; fail fast when resources are insufficient; auto-unload after a configurable idle period (default 1 hour) to release resources.

Clients always talk to one fixed endpoint:

```text
http://server:8000/v1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://server:8000/v1", api_key="...")
client.chat.completions.create(model="qwen2.5-7b", messages=[...])  # on-demand start
client.chat.completions.create(model="llama3.1-8b", messages=[...])  # automatic backend switch
```

## Features

- **Backend-agnostic**: a single `command_template` adapts any OpenAI-compatible server (SGLang, vLLM, llama.cpp, OpenVINO, ...). No backend-specific code, no hardcoded flags.
- **On-demand lifecycle**: the gateway spawns, health-checks, stops and switches the backend process. The backend is never left running when idle.
- **Resource-aware**: dual gates — GPU VRAM (NVML, skipped automatically when no NVIDIA driver) + system RAM (psutil, cross-platform). Never hard-start into an OOM.
- **No preemption**: switching is refused while requests are in flight — streaming connections are never cut (unless you force-stop).
- **Formal state machine** (`STOPPED / STARTING / RUNNING / STOPPING / SWITCHING`) with a global lifecycle lock; racing requests can never start two backends.
- **Idle watchdog**: auto-unload after a configurable idle timeout (default 1 hour, never hardcoded — YAML or `LLM_GATEWAY_IDLE_TIMEOUT_SECONDS`).
- **Token usage tracking**: per-request tokens (plain + streaming) recorded to SQLite, with a built-in usage dashboard.
- **API-key auth**: optional `Authorization: Bearer` gate for all data endpoints.
- **Windows-friendly**: pure asyncio, no Unix-only dependencies; RAM gating works on CPU/NPU machines (e.g. Intel Core Ultra + OpenVINO).

## Architecture

```text
Client
  │
  │ OpenAI-compatible API (fixed endpoint :8000)
  ▼
llm-gateway :8000
  │
  ├── request routing (by `model` field)
  ├── state machine STOPPED/STARTING/RUNNING/STOPPING/SWITCHING
  ├── resource gates (VRAM via NVML optional + RAM via psutil)
  ├── backend lifecycle (start/stop/switch, serialized by one global lock)
  ├── health polling (never a blind sleep)
  ├── active_requests / last_activity accounting
  └── idle watchdog (configurable timeout)
          │
          ▼
     backend 127.0.0.1:30000 (spawned on demand, managed by the gateway)
     SGLang / vLLM / llama.cpp / OpenVINO ... (one command_template line)
          │
          ▼
     GPU (NVML) / CPU / NPU (RAM)
```

## Quickstart

```bash
pip install -e ".[dev]"
cp config.example.yaml config.yaml      # then edit
python -m llm_gateway --config config.yaml
# or: llm-gateway --config config.yaml
```

```bash
# Triggers a cold start of the backend (first request waits for model load;
# set your client timeout >= backend.startup_timeout_seconds)
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"Hello"}]}'

curl http://127.0.0.1:8000/gateway/status   # state, model, VRAM + RAM
```

## Backend integration

The launch command is a **template** (`backend.command_template`) with placeholders:
`{python} {model_path} {model_name} {host} {port} {device} {extra_args}` (all quoted automatically; `{extra_args}` is appended if the template omits it).

```yaml
# SGLang (the default template — no config needed for this)
command_template: "{python} -m sglang.launch_server --model-path {model_path} --host {host} --port {port} --tp-size 1 {extra_args}"

# vLLM
command_template: "{python} -m vllm.entrypoints.openai.api_server --model {model_path} --host {host} --port {port} {extra_args}"

# llama.cpp (Linux / macOS / Windows — an .exe path works directly)
command_template: "llama-server -m {model_path} --host {host} --port {port} {extra_args}"

# OpenVINO GenAI (custom OpenAI-compatible server, CPU/NPU)
command_template: "{python} -m my_openvino_server --model {model_path} --port {port} {extra_args}"
```

A backend only needs to (1) expose an OpenAI-compatible API and (2) return 200 on `health_path` when ready.

> **vLLM caveat**: vLLM strictly validates the `model` field against `--served-model-name` (defaults to the model path). Set `--served-model-name <registry name>` in `extra_args` so it matches the name clients send.

### Mixing backends in one registry

The global template applies to every model **unless a model defines its own**
(`models.<name>.backend.command_template`, plus `backend.health_path` for
backends whose readiness path differs). Only one model runs at a time, so the
shared internal port never collides:

```yaml
models:
  vllm-qwen2.5-7b:
    path: /models/Qwen2.5-7B-Instruct
    required_vram_gib: 18
    backend:
      command_template: "{python} -m vllm.entrypoints.openai.api_server --model {model_path} --host {host} --port {port} {extra_args}"
      extra_args: ["--served-model-name", "vllm-qwen2.5-7b"]
  llama3.2-3b-gguf:
    path: C:\models\llama-3.2-3b-instruct-q8.gguf
    required_ram_gib: 8
    backend:
      command_template: "llama-server -m {model_path} --host {host} --port {port} {extra_args}"
      extra_args: ["--ctx-size", 8192]
```

Numbers in `extra_args` are coerced to strings automatically (no quoting needed).

## Resource gates (VRAM + RAM)

Per model: `required_vram_gib` (GPU) and `required_ram_gib` (system RAM); threshold = requirement + `memory.safety_margin_gib`.

- **VRAM** (NVML): checked **only when an NVIDIA driver is present**. On CPU/NPU machines the VRAM gate is skipped automatically and only RAM is enforced — exactly what you want for Windows + OpenVINO / llama.cpp-CPU.
- **RAM** (psutil `available`): Linux / Windows / macOS.
- `0` (default) disables that gate.
- After a switch, the decision uses the **re-read values after the old backend has exited and memory has actually settled**.

```text
model needs VRAM 40G + 4G margin = 44G; free 44.1G ✓ start
model needs RAM  32G + 4G margin = 36G; available 34G ✗ 503 insufficient_memory
```

## Ports & routes

| Port | Bind | Purpose | Config |
|---|---|---|---|
| **8000** | `0.0.0.0` (default) | the only client-facing entry (OpenAI API + management) | `server.port` / `LLM_GATEWAY_PORT` |
| **30000** | `127.0.0.1` (default) | internal backend port, reachable only from the gateway | `backend.port` |

### OpenAI-compatible surface (client-facing, API-key protected)

| Method | Path | Behavior |
|---|---|---|
| POST | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, any `/v1/*` | gated on the `model` field (start / switch / direct route) then forwarded verbatim, SSE included |
| GET | `/v1/models` | registry listing (available even while the backend is stopped) |
| GET | other `/v1/*` | forwarded only when a model is loaded |

Streaming requests get `stream_options.include_usage=true` injected automatically so real token counts are captured; transparent to clients.

### Management surface (`/gateway/*`, API-key protected)

| Method | Path | Behavior |
|---|---|---|
| GET | `/gateway/status` | state machine + active requests + idle seconds + memory (vram/ram) |
| GET | `/gateway/models` | registry + per-model loaded state |
| POST | `/gateway/stop` | graceful unload — accepted in any state while idle (cancels an in-flight start/switch); 503 when busy |
| POST | `/gateway/force-stop` | unconditional teardown (cuts in-flight streaming); use to reclaim resources |
| POST | `/gateway/preload/{model}` | warm a model and return when ready |
| GET | `/gateway/usage` | usage aggregates: `model` / `since` / `until` / `group_by=hour\|day\|none` / `limit` |
| GET | `/gateway/dashboard` | built-in usage dashboard (self-contained HTML/JS, no CDN) |

`/docs`, `/redoc`, `/openapi.json` are the FastAPI built-ins (schema only, no auth).

### API-key auth

Set `server.api_key` (string or list) or `LLM_GATEWAY_API_KEY` (comma-separated). All `/v1/*` and `/gateway/*` data endpoints then require `Authorization: Bearer <key>` (401 `authentication_error` otherwise). Exempt: `/docs`, `/redoc`, `/openapi.json`, `/gateway/dashboard` (static shell; the page prompts for the key on 401). **No key configured = open access.**

## State machine & concurrency

```text
STOPPED ──start(model)──▶ STARTING ──health ok──▶ RUNNING(model)
   ▲                        │                       │
   │◀──failed/timeout───────┘                       │ stop/idle
   │                                                ▼
   └─────────────────────────────────────────── STOPPING ──▶ STOPPED
RUNNING(a) ──switch(b)──▶ SWITCHING(a→b) ──▶ RUNNING(b)  (failure → STOPPED)
```

| State | request for the starting model | request for another model |
|---|---|---|
| STOPPED | starts the backend | starts the backend (serialized) |
| STARTING(m) | waits for startup, then forwards | 503 `model_switch_busy` |
| RUNNING(m) | **direct route** (no lock) | busy → busy; idle → switch |
| SWITCHING(a→b) | waits for the switch | 503 `model_switch_busy` |
| STOPPING | 503 `backend_unavailable` | same |

A client disconnecting mid-start never wedges the state machine (cancellation is cleaned up; state returns to STOPPED).

## Cold start & timeouts

A request that misses the loaded model waits: backend start → health ready → forward. Three clocks matter:

| clock | default | note |
|---|---|---|
| model load time | measured: 86–188 s for a 35B FP8 via SGLang | the actual wait |
| `backend.startup_timeout_seconds` | 180 (600 in the shipped example) | server-side backstop → 503 `backend_startup_timeout`, never hangs forever |
| client timeout | openai-python 600 s; **httpx/requests 5 s** | the real risk: set `timeout=600` or `None` |

Concurrent requests for the same model share one startup; requests for a different model get an immediate 503 busy.

## Errors

| HTTP | error.type | meaning |
|---|---|---|
| 503 | `insufficient_memory` | VRAM and/or RAM below requirement (message says which) |
| 503 | `model_switch_busy` | model busy / transition in flight |
| 404 | `model_not_found` | unknown model |
| 503 | `backend_startup_failed` / `backend_startup_timeout` | backend failed / timed out during startup |
| 503 | `backend_unavailable` | backend unreachable / stopping |
| 400 | `invalid_request` | malformed body / missing `model` |
| 401 | `authentication_error` | missing or wrong API key |

## Configuration

```yaml
server:   # manager listen address + optional api_key
backend:  # internal port, health path, command_template, env
memory:   # device (CUDA only), safety_margin_gib, release timeout
idle:     # timeout_seconds (configurable, never hardcoded), check interval
usage:    # SQLite token tracking + dashboard (enabled / db_path)
models:   # per model: path, required_vram_gib, required_ram_gib, backend.extra_args/env
```

Env overrides: `LLM_GATEWAY_CONFIG`, `LLM_GATEWAY_HOST`, `LLM_GATEWAY_PORT`, `LLM_GATEWAY_API_KEY`, `LLM_GATEWAY_IDLE_TIMEOUT_SECONDS`. Config sections and env variables from pre-0.4 versions are still accepted as deprecated aliases for a smooth migration (see `llm_gateway/config.py`).

## Windows support

- Pure Python 3.10+ / asyncio; no Unix-only dependencies (no signal/fcntl/resource); psutil reads memory on all platforms.
- `terminate()` is already a hard kill on Windows; the SIGTERM→SIGKILL escalation flow still works.
- Templates accept `.exe` paths directly (quote paths with spaces).
- No NVIDIA driver (Intel CPU/NPU) → VRAM gate skipped, RAM gate active.
- Run via console, NSSM, or `deploy/llm-gateway.bat` (Linux: `deploy/llm-gateway.service`).

## Development

```bash
pip install -e ".[dev]"
pytest -v     # 97 tests, fake memory/runner — no GPU required
```

Layout:

```text
llm_gateway/
├── __main__.py     CLI entry
├── api.py          FastAPI: /v1 proxy + /gateway management + usage/dashboard + auth
├── config.py       YAML config (backend template / dual memory gates / legacy aliases)
├── dashboard.py    usage dashboard page (inline HTML/CSS/JS)
├── errors.py       typed error hierarchy
├── memory.py       VRAM (NVML optional) + RAM (psutil) monitor
├── manager.py      state machine + lifecycle lock + watchdog + accounting
├── proxy.py        OpenAI-compatible reverse proxy (SSE + usage capture)
├── runner.py       backend subprocess: template command / health / stop
├── state.py        state enum
└── usage.py        SQLite token usage recording & aggregation
```

## License

MIT
