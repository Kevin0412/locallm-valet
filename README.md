# locallm-valet

**English** | [简体中文](README.zh-CN.md)

A **local LLM valet**: it parks (unloads) your model when idle and brings it back on demand. On a single device with a single active model, it manages **any OpenAI-compatible inference backend on your own machine** — SGLang / vLLM / llama.cpp / OpenVINO / ...

> Like a valet parking your car: when the model isn't needed, it is unloaded and the GPU/CPU/NPU is returned to you; when a request arrives, the model is started (only if VRAM/RAM is sufficient), health-checked and served — all driven by the OpenAI API `model` field. Switches are refused while the model is busy, and after a configurable idle period (default 1 hour) the model is parked again automatically.

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
- **Idle watchdog**: auto-unload after a configurable idle timeout (default 1 hour, never hardcoded — YAML or `LOCALLM_VALET_IDLE_TIMEOUT_SECONDS`).
- **Token usage tracking**: per-request tokens (plain + streaming) recorded to SQLite, with a built-in usage dashboard.
- **API-key auth**: optional `Authorization: Bearer` gate for all data endpoints.
- **Windows-friendly**: pure asyncio, no Unix-only dependencies; RAM gating works on CPU/NPU machines (e.g. Intel Core Ultra + OpenVINO).

## Architecture

```text
Client
  │
  │ OpenAI-compatible API (fixed endpoint :8000)
  ▼
locallm-valet :8000
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
# or: pip install locallm-valet
cp config.example.yaml config.yaml      # then edit
python -m locallm_valet --config config.yaml
# or: locallm-valet --config config.yaml
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

- **VRAM** (NVML): the **primary gate for GPU backends** — a hard failure if insufficient. Checked only when an NVIDIA driver is present; on CPU/NPU machines it is skipped automatically.
- **RAM** (psutil `available`): **opt-in** (`required_ram_gib > 0`). Mainly for CPU/NPU backends (llama.cpp / OpenVINO) where RAM *is* the compute resource. For GPU serving you usually leave it 0 — VRAM sufficiency is what matters; a tight RAM situation makes loading slow (swap), not impossible.
- `0` (default) disables that gate.
- After a switch, the decision uses the **re-read values after the old backend has exited and memory has actually settled**.

```text
model needs VRAM 40G + 4G margin = 44G; free 44.1G ✓ start
model needs RAM  32G + 4G margin = 36G; available 34G ✗ 503 insufficient_memory
```

## Ports & routes

| Port | Bind | Purpose | Config |
|---|---|---|---|
| **8000** | `0.0.0.0` (default) | the only client-facing entry (OpenAI API + management) | `server.port` / `LOCALLM_VALET_PORT` |
| **30000** | `127.0.0.1` (default) | internal backend port, reachable only from the gateway | `backend.port` |

### OpenAI-compatible surface (client-facing, API-key protected)

| Method | Path | Behavior |
|---|---|---|
| POST | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, any `/v1/*` | gated on the `model` field (start / switch / direct route) then forwarded verbatim, SSE included |
| GET | `/v1/models` | registry listing (available even while the backend is stopped); each entry carries `context_length` (declared `--context-length`) and `max_context_tokens` (real KV capacity — `null` until that model has been loaded once) |
| GET | other `/v1/*` | forwarded only when a model is loaded |

Streaming requests get `stream_options.include_usage=true` injected automatically so real token counts are captured; transparent to clients.

### Management surface (`/gateway/*`, API-key protected)

| Method | Path | Behavior |
|---|---|---|
| GET | `/gateway/status` | state machine + active requests + idle seconds + memory (vram/ram) + `max_context_tokens` (real KV capacity of the loaded model, probed at load time — the configured `--context-length` is only a declaration; the real limit differs per machine, e.g. ~158K tokens for a 27B dense FP8 on 48G) |
| GET | `/gateway/models` | registry + per-model loaded state + `max_context_tokens` (only for the loaded model; `null` until it has been loaded once) |
| POST | `/gateway/stop` | graceful unload — accepted in any state while idle (cancels an in-flight start/switch); 503 when busy |
| POST | `/gateway/force-stop` | unconditional teardown (cuts in-flight streaming); use to reclaim resources |
| POST | `/gateway/preload/{model}` | warm a model and return when ready |
| GET | `/gateway/usage` | usage aggregates: `model` / `since` / `until` / `group_by=hour\|day\|none` / `limit` |
| GET | `/gateway/dashboard` | built-in usage dashboard (self-contained HTML/JS, no CDN) |

`/docs`, `/redoc`, `/openapi.json` are the FastAPI built-ins (schema only, no auth).

### API-key auth

Set `server.api_key` (string or list) or `LOCALLM_VALET_API_KEY` (comma-separated). All `/v1/*` and `/gateway/*` data endpoints then require `Authorization: Bearer <key>` (401 `authentication_error` otherwise). Exempt: `/docs`, `/redoc`, `/openapi.json`, `/gateway/dashboard` (static shell; the page prompts for the key on 401). **No key configured = open access.**

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

Env overrides: `LOCALLM_VALET_CONFIG`, `LOCALLM_VALET_HOST`, `LOCALLM_VALET_PORT`, `LOCALLM_VALET_API_KEY`, `LOCALLM_VALET_IDLE_TIMEOUT_SECONDS`. Env variables and config sections from earlier project names are still accepted as deprecated aliases for a smooth migration (see `locallm_valet/config.py`).

## Windows support

- Pure Python 3.10+ / asyncio; no Unix-only dependencies (no signal/fcntl/resource); psutil reads memory on all platforms.
- `terminate()` is already a hard kill on Windows; the SIGTERM→SIGKILL escalation flow still works.
- Templates accept `.exe` paths directly (quote paths with spaces).
- No NVIDIA driver (Intel CPU/NPU) → VRAM gate skipped, RAM gate active.
- Run via console, NSSM, or `deploy/locallm-valet.bat` (Linux: `deploy/locallm-valet.service`).

## Development

```bash
pip install -e ".[dev]"
# or: pip install locallm-valet
pytest -v     # 97 tests, fake memory/runner — no GPU required
```

Layout:

```text
locallm_valet/
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

## Benchmark — 一键量化质量检测

验证本地模型的量化降级（Q8→Q4→Q3 精度损失）。

### 单模型跑分

```bash
python -m locallm_valet benchmark run \
  --model Qwen3-1.7B-Q8_0 \
  --dataset builtin \
  --base-url http://127.0.0.1:8000/v1

# 结果 → benchmark_results/benchmark_<model>.md
# 看板  → http://127.0.0.1:8000/gateway/benchmark
```

### 同模型不同量化对比

```bash
python -m locallm_valet benchmark compare \
  --models Qwen3-1.7B-Q8_0 Qwen3-1.7B-Q4_K_M \
  --labels "Q8_0" "Q4_K_M" \
  --dataset builtin \
  --base-url http://127.0.0.1:8000/v1

# 结果 → benchmark_results/benchmark_comparison.md
```

### 内置数据集 (`builtin`)

48 道题，覆盖 **fact / reasoning / math / chinese (中文) / instruction / coding** 六个维度。零依赖零下载，即跑即用。

### 可用的模型名

| GGUF (llama.cpp CPU) | OpenVINO (GPU/NPU) |
|---|---|
| `gemma-4-e2b`, `gemma-4-e4b`, `gemma-3-4b-it` | `qwen3-1.7b` (GPU), `qwen3-1.7b-npu` (NPU) |
| `Qwen3-1.7B-Q8_0`, `Qwen3-1.7B-Q5_K_M`, `Qwen3-1.7B-Q4_K_M` | `llama-3.2-1b` (NPU), `qwen2.5-1.5b` (NPU) |
| `Qwen3.5-0.8B`, `Qwen3.5-2B`, `Qwen3.5-4B` | |

## License

MIT
