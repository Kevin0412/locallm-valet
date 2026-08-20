"""FastAPI application: OpenAI-compatible proxy + gateway management API.

Client-facing surface (fixed, independent of the backend):

- ``POST /v1/chat/completions``, ``POST /v1/completions``, ``POST
  /v1/responses``, ... — any ``/v1/*`` POST is gated on its ``model`` field
  and forwarded verbatim to the backend (including SSE streaming).
- ``GET /v1/models`` — the configured registry, so OpenAI clients can list
  models even while the backend is stopped.
- ``GET /gateway/status``, ``GET /gateway/models``, ``POST /gateway/stop``,
  ``POST /gateway/preload/{model}`` — management surface.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import __version__
from .config import Config, ConfigError, load_config
from .dashboard import DASHBOARD_HTML
from .errors import InvalidRequest, ManagerError, BackendUnavailable
from .memory import MemoryMonitor
from .manager import ModelManager
from .proxy import Proxy
from .runner import BackendRunner
from .state import State
from .usage import UsageRecorder, extract_usage_from_json
from .benchmark.job import current_job, start_job, pause_job, resume_job, stop_job

logger = logging.getLogger(__name__)


def build_manager(config: Config):
    """Wire the pool monitor + per-slot backend runners into a SlotManager.

    Every slot owns a ModelManager with its own backend port and state
    machine; shared resource pools gate starts globally.
    """
    from .slot_manager import SlotManager
    return SlotManager(config)


def create_app(
    config: Config | None = None,
    manager: ModelManager | None = None,
    proxy: Proxy | None = None,
    recorder: UsageRecorder | None = None,
) -> FastAPI:
    """App factory. ``manager``/``proxy``/``recorder`` can be injected (tests
    use fakes / in-memory databases)."""

    if config is None:
        config = load_config()
    manager = manager or build_manager(config)
    proxy = proxy or Proxy(config.backend.base_url)
    recorder = recorder or (UsageRecorder(config.usage.db_path) if config.usage.enabled else None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.start()
        try:
            yield
        finally:
            await manager.shutdown()
            await proxy.aclose()
            if recorder is not None:
                recorder.close()

    app = FastAPI(title="locallm-valet", version=__version__, lifespan=lifespan)

    if config.server.auth_enabled:
        _AUTH_EXEMPT_PREFIXES = ("/docs", "/redoc", "/openapi.json")

        def _check_credentials(request: Request) -> bool:
            """Accept ``Authorization: Bearer <key>``, ``x-api-key: <key>``
            (Anthropic clients) or ``Authorization: Basic base64(user:pass)``."""
            auth = request.headers.get("authorization", "")
            scheme, _, cred = auth.partition(" ")
            scheme = scheme.lower()
            if scheme == "bearer" and cred.strip() in config.server.api_keys:
                return True
            if scheme == "basic" and config.server.username and config.server.password:
                import base64
                try:
                    decoded = base64.b64decode(cred.strip()).decode("utf-8")
                except Exception:
                    return False
                user, _, pw = decoded.partition(":")
                if user == config.server.username and pw == config.server.password:
                    return True
            if request.headers.get("x-api-key", "") in config.server.api_keys:
                return True
            return False

        @app.middleware("http")
        async def _api_key_auth(request: Request, call_next):
            """Auth gate.

            Protected: every ``/v1/*`` request (the OpenAI-compatible surface,
            i.e. model access) and every write on ``/gateway/*`` (stop /
            force-stop / preload / benchmark run — anything that drives model
            inference or tears the service down).

            Open: read-only ``GET /gateway/*`` (status / models / usage /
            dashboard / benchmark page & progress) and docs pages.
            """
            path = request.url.path
            if path.startswith(_AUTH_EXEMPT_PREFIXES):
                return await call_next(request)
            if request.method == "GET" and path.startswith("/gateway/"):
                return await call_next(request)
            if _check_credentials(request):
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "type": "authentication_error",
                        "message": "invalid or missing credentials (Authorization: Bearer <key> or Basic <user:pass>)",
                        "code": "invalid_api_key",
                    }
                },
                headers={"WWW-Authenticate": 'Bearer, Basic realm="locallm-valet"'},
            )

    @app.exception_handler(ManagerError)
    async def _manager_error_handler(request: Request, exc: ManagerError):
        return JSONResponse(status_code=exc.http_status, content={"error": exc.to_payload()})

    async def _route_with_gate(request: Request, body: bytes, payload: dict) -> object:
        """Gate on ``model``, then forward; owns active-request accounting
        and usage recording."""

        model_name = payload.get("model")
        if not isinstance(model_name, str) or not model_name:
            raise InvalidRequest("missing or invalid 'model' field in request body")
        started = time.monotonic()
        is_stream = bool(payload.get("stream"))
        # Only chat/completions speaks the OpenAI stream_options contract;
        # Responses (/v1/responses) and Anthropic (/v1/messages) have their
        # own streaming shape — pass those through verbatim (routing only,
        # no protocol rewriting).
        if is_stream and request.url.path == "/v1/chat/completions":
            # The backend only sends the final SSE usage frame when explicitly
            # requested; inject include_usage so our token accounting (and the
            # client, transparently) gets real numbers.
            opts = payload.get("stream_options")
            if not isinstance(opts, dict):
                opts = {}
                payload["stream_options"] = opts
            opts["include_usage"] = True
            body = json.dumps(payload).encode()

        def record_usage(usage: dict | None, status: int | None) -> None:
            if recorder is None:
                return
            try:
                recorder.record(
                    model=model_name,
                    endpoint=request.url.path,
                    stream=is_stream,
                    status=status,
                    prompt_tokens=(usage or {}).get("prompt_tokens", 0),
                    completion_tokens=(usage or {}).get("completion_tokens", 0),
                    duration_ms=(time.monotonic() - started) * 1000.0,
                )
            except Exception:  # noqa: BLE001 - recording never breaks proxying
                logger.exception("usage recording failed for %s", model_name)

        await manager.ensure_loaded(model_name)
        slot_mgr = manager.get_slot_manager(model_name)
        base_url = manager.base_url_for(model_name)
        slot_mgr.request_started()
        if is_stream:
            # proxy.stream owns the finish callback: it fires on send failure
            # or when the SSE stream is fully consumed / closed by the client.
            return await proxy.stream(
                request.method, request.url.path, request.headers, body,
                on_finished=slot_mgr.request_finished,
                on_usage=record_usage if recorder is not None else None,
                base_url=base_url,
            )
        try:
            resp = await proxy.plain(request.method, request.url.path, request.headers, body,
                                     base_url=base_url)
            record_usage(extract_usage_from_json(resp.body), resp.status_code)
            return resp
        finally:
            slot_mgr.request_finished()

    # ------------------------------------------------------------- /v1

    @app.get("/v1/models")
    async def list_models():
        data = []
        model_status = {m["name"]: m for m in manager.models_status()}
        for name, spec in manager.cfg.models.items():
            ms = model_status.get(name, {})
            data.append(
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "locallm-valet",
                    "slot": ms.get("slot", spec.slot),
                    # Declared context (from --context-length) vs the real KV
                    # capacity probed at load time (unknown until loaded).
                    "context_length": spec.configured_context_length(),
                    "max_context_tokens": ms.get("max_context_tokens") if ms.get("loaded") else None,
                }
            )
        return {"object": "list", "data": data}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def v1_proxy(path: str, request: Request):
        query = request.url.query
        path_and_query = request.url.path + (f"?{query}" if query else "")
        if request.method == "GET":
            # No model field in a GET; forward to the first slot with a loaded
            # model (multi-slot: any RUNNING slot works for backend-side GETs).
            for slot_mgr in manager.slots.values():
                if slot_mgr.state is State.RUNNING and slot_mgr.current_model:
                    started = time.monotonic()
                    slot_mgr.request_started()
                    base_url = f"http://{config.backend.host}:{slot_mgr.runner.cfg.port}"
                    try:
                        resp = await proxy.plain("GET", path_and_query, request.headers, b"",
                                                 base_url=base_url)
                        if recorder is not None:
                            try:
                                recorder.record(
                                    model=slot_mgr.current_model or "",
                                    endpoint=request.url.path,
                                    stream=False,
                                    status=resp.status_code,
                                    duration_ms=(time.monotonic() - started) * 1000.0,
                                )
                            except Exception:  # noqa: BLE001
                                logger.exception("usage recording failed for GET %s", request.url.path)
                        return resp
                    finally:
                        slot_mgr.request_finished()
            raise BackendUnavailable("no model loaded; POST with a 'model' field to load one")

        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except ValueError:
            raise InvalidRequest("request body must be valid JSON") from None
        if not isinstance(payload, dict):
            raise InvalidRequest("request body must be a JSON object")
        return await _route_with_gate(request, body, payload)

    # ---------------------------------------------------------- /gateway

    @app.get("/gateway/status")
    async def gateway_status():
        return manager.status()

    @app.get("/gateway/models")
    async def gateway_models():
        return {
            "slots": {name: {"state": m.state.value, "model": m.current_model}
                      for name, m in manager.slots.items()},
            "models": manager.models_status(),
        }

    @app.post("/gateway/stop")
    async def gateway_stop():
        """正常关闭所有槽：空闲时（任意状态）接受并清资源；正在服务时 503。"""
        result = await manager.stop(reason="manual")
        return {"slots": result}

    @app.post("/gateway/force-stop")
    async def gateway_force_stop():
        """强制关闭所有槽：无条件清资源，即使有活跃请求（会切断流式连接）。"""
        result = await manager.stop(reason="manual (forced)", force=True)
        return {"slots": result}

    @app.post("/gateway/preload/{model_name}")
    async def gateway_preload(model_name: str):
        slot_mgr = manager.get_slot_manager(model_name)
        await slot_mgr.ensure_loaded(model_name)
        return {
            "slot": slot_mgr.slot_name,
            "state": slot_mgr.state.value,
            "model": slot_mgr.current_model,
            "active_requests": slot_mgr.active_requests,
        }

    # ------------------------------------------------------ usage / dashboard

    def _parse_time(value: str | None) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            raise InvalidRequest(f"invalid time value {value!r}: use epoch seconds or ISO8601") from None

    if recorder is not None:

        @app.get("/gateway/usage")
        async def gateway_usage(
            model: str | None = None,
            since: str | None = None,
            until: str | None = None,
            group_by: str = "hour",
            limit: int = 50,
        ):
            if group_by not in ("hour", "day", "none"):
                raise InvalidRequest(f"group_by must be one of hour|day|none, got {group_by!r}")
            return recorder.query(
                model=model or None,
                since=_parse_time(since),
                until=_parse_time(until),
                group_by=None if group_by == "none" else group_by,
                limit=max(1, min(int(limit), 500)),
            )

        @app.get("/gateway/benchmark", response_class=HTMLResponse)
        async def gateway_benchmark():
            """Benchmark page (bilingual, one-click run / pause / resume)."""
            return HTMLResponse(_render_benchmark_page(config))

        @app.get("/gateway/benchmark/status")
        async def gateway_benchmark_status():
            """Read-only progress of the current benchmark job."""
            return current_job().status()

        @app.post("/gateway/benchmark/run")
        async def gateway_benchmark_run(request: Request):
            """Start a benchmark job: POST {"dataset": "mmlu", "models": [...]}.
            Drives model inference — auth-gated (see middleware)."""
            body = await request.body()
            try:
                payload = json.loads(body) if body else {}
            except ValueError:
                raise InvalidRequest("request body must be valid JSON") from None
            dataset = payload.get("dataset", "mmlu")
            models = payload.get("models")
            if not isinstance(models, list) or not models:
                raise InvalidRequest("'models' must be a non-empty list of model names")
            job = start_job(
                dataset=dataset,
                models=[str(m) for m in models],
                base_url=f"http://127.0.0.1:{config.server.port}/v1",
                max_tokens=int(payload.get("max_tokens", 256)),
                sample=payload.get("sample"),
                concurrency=int(payload.get("concurrency", 1)),
                slot_of={m: config.models[m].slot for m in models if m in config.models},
            )
            return job.status()

        @app.post("/gateway/benchmark/pause")
        async def gateway_benchmark_pause():
            return pause_job().status()

        @app.post("/gateway/benchmark/resume")
        async def gateway_benchmark_resume():
            return resume_job().status()

        @app.post("/gateway/benchmark/stop")
        async def gateway_benchmark_stop():
            return stop_job().status()

        @app.get("/gateway/dashboard", response_class=HTMLResponse)
        async def gateway_dashboard():
            return HTMLResponse(DASHBOARD_HTML)

    return app


# ---------------------------------------------------------------------------
# Benchmark page rendering
# ---------------------------------------------------------------------------

def _render_benchmark_page(config: Config) -> str:
    """Render the benchmark page from the shared design system: dataset/model
    picker, one-click run with pause/resume, live progress, results table."""
    from pathlib import Path
    from collections import defaultdict

    from .frontend import page
    from .benchmark.dataset import list_datasets
    from .benchmark.job import current_job

    models = sorted(config.models.keys())
    datasets = list_datasets()
    job = current_job().status()

    # Aggregate scored JSONL by model_name
    rows = ""
    results_dir = Path("benchmark_results")
    if results_dir.is_dir():
        ms: dict = defaultdict(lambda: {"t": 0, "c": 0, "cat": defaultdict(lambda: [0, 0]), "lat": [], "tps": []})
        for f in sorted(results_dir.glob("*_results.jsonl")):
            for line in f.read_text("utf-8").strip().splitlines():
                if not line:
                    continue
                r = json.loads(line)
                m = r.get("model_name", "?")
                ms[m]["t"] += 1
                if r.get("is_correct") is True:
                    ms[m]["c"] += 1
                cat = r.get("category", "?")
                ms[m]["cat"][cat][0] += 1
                if r.get("is_correct") is True:
                    ms[m]["cat"][cat][1] += 1
                if r.get("latency_ms"):
                    ms[m]["lat"].append(r["latency_ms"])
                if r.get("tps"):
                    ms[m]["tps"].append(r["tps"])

        def _tag(cn: str, pct: float) -> str:
            cls = "ok" if pct >= 60 else ("warn" if pct >= 30 else "err")
            return f'<span class="tag {cls}">{cn} {pct:.0f}%</span>'

        def _row(name: str, s: dict) -> str:
            acc = round(s["c"] / s["t"] * 100, 1) if s["t"] else 0
            lat = round(sum(s["lat"]) / len(s["lat"]), 1) if s["lat"] else "-"
            tps = round(sum(s["tps"]) / len(s["tps"]), 2) if s["tps"] else "-"
            tags = "".join(
                _tag(cn, round(c[1] / c[0] * 100, 1))
                for cn in ("fact", "reasoning", "math", "chinese", "instruction", "coding")
                if (c := s["cat"].get(cn)) and c[0]
            )
            return (f'<tr><td>{name}</td><td class="num" style="font-weight:650">{acc}%</td>'
                    f'<td class="num">{s["c"]}/{s["t"]}</td><td>{tags}</td>'
                    f'<td class="num">{lat}</td><td class="num">{tps}</td></tr>')

        rows = "".join(
            _row(name, s)
            for name, s in sorted(ms.items(), key=lambda x: -x[1]["c"] / max(x[1]["t"], 1))
        )

    # Group models by slot for a friendlier selector
    from collections import defaultdict as _dd
    slot_models: dict = _dd(list)
    for m in models:
        slot_models[config.models[m].slot].append(m)

    slot_blocks = []
    for slot_name in sorted(slot_models):
        opts = "".join(
            f'<label class="bm-model" title="{slot_name}"><input type="checkbox" name="bmModel" value="{m}">{m}</label>'
            for m in slot_models[slot_name]
        )
        slot_blocks.append(
            f'<div class="bm-slot"><div class="bm-slot-name">{slot_name}</div><div class="bm-slot-models">{opts}</div></div>'
        )
    model_picker = "".join(slot_blocks)
    dataset_opts = "".join(f'<option value="{d}"{" selected" if d == "mmlu" else ""}>{d}</option>' for d in datasets)

    body = f"""
<main>
  <h1 class="page-title" data-i18n="bm_title">模型评测 Benchmark</h1>
  <p class="page-sub" data-i18n="bm_sub">基于公认题库（MMLU / MMLU-Pro / BFCL / MMStar / OCRBench）评估模型能力</p>

  <div class="panel">
    <h2 data-i18n="run_title">运行评测</h2>
    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">
      <select id="dsSel">{dataset_opts}</select>
      <input type="number" id="sampleSel" min="0" step="100" placeholder="sample (0=full)" value="" style="width:130px"
             title="采样条数，0/空 = 全量；有 sample_N_indices.json 时用固定抽样">
      <input type="number" id="concSel" min="1" max="32" value="1" style="width:70px"
             title="并发/批大小：llama.cpp 单槽用 1；SGLang/vLLM 等支持并发的后端可调高（如 4-8）">
      <span class="spacer"></span>
      <button id="selectAllBtn" class="icon-btn" data-i18n="select_all">全选</button>
      <button id="selectNoneBtn" class="icon-btn" data-i18n="select_none">清空</button>
      <button id="runBtn" class="primary" data-i18n="start">开始</button>
      <button id="pauseBtn" data-i18n="pause">暂停</button>
      <button id="resumeBtn" data-i18n="resume" disabled>继续</button>
      <button id="stopBtn" class="danger" data-i18n="stop" disabled>停止</button>
    </div>
    <div class="bm-picker">{model_picker}</div>
    <div class="progress" id="progWrap" style="display:none"><i id="progBar"></i></div>
    <div id="progText" class="muted" style="font-size:12px;margin-top:6px"></div>
    <div id="jobErr" class="err-text" style="margin-top:6px"></div>
  </div>

  <div class="panel">
    <h2 data-i18n="results_title">评测结果</h2>
    <table>
      <thead><tr>
        <th data-i18n="model">模型</th>
        <th class="num" data-i18n="accuracy">准确率</th>
        <th class="num" data-i18n="correct_total">正确/总数</th>
        <th data-i18n="category">分项</th>
        <th class="num" data-i18n="avg_lat">平均耗时 ms</th>
        <th class="num" data-i18n="avg_tps">吞吐 tok/s</th>
      </tr></thead>
      <tbody id="resultsBody">{rows or '<tr><td colspan="6" class="empty" data-i18n="empty">暂无数据</td></tr>'}</tbody>
    </table>
  </div>
</main>
<style>
.bm-picker {{ display:flex; flex-direction:column; gap:6px; margin-top:12px; }}
.bm-slot {{ display:flex; align-items:flex-start; gap:10px; }}
.bm-slot-name {{ min-width:70px; font-size:12px; color:var(--fg-3); padding-top:3px; }}
.bm-slot-models {{ display:flex; flex-wrap:wrap; gap:6px; }}
.bm-model {{ display:inline-flex; align-items:center; gap:5px; background:var(--bg-soft);
  border:1px solid var(--border-soft); border-radius:6px; padding:4px 9px; font-size:12px; cursor:pointer; }}
.bm-model:hover {{ border-color:var(--accent); }}
.bm-model input {{ accent-color: var(--accent); }}
</style>
"""
    return page("模型评测", "Benchmark", active="benchmark", body=body,
                extra_js=_benchmark_js())


def _benchmark_js() -> str:
    return r"""
let jobTimer = null;

function stopPolling() {
  if (jobTimer) { clearInterval(jobTimer); jobTimer = null; }
}

async function refreshJob() {
  try {
    const r = await authedFetch('/gateway/benchmark/status');
    const j = await r.json();
    const wrap = $('progWrap'), bar = $('progBar'), txt = $('progText');
    const runBtn = $('runBtn'), pauseBtn = $('pauseBtn'), resumeBtn = $('resumeBtn'), stopBtn = $('stopBtn');
    if (j.state === 'running' || j.state === 'paused') {
      wrap.style.display = 'block';
      const pct = j.total_items ? Math.round(100 * j.done_items / j.total_items) : 0;
      bar.style.width = pct + '%';
      txt.textContent = (j.state === 'paused' ? i18n('paused') : i18n('running')) +
        ' · ' + j.dataset + ' · ' + j.current_model + ' · ' + j.done_items + '/' + j.total_items;
      runBtn.disabled = true; pauseBtn.disabled = (j.state !== 'running');
      resumeBtn.disabled = (j.state !== 'paused'); stopBtn.disabled = false;
    } else if (j.state === 'done' || j.state === 'stopped' || j.state === 'error') {
      wrap.style.display = 'block';
      bar.style.width = '100%';
      txt.textContent = j.state.toUpperCase() + (j.error ? ' · ' + j.error : '');
      runBtn.disabled = false; pauseBtn.disabled = true; resumeBtn.disabled = true; stopBtn.disabled = true;
      stopPolling();
    }
  } catch (e) {}
}

async function runBench() {
  const boxes = document.querySelectorAll('input[name="bmModel"]:checked');
  const models = [...boxes].map(b => b.value);
  if (!models.length) models.push(...document.querySelectorAll('input[name="bmModel"]')).map(b => b.value);
  const payload = {
    dataset: $('dsSel').value,
    models,
    sample: parseInt($('sampleSel').value, 10) || null,
    concurrency: parseInt($('concSel').value, 10) || 1,
  };
  const r = await authedFetch('/gateway/benchmark/run', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    $('jobErr').textContent = (d.error && d.error.message) || 'HTTP ' + r.status;
    return;
  }
  $('jobErr').textContent = '';
  stopPolling();
  refreshJob();
  jobTimer = setInterval(refreshJob, 2000);
}

function setAllModels(checked) {
  document.querySelectorAll('input[name="bmModel"]').forEach(b => b.checked = checked);
}
$('selectAllBtn').onclick = () => setAllModels(true);
$('selectNoneBtn').onclick = () => setAllModels(false);
$('runBtn').onclick = runBench;
$('pauseBtn').onclick = async () => { await authedFetch('/gateway/benchmark/pause', { method: 'POST' }); refreshJob(); };
$('resumeBtn').onclick = async () => { await authedFetch('/gateway/benchmark/resume', { method: 'POST' }); refreshJob(); };
$('stopBtn').onclick = async () => { await authedFetch('/gateway/benchmark/stop', { method: 'POST' }); refreshJob(); };

// Poll only while a job is actually running — an idle page must not spam
// /gateway/benchmark/status every few seconds.
(async function init() {
  try {
    const j = await (await authedFetch('/gateway/benchmark/status')).json();
    if (j.state === 'running' || j.state === 'paused') {
      refreshJob();
      jobTimer = setInterval(refreshJob, 2000);
    } else {
      refreshJob();  // renders the terminal state once, then stays quiet
    }
  } catch (e) {}
})();
"""


def _make_default_app() -> FastAPI | None:
    try:
        return create_app()
    except ConfigError as exc:
        logger.error("cannot create default app (no usable config): %s", exc)
        return None


class _LazyApp:
    """Lazy default app — only inits when accessed, so benchmark CLI
    doesn't trigger a spurious config load on import."""

    _instance: FastAPI | None = None

    def __getattr__(self, name: str):
        if self._instance is None:
            self._instance = _make_default_app()
            if self._instance is None:
                raise RuntimeError(
                    "no valid config found; use `python -m locallm_valet --config config.yaml`"
                )
        return getattr(self._instance, name)


# For `uvicorn locallm_valet.api:app`; the lazy wrapper means import-time
# side effects (config load) don't happen until uvicorn actually serves.
# Prefer `python -m locallm_valet --config config.yaml`.
app: FastAPI = _LazyApp()  # type: ignore[assignment]
