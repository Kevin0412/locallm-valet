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

logger = logging.getLogger(__name__)


def build_manager(config: Config) -> ModelManager:
    """Wire the real memory monitor + backend runner into a manager."""

    memory = MemoryMonitor(config.memory.device)
    runner = BackendRunner(config.backend, device=config.memory.device)
    return ModelManager(config, memory, runner)


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

    if config.server.api_keys:
        _AUTH_EXEMPT_PREFIXES = ("/docs", "/redoc", "/openapi.json")

        @app.middleware("http")
        async def _api_key_auth(request: Request, call_next):
            """Bearer API-key gate.

            Protected: every ``/v1/*`` request (the OpenAI-compatible surface)
            and every write on ``/gateway/*`` (stop / force-stop / preload —
            the operations that can tear down the service).

            Open: read-only ``GET /gateway/*`` (status / models / usage /
            dashboard) — monitoring data is not a secret, so the dashboard
            works without a key.  Docs pages are exempt (schema only).
            """
            path = request.url.path
            if path.startswith(_AUTH_EXEMPT_PREFIXES):
                return await call_next(request)
            if request.method == "GET" and path.startswith("/gateway/"):
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            if not token or token not in config.server.api_keys:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "type": "authentication_error",
                            "message": "invalid or missing API key (Authorization: Bearer <key>)",
                            "code": "invalid_api_key",
                        }
                    },
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return await call_next(request)

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
        if is_stream:
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
        manager.request_started()
        if is_stream:
            # proxy.stream owns the finish callback: it fires on send failure
            # or when the SSE stream is fully consumed / closed by the client.
            return await proxy.stream(
                request.method, request.url.path, request.headers, body,
                on_finished=manager.request_finished,
                on_usage=record_usage if recorder is not None else None,
            )
        try:
            resp = await proxy.plain(request.method, request.url.path, request.headers, body)
            record_usage(extract_usage_from_json(resp.body), resp.status_code)
            return resp
        finally:
            manager.request_finished()

    # ------------------------------------------------------------- /v1

    @app.get("/v1/models")
    async def list_models():
        data = []
        for name, spec in manager.cfg.models.items():
            loaded = manager.state is State.RUNNING and manager.current_model == name
            data.append(
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "locallm-valet",
                    # Declared context (from --context-length) vs the real KV
                    # capacity probed at load time (unknown until loaded).
                    "context_length": spec.configured_context_length(),
                    "max_context_tokens": manager.max_context_tokens if loaded else None,
                }
            )
        return {"object": "list", "data": data}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def v1_proxy(path: str, request: Request):
        query = request.url.query
        path_and_query = request.url.path + (f"?{query}" if query else "")
        if request.method == "GET":
            # No model field in a GET; only forward when something is loaded.
            if manager.state is State.RUNNING:
                started = time.monotonic()
                manager.request_started()
                try:
                    resp = await proxy.plain("GET", path_and_query, request.headers, b"")
                    if recorder is not None:
                        try:
                            recorder.record(
                                model=manager.current_model or "",
                                endpoint=request.url.path,
                                stream=False,
                                status=resp.status_code,
                                duration_ms=(time.monotonic() - started) * 1000.0,
                            )
                        except Exception:  # noqa: BLE001
                            logger.exception("usage recording failed for GET %s", request.url.path)
                    return resp
                finally:
                    manager.request_finished()
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
            "state": manager.state.value,
            "model": manager.current_model,
            "models": manager.models_status(),
        }

    @app.post("/gateway/stop")
    async def gateway_stop():
        """正常关闭：空闲时（任意状态）接受并清显存；正在服务时 503。"""
        await manager.stop(reason="manual")
        return {"state": manager.state.value, "model": manager.current_model}

    @app.post("/gateway/force-stop")
    async def gateway_force_stop():
        """强制关闭：无条件清显存，即使有活跃请求（会切断流式连接）。"""
        await manager.stop(reason="manual (forced)", force=True)
        return {
            "state": manager.state.value,
            "model": manager.current_model,
            "active_requests": manager.active_requests,
        }

    @app.post("/gateway/preload/{model_name}")
    async def gateway_preload(model_name: str):
        await manager.ensure_loaded(model_name)
        return {
            "state": manager.state.value,
            "model": manager.current_model,
            "active_requests": manager.active_requests,
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
            """Render a benchmark results page from scored JSONL files in benchmark_results/."""
            return HTMLResponse(_render_benchmark_page())

        @app.get("/gateway/dashboard", response_class=HTMLResponse)
        async def gateway_dashboard():
            return HTMLResponse(DASHBOARD_HTML)

    return app


# ---------------------------------------------------------------------------
# Benchmark page rendering
# ---------------------------------------------------------------------------

def _render_benchmark_page() -> str:
    """Scan benchmark_results/ for scored JSONL files and render an HTML page
    with model-by-model accuracy comparison."""
    import os
    from pathlib import Path

    results_dir = Path("benchmark_results")
    if not results_dir.is_dir():
        return "<html><body><h1>Benchmark</h1><p>No benchmark results yet. Run <code>python -m locallm_valet benchmark run --model &lt;name&gt;</code> to generate.</p></body></html>"

    files = sorted(results_dir.glob("*_results.jsonl"))
    if not files:
        return "<html><body><h1>Benchmark</h1><p>No scored JSONL files found in <code>benchmark_results/</code>.</p></body></html>"

    rows: list[str] = []
    for f in files:
        model_name = f.stem.replace("_results", "")
        label = model_name  # human-readable, maybe improve later

        total = correct = 0
        cats: dict[str, dict] = {}
        latencies: list[float] = []
        tps_list: list[float] = []

        for line in f.read_text(encoding="utf-8").strip().splitlines():
            if not line:
                continue
            d = json.loads(line)
            cat = d.get("category", "?")
            is_correct = d.get("is_correct")
            lat = d.get("latency_ms")
            tps = d.get("tps")
            total += 1
            if is_correct is True:
                correct += 1
            if cat not in cats:
                cats[cat] = {"total": 0, "correct": 0}
            cats[cat]["total"] += 1
            if is_correct is True:
                cats[cat]["correct"] += 1
            if lat is not None:
                latencies.append(lat)
            if tps is not None:
                tps_list.append(tps)

        acc = round(correct / total * 100, 1) if total else 0
        avg_lat = round(sum(latencies) / len(latencies), 1) if latencies else "-"
        avg_tps = round(sum(tps_list) / len(tps_list), 2) if tps_list else "-"

        # Category breakdown tags
        cat_tags = ""
        for cat_name in ("fact", "reasoning", "math", "chinese", "instruction", "coding"):
            if cat_name not in cats:
                continue
            c = cats[cat_name]
            pct = round(c["correct"] / c["total"] * 100, 1) if c["total"] else 0
            color = "#3ecf8e" if pct >= 60 else ("#ffb454" if pct >= 30 else "#ff6b6b")
            cat_tags += f"<span style=\"display:inline-block;padding:1px 8px;border-radius:4px;background:{color}22;color:{color};font-size:12px;margin-right:4px\">{cat_name}&nbsp;{pct}%</span>"

        rows.append(f"""<tr>
<td style="font-weight:600">{label}</td>
<td style="text-align:center;font-weight:700;font-size:18px">{acc}%</td>
<td style="text-align:center">{correct}/{total}</td>
<td>{cat_tags}</td>
<td style="text-align:center">{avg_lat}</td>
<td style="text-align:center">{avg_tps}</td>
<td><a href="{f.name.replace('_results.jsonl','_report.md').replace('.jsonl','')}" style="color:var(--accent)" target="_blank">report</a></td>
</tr>""")

    # Also link to any comparison report
    compare_link = ""
    compare_files = list(results_dir.glob("*_comparison.md"))
    if compare_files:
        compare_link = f"""<p style="margin-top:16px">📊 <a href="{compare_files[-1].name}" target="_blank" style="color:var(--accent)">Cross-model comparison report</a></p>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>locallm-valet · Benchmark</title>
<style>
* {{box-sizing:border-box;margin:0;padding:0}}
body {{background:#0f1115;color:#e6e8ee;font:14px/1.5 "SF Mono",Consolas,"Microsoft YaHei",monospace;padding:20px}}
h1 {{font-size:18px;margin-bottom:12px}}
p, ol {{margin-bottom:12px;color:#8b93a3}}
code {{background:#1e222c;padding:1px 6px;border-radius:4px;font-size:13px}}
a {{color:#4f8cff;text-decoration:none}}
a:hover {{text-decoration:underline}}
table {{width:100%;border-collapse:collapse;font-size:13px}}
th, td {{text-align:left;padding:8px 10px;border-bottom:1px solid #262b36;white-space:nowrap}}
th {{color:#8b93a3;font-weight:600}}
tr:hover {{background:#171a21}}
</style>
</head>
<body>
<h1>📊 Benchmark Results</h1>
<p>Run via <code>python -m locallm_valet benchmark run/compare --model ...</code> — see README for details.</p>
<table>
<thead><tr>
<th>Model</th><th style="text-align:center">Accuracy</th><th style="text-align:center">Correct/Total</th>
<th>Category Breakdown</th><th style="text-align:center">Avg Latency</th><th style="text-align:center">Avg TPS</th><th>Report</th>
</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
{compare_link}
<ol style="padding-left:20px">
<li>Run once: <code>python -m locallm_valet benchmark run --model &lt;name&gt; --dataset builtin</code></li>
<li>Compare across quantizations: <code>python -m locallm_valet benchmark compare --models Q4 Q8 --labels "Q4_K_M" "Q8_0"</code></li>
<li>Refresh this page to see updated results.</li>
</ol>
</body>
</html>"""


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
