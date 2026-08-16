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
        _AUTH_EXEMPT_EXACT = ("/gateway/dashboard",)

        @app.middleware("http")
        async def _api_key_auth(request: Request, call_next):
            """Bearer API-key gate for /v1/* and /gateway/* data endpoints.

            The dashboard page itself is exempt (static shell, no data); its
            JS prompts for the key and sends it on data fetches.  Docs pages
            are exempt (schema only).  Everything else requires a valid key
            when ``server.api_keys`` is configured.
            """
            path = request.url.path
            if path.startswith(_AUTH_EXEMPT_PREFIXES) or path in _AUTH_EXEMPT_EXACT:
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
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "locallm-valet"}
                for name in manager.cfg.models
            ],
        }

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

        @app.get("/gateway/dashboard", response_class=HTMLResponse)
        async def gateway_dashboard():
            return HTMLResponse(DASHBOARD_HTML)

    return app


def _make_default_app() -> FastAPI | None:
    try:
        return create_app()
    except ConfigError as exc:
        logger.error("cannot create default app (no usable config): %s", exc)
        return None


# For `uvicorn locallm_valet.api:app`; None until a config is available.
# Prefer `python -m locallm_valet --config config.yaml` (or the
# `locallm-valet` console script), which reports config errors clearly.
app = _make_default_app()
