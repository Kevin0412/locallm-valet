"""FastAPI application: OpenAI-compatible proxy + gateway management API.

Client-facing surface (fixed, independent of what SGLang does):

- ``POST /v1/chat/completions``, ``POST /v1/completions``, ``POST
  /v1/responses``, ... — any ``/v1/*`` POST is gated on its ``model`` field
  and forwarded verbatim to SGLang (including SSE streaming).
- ``GET /v1/models`` — the configured registry, so OpenAI clients can list
  models even while SGLang is stopped.
- ``GET /gateway/status``, ``GET /gateway/models``, ``POST /gateway/stop``,
  ``POST /gateway/preload/{model}`` — management surface.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .config import Config, ConfigError, load_config
from .errors import InvalidRequest, ManagerError, SglangUnavailable
from .gpu import GpuMonitor
from .manager import ModelManager
from .proxy import Proxy
from .runner import SglangRunner
from .state import State

logger = logging.getLogger(__name__)


def build_manager(config: Config) -> ModelManager:
    """Wire the real GPU monitor + SGLang runner into a manager."""

    gpu = GpuMonitor(config.gpu.device)
    runner = SglangRunner(config.sglang, gpu_device=config.gpu.device)
    return ModelManager(config, gpu, runner)


def create_app(
    config: Config | None = None,
    manager: ModelManager | None = None,
    proxy: Proxy | None = None,
) -> FastAPI:
    """App factory. ``manager``/``proxy`` can be injected (tests use fakes)."""

    if config is None:
        config = load_config()
    manager = manager or build_manager(config)
    proxy = proxy or Proxy(config.sglang.base_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.start()
        try:
            yield
        finally:
            await manager.shutdown()
            await proxy.aclose()

    app = FastAPI(title="sglang-manager", version=__version__, lifespan=lifespan)

    @app.exception_handler(ManagerError)
    async def _manager_error_handler(request: Request, exc: ManagerError):
        return JSONResponse(status_code=exc.http_status, content={"error": exc.to_payload()})

    async def _route_with_gate(request: Request, body: bytes, payload: dict) -> object:
        """Gate on ``model``, then forward; owns active-request accounting."""

        model_name = payload.get("model")
        if not isinstance(model_name, str) or not model_name:
            raise InvalidRequest("missing or invalid 'model' field in request body")
        await manager.ensure_loaded(model_name)
        manager.request_started()
        if payload.get("stream"):
            # proxy.stream owns the finish callback: it fires on send failure
            # or when the SSE stream is fully consumed / closed by the client.
            return await proxy.stream(
                request.method, request.url.path, request.headers, body, manager.request_finished
            )
        try:
            return await proxy.plain(request.method, request.url.path, request.headers, body)
        finally:
            manager.request_finished()

    # ------------------------------------------------------------- /v1

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "sglang-manager"}
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
                manager.request_started()
                try:
                    return await proxy.plain("GET", path_and_query, request.headers, b"")
                finally:
                    manager.request_finished()
            raise SglangUnavailable("no model loaded; POST with a 'model' field to load one")

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
        await manager.stop(reason="manual")
        return {"state": manager.state.value, "model": manager.current_model}

    @app.post("/gateway/preload/{model_name}")
    async def gateway_preload(model_name: str):
        await manager.ensure_loaded(model_name)
        return {
            "state": manager.state.value,
            "model": manager.current_model,
            "active_requests": manager.active_requests,
        }

    return app


def _make_default_app() -> FastAPI | None:
    try:
        return create_app()
    except ConfigError as exc:
        logger.error("cannot create default app (no usable config): %s", exc)
        return None


# For `uvicorn sglang_manager.api:app`; None until a config is available.
# Prefer `python -m sglang_manager --config config.yaml` (or the
# `sglang-manager` console script), which reports config errors clearly.
app = _make_default_app()
