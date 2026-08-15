"""API-level tests: full stack through ASGI with a fake upstream SGLang."""

import asyncio

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pytest_asyncio import fixture as async_fixture

from sglang_manager.api import create_app
from sglang_manager.errors import SglangStartupTimeout
from sglang_manager.manager import ModelManager
from sglang_manager.proxy import Proxy

from .conftest import make_config


def make_upstream_app() -> FastAPI:
    """A miniature SGLang: /health + /v1/chat/completions (SSE included)."""

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if body.get("stream"):
            async def gen():
                for i in range(3):
                    yield f"data: {{\"chunk\": {i}}}\n\n"
                    await asyncio.sleep(0.01)
                yield ('data: {"id":"cmpl-1","object":"chat.completion.chunk","choices":[],'
                       '"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}\n\n')
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return {
            "id": "cmpl-1",
            "object": "chat.completion",
            "model": body["model"],
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hello from upstream"}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    @app.post("/v1/{path:path}")
    async def other(request: Request, path: str):
        body = await request.json()
        return {"model": body["model"], "path": path, "forwarded": True}

    return app


@async_fixture
async def stack(gpu, runner):
    cfg = make_config()
    manager = ModelManager(cfg, gpu=gpu, runner=runner)
    upstream = make_upstream_app()
    proxy = Proxy("http://127.0.0.1:30000")
    proxy._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=upstream), timeout=None)
    app = create_app(config=cfg, manager=manager, proxy=proxy)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://manager") as client:
            yield client, manager, runner, gpu


async def test_chat_completion_start_on_demand(stack):
    client, manager, runner, gpu = stack
    gpu.free_g = 40
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "qwen"
    assert data["choices"][0]["message"]["content"] == "hello from upstream"
    assert manager.state.value == "running"
    assert manager.current_model == "qwen"
    assert runner.starts == ["qwen"]
    assert manager.active_requests == 0


async def test_second_request_routes_without_restart(stack):
    client, manager, runner, gpu = stack
    gpu.free_g = 40
    await client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    await client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    assert runner.starts == ["qwen"]
    assert runner.stops == 0


async def test_get_models_lists_registry(stack):
    client, *_ = stack
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}
    assert ids == {"qwen", "gemma"}


async def test_unknown_model_404(stack):
    client, *_ = stack
    resp = await client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "model_not_found"


async def test_missing_model_400(stack):
    client, *_ = stack
    resp = await client.post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request"


async def test_non_json_body_400(stack):
    client, *_ = stack
    resp = await client.post("/v1/chat/completions", content=b"not json", headers={"content-type": "text/plain"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request"


async def test_insufficient_vram_503(stack):
    client, manager, runner, gpu = stack
    gpu.free_g = 10
    resp = await client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    assert resp.status_code == 503
    err = resp.json()["error"]
    assert err["type"] == "insufficient_gpu_memory"
    assert "34.0 GiB" in err["message"]
    assert manager.state.value == "stopped"
    assert runner.starts == []


async def test_startup_timeout_503(stack):
    client, manager, runner, gpu = stack
    gpu.free_g = 40
    runner.health_mode = "timeout"
    resp = await client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "sglang_startup_timeout"
    assert manager.state.value == "stopped"


async def test_switch_via_api(stack):
    client, manager, runner, gpu = stack
    gpu.free_g = 40
    await client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    resp = await client.post("/v1/chat/completions", json={"model": "gemma", "messages": []})
    assert resp.status_code == 200
    assert manager.current_model == "gemma"
    assert runner.starts == ["qwen", "gemma"]
    assert runner.stops == 1


async def test_streaming_proxy(stack):
    client, manager, runner, gpu = stack
    gpu.free_g = 40
    async with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "qwen", "messages": [], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = b"".join([chunk async for chunk in resp.aiter_bytes()])
    assert b"chunk" in body and b"[DONE]" in body
    # accounting: only after the SSE stream fully closed is the request "done"
    assert manager.active_requests == 0
    assert manager.state.value == "running"


async def test_gateway_status(stack):
    client, manager, runner, gpu = stack
    gpu.free_g = 40
    await client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    resp = await client.get("/gateway/status")
    data = resp.json()
    assert data["state"] == "running"
    assert data["model"] == "qwen"
    assert data["active_requests"] == 0
    assert data["idle_timeout_seconds"] == 3600
    assert data["gpu"]["free_gib"] == 40
    assert data["gpu"]["total_gib"] == 48


async def test_gateway_models(stack):
    client, *_ = stack
    resp = await client.get("/gateway/models")
    data = resp.json()
    by_name = {m["name"]: m for m in data["models"]}
    assert set(by_name) == {"qwen", "gemma"}
    assert by_name["qwen"]["required_vram_gib"] == 30
    assert by_name["qwen"]["loaded"] is False


async def test_gateway_stop_then_restart_on_demand(stack):
    client, manager, runner, gpu = stack
    gpu.free_g = 40
    await client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    resp = await client.post("/gateway/stop")
    assert resp.status_code == 200
    assert resp.json()["state"] == "stopped"
    resp = await client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    assert resp.status_code == 200
    assert runner.starts == ["qwen", "qwen"]
    assert runner.stops == 1


async def test_gateway_stop_refused_when_busy(stack):
    client, manager, runner, gpu = stack
    gpu.free_g = 40
    await client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    manager.request_started()
    resp = await client.post("/gateway/stop")
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "model_switch_busy"
    assert manager.state.value == "running"
    manager.request_finished()


async def test_gateway_force_stop_when_busy(stack):
    client, manager, runner, gpu = stack
    gpu.free_g = 40
    await client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    manager.request_started()  # in-flight request
    resp = await client.post("/gateway/force-stop")
    assert resp.status_code == 200
    assert resp.json()["state"] == "stopped"
    assert manager.state.value == "stopped"
    assert runner.stops == 1
    manager.request_finished()


async def test_gateway_stop_cancels_starting(stack):
    """Idle stop during a pending startup: accepted, startup aborted,
    the in-flight request fails cleanly."""
    client, manager, runner, gpu = stack
    gpu.free_g = 40
    runner.health_delay = 0.3
    task = asyncio.create_task(
        client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    )
    await asyncio.sleep(0.05)
    assert manager.state.value == "starting"
    resp = await client.post("/gateway/stop")
    assert resp.status_code == 200
    assert resp.json()["state"] == "stopped"
    r = await task
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "sglang_unavailable"
    assert manager.state.value == "stopped"


async def test_gateway_preload(stack):
    client, manager, runner, gpu = stack
    gpu.free_g = 40
    resp = await client.post("/gateway/preload/gemma")
    assert resp.status_code == 200
    assert resp.json()["model"] == "gemma"
    assert manager.state.value == "running"
    assert runner.starts == ["gemma"]
    assert manager.active_requests == 0


async def test_other_v1_post_paths_forwarded(stack):
    """Any /v1/* POST is gated and forwarded transparently."""
    client, manager, runner, gpu = stack
    gpu.free_g = 40
    resp = await client.post("/v1/completions", json={"model": "qwen", "prompt": "x"})
    assert resp.status_code == 200
    assert resp.json()["model"] == "qwen"
    assert manager.current_model == "qwen"
