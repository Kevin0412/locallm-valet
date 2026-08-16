"""Usage recording / aggregation tests: recorder unit tests + API integration
(plain & streaming token accounting, dashboard page)."""

import httpx
import pytest
from fastapi import Request
from pytest_asyncio import fixture as async_fixture

from locallm_valet.api import create_app
from locallm_valet.errors import InvalidRequest
from locallm_valet.manager import ModelManager
from locallm_valet.proxy import Proxy
from locallm_valet.usage import SseUsageScanner, UsageRecorder, extract_usage_from_json

from .conftest import FakeMemory, FakeRunner, make_config
from .test_api import make_upstream_app


# ------------------------------------------------------------- unit: recorder

def test_record_and_query(tmp_path):
    r = UsageRecorder(str(tmp_path / "u.db"))
    r.record(model="qwen", endpoint="/v1/chat/completions", status=200,
             prompt_tokens=10, completion_tokens=5, ts_epoch=1000)
    r.record(model="qwen", endpoint="/v1/chat/completions", stream=True, status=200,
             prompt_tokens=7, completion_tokens=3, ts_epoch=2000)
    r.record(model="gemma", endpoint="/v1/completions", status=200,
             prompt_tokens=1, completion_tokens=1, ts_epoch=3000)

    q = r.query()
    assert q["summary"]["requests"] == 3
    assert q["summary"]["prompt_tokens"] == 18
    assert q["summary"]["completion_tokens"] == 9
    assert q["summary"]["total_tokens"] == 27
    assert [m["model"] for m in q["by_model"]] == ["qwen", "gemma"]
    assert q["by_model"][0]["requests"] == 2
    assert len(q["recent"]) == 3
    assert q["recent"][0]["model"] == "gemma"  # newest first
    assert q["recent"][2]["stream"] is False

    assert r.query(model="qwen")["summary"]["requests"] == 2
    assert r.query(since=2500)["summary"]["requests"] == 1
    assert r.query(until=1500)["summary"]["requests"] == 1
    r.close()


def test_series_hour_and_day(tmp_path):
    r = UsageRecorder(str(tmp_path / "u.db"))
    r.record(model="q", endpoint="/x", prompt_tokens=1, ts_epoch=0)
    r.record(model="q", endpoint="/x", prompt_tokens=2, ts_epoch=3599)
    r.record(model="q", endpoint="/x", prompt_tokens=4, ts_epoch=3600)

    q = r.query(group_by="hour")
    assert len(q["series"]) == 2
    assert q["series"][0]["requests"] == 2
    assert q["series"][0]["prompt_tokens"] == 3
    assert q["series"][1]["requests"] == 1
    assert q["series"][0]["bucket_epoch"] == 0

    qd = r.query(group_by="day")
    assert len(qd["series"]) == 1
    assert qd["series"][0]["requests"] == 3
    r.close()


def test_extract_usage_from_json():
    body = b'{"id":"x","usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}'
    assert extract_usage_from_json(body) == {
        "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7,
    }
    assert extract_usage_from_json(b'{"choices": []}') is None
    assert extract_usage_from_json(b"not json") is None
    assert extract_usage_from_json(b"") is None


def test_sse_scanner_fragmented_frames():
    """Frames split across arbitrary chunk boundaries must still be parsed."""
    stream = (
        b'data: {"id":"1","choices":[{"delta":{"content":"a"}}]}\n\n'
        b'data: {"id":"1","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}\n\n'
        b"data: [DONE]\n\n"
    )
    s = SseUsageScanner()
    for i in range(0, len(stream), 3):  # deliberate fragmentation
        s.feed(stream[i:i + 3])
    assert s.finish() == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}

    s2 = SseUsageScanner()
    s2.feed(b'data: {"choices":[]}\n\ndata: [DONE]\n\n')
    assert s2.finish() is None


def test_sse_scanner_last_usage_wins():
    s = SseUsageScanner()
    s.feed(b'data: {"usage":{"prompt_tokens":1}}\n\n')
    s.feed(b'data: {"usage":{"prompt_tokens":9}}\n\n')
    assert s.finish() == {"prompt_tokens": 9}


def test_recorder_in_memory():
    r = UsageRecorder(":memory:")
    r.record(model="q", endpoint="/x", prompt_tokens=2)
    assert r.query()["summary"]["requests"] == 1
    r.close()


# ------------------------------------------------------------- API integration

@async_fixture
async def stack(memory, runner):
    cfg = make_config()
    manager = ModelManager(cfg, memory=memory, runner=runner)
    upstream = make_upstream_app()
    proxy = Proxy("http://127.0.0.1:30000")
    proxy._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=upstream), timeout=None)
    app = create_app(config=cfg, manager=manager, proxy=proxy)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://manager") as client:
            yield client, manager, runner, memory


async def test_plain_request_recorded(stack):
    client, manager, runner, memory = stack
    memory.free_g = 40
    await client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    data = (await client.get("/gateway/usage")).json()
    assert data["summary"]["requests"] == 1
    assert data["summary"]["prompt_tokens"] == 10
    assert data["summary"]["completion_tokens"] == 5
    assert data["summary"]["total_tokens"] == 15
    rec = data["recent"][0]
    assert rec["model"] == "qwen"
    assert rec["endpoint"] == "/v1/chat/completions"
    assert rec["stream"] is False
    assert rec["status"] == 200
    assert rec["duration_ms"] >= 0


async def test_stream_request_recorded(stack):
    client, manager, runner, memory = stack
    memory.free_g = 40
    async with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "qwen", "messages": [], "stream": True},
    ) as resp:
        body = b"".join([c async for c in resp.aiter_bytes()])
    assert b"[DONE]" in body
    data = (await client.get("/gateway/usage")).json()
    rec = data["recent"][0]
    assert rec["stream"] is True
    assert rec["prompt_tokens"] == 7   # from the SSE final usage frame
    assert rec["completion_tokens"] == 3
    assert rec["status"] == 200
    assert data["summary"]["requests"] == 1


async def test_usage_model_filter_and_grouping(stack):
    client, manager, runner, memory = stack
    memory.free_g = 40
    await client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    await client.post("/v1/chat/completions", json={"model": "gemma", "messages": []})
    q = (await client.get("/gateway/usage", params={"model": "qwen"})).json()
    assert q["summary"]["requests"] == 1
    assert q["by_model"][0]["model"] == "qwen"
    q = (await client.get("/gateway/usage", params={"group_by": "hour"})).json()
    assert len(q["series"]) == 1
    assert q["series"][0]["requests"] == 2
    q = (await client.get("/gateway/usage", params={"group_by": "none"})).json()
    assert q["series"] == []


async def test_usage_bad_time_param(stack):
    client, *_ = stack
    resp = await client.get("/gateway/usage", params={"since": "not-a-time"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request"


async def test_dashboard_page(stack):
    client, *_ = stack
    resp = await client.get("/gateway/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "用量看板" in resp.text


async def test_usage_disabled_no_endpoints():
    cfg = make_config()
    cfg.usage.enabled = False
    app = create_app(config=cfg, manager=ModelManager(cfg, memory=FakeMemory(), runner=FakeRunner()))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://manager") as client:
        assert (await client.get("/gateway/usage")).status_code == 404
        assert (await client.get("/gateway/dashboard")).status_code == 404
