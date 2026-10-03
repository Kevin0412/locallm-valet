"""Typed decision routing, lifecycle accounting and native scorer input contracts."""

import copy
import json

import httpx
import pytest
from fastapi import FastAPI, Request

from locallm_valet.api import create_app
from locallm_valet.config import ConfigError, load_config
from locallm_valet.jev_api import DecisionService, InputError, UpstreamError, checked_result
from locallm_valet.manager import ModelManager
from locallm_valet.proxy import Proxy
from locallm_valet.usage import UsageRecorder

from .conftest import FakeMemory, FakeRunner, make_config


def test_protocol_parsing_and_capabilities(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "models:\n  jev:\n    path: /local/model\n    backend: {api_protocol: systemone}\n"
    )
    cfg = load_config(config)
    assert cfg.models["jev"].capabilities == [
        "typed_decision",
        "choice",
        "noul",
        "score",
    ]
    config.write_text(config.read_text().replace("systemone", "unknown"))
    with pytest.raises(ConfigError, match="api_protocol"):
        load_config(config)


@pytest.mark.asyncio
async def test_typed_routing_auth_and_usage_are_preserved():
    cfg = make_config()
    cfg.benchmark.enabled = False
    cfg.server.api_keys = ["test-key"]
    cfg.models["qwen"].backend.api_protocol = "systemone"
    runner = FakeRunner()
    manager = ModelManager(cfg, memory=FakeMemory(), runner=runner)
    recorder = UsageRecorder(":memory:")
    upstream = FastAPI()
    received = []

    @upstream.post("/v1/systemone")
    async def evaluate(request: Request):
        payload = await request.json()
        received.append(payload)
        return {
            "model": payload["model"],
            "answers": {"custom_name": {"type": "noul", "noul": 0.8}},
            "usage": {"input_tokens": 17, "output_tokens": 0},
        }

    proxy = Proxy("http://backend")
    await proxy._client.aclose()
    proxy._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream), base_url="http://backend"
    )
    app = create_app(cfg, manager=manager, proxy=proxy, recorder=recorder)
    body = {
        "model": "qwen",
        "state": "Context",
        "questions": {
            "custom_name": {"type": "noul", "instructions": "Is it relevant?"}
        },
    }
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.post("/v1/systemone", json=body)).status_code == 401
            assert not runner.starts
            headers = {"Authorization": "Bearer test-key"}
            response = await client.post("/v1/systemone", json=body, headers=headers)
            assert response.status_code == 200
            assert response.json()["answers"]["custom_name"]["noul"] == 0.8
            assert received == [body]
            assert manager.active_requests == 0
            registry = (await client.get("/v1/models", headers=headers)).json()["data"]
            assert (
                next(row for row in registry if row["id"] == "qwen")["api_protocol"]
                == "systemone"
            )
        assert recorder.query()["summary"]["prompt_tokens"] == 17
    finally:
        await manager.shutdown()
        await proxy.aclose()
        recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,model,stream",
    [
        ("/v1/chat/completions", "qwen", False),
        ("/v1/systemone", "qwen", True),
        ("/v1/systemone", "gemma", False),
    ],
)
async def test_wrong_protocol_is_rejected_before_model_loading(path, model, stream):
    cfg = make_config()
    cfg.usage.enabled = False
    cfg.benchmark.enabled = False
    cfg.models["qwen"].backend.api_protocol = "systemone"
    runner = FakeRunner()
    manager = ModelManager(cfg, memory=FakeMemory(), runner=runner)
    proxy = Proxy("http://backend")
    app = create_app(cfg, manager=manager, proxy=proxy)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(path, json={"model": model, "stream": stream})
            assert response.status_code == 400
            assert not runner.starts
    finally:
        await proxy.aclose()


def scorer(body):
    question = body["questions"]["answer"]
    kind = question["type"]
    if kind == "noul":
        answer = {"type": kind, "noul": 0.8}
    elif kind == "score":
        answer = {"type": kind, "probabilities": {"0": 0.25, "1": 0.75}, "score": 0.75}
    else:
        keys = list(question["criteria"])
        answer = {
            "type": kind,
            "probabilities": {keys[0]: 0.25, keys[1]: 0.75},
            "choice": keys[1],
        }
    return {
        "model": "native",
        "answers": {"answer": answer},
        "usage": {"input_tokens": 5, "output_tokens": 0},
    }


def test_named_questions_compile_and_usage_aggregates_without_mutating_input():
    seen = []

    def predict(body):
        seen.append(body)
        return scorer(body)

    service = DecisionService(
        "registered", "native", lambda body: {"supported": True}, predict, 16
    )
    body = {
        "model": "registered",
        "state": {"message": "hello"},
        "questions": {
            "is_valid": {"type": "noul", "instructions": "Relevant?"},
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"a": None, "b": "Billing"},
            },
            "rating": {
                "type": "score",
                "instructions": "Rate",
                "criteria": ["Bad", "Good"],
            },
        },
    }
    original = copy.deepcopy(body)
    result = service.evaluate(body)
    assert body == original
    assert result["model"] == "registered"
    assert set(result["answers"]) == set(body["questions"])
    assert result["usage"] == {"input_tokens": 15, "output_tokens": 0}
    assert all(json.loads(row["state"]) == body["state"] for row in seen)


def test_later_invalid_question_prevents_all_inference():
    calls = []
    service = DecisionService(
        "registered", "native", lambda body: {"supported": False}, calls.append, 16
    )
    with pytest.raises(ValueError):
        service.evaluate(
            {
                "model": "registered",
                "state": "A state",
                "questions": {
                    "valid": {"type": "noul", "instructions": "Good?"},
                    "invalid": {
                        "type": "score",
                        "instructions": "Rate",
                        "criteria": ["Only one"],
                    },
                },
            }
        )
    assert not calls


def test_rejections_name_their_reason_and_split_caller_from_upstream_errors():
    """A 422 must be explainable: caller mistakes are InputError, a malformed
    upstream answer is UpstreamError (so the service can answer 500)."""

    def predict(body):
        return scorer(body)

    service = DecisionService(
        "registered", "native", lambda body: {"supported": True}, predict, 16
    )
    with pytest.raises(InputError, match="Choice requires 2 to 16 options"):
        service.evaluate(
            {
                "model": "registered",
                "state": "A state",
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {"only": "One option"},
                    }
                },
            }
        )
    with pytest.raises(InputError, match="1 to 32 named questions"):
        service.evaluate(
            {"model": "registered", "state": "A state", "questions": {}}
        )

    def too_long(body):
        raise ValueError("Input exceeds token limit; no truncation")

    with pytest.raises(InputError, match="Input exceeds token limit"):
        DecisionService("registered", "native", too_long, predict, 16).evaluate(
            {
                "model": "registered",
                "state": "A state",
                "questions": {"q": {"type": "noul", "instructions": "Good?"}},
            }
        )
    with pytest.raises(UpstreamError, match="probability mass"):
        checked_result(
            {
                "model": "native",
                "answers": {"answer": {"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.9}}},
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
            "native",
            "choice",
            {"a": "A", "b": "B"},
        )
    with pytest.raises(UpstreamError, match="model identity"):
        checked_result({"model": "other"}, "native", "noul", {"false": "No", "true": "Yes"})


@pytest.mark.asyncio
async def test_benchmark_page_and_run_endpoint_exclude_typed_models():
    """Chat benchmarks cannot drive a typed-decision model: it must neither be
    offered on the page nor accepted by the runner."""
    cfg = make_config()
    cfg.benchmark.enabled = True
    cfg.benchmark.db_path = ":memory:"
    cfg.models["qwen"].backend.api_protocol = "systemone"
    runner = FakeRunner()
    manager = ModelManager(cfg, memory=FakeMemory(), runner=runner)
    proxy = Proxy("http://backend")
    app = create_app(cfg, manager=manager, proxy=proxy)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            page = (await client.get("/gateway/benchmark")).text
            assert "gemma" in page
            assert "qwen" not in page
            response = await client.post(
                "/gateway/benchmark/run", json={"dataset": "mmlu", "models": ["qwen"]}
            )
            assert response.status_code == 400
            assert "text-generation" in response.json()["error"]["message"]
            assert not runner.starts
    finally:
        await manager.shutdown()
        await proxy.aclose()
