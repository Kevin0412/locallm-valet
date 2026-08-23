# -*- coding: utf-8 -*-
"""Regression tests for benchmark CLI params, persistence, BFCL AST scoring,
and thinking-mode defaults — every bug hit during the overnight runs."""
import argparse
import json

import pytest

from locallm_valet.benchmark.cli import _persist_model_results, build_subparser
from locallm_valet.benchmark.schema import BenchmarkItem, BenchmarkResult


def _parse(argv):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    build_subparser(sub)
    return p.parse_args(argv)


def test_thinking_defaults_on():
    for argv in (["benchmark", "run", "--model", "m"],
                 ["benchmark", "all"],
                 ["benchmark", "compare", "--models", "a", "b"]):
        assert _parse(argv).thinking is True, argv


def test_no_thinking_opt_out():
    assert _parse(["benchmark", "all", "--no-thinking"]).thinking is False


def test_skip_models_accepts_multiple():
    args = _parse(["benchmark", "all", "--skip-models", "a", "b", "c"])
    assert args.skip_models == ["a", "b", "c"]


def test_max_tokens_and_retries_defaults():
    a = _parse(["benchmark", "all"])
    assert a.max_tokens == 64000
    assert a.retries == 2
    b = _parse(["benchmark", "all", "--max-tokens", "123", "--retries", "5"])
    assert b.max_tokens == 123 and b.retries == 5


def test_persist_new_overwrites_old(tmp_path):
    """Fresh results for the same (model, item_id) must replace stale rows."""
    def mk(correct, thinking):
        it = BenchmarkItem(item_id="q1", category="fact", question="q", ground_truth="B")
        r = BenchmarkResult(item=it, model_name="m", raw_response="B")
        r.is_correct, r.thinking = correct, thinking
        return [r]

    _persist_model_results(mk(True, False), "t", str(tmp_path))     # stale no-think
    _persist_model_results(mk(False, True), "t", str(tmp_path))     # fresh think
    rows = [json.loads(l) for l in (tmp_path / "t_results.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["thinking"] is True and rows[0]["is_correct"] is False


def test_persist_failure_never_clobbers_good(tmp_path):
    """Regression: HTTP-error / empty records from a broken run must not
    overwrite previously-good answers for the same (model, item_id)."""
    def mk(raw, item_id="q1", model="m", correct=None):
        it = BenchmarkItem(item_id=item_id, category="fact", question="q", ground_truth="B")
        r = BenchmarkResult(item=it, model_name=model, raw_response=raw)
        r.is_correct = correct
        return [r]

    # good answer first
    _persist_model_results(mk("B", correct=True), "t", str(tmp_path))
    # broken rerun: HTTP error for the same item — must not replace it
    _persist_model_results(mk("[HTTP 500] boom", correct=False), "t", str(tmp_path))
    rows = [json.loads(l) for l in (tmp_path / "t_results.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["is_correct"] is True
    assert "HTTP" not in str(rows[0]["raw_response"])


def test_persist_keeps_other_models(tmp_path):
    """A run for one model must leave other models' rows untouched."""
    def mk(model, item_id, raw="ok", correct=True):
        it = BenchmarkItem(item_id=item_id, category="fact", question="q", ground_truth="B")
        r = BenchmarkResult(item=it, model_name=model, raw_response=raw)
        r.is_correct = correct
        return [r]

    _persist_model_results(mk("m-a", "q1") + mk("m-b", "q1"), "t", str(tmp_path))
    # rerun only m-b with a fresh (different) answer for the same item
    _persist_model_results(mk("m-b", "q1", raw="C", correct=False), "t", str(tmp_path))
    rows = [json.loads(l) for l in (tmp_path / "t_results.jsonl").read_text().splitlines() if l.strip()]
    by = {r["model_name"]: r for r in rows}
    assert set(by) == {"m-a", "m-b"}
    assert by["m-a"]["is_correct"] is True          # untouched
    assert by["m-b"]["is_correct"] is False         # updated
    assert by["m-b"]["raw_response"] == "C"


def test_bfcl_ast_scoring():
    from locallm_valet.benchmark.scorer import score_result
    fn = {"name": "get_time", "parameters": {"type": "object", "properties": {
        "tz": {"type": "string"}}, "required": ["tz"]}}
    item = BenchmarkItem(item_id="s", category="bfcl_simple", question="q",
                         ground_truth="(AST)", meta={"functions": [fn], "check": "ast"})

    def call(name, args, n=1):
        r = BenchmarkResult(item=item, model_name="m")
        r.tool_calls = [{"function": {"name": name, "arguments": json.dumps(args)}} for _ in range(n)]
        score_result(r)
        return r

    assert call("get_time", {"tz": "UTC"}).is_correct is True
    assert call("get_time", {"tz": "UTC"}, n=2).is_correct is False      # simple: 1 call
    assert call("nope", {"tz": "UTC"}).is_correct is False               # unknown fn
    assert call("get_time", {}).is_correct is False                      # missing required
    assert call("get_time", {"tz": 123}).is_correct is False             # wrong type


def test_bfcl_parallel_allows_multiple():
    from locallm_valet.benchmark.scorer import score_result
    fn = {"name": "f", "parameters": {"type": "object", "properties": {}, "required": []}}
    item = BenchmarkItem(item_id="p", category="bfcl_parallel", question="q",
                         ground_truth="(AST)", meta={"functions": [fn], "check": "ast"})
    r = BenchmarkResult(item=item, model_name="m")
    r.tool_calls = [{"function": {"name": "f", "arguments": {}}} for _ in range(3)]
    score_result(r)
    assert r.is_correct is True
