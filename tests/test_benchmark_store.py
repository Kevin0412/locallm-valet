"""SQLite benchmark store tests: result recording/dedup, aggregation, speed
records, and the JSONL backfill (incl. the legacy-file superseded rule)."""

import json

from locallm_valet.benchmark.schema import BenchmarkItem, BenchmarkResult
from locallm_valet.benchmark.store import BenchmarkStore, backfill_jsonl


def _result(model, item_id, dataset, *, correct, category="fact", thinking=True):
    item = BenchmarkItem(item_id=item_id, category=category,
                         question="Q", ground_truth="A", choices=[])
    r = BenchmarkResult(item=item, model_name=model)
    r.thinking = thinking
    r.is_correct = correct
    return r


def test_record_dedup_and_aggregate(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    # Two items for modelA; re-record item1 (idempotent, still one row).
    store.record_results("modelA", "mmlu", [
        _result("modelA", "i1", "mmlu", correct=True),
        _result("modelA", "i2", "mmlu", correct=False),
    ])
    store.record_results("modelA", "mmlu", [
        _result("modelA", "i1", "mmlu", correct=True),
    ])

    agg = store.query_aggregate()
    a = agg["mmlu"]["modelA"]
    assert a["t"] == 2
    assert a["c"] == 1
    assert a["thinking"] == "thinking"
    assert a["cat"]["fact"] == [2, 1]

    # thinking mode is part of the dedup key: the same item in a different
    # mode is a distinct row.
    store.record_results("modelA", "mmlu", [
        _result("modelA", "i1", "mmlu", correct=False, thinking=False),
    ])
    agg = store.query_aggregate()
    a = agg["mmlu"]["modelA"]
    assert a["t"] == 3
    assert a["c"] == 1
    assert a["thinking"] == "mixed"
    store.close()


def test_record_speed_and_load(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    store.record_speed("modelA", {"prefill_tps": 6000.0, "decode_tps": 150.0,
                                  "samples": 8, "steady": 6, "ts": "2026-01-01T00:00:00"})
    store.record_speed("modelA", {"prefill_tps": 6100.0, "decode_tps": 155.0})  # upsert
    speeds = store.load_speeds()
    assert speeds["modelA"]["prefill_tps"] == 6100.0
    assert speeds["modelA"]["decode_tps"] == 155.0
    assert speeds["modelA"].get("ts")  # retained from the first record
    store.close()


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _rec(model, item_id, correct, *, thinking=True, category="fact"):
    return {
        "model_name": model, "item_id": item_id, "category": category,
        "question": "Q", "ground_truth": "A", "choices": [],
        "thinking": thinking, "is_correct": correct, "raw_response": "R",
        "extracted_answer": "A", "score_detail": "",
        "latency_ms": 10.0, "tps": 5.0, "ttft_ms": 2.0,
        "prompt_tokens": 10, "completion_tokens": 5,
    }


def test_backfill_supersedes_legacy(tmp_path):
    """A per-model file is authoritative; legacy same-(model,dataset) rows are
    skipped, but other models in the legacy file are still ingested."""
    d = tmp_path / "results"
    d.mkdir()
    _write_jsonl(d / "modelA_mmlu_results.jsonl", [
        _rec("modelA", "i1", True), _rec("modelA", "i2", False),
    ])
    # Legacy file: modelA (superseded → skipped) + modelB (kept).
    _write_jsonl(d / "mmlu_results.jsonl", [
        _rec("modelA", "i0", True),   # same (model,dataset) as per-model file → skip
        _rec("modelB", "i7", True),
    ])

    store = BenchmarkStore(str(tmp_path / "b.db"))
    counts = backfill_jsonl(store, str(d))
    assert counts["mmlu"] == 3  # 2 (modelA per-model) + 1 (modelB legacy)

    agg = store.query_aggregate()
    assert agg["mmlu"]["modelA"]["t"] == 2
    assert agg["mmlu"]["modelA"]["c"] == 1
    assert agg["mmlu"]["modelB"]["t"] == 1
    assert agg["mmlu"]["modelB"]["c"] == 1
    store.close()


def test_backfill_no_results_dir(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    assert backfill_jsonl(store, str(tmp_path / "nope")) == {}
    assert store.query_aggregate() == {}
    store.close()
