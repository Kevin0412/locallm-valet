"""Durable benchmark job tests: the persisted question queue (benchmark_job +
benchmark_job_item) and the id-driven start/pause/resume/stop lifecycle."""

import time

from locallm_valet.benchmark.schema import BenchmarkItem, BenchmarkResult
from locallm_valet.benchmark.store import BenchmarkStore, item_from_row


def _item(n, category="fact"):
    return BenchmarkItem(item_id=f"i{n}", category=category, question="Q",
                         ground_truth="A", choices=[])


# ------------------------------------------------------------- store job queue

def test_store_job_queue(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    items = [_item(n) for n in range(5)]
    assert store.create_job(job_id="j1", dataset="d", models=["m1", "m2"],
                            items=items) == 10
    job = store.get_job("j1")
    assert job["status"] == "running"
    assert job["total_items"] == 10
    assert job["models"] == ["m1", "m2"]

    claimed = store.claim_items("j1", "m1", limit=2)
    assert len(claimed) == 2
    assert all(c["status"] == "in_progress" for c in claimed)
    assert [c["item_id"] for c in claimed] == ["i0", "i1"]

    store.finish_item("j1", claimed[0]["id"],
                      {"is_correct": True, "model_name": "m1", "raw_response": "x"})
    assert store.get_job("j1")["done_items"] == 1

    store.fail_item("j1", claimed[1]["id"], "boom")
    assert store.job_items("j1", model_name="m1", status="failed")[0]["item_id"] == "i1"

    # resume helper: in-progress items from an interrupted run return to pending
    store.claim_items("j1", "m1", limit=1)          # i2 becomes in_progress
    assert store.reset_in_progress("j1", "m1") == 1
    pending = store.job_items("j1", model_name="m1", status="pending")
    assert {p["item_id"] for p in pending} == {"i2", "i3", "i4"}

    assert store.list_jobs()[0]["id"] == "j1"
    store.close()


def test_item_from_row_roundtrip(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    items = [_item(0)]
    store.create_job(job_id="j", dataset="d", models=["m"], items=items)
    row = store.job_items("j")[0]
    rebuilt = item_from_row(row)
    assert rebuilt.item_id == "i0"
    assert rebuilt.ground_truth == "A"
    store.close()


# ------------------------------------------------------------- job lifecycle

def test_job_lifecycle_start_resume(monkeypatch, tmp_path):
    from locallm_valet.benchmark import job as job_module
    from locallm_valet.benchmark import runner as runner_module

    job_module.configure_store(str(tmp_path / "b.db"))

    def fake_run(item, **kw):
        r = BenchmarkResult(item=item, model_name=kw.get("model_name", ""))
        r.is_correct = True
        r.raw_response = "ok"
        r.latency_ms = 1.0
        return r

    def fake_probe(**kw):
        return None

    monkeypatch.setattr(job_module, "run_single_item", fake_run)
    monkeypatch.setattr(runner_module, "probe_single_request_stats", fake_probe)
    monkeypatch.setattr(runner_module, "save_speed", lambda *a, **k: None)

    job = job_module.start_job(dataset="smoke", models=["m1"], max_tokens=16,
                               concurrency=2, enable_thinking=False,
                               base_url="http://127.0.0.1:9/v1")
    assert job.job_id

    state = job.status()
    for _ in range(200):
        state = job.status()
        if state["state"] == "done":
            break
        time.sleep(0.05)
    assert state["state"] == "done"
    assert state["total_items"] == 6          # smoke = 6 items
    assert state["done_items"] == 6

    store = job_module.get_store()
    assert len(store.job_items(job.job_id, model_name="m1", status="done")) == 6
    # Results are mirrored into the dashboard-facing benchmark_result table.
    agg = store.query_aggregate()
    assert agg["smoke"]["m1"]["t"] == 6

    # A paused job stays resumable via its id.
    job.control.paused = True
    assert job.status()["state"] == "paused"
