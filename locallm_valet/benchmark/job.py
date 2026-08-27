# -*- coding: utf-8 -*-
"""Durable, id-driven benchmark job scheduler.

Every run gets a job id and its full per-model question queue is persisted in
SQLite (``benchmark_job`` + ``benchmark_job_item``).  Items are executed with a
small thread pool so an answer is acknowledged while the next item's request is
already in flight (pipelining).  A job can be paused / resumed / stopped at any
time via its id; on resume the pending queue is re-read from the database, so a
job survives a server restart.

Benchmark requests are tagged ``x-locallm-benchmark: <job_id>`` so the gateway
can identify them.  When external (non-benchmark) requests are active the
scheduler throttles its concurrency down to 1 — downgrade-priority preemption:
benchmark work yields the backend to real requests instead of pausing.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .dataset import get_dataset
from .runner import JobControl, run_single_item
from .scorer import score_result
from .store import BenchmarkStore, item_from_row
from .schema import BenchmarkItem, BenchmarkResult

logger = logging.getLogger("locallm_valet.benchmark.job")


# ---------------------------------------------------------------------------
# SQLite store integration
# ---------------------------------------------------------------------------

_store_path: Optional[str] = None
_store: Optional[BenchmarkStore] = None
_store_lock = threading.Lock()


def configure_store(db_path: Optional[str]) -> None:
    """Point the job's SQLite store at ``db_path`` (called from create_app)."""
    global _store_path, _store
    with _store_lock:
        _store_path = db_path
        _store = None


def get_store() -> BenchmarkStore:
    """Lazily open the process-wide benchmark store (thread-safe)."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = BenchmarkStore(_store_path)
    return _store


# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------

_JOBS: dict[str, "BenchmarkJob"] = {}
_JOB_LOCK = threading.Lock()
_LAST_JOB_ID: Optional[str] = None


@dataclass
class BenchmarkJob:
    """Live state for one durable benchmark job (registry entry)."""

    job_id: str
    dataset: str
    models: list[str]
    base_url: str
    api_key: str
    max_tokens: int
    sample: Optional[int]
    concurrency: int
    enable_thinking: bool
    store: BenchmarkStore
    slot_of: dict
    output_dir: str
    control: JobControl = field(default_factory=JobControl)
    _threads: list[threading.Thread] = field(default_factory=list)
    current_item: int = 0
    speeds: dict = field(default_factory=dict)   # {model: single-request tok/s}

    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def _live_state(self, row: dict) -> str:
        if self.control.cancel:
            return "stopped"
        if self.control.paused:
            return "paused"
        if self.running():
            return "running"
        return row.get("status", "idle")

    def status(self) -> dict:
        row = self.store.get_job(self.job_id) or {}
        return {
            "job_id": self.job_id,
            "state": self._live_state(row),
            "dataset": self.dataset,
            "models": self.models,
            "sample": self.sample,
            "concurrency": self.concurrency,
            "enable_thinking": self.enable_thinking,
            "current_model": row.get("current_model", ""),
            "current_item": self.current_item,
            "total_items": row.get("total_items", 0),
            "done_items": row.get("done_items", 0),
            "error": row.get("error", ""),
            "created_at": row.get("created_at"),
            "speeds": self.speeds,
        }


def current_job() -> BenchmarkJob:
    """Return the most recently started job, or a placeholder if none yet."""
    with _JOB_LOCK:
        if _LAST_JOB_ID and _LAST_JOB_ID in _JOBS:
            return _JOBS[_LAST_JOB_ID]
    return BenchmarkJob(
        job_id="", dataset="", models=[], base_url="", api_key="", max_tokens=256,
        sample=None, concurrency=1, enable_thinking=False, store=get_store(),
        slot_of={}, output_dir="benchmark_results",
    )


def get_job(job_id: str) -> Optional[BenchmarkJob]:
    with _JOB_LOCK:
        return _JOBS.get(job_id)


def _register(job: BenchmarkJob) -> None:
    global _LAST_JOB_ID
    with _JOB_LOCK:
        _JOBS[job.job_id] = job
        _LAST_JOB_ID = job.job_id


# ---------------------------------------------------------------------------
# Start / resume / pause / stop
# ---------------------------------------------------------------------------

def start_job(
    *,
    dataset: str,
    models: list[str],
    base_url: str,
    max_tokens: int,
    sample: Optional[int] = None,
    concurrency: int = 1,
    enable_thinking: bool = False,
    api_key: str = "",
    output_dir: str = "benchmark_results",
    slot_of: Optional[dict] = None,
) -> BenchmarkJob:
    """Start a durable benchmark job and persist its full question queue.

    Returns the (already registered) job; its id is in ``job.job_id``.
    """
    store = get_store()
    items = get_dataset(dataset, sample)
    job_id = uuid.uuid4().hex[:12]
    store.create_job(
        job_id=job_id, dataset=dataset, models=list(models), items=items,
        sample=sample, concurrency=concurrency, max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )
    job = BenchmarkJob(
        job_id=job_id, dataset=dataset, models=list(models), base_url=base_url,
        api_key=api_key, max_tokens=max_tokens, sample=sample,
        concurrency=concurrency, enable_thinking=enable_thinking, store=store,
        slot_of=slot_of or {}, output_dir=output_dir,
    )
    _register(job)
    _spawn(job)
    logger.info("benchmark job %s started: dataset=%s models=%s items=%d",
                job_id, dataset, models, len(items))
    return job


def pause_job(job_id: str) -> Optional[BenchmarkJob]:
    store = get_store()
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.control.paused = True
    store.update_job(job_id, status="paused")
    logger.info("benchmark job %s paused", job_id)
    return job


def resume_job(job_id: str, base_url: str, api_key: str) -> Optional[BenchmarkJob]:
    store = get_store()
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.control.paused = False
            job.control.cancel = False
            job.base_url = base_url or job.base_url
            job.api_key = api_key or job.api_key
            store.update_job(job_id, status="running")
            if not job.running():
                _spawn(job)
            return job
    # Not in the live registry (e.g. server restarted) — rebuild from storage.
    row = store.get_job(job_id)
    if row is None:
        return None
    job = _rebuild_from_row(row, base_url=base_url, api_key=api_key)
    _register(job)
    store.update_job(job_id, status="running")
    _spawn(job)
    logger.info("benchmark job %s resumed (rebuilt from storage)", job_id)
    return job


def stop_job(job_id: str) -> Optional[BenchmarkJob]:
    store = get_store()
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.control.cancel = True
            job.control.paused = False
    store.update_job(job_id, status="stopped")
    logger.info("benchmark job %s stopped", job_id)
    return job


def _rebuild_from_row(row: dict, base_url: str, api_key: str) -> BenchmarkJob:
    return BenchmarkJob(
        job_id=row["id"], dataset=row["dataset"], models=list(row["models"]),
        base_url=base_url, api_key=api_key, max_tokens=row.get("max_tokens", 256),
        sample=row.get("sample"), concurrency=row.get("concurrency", 1),
        enable_thinking=bool(row.get("enable_thinking")), store=get_store(),
        slot_of={}, output_dir="benchmark_results",
    )


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

def _spawn(job: BenchmarkJob) -> None:
    """Group models by slot and drive them from a single worker thread."""
    groups: dict[str, list[str]] = defaultdict(list)
    for model_name in job.models:
        groups[job.slot_of.get(model_name, "cpu")].append(model_name)

    t = threading.Thread(
        target=_driver, args=(job, groups),
        name=f"benchmark-job-{job.job_id}", daemon=True,
    )
    t.start()
    job._threads = [t]
    logger.info("benchmark job %s: started driver thread (%d slot(s))",
                job.job_id, len(groups))


def _driver(job: BenchmarkJob, groups: dict[str, list[str]]) -> None:
    """Start one worker per slot, join them all, then settle the job status."""
    slot_threads: list[threading.Thread] = []
    for slot_name, slot_models in groups.items():
        st = threading.Thread(
            target=_slot_worker, args=(job, slot_models),
            name=f"benchmark-job-{job.job_id}-{slot_name}", daemon=True,
        )
        st.start()
        slot_threads.append(st)
        logger.info("benchmark job %s: slot %s started (%d models)",
                    job.job_id, slot_name, len(slot_models))

    for st in slot_threads:
        st.join()

    # A slot worker already recorded an error (with cancel set) — keep it.
    row = job.store.get_job(job.job_id) or {}
    if row.get("status") == "error":
        return
    if job.control.cancel:
        job.store.update_job(job.job_id, status="stopped")
    else:
        job.store.update_job(job.job_id, status="done")
        logger.info("benchmark job %s finished (done)", job.job_id)


def _slot_worker(job: BenchmarkJob, models: list[str]) -> None:
    """Run one slot's models serially (the slot backend holds one at a time)."""
    try:
        for model_name in models:
            if job.control.cancel:
                return
            job.store.update_job(job.job_id, current_model=model_name)
            logger.info("benchmark job %s: model=%s", job.job_id, model_name)
            _run_model(job, model_name)
            if job.control.cancel:
                return
            _probe_speed(job, model_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("benchmark job %s slot failed: %s", job.job_id, exc)
        job.store.update_job(job.job_id, status="error", error=str(exc))
        job.control.cancel = True
        return


def _run_model(job: BenchmarkJob, model_name: str) -> None:
    """Run every pending item for one model with a pipelined thread pool."""
    from concurrent.futures import ThreadPoolExecutor

    store = job.store
    # On (re)start/resume, any in-progress items from an interrupted run are
    # returned to pending so they are re-claimed (never silently skipped).
    store.reset_in_progress(job.job_id, model_name)

    n_workers = max(1, int(job.concurrency))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_item_worker, job, model_name) for _ in range(n_workers)]
        for f in futures:
            f.result()


def _item_worker(job: BenchmarkJob, model_name: str) -> None:
    """Continuously claim and run items until none remain for this model."""
    store = job.store
    while True:
        job.control.wait_if_paused()
        if job.control.cancel:
            return
        rows = store.claim_items(job.job_id, model_name, limit=1)
        if not rows:
            return
        row = rows[0]
        if job.control.cancel:
            return
        job.current_item = row["ord"]
        item = item_from_row(row)
        # Downgrade-priority preemption: yield to external (non-benchmark)
        # requests by backing off while any are active.
        try:
            from ..priority import external_active
            if external_active() > 0:
                time.sleep(0.05)
        except Exception:  # noqa: BLE001
            pass
        result = run_single_item(
            item,
            model_name=model_name,
            base_url=job.base_url,
            api_key=job.api_key,
            max_tokens=job.max_tokens,
            control=job.control,
            enable_thinking=job.enable_thinking,
            extra_headers={"x-locallm-benchmark": job.job_id},
        )
        if result.is_correct is None:
            try:
                score_result(result)
            except Exception:  # noqa: BLE001
                result.is_correct = False
        # Acknowledge the answer in the durable queue, then mirror it into the
        # benchmark_result table the dashboard reads.
        store.finish_item(job.job_id, row["id"], result.to_dict())
        try:
            store.record_results(model_name, job.dataset, [result], source="job")
        except Exception:  # noqa: BLE001
            logger.exception("benchmark result mirror failed for %s", model_name)


def _probe_speed(job: BenchmarkJob, model_name: str) -> None:
    """Single-request throughput probe (strictly serial, after a model finishes)."""
    from .runner import probe_single_request_stats, save_speed
    if job.control.cancel:
        return
    try:
        stats = probe_single_request_stats(
            model_name=model_name, base_url=job.base_url, api_key=job.api_key,
        )
        try:
            save_speed(model_name, stats)
            job.store.record_speed(model_name, stats)
        except Exception:  # noqa: BLE001
            logger.warning("speed SQLite save failed for %s", model_name)
        job.speeds[model_name] = stats
        if stats:
            logger.info("single-request throughput %s: prefill=%.0f decode=%.0f tok/s",
                        model_name, stats["prefill_tps"], stats["decode_tps"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("speed probe failed for %s: %s", model_name, exc)
