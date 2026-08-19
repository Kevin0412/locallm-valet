# -*- coding: utf-8 -*-
"""Benchmark job runner — runs a benchmark in a background thread so the web
UI can start / pause / resume / stop it and poll progress.

Only one job runs at a time (single device, single active model — a core V1
constraint, same as the valet itself).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .dataset import get_dataset
from .runner import JobControl, run_benchmark
from .scorer import score_result
from .schema import BenchmarkItem, BenchmarkResult

logger = logging.getLogger("locallm_valet.benchmark.job")


@dataclass
class BenchmarkJob:
    """State of the current (or last) benchmark job."""

    dataset: str = ""
    models: list[str] = field(default_factory=list)
    base_url: str = "http://127.0.0.1:8000/v1"
    max_tokens: int = 256
    sample: Optional[int] = None
    concurrency: int = 1
    state: str = "idle"          # idle | running | paused | done | error | stopped
    current_model: str = ""
    current_item: int = 0
    total_items: int = 0
    done_items: int = 0
    error: str = ""
    started_at: float = 0.0
    finished_at: Optional[float] = None
    _control: JobControl = field(default_factory=JobControl)
    _thread: Optional[threading.Thread] = None
    _results: list[BenchmarkResult] = field(default_factory=list)

    def status(self) -> dict:
        return {
            "state": self.state,
            "dataset": self.dataset,
            "models": self.models,
            "sample": self.sample,
            "concurrency": self.concurrency,
            "current_model": self.current_model,
            "current_item": self.current_item,
            "total_items": self.total_items,
            "done_items": self.done_items,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


_JOB = BenchmarkJob()
_LOCK = threading.Lock()


def current_job() -> BenchmarkJob:
    return _JOB


def start_job(
    *,
    dataset: str,
    models: list[str],
    base_url: str,
    max_tokens: int,
    sample: Optional[int] = None,
    concurrency: int = 1,
    output_dir: str = "benchmark_results",
) -> BenchmarkJob:
    """Start a benchmark job in a background thread (no-op if already running)."""
    global _JOB
    with _LOCK:
        if _JOB.running():
            return _JOB
        job = BenchmarkJob(
            dataset=dataset,
            models=list(models),
            base_url=base_url,
            max_tokens=max_tokens,
            sample=sample,
            concurrency=concurrency,
            state="running",
            started_at=time.time(),
            _control=JobControl(),
            _results=[],
        )
        _JOB = job
        thread = threading.Thread(
            target=_worker,
            args=(job, output_dir),
            name="benchmark-job",
            daemon=True,
        )
        job._thread = thread
        thread.start()
        return job


def _worker(job: BenchmarkJob, output_dir: str) -> None:
    try:
        items = get_dataset(job.dataset, job.sample)
    except Exception as exc:  # noqa: BLE001 - surfaced via job.error
        logger.exception("benchmark dataset load failed")
        job.state = "error"
        job.error = str(exc)
        job.finished_at = time.time()
        return

    job.total_items = len(items) * len(job.models)
    all_results: list[BenchmarkResult] = []

    for model_name in job.models:
        if job._control.cancel:
            job.state = "stopped"
            break
        job.current_model = model_name
        job.current_item = 0
        logger.info("benchmark job: model=%s items=%d", model_name, len(items))
        try:
            results = run_benchmark(
                items=items,
                model_name=model_name,
                base_url=job.base_url,
                max_tokens=job.max_tokens,
                concurrency=job.concurrency,
                control=job._control,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("benchmark model run failed: %s", model_name)
            job.state = "error"
            job.error = f"{model_name}: {exc}"
            break

        for r in results:
            if r.is_correct is None:
                try:
                    score_result(r)
                except Exception:  # noqa: BLE001
                    r.is_correct = False
        all_results.extend(results)
        job.done_items += len(results)
        # Persist per-model immediately so partial results are visible while
        # the rest of the run continues.
        try:
            _save_model(job, results, output_dir)
        except Exception as exc:  # noqa: BLE001
            logger.exception("benchmark save failed for %s", model_name)
            job.error = job.error or f"save {model_name}: {exc}"
        if job._control.cancel:
            job.state = "stopped"
            break

    job._results = all_results

    # Persist results regardless of stop/cancel
    try:
        _save_results(job, all_results, output_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("benchmark save failed")
        job.error = job.error or f"save: {exc}"

    if job.state not in ("stopped", "error"):
        job.state = "done"
    job.finished_at = time.time()
    logger.info("benchmark job finished: state=%s results=%d", job.state, len(all_results))


def _save_results(job: BenchmarkJob, results: list[BenchmarkResult], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Group per model for the benchmark page to read
    by_model: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_model.setdefault(r.model_name, []).append(r)

    for model_name, rlist in by_model.items():
        _write_model_jsonl(out, model_name, rlist)
    logger.info("saved %d result file(s) under %s", len(by_model), out)


def _save_model(job: BenchmarkJob, results: list[BenchmarkResult], output_dir: str) -> None:
    """Persist one model's results as its own JSONL file immediately."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_model_jsonl(out, results[0].model_name if results else "unknown", results)
    logger.info("saved %d results for %s under %s", len(results),
                results[0].model_name if results else "?", out)


def _write_model_jsonl(out: Path, model_name: str, results: list[BenchmarkResult]) -> None:
    path = out / f"{model_name}_results.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def pause_job() -> BenchmarkJob:
    with _LOCK:
        if _JOB.state == "running":
            _JOB._control.paused = True
            _JOB.state = "paused"
    return _JOB


def resume_job() -> BenchmarkJob:
    with _LOCK:
        if _JOB.state == "paused":
            _JOB._control.paused = False
            _JOB.state = "running"
    return _JOB


def stop_job() -> BenchmarkJob:
    with _LOCK:
        _JOB._control.cancel = True
        _JOB._control.paused = False
        _JOB.state = "stopped"
    return _JOB