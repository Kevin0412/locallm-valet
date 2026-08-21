# -*- coding: utf-8 -*-
"""Benchmark job runner — runs a benchmark in background threads so the web
UI can start / pause / resume / stop it and poll progress.

Concurrency model (multi-slot): models are grouped by their slot; each slot
runs on its own worker thread, and within a slot models are serialized (the
slot's single backend can hold one model at a time). Different slots run in
parallel — CPU / NPU / GPU models benchmark simultaneously, matching the
valet's multi-slot serving architecture.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
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
    api_key: str = ""
    max_tokens: int = 256
    sample: Optional[int] = None
    concurrency: int = 1
    enable_thinking: bool = False
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
    speeds: dict = field(default_factory=dict)   # {model: single-request tok/s}

    def status(self) -> dict:
        return {
            "state": self.state,
            "dataset": self.dataset,
            "models": self.models,
            "sample": self.sample,
            "concurrency": self.concurrency,
            "enable_thinking": self.enable_thinking,
            "current_model": self.current_model,
            "current_item": self.current_item,
            "total_items": self.total_items,
            "done_items": self.done_items,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "speeds": self.speeds,
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
    enable_thinking: bool = False,
    api_key: str = "",
    output_dir: str = "benchmark_results",
    slot_of: Optional[dict] = None,
) -> BenchmarkJob:
    """Start a benchmark job in background threads (no-op if already running).

    ``enable_thinking``: False = non-thinking mode (default, fast);
    True = thinking mode (slower, higher quality) — recorded per result so
    both deployment modes can be compared.

    ``api_key``: forwarded to the runner so requests stay authenticated when
    the valet has auth enabled (Bearer header).

    ``slot_of``: optional mapping model_name → slot_name, used to group models
    for cross-slot parallel execution. When absent, all models run serially on
    one worker (single-slot behaviour).
    """
    global _JOB
    with _LOCK:
        if _JOB.running():
            return _JOB
        job = BenchmarkJob(
            dataset=dataset,
            models=list(models),
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            sample=sample,
            concurrency=concurrency,
            enable_thinking=enable_thinking,
            state="running",
            started_at=time.time(),
            _control=JobControl(),
            _results=[],
        )
        _JOB = job
        thread = threading.Thread(
            target=_worker,
            args=(job, output_dir, slot_of or {}),
            name="benchmark-job",
            daemon=True,
        )
        job._thread = thread
        thread.start()
        return job


def _worker(job: BenchmarkJob, output_dir: str, slot_of: dict) -> None:
    try:
        items = get_dataset(job.dataset, job.sample)
    except Exception as exc:  # noqa: BLE001 - surfaced via job.error
        logger.exception("benchmark dataset load failed")
        job.state = "error"
        job.error = str(exc)
        job.finished_at = time.time()
        return

    job.total_items = len(items) * len(job.models)

    # Group models by slot: each slot runs on its own thread (parallel across
    # slots, serialized within a slot). Without slot info, everything lands in
    # one "cpu" group — single-worker serial behaviour.
    groups: dict[str, list[str]] = defaultdict(list)
    for model_name in job.models:
        groups[slot_of.get(model_name, "cpu")].append(model_name)

    threads = []
    for slot_name, slot_models in groups.items():
        t = threading.Thread(
            target=_slot_worker,
            args=(job, slot_name, slot_models, items, output_dir),
            name=f"benchmark-slot-{slot_name}",
            daemon=True,
        )
        t.start()
        threads.append(t)
        logger.info("benchmark slot thread started: %s (%d models)", slot_name, len(slot_models))

    for t in threads:
        t.join()

    if job.state not in ("stopped", "error"):
        job.state = "done"
    job.finished_at = time.time()
    logger.info("benchmark job finished: state=%s total_results=%d",
                job.state, len(job._results))


def _slot_worker(job: BenchmarkJob, slot_name: str, models: list[str],
                 items: list[BenchmarkItem], output_dir: str) -> None:
    """Run one slot's models serially (the slot backend holds one at a time);
    multiple slots run concurrently."""
    for model_name in models:
        if job._control.cancel:
            job.state = "stopped"
            return
        job.current_model = model_name
        job.current_item = 0
        logger.info("benchmark job [%s]: model=%s items=%d", slot_name, model_name, len(items))
        try:
            results = run_benchmark(
                items=items,
                model_name=model_name,
                base_url=job.base_url,
                api_key=job.api_key,
                max_tokens=job.max_tokens,
                concurrency=job.concurrency,
                control=job._control,
                enable_thinking=job.enable_thinking,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("benchmark model run failed: %s", model_name)
            job.state = "error"
            job.error = f"{model_name}: {exc}"
            return

        for r in results:
            if r.is_correct is None:
                try:
                    score_result(r)
                except Exception:  # noqa: BLE001
                    r.is_correct = False
        with _LOCK:
            job._results.extend(results)
            job.done_items += len(results)
        # Persist per-model immediately so partial results are visible while
        # the rest of the run continues. File name carries model + dataset so
        # the results table can distinguish both.
        try:
            _save_model(job, results, output_dir)
        except Exception as exc:  # noqa: BLE001
            logger.exception("benchmark save failed for %s", model_name)
            job.error = job.error or f"save {model_name}: {exc}"
        # Single-request throughput probe: the per-item tps above was measured
        # under batched decode; probe strictly serially for a clean figure.
        if not job._control.cancel:
            try:
                from .runner import probe_single_request_stats, save_speed

                stats = probe_single_request_stats(
                    model_name=model_name, base_url=job.base_url,
                    api_key=job.api_key,
                )
                save_speed(model_name, stats)
                job.speeds[model_name] = stats
                if stats:
                    logger.info("single-request throughput %s: prefill=%.0f decode=%.0f tok/s",
                                model_name, stats["prefill_tps"], stats["decode_tps"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("speed probe failed for %s: %s", model_name, exc)
        if job._control.cancel:
            job.state = "stopped"
            return


def _save_results(job: BenchmarkJob, results: list[BenchmarkResult], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Group per model for the benchmark page to read
    by_model: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_model.setdefault(r.model_name, []).append(r)

    for model_name, rlist in by_model.items():
        _write_model_jsonl(out, model_name, job.dataset, rlist)
    logger.info("saved %d result file(s) under %s", len(by_model), out)


def _save_model(job: BenchmarkJob, results: list[BenchmarkResult], output_dir: str) -> None:
    """Persist one model's results as its own JSONL file immediately."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_model_jsonl(out, results[0].model_name if results else "unknown",
                       job.dataset, results)
    logger.info("saved %d results for %s under %s", len(results),
                results[0].model_name if results else "?", out)


def _write_model_jsonl(out: Path, model_name: str, dataset: str,
                       results: list[BenchmarkResult]) -> None:
    # {model}_{dataset}_results.jsonl — the dataset is explicit in the name,
    # so the results page groups by model_name and shows the dataset, and a
    # later same-model run on another dataset cannot overwrite it.
    safe_dataset = "".join(c for c in dataset if c.isalnum() or c in "_-")
    path = out / f"{model_name}_{safe_dataset}_results.jsonl"
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