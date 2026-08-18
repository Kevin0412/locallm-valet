# -*- coding: utf-8 -*-
"""Benchmark runner — sends items to the valet's OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import httpx

from .schema import BenchmarkItem, BenchmarkResult

logger = logging.getLogger("locallm_valet.benchmark.runner")


class JobControl:
    """Cooperative pause/cancel flags for a benchmark job.

    The runner checks these between items so the web UI can pause, resume
    or stop a long benchmark run without killing the process.
    """

    __slots__ = ("paused", "cancel")

    def __init__(self) -> None:
        self.paused = False
        self.cancel = False

    def wait_if_paused(self) -> None:
        """Block while paused (checks cancellation too)."""
        import time as _t
        while self.paused:
            if self.cancel:
                return
            _t.sleep(0.2)


def run_benchmark(
    *,
    items: list[BenchmarkItem],
    model_name: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "",
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout_s: int = 180,
    concurrency: int = 1,
    save_responses: bool = True,
    control: JobControl | None = None,
) -> list[BenchmarkResult]:
    """Run a list of benchmark items through the valet API.

    Args:
        items: Benchmark items to evaluate.
        model_name: The model registry name as the valet knows it.
        base_url: Valet's OpenAI-compatible endpoint.
        api_key: API key if the valet requires one.
        max_tokens: Max generation tokens.
        temperature: 0.0 = greedy (recommended for benchmark reproducibility).
        timeout_s: Per-request timeout.
        concurrency: Number of concurrent requests (1 = serial, simplest).
        save_responses: If True, record raw response text in the result.
        control: Optional JobControl for pause/resume/stop from the web UI.

    Returns:
        List of BenchmarkResult, one per item.
    """
def _run_one(item: BenchmarkItem, model_name: str, chat_url: str, headers: dict,
             max_tokens: int, temperature: float, timeout_s: int,
             save_responses: bool, total: int) -> BenchmarkResult:
    """Run a single benchmark item (used directly or via thread pool)."""
    messages = [{"role": "user", "content": item.question}]
    t0 = time.monotonic()
    result = BenchmarkResult(item=item, model_name=model_name)
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_s)) as client:
            resp = client.post(
                chat_url,
                headers=headers,
                json={
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    # Qwen3-style models default to thinking mode, which
                    # burns the token budget on reasoning_content and
                    # leaves content empty — disable for benchmarking so
                    # the answer is actually produced.
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            elapsed = time.monotonic() - t0

        if resp.status_code != 200:
            result.raw_response = f"[HTTP {resp.status_code}] {resp.text[:300]}"
            result.latency_ms = round(elapsed * 1000, 1)
            return result

        data = resp.json()
        choices = data.get("choices", [])
        raw_text = ""
        if choices:
            raw_text = (choices[0].get("message", {}) or {}).get("content", "")
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        if save_responses:
            result.raw_response = raw_text
        result.latency_ms = round(elapsed * 1000, 1)
        if completion_tokens > 0 and elapsed > 0:
            result.tps = round(completion_tokens / elapsed, 2)

    except httpx.TimeoutException:
        result.raw_response = "[TIMEOUT]"
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
    except Exception as exc:
        result.raw_response = f"[ERROR] {exc}"
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)

    return result


def run_benchmark(
    *,
    items: list[BenchmarkItem],
    model_name: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "",
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout_s: int = 180,
    concurrency: int = 1,
    save_responses: bool = True,
    control: JobControl | None = None,
) -> list[BenchmarkResult]:
    """Run a list of benchmark items through the valet API.

    Args:
        items: Benchmark items to evaluate.
        model_name: The model registry name as the valet knows it.
        base_url: Valet's OpenAI-compatible endpoint.
        api_key: API key if the valet requires one.
        max_tokens: Max generation tokens.
        temperature: 0.0 = greedy (recommended for benchmark reproducibility).
        timeout_s: Per-request timeout.
        concurrency: Number of parallel requests. Backends that support batching
                     (SGLang/vLLM) benefit directly; llama.cpp single-slot
                     servers just queue. Defaults to 1 (serial).
        save_responses: If True, record raw response text in the result.
        control: Optional JobControl for pause/resume/stop from the web UI.

    Returns:
        List of BenchmarkResult, one per item (input order preserved).
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    total = len(items)

    if concurrency <= 1:
        results: list[BenchmarkResult] = []
        for idx, item in enumerate(items):
            if control is not None:
                if control.cancel:
                    logger.info("[%d/%d] job cancelled, stopping", idx + 1, total)
                    break
                control.wait_if_paused()
            logger.info("[%d/%d] %s (%s) — sending...",
                        idx + 1, total, item.item_id, item.category)
            results.append(_run_one(item, model_name, chat_url, headers,
                                    max_tokens, temperature, timeout_s,
                                    save_responses, total))
        return results

    # Concurrent path: submit in chunks of `concurrency`, checking pause/cancel
    # between chunks so the web UI can interrupt a long run.
    from concurrent.futures import ThreadPoolExecutor
    results = [None] * total  # type: list[BenchmarkResult | None]
    pos = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while pos < total:
            if control is not None:
                if control.cancel:
                    logger.info("job cancelled at item %d, stopping", pos)
                    break
                control.wait_if_paused()
            chunk = items[pos:pos + concurrency]
            futures = {pool.submit(_run_one, item, model_name, chat_url, headers,
                                   max_tokens, temperature, timeout_s,
                                   save_responses, total): (pos + i)
                       for i, item in enumerate(chunk)}
            for fut in futures:
                idx = futures[fut]
                results[idx] = fut.result()
            pos += len(chunk)

    return [r for r in results if r is not None]