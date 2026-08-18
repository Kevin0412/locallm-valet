# -*- coding: utf-8 -*-
"""Benchmark runner — sends items to the valet's OpenAI-compatible API.

Requests are retried on timeouts / transport errors / transient 5xx so that
slow local models never silently drop questions from a run.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Optional

import httpx

from .schema import BenchmarkItem, BenchmarkResult

logger = logging.getLogger("locallm_valet.benchmark.runner")

# HTTP statuses that are safe to retry — the server was reachable but
# transiently unable to serve (rate-limited, mid-switch, overloaded).
_RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def _post_with_retry(
    client: httpx.Client,
    chat_url: str,
    headers: dict,
    payload: dict,
    retries: int,
    backoff: float = 1.5,
) -> httpx.Response:
    """POST ``payload`` to ``chat_url``, retrying transient failures.

    Retries on httpx timeouts, transport-level errors and retryable HTTP
    statuses, sleeping ``backoff * (attempt + 1)`` between attempts.
    Raises the last exception when ``retries`` are exhausted; retryable
    HTTP responses are returned as-is on the final attempt so the caller
    can record the exact status instead of losing the question silently.
    """
    attempt = 0
    while True:
        try:
            resp = client.post(chat_url, headers=headers, json=payload)
            if resp.status_code not in _RETRYABLE_HTTP:
                return resp
            if attempt >= retries:
                return resp  # final attempt — surface the status to the caller
            wait = backoff * (attempt + 1)
            logger.warning(
                "HTTP %d — retry in %.1fs (attempt %d/%d)",
                resp.status_code, wait, attempt + 1, retries,
            )
            time.sleep(wait)
        except httpx.TimeoutException as exc:
            if attempt >= retries:
                raise
            wait = backoff * (attempt + 1)
            logger.warning(
                "Timeout (%s) — retry in %.1fs (attempt %d/%d)",
                exc.__class__.__name__, wait, attempt + 1, retries,
            )
            time.sleep(wait)
        except httpx.HTTPError as exc:
            if attempt >= retries:
                raise
            wait = backoff * (attempt + 1)
            logger.warning(
                "Transport error (%s) — retry in %.1fs (attempt %d/%d)",
                exc.__class__.__name__, wait, attempt + 1, retries,
            )
            time.sleep(wait)
        attempt += 1


def _run_one(
    item: BenchmarkItem,
    model_name: str,
    chat_url: str,
    headers: dict,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    retries: int,
    save_responses: bool,
    total: int,
) -> BenchmarkResult:
    """Run a single benchmark item (directly, or via the thread pool)."""
    messages = [{"role": "user", "content": item.question}]
    t0 = time.monotonic()
    result = BenchmarkResult(item=item, model_name=model_name)
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_s)) as client:
            resp = _post_with_retry(
                client=client,
                chat_url=chat_url,
                headers=headers,
                payload={
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                retries=retries,
            )
            elapsed = time.monotonic() - t0

        if resp.status_code != 200:
            logger.warning(
                "[%s] HTTP %d: %s",
                item.item_id, resp.status_code, resp.text[:200],
            )
            result.raw_response = f"[HTTP {resp.status_code}] {resp.text[:300]}"
            result.latency_ms = round(elapsed * 1000, 1)
            return result

        data = resp.json()
        choices = data.get("choices", [])
        raw_text = ""
        if choices:
            msg = choices[0].get("message", {}) or {}
            raw_text = msg.get("content", "")
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        if save_responses:
            result.raw_response = raw_text
        result.latency_ms = round(elapsed * 1000, 1)
        if completion_tokens > 0 and elapsed > 0:
            result.tps = round(completion_tokens / elapsed, 2)

        logger.info(
            "[%s] OK tok=%d lat=%.1fs tps=%.1f",
            item.item_id, completion_tokens, elapsed, result.tps or 0,
        )

    except httpx.TimeoutException:
        logger.warning(
            "[%s] TIMEOUT after %d retries (%ds)",
            item.item_id, retries, timeout_s,
        )
        result.raw_response = "[TIMEOUT]"
        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)

    except Exception as exc:  # noqa: BLE001 - record and continue
        logger.warning("[%s] ERROR: %s", item.item_id, exc)
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
    retries: int = 2,
    save_responses: bool = True,
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
        retries: Extra attempts per item on timeout / transport error /
            retryable 5xx (0 = no retries). Retrying keeps slow local models
            from silently dropping questions.
        save_responses: If True, record raw response text in the result.

    Returns:
        List of BenchmarkResult, one per item, in the original item order.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    total = len(items)

    def run_one(item: BenchmarkItem) -> BenchmarkResult:
        return _run_one(
            item, model_name, chat_url, headers, max_tokens, temperature,
            timeout_s, retries, save_responses, total,
        )

    if concurrency <= 1:
        return [run_one(item) for item in items]

    results: list[Optional[BenchmarkResult]] = [None] * total
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx = {
            pool.submit(run_one, item): idx
            for idx, item in enumerate(items)
        }
        done = 0
        for fut in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001 - a broken item must not kill the run
                logger.error("[item %d] future failed: %s", idx, exc)
            done += 1
            if done % 25 == 0 or done == total:
                logger.info("Progress: %d/%d", done, total)

    return [r for r in results if r is not None]
