# -*- coding: utf-8 -*-
"""Benchmark runner — sends items to the valet's OpenAI-compatible API.

Requests are retried on timeouts / transport errors / transient 5xx so that
slow local models never silently drop questions from a run. Records per-request
latency, TTFT (when the backend exposes timings), decode throughput and output
tokens, plus the thinking mode used so accuracy can be compared across
non-thinking / thinking deployment modes.
"""

from __future__ import annotations

import concurrent.futures
import json
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
    control: JobControl | None = None,
    enable_thinking: bool = True,
) -> BenchmarkResult:
    """Run a single benchmark item (directly, or via the thread pool).

    ``enable_thinking=True`` (default) lets the model think — the mode the
    official model cards are benchmarked in; ``False`` disables reasoning
    (fast but understates ability — only for quick sanity runs),
    burns output tokens on reasoning_content). The mode is recorded on the
    result so accuracy/latency can be compared across modes.
    """
    if control is not None:
        control.wait_if_paused()
        if control.cancel:
            result = BenchmarkResult(item=item, model_name=model_name)
            result.raw_response = "[CANCELLED]"
            result.score_detail = "cancelled by user before request"
            return result
    messages = []
    if item.system:
        messages.append({"role": "system", "content": item.system})
    messages.append({"role": "user", "content": item.question})
    t0 = time.monotonic()
    result = BenchmarkResult(item=item, model_name=model_name)
    result.thinking = enable_thinking
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_s)) as client:
            req_body: dict = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                # Thinking-capable models (Qwen3 via SGLang, gemma-4 via
                # llama.cpp) default to a reasoning mode that burns the token
                # budget on reasoning_content. Both backends honor this
                # template variable, so we make the mode explicit and
                # measurable.
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
            # BFCL-style function calling: pass the tool schemas so the model
            # can actually emit tool_calls.
            tools = item.meta.get("tools") if item.meta else None
            if tools:
                req_body["tools"] = tools
                req_body["tool_choice"] = "auto"
            resp = _post_with_retry(
                client=client,
                chat_url=chat_url,
                headers=headers,
                payload=req_body,
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
        reasoning_text = ""
        tool_calls = None
        if choices:
            msg = choices[0].get("message", {}) or {}
            raw_text = msg.get("content", "") or ""
            reasoning_text = msg.get("reasoning_content") or msg.get("reasoning") or ""
            tool_calls = msg.get("tool_calls")
            if not raw_text.strip():
                # Safety net: if content is empty (e.g. the backend still
                # routed output to reasoning_content), keep that text so an
                # answer is never silently lost.
                raw_text = reasoning_text
        # Keep tool calls for BFCL scoring (stored in raw_response as JSON so
        # the scorer can compare against expected_tool_calls).
        if tool_calls:
            result.raw_response = json.dumps(tool_calls, ensure_ascii=False)
            result.tool_calls = tool_calls
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)

        if save_responses and not tool_calls:
            result.raw_response = raw_text
        result.latency_ms = round(elapsed * 1000, 1)
        result.prompt_tokens = prompt_tokens
        result.completion_tokens = completion_tokens
        result.reasoning_tokens = usage.get("reasoning_tokens") or len(reasoning_text) or 0
        # Cache report: SGLang --enable-cache-report and vLLM
        # --enable-prompt-tokens-details both surface the prompt-cache hit as
        # usage.prompt_tokens_details.cached_tokens. llama.cpp exposes the
        # cached share implicitly via timings.prompt_n (handled in _probe_once);
        # per-request it reports it through prompt_tokens_details too.
        ptd = usage.get("prompt_tokens_details") or {}
        if isinstance(ptd, dict):
            cached = ptd.get("cached_tokens") or 0
        else:
            cached = 0
        result.cached_tokens = int(cached)
        if completion_tokens > 0 and elapsed > 0:
            result.tps = round(completion_tokens / elapsed, 2)

        # TTFT: backend-provided timings win (llama.cpp returns
        # `timings.prompt_ms` ≈ prefill/TTFT, `timings.predicted_ms` ≈ decode).
        timings = data.get("timings") or {}
        prompt_ms = timings.get("prompt_ms")
        predicted_ms = timings.get("predicted_ms")
        if isinstance(prompt_ms, (int, float)) and prompt_ms > 0:
            result.ttft_ms = round(float(prompt_ms), 1)
        if isinstance(predicted_ms, (int, float)) and predicted_ms > 0:
            result.decode_ms = round(float(predicted_ms), 1)
            if completion_tokens > 0 and predicted_ms > 0:
                result.tps = round(completion_tokens / (predicted_ms / 1000.0), 2)

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
    retries: int = 2,
    save_responses: bool = True,
    control: JobControl | None = None,
    enable_thinking: bool = True,
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
        retries: Extra attempts per item on timeout / transport error /
            retryable 5xx (0 = no retries). Retrying keeps slow local models
            from silently dropping questions.
        save_responses: If True, record raw response text in the result.
        control: Optional JobControl for pause/resume/stop from the web UI.
        enable_thinking: False = non-thinking mode (fast, low latency);
                         True = thinking mode (higher quality, slower).
                         The mode is recorded per result so both can be
                         compared side by side.

    Returns:
        List of BenchmarkResult, one per item (input order preserved).
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    total = len(items)

    def run_one(item: BenchmarkItem) -> BenchmarkResult:
        return _run_one(
            item, model_name, chat_url, headers, max_tokens, temperature,
            timeout_s, retries, save_responses, total, control,
            enable_thinking=enable_thinking,
        )

    if concurrency <= 1:
        results: list[BenchmarkResult] = []
        for idx, item in enumerate(items):
            if control is not None and control.cancel:
                logger.info("job cancelled at item %d, stopping", idx)
                break
            results.append(run_one(item))
        return results

    results: list[Optional[BenchmarkResult]] = [None] * total
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx: dict = {}
        for idx, item in enumerate(items):
            if control is not None and control.cancel:
                logger.info("job cancelled at item %d, stopping", idx)
                break
            future_to_idx[pool.submit(run_one, item)] = idx
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


def _probe_once(
    *,
    chat_url: str,
    headers: dict,
    model_name: str,
    max_tokens: int,
    timeout_s: int,
) -> tuple[float, float] | None:
    """One streaming probe request; returns (prefill_tps, decode_tps).

    llama.cpp responses carry a ``timings`` field with exact prompt/predicted
    ms — used directly. Other backends (SGLang/vLLM) are split via SSE
    timing: time-to-first-content-chunk ≈ prefill (+ first-token decode),
    the remaining time is decode.
    """
    unit = "The quick brown fox jumps over the lazy dog while the rain falls on the quiet village streets. "
    prompt = (unit * 18)[:1000]  # ~180-token prompt

    t0 = time.monotonic()
    first_ts: float | None = None
    last_delta_ts: float | None = None
    usage: dict = {}
    timings: dict = {}
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_s)) as client:
            with client.stream(
                "POST",
                chat_url,
                headers=headers,
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            ) as resp:
                if resp.status_code != 200:
                    logger.warning("[speed-probe] HTTP %d: %s",
                                   resp.status_code, resp.text[:120])
                    return None
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        d = json.loads(data)
                    except Exception:  # noqa: BLE001
                        continue
                    if d.get("usage"):
                        usage = d["usage"]
                    ch = (d.get("choices") or [{}])[0]
                    if ch.get("timings"):
                        timings = ch["timings"]
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        if first_ts is None:
                            first_ts = time.monotonic()
                        last_delta_ts = time.monotonic()
    except httpx.HTTPError as exc:
        logger.warning("[speed-probe] %s failed: %s", model_name, exc)
        return None

    total = time.monotonic() - t0
    if first_ts is None or total <= 0:
        return None

    # llama.cpp timings are the exact per-phase figures
    if timings.get("predicted_ms"):
        try:
            ptok = int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0)
            dtok = int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)
            p_ms = float(timings.get("prompt_ms") or 0)
            d_ms = float(timings.get("predicted_ms") or 0)
            prefill = ptok / (p_ms / 1000.0) if p_ms > 0 and ptok > 0 else None
            decode = dtok / (d_ms / 1000.0) if d_ms > 0 and dtok > 0 else None
            if prefill is not None and decode is not None:
                return (round(prefill, 1), round(decode, 1))
        except (TypeError, ValueError):
            pass

    # Generic SSE decomposition
    ptok = usage.get("prompt_tokens") or 0
    dtok = usage.get("completion_tokens") or 0
    ttft = first_ts - t0
    dec_time = (last_delta_ts or total) - first_ts
    if ttft > 0 and ptok > 0 and dec_time > 0 and dtok > 0:
        return (round(ptok / ttft, 1), round(dtok / dec_time, 1))
    return None


def probe_single_request_stats(
    *,
    model_name: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "",
    max_tokens: int = 256,
    samples: int = 8,
    warmup: int = 2,
    timeout_s: int = 180,
) -> dict | None:
    """Measure single-request prefill & decode throughput (tok/s each).

    The per-item ``tps`` recorded during a benchmark run is measured while
    other requests share the backend (batched decode), so it understates a
    model's real single-request throughput. This probe streams ``samples``
    requests strictly one after another (no concurrency), discards the first
    ``warmup`` of them (CUDA-graph capture / cache warmup skew the first
    requests), and returns the arithmetic mean of the steady samples —
    clean single-request figures.
    """
    import statistics

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    chat_url = f"{base_url.rstrip('/')}/chat/completions"

    prefill: list[float] = []
    decode: list[float] = []
    for i in range(samples):
        once = _probe_once(
            chat_url=chat_url, headers=headers, model_name=model_name,
            max_tokens=max_tokens, timeout_s=timeout_s,
        )
        if once is None:
            continue
        p, d = once
        prefill.append(p)
        decode.append(d)
        logger.info("[speed-probe] %s sample %d/%d: prefill=%.0f tok/s decode=%.0f tok/s",
                    model_name, i + 1, samples, p, d)

    steady_p = prefill[warmup:]
    steady_d = decode[warmup:]
    if not steady_p or not steady_d:
        return None
    return {
        "prefill_tps": round(statistics.fmean(steady_p), 1),
        "decode_tps": round(statistics.fmean(steady_d), 1),
        "samples": len(prefill),
        "steady": len(steady_p),
    }


_SPEEDS_FILE = "benchmark_results/speeds.json"


def save_speed(model_name: str, stats: dict | None) -> None:
    """Persist a model's single-request throughput (benchmark_results/speeds.json).

    ``stats`` carries ``prefill_tps`` and ``decode_tps``; a plain legacy
    ``tps`` value is kept for backward compatibility.
    """
    from pathlib import Path

    p = Path(_SPEEDS_FILE)
    data: dict = {}
    if p.is_file():
        try:
            data = json.loads(p.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
    if stats:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if "prefill_tps" in stats or "decode_tps" in stats:
            entry.update({"prefill_tps": stats.get("prefill_tps"),
                          "decode_tps": stats.get("decode_tps")})
        if stats.get("samples"):
            entry["samples"] = stats["samples"]
            entry["steady"] = stats.get("steady")
        if stats.get("tps"):
            entry["tps"] = stats["tps"]
        data[model_name] = entry
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def load_speeds() -> dict:
    """Read persisted single-request throughputs: {model: {prefill_tps, decode_tps, ts}}."""
    from pathlib import Path

    p = Path(_SPEEDS_FILE)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return {}
