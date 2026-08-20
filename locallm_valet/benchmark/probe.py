# -*- coding: utf-8 -*-
"""Speed probe — TTFT / prefill / decode / cold-start / concurrency metrics.

Unlike the accuracy benchmark (which measures *quality*), this probes the
serving characteristics of a model on a given slot:

- TTFT (time to first token) via a streaming request
- decode throughput (tokens/s) from the stream
- prefill scaling across prompt lengths (1K / 8K / 32K)
- cold start (first request after load, including model load time)
- concurrency scaling (1 / 2 / 4 parallel streams)

Output: a dict of metrics, plus a JSONL row per probe for the results page.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger("locallm_valet.benchmark.probe")

DEFAULT_LENGTHS = [1024, 8192, 32768]


def _make_prompt(n_tokens: int, model_name: str) -> str:
    """A filler prompt of roughly n tokens (English words ≈ 0.75 tok each)."""
    word = "lorem ipsum dolor sit amet consectetur adipiscing elit "
    # ~0.75 tokens/word → need ~1.33 words per token
    words = max(1, int(n_tokens * 1.3))
    text = (word * (words // len(word.split()) + 1))
    return " ".join(text.split()[:words]) + "\nReply with the single word OK."


def _stream_once(client: httpx.Client, chat_url: str, headers: dict,
                 model_name: str, prompt: str, max_tokens: int,
                 timeout_s: float) -> dict:
    """One streaming request (sync); returns TTFT, decode tps, output tokens."""
    started = time.monotonic()
    ttft: Optional[float] = None
    n_chars = 0
    with client.stream(
        "POST", chat_url, headers=headers, json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }, timeout=httpx.Timeout(timeout_s),
    ) as resp:
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}", "latency_ms": (time.monotonic() - started) * 1000}
        for line in resp.iter_lines():
            if ttft is None:
                ttft = (time.monotonic() - started) * 1000.0
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                n_chars += len(line)
    total_ms = (time.monotonic() - started) * 1000.0
    return {
        "ttft_ms": round(ttft, 1) if ttft else None,
        "decode_ms": round(total_ms - (ttft or total_ms), 1),
        "total_ms": round(total_ms, 1),
        "out_tokens": max(1, n_chars // 4),  # rough char→token estimate
    }


def probe_speed(
    *,
    model_name: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "",
    prompt_lengths: list[int] | None = None,
    concurrency_levels: list[int] | None = None,
    include_cold_start: bool = True,
    max_tokens: int = 64,
    timeout_s: float = 300.0,
) -> dict:
    """Run the speed probe suite for one model (sync; call in a thread)."""
    lengths = prompt_lengths or DEFAULT_LENGTHS
    concurrency_levels = concurrency_levels or [1, 2, 4]
    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    results: dict = {"model": model_name, "metrics": {}}

    with httpx.Client(timeout=httpx.Timeout(timeout_s)) as client:

        # --- cold start: first request includes model load ---
        if include_cold_start:
            t0 = time.monotonic()
            r = client.post(chat_url, headers=headers, json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 8, "temperature": 0.0,
            })
            cold_ms = (time.monotonic() - t0) * 1000
            results["metrics"]["cold_start_ms"] = round(cold_ms, 1)
            results["metrics"]["cold_start_status"] = r.status_code
            logger.info("cold start: %d ms (HTTP %d)", cold_ms, r.status_code)

        # --- TTFT + decode via streaming ---
        for n_tokens in lengths:
            prompt = _make_prompt(n_tokens, model_name)
            try:
                m = _stream_once(client, chat_url, headers, model_name,
                                 prompt, max_tokens, timeout_s)
            except Exception as exc:  # noqa: BLE001
                m = {"error": str(exc)}
            label = f"prefill_{n_tokens // 1024}k" if n_tokens >= 1024 else f"prefill_{n_tokens}"
            results["metrics"][label] = m
            logger.info("probe %s: %s", label, m)

        # --- concurrency scaling (parallel streams) ---
        from concurrent.futures import ThreadPoolExecutor
        for conc in concurrency_levels:
            prompt = _make_prompt(1024, model_name)
            try:
                with ThreadPoolExecutor(max_workers=conc) as pool:
                    outcomes = list(pool.map(
                        lambda _: _stream_once(client, chat_url, headers, model_name,
                                               prompt, max_tokens, timeout_s),
                        range(conc)))
                ok = [o for o in outcomes if "error" not in o]
                if ok:
                    avg_ttft = sum(o["ttft_ms"] for o in ok if o.get("ttft_ms")) / len(ok)
                    results["metrics"][f"conc{conc}_ttft_ms"] = round(avg_ttft, 1)
                    results["metrics"][f"conc{conc}_out_tok"] = sum(o["out_tokens"] for o in ok)
            except Exception as exc:  # noqa: BLE001
                results["metrics"][f"conc{conc}"] = {"error": str(exc)}

    return results


def probe_to_jsonl(probe: dict, path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(probe, ensure_ascii=False) + "\n")
