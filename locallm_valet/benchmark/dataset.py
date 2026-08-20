# -*- coding: utf-8 -*-
"""Benchmark datasets — smoke (built-in) + cached from datasets_eval.txz.

Datasets (all from local cache, zero downloads after extraction):
  - "smoke"      — 6 built-in items, no cache needed
  - "mmlu"       — MMLU (57 subjects, 5-shot)
  - "mmlu_pro"   — MMLU-Pro
  - "bfcl"       — Berkeley Function Calling Leaderboard
  - "mmstar"     — MMStar
  - "ocrbench"   — OCRBench

Extract the archive to the project root first:
  tar -xJf datasets_eval.txz -C locallm-valet/dataset_cache
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from .schema import BenchmarkItem

logger = logging.getLogger("locallm_valet.benchmark.dataset")

# ---------------------------------------------------------------------------
# Built-in smoke dataset (from smoke.json, zero deps)
# ---------------------------------------------------------------------------

_SMOKE_JSON = Path(__file__).resolve().parent / "smoke.json"


def _get_smoke(sample: int | None = None) -> list[BenchmarkItem]:
    with open(_SMOKE_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items: list[BenchmarkItem] = []
    for d in raw:
        lang = "zh" if any("\u4e00" <= c <= "\u9fff" for c in d.get("question", "")) else "en"
        items.append(BenchmarkItem(
            item_id=d["item_id"], category=d["category"],
            question=d["question"], ground_truth=d["ground_truth"],
            choices=d.get("choices", []), language=lang,
        ))
    return items


# ---------------------------------------------------------------------------
# Local cache from datasets_eval.txz
# ---------------------------------------------------------------------------

# dataset_cache/ is at the project root (two levels up from benchmark/)
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "dataset_cache" / "dataset"


def _load_processed(name: str, sample: int | None = None) -> tuple[list[dict], Optional[dict]]:
    """Load processed.json (FULL data) + dev_examples.json if present.

    ``sample``: when set, use the matching ``sample_{sample}_indices.json``
    file if present (fixed, reproducible subset), otherwise STRATIFY by
    subject/category and sample proportionally. ``None`` = full dataset.

    IMPORTANT: processed.json always holds the FULL dataset — sampling is
    purely an index into it, so a later ``sample=None`` run uses everything.
    """
    p = _CACHE_DIR / name / "processed.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"Dataset '{name}' not cached at {p}. "
            f"Extract datasets_eval.txz to {_CACHE_DIR.parent}"
        )
    with open(p, "r", encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    if sample is not None and 0 < sample < len(items):
        # 1) fixed indices shipped in the archive (already stratified by the
        #    upstream generator where available)
        idx_p = _CACHE_DIR / name / f"sample_{sample}_indices.json"
        if idx_p.is_file():
            with open(idx_p, "r", encoding="utf-8") as f:
                indices: list[int] = json.load(f)
            indices = [i for i in indices if 0 <= i < len(items)]
            if indices:
                items = [items[i] for i in indices]
                logger.info("Sampled %d items via %s", len(items), idx_p.name)
                _write_sample_indices(_CACHE_DIR / name, sample, indices)
                return _finish_load(name, items)

        # 2) no fixed indices → stratified random sample by subject/category
        items = _stratified_sample(items, sample)
        logger.info("Stratified-sampled %d items (seed=42, by subject/category)", len(items))

    return _finish_load(name, items)


def _finish_load(name: str, items: list[dict]) -> tuple[list[dict], Optional[dict]]:
    dev: Optional[dict] = None
    dev_p = _CACHE_DIR / name / "dev_examples.json"
    if dev_p.is_file():
        with open(dev_p, "r", encoding="utf-8") as f:
            dev = json.load(f)
    logger.info("Loaded %d items from %s", len(items), name)
    return items, dev


def _stratified_sample(items: list[dict], n: int) -> list[dict]:
    """Proportional stratified sample by subject (MMLU) or category.

    Keeps the distribution of subjects/categories the same as the full set,
    instead of a flat random draw which can over/under-represent a subject.
    """
    import collections
    random.seed(42)
    key_of = lambda r: r.get("subject") or r.get("category") or "other"
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(items):
        groups[key_of(r)].append(i)

    chosen: list[int] = []
    total = len(items)
    # round-robin across groups so every group contributes, weighted by size
    for g, idxs in groups.items():
        share = max(1, round(n * len(idxs) / total))
        chosen.extend(random.sample(idxs, min(share, len(idxs))))
    # top up / trim to exactly n, keeping the stratified distribution close
    random.shuffle(chosen)
    return [items[i] for i in chosen[:n]]


def _write_sample_indices(cache_dir, sample: int, indices: list[int]) -> None:
    """Persist the sample indices for reproducibility."""
    try:
        out = cache_dir / f"sample_{sample}_indices.json"
        if not out.is_file():
            with open(out, "w", encoding="utf-8") as f:
                json.dump(indices, f)
            logger.info("wrote %s", out)
    except Exception:  # noqa: BLE001 - cache write is best-effort
        logger.warning("could not persist sample indices for %s", sample)


# ---------------------------------------------------------------------------
# Auto-detect format and convert to BenchmarkItem list
# ---------------------------------------------------------------------------

def _convert(items_raw: list[dict], dev: Optional[dict] = None) -> list[BenchmarkItem]:
    if not items_raw:
        return []
    keys = set(items_raw[0].keys())

    # HF raw format: question, choices(4 strings), answer(0-3 int), optional subject
    if {"question", "choices", "answer"} <= keys:
        return _from_hf_raw(items_raw, dev)

    # Reference benchmark converted: item_id, subject, ground_truth, options(optional)
    if {"item_id", "subject", "ground_truth"} <= keys:
        return _from_ref_converted(items_raw)

    # Unknown: best effort
    logger.warning("Unknown format (keys=%s), falling back to generic conversion", sorted(keys))
    return _from_generic(items_raw)


def _from_hf_raw(items_raw: list[dict], dev: Optional[dict]) -> list[BenchmarkItem]:
    out = []
    for i, row in enumerate(items_raw):
        subject = row.get("subject", f"subject_{i}")
        question = row["question"]
        choices: list[str] = row["choices"]
        answer_idx = row["answer"]
        answer_letter = chr(ord("A") + answer_idx)

        # Build 5-shot prompt if dev examples available
        fewshot_lines = []
        if dev and subject in dev:
            for ex in (dev[subject] or [])[:5]:
                q = ex["question"]
                c = ex["choices"]
                a = chr(ord("A") + ex["answer"])
                fewshot_lines.append(f"Question: {q}\nA. {c[0]}  B. {c[1]}  C. {c[2]}  D. {c[3]}\nAnswer: {a}")
        fewshot_text = "\n\n".join(fewshot_lines)
        if fewshot_text:
            fewshot_text += "\n\n"

        prompt = (
            f"The following are multiple choice questions (with answers) about {subject}.\n\n"
            f"{fewshot_text}"
            f"Question: {question}\n"
            f"A. {choices[0]}  B. {choices[1]}  C. {choices[2]}  D. {choices[3]}\n"
            f"Answer:"
        )

        lang = "zh" if any("\u4e00" <= c <= "\u9fff" for c in prompt) else "en"
        out.append(BenchmarkItem(
            item_id=row.get("item_id", f"{subject}_{i}"),
            # MMLU's real grouping is the subject; use it as the category so
            # per-category stats reflect actual subjects, not a fake "fact".
            category=subject if subject else row.get("category", "fact"),
            question=prompt,
            ground_truth=answer_letter,
            choices=[f"A. {choices[0]}", f"B. {choices[1]}",
                     f"C. {choices[2]}", f"D. {choices[3]}"],
            language=lang,
        ))
    return out


def _from_ref_converted(items_raw: list[dict]) -> list[BenchmarkItem]:
    out = []
    for row in items_raw:
        msgs = row.get("messages") or []
        system = ""
        user_parts: list[str] = []
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                user_parts.append(content)
            elif role == "assistant" and "fewshot" in content.lower():
                # keep few-shot assistant turns? they belong to the prompt;
                # simplest: ignore (the messages already include them for
                # upstream-faithful runs — see below)
                pass
        # Rebuild the user turn with the answer options inline if present,
        # matching the original "A. .. B. .." layout the scorer expects.
        choices_raw = row.get("options", [])
        if choices_raw and user_parts:
            user_parts[-1] = user_parts[-1] + "\n\n" + "\n".join(
                f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(choices_raw)
            )
        elif not user_parts:
            user_parts.append(row.get("question", row.get("description", "")))
        question_text = "\n".join(user_parts)

        gt = row["ground_truth"]
        choices = [f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(choices_raw)]

        lang = "zh" if any("\u4e00" <= c <= "\u9fff" for c in question_text) else "en"
        # Keep the real subject/category (MMLU subject, MMLU-Pro category) so
        # per-category statistics actually mean something.
        category = (row.get("category") or row.get("subject") or "fact")
        out.append(BenchmarkItem(
            item_id=row["item_id"], category=category,
            question=question_text, ground_truth=gt,
            choices=choices, system=system, language=lang,
        ))
    return out


def _from_generic(items_raw: list[dict]) -> list[BenchmarkItem]:
    out = []
    for i, row in enumerate(items_raw):
        gt = row.get("ground_truth", row.get("answer", row.get("expected", "")))
        q = row.get("question", row.get("input", row.get("prompt", str(row))))
        choices = row.get("choices", row.get("options", []))
        if isinstance(gt, int):
            gt = chr(ord("A") + gt)
        lang = "zh" if any("\u4e00" <= c <= "\u9fff" for c in str(q)) else "en"
        out.append(BenchmarkItem(
            item_id=row.get("item_id", f"item_{i}"),
            category=row.get("category", "fact"),
            question=str(q)[:2000],
            ground_truth=str(gt),
            choices=choices,
            language=lang,
        ))
    return out


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def _get_mmlu(sample: int | None = None) -> list[BenchmarkItem]:
    items_raw, dev = _load_processed("mmlu", sample)
    return _convert(items_raw, dev)

def _get_mmlu_pro(sample: int | None = None) -> list[BenchmarkItem]:
    items_raw, dev = _load_processed("mmlu_pro", sample)
    return _convert(items_raw, dev)

def _get_mmstar(sample: int | None = None) -> list[BenchmarkItem]:
    logger.warning(
        "MMStar is a VISION benchmark; the text-only runner cannot evaluate it "
        "meaningfully — results will be garbage. Use a multimodal backend."
    )
    items_raw, dev = _load_processed("mmstar", sample)
    return _convert(items_raw, dev)

def _get_ocrbench(sample: int | None = None) -> list[BenchmarkItem]:
    logger.warning(
        "OCRBench is a VISION/OCR benchmark; the text-only runner cannot "
        "evaluate it meaningfully — results will be garbage. Use a multimodal "
        "backend."
    )
    items_raw, dev = _load_processed("ocrbench", sample)
    return _convert(items_raw, dev)


def _get_bfcl(sample: int | None = None) -> list[BenchmarkItem]:
    """BFCL function-calling items: keep the tool schemas + expected calls in
    meta so the runner can pass tools and the scorer can compare calls."""
    items_raw, _ = _load_processed("bfcl", sample)
    out = []
    for row in items_raw:
        system = ""
        user_parts = []
        for m in (row.get("messages") or []):
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                user_parts.append(content)
        question_text = "\n".join(user_parts) or row.get("item_id", "")
        out.append(BenchmarkItem(
            item_id=row["item_id"],
            category=str(row.get("category", "bfcl")),
            question=question_text,
            ground_truth=_encode_expected(row.get("expected_tool_calls") or []),
            system=system,
            meta={
                "tools": row.get("tools") or [],
                "expected_tool_calls": row.get("expected_tool_calls") or [],
            },
        ))
    return out


def _encode_expected(expected: list) -> str:
    """Human-readable ground truth for reports."""
    parts = []
    for e in expected:
        fn = e.get("function") or e
        parts.append(fn.get("name", "?"))
    return ", ".join(parts) if parts else "(no call)"


# ---------------------------------------------------------------------------
# Code benchmarks (HumanEval / MBPP) — downloaded from GitHub, cached locally.
# ---------------------------------------------------------------------------

_CODE_CACHE = _CACHE_DIR / "code"


def _github_get(url: str) -> bytes:
    """Fetch a raw GitHub file, honouring the proxy from proxy.txt if set.
    Caches the raw bytes under dataset_cache/code/."""
    import hashlib
    _CODE_CACHE.mkdir(parents=True, exist_ok=True)
    cache_f = _CODE_CACHE / hashlib.sha1(url.encode()).hexdigest()
    if cache_f.is_file():
        return cache_f.read_bytes()

    import os
    import httpx
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("http_proxy")
    try:
        r = httpx.get(url, timeout=90, proxy=proxy)
        r.raise_for_status()
    except Exception:
        # no proxy configured → try direct; Windows boxes often need the proxy
        r = httpx.get(url, timeout=90)
        r.raise_for_status()
    cache_f.write_bytes(r.content)
    return r.content


def _load_humaneval(sample: int | None = None) -> list[BenchmarkItem]:
    """HumanEval (openai, 164 tasks): generate the function body; scored by
    executing the hidden test with the generated candidate (pass@1)."""
    import gzip
    raw = _github_get(
        "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    )
    rows = [json.loads(l) for l in gzip.decompress(raw).decode("utf-8").splitlines()]

    if sample is not None and 0 < sample < len(rows):
        random.seed(42)
        rows = random.sample(rows, sample)

    items = []
    for row in rows:
        prompt = row["prompt"].rstrip()
        items.append(BenchmarkItem(
            item_id=row["task_id"],
            category="coding",
            question=(
                f"Complete the following Python function. Return ONLY the code, "
                f"no explanation, no markdown fences.\n\n{prompt}"
            ),
            ground_truth=row.get("canonical_solution", ""),
            meta={
                "prompt": row["prompt"],
                "entry_point": row["entry_point"],
                "test": row["test"],
            },
        ))
    logger.info("Loaded %d HumanEval items", len(items))
    return items


def _load_mbpp(sample: int | None = None) -> list[BenchmarkItem]:
    """MBPP (google, 974 tasks): implement a described function; scored by
    executing the provided assert list against the generated solution."""
    raw = _github_get(
        "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
    )
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines()]

    if sample is not None and 0 < sample < len(rows):
        random.seed(42)
        rows = random.sample(rows, sample)

    items = []
    for row in rows:
        items.append(BenchmarkItem(
            item_id=f"MBPP_{row['task_id']}",
            category="coding",
            question=(
                f"Write a Python function that satisfies: {row['text']}\n"
                f"Return ONLY the code, no explanation, no markdown fences."
            ),
            ground_truth=row.get("code", ""),
            meta={
                "test_setup": row.get("test_setup_code", ""),
                "test_list": row.get("test_list", []),
            },
        ))
    logger.info("Loaded %d MBPP items", len(items))
    return items


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DATASETS: dict[str, callable] = {
    "smoke": _get_smoke,
    "mmlu": _get_mmlu,
    "mmlu_pro": _get_mmlu_pro,
    "bfcl": _get_bfcl,
    "mmstar": _get_mmstar,
    "ocrbench": _get_ocrbench,
    "humaneval": _load_humaneval,
    "mbpp": _load_mbpp,
}


def get_dataset(name: str = "mmlu", sample: int | None = None) -> list[BenchmarkItem]:
    builder = _DATASETS.get(name)
    if builder is None:
        raise KeyError(f"Unknown dataset: {name!r}. Available: {', '.join(list_datasets())}")
    return builder(sample)


def list_datasets() -> list[str]:
    return sorted(_DATASETS.keys())