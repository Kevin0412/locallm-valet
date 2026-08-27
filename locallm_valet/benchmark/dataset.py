# -*- coding: utf-8 -*-
"""Benchmark datasets — smoke (built-in) + datasets cached on disk.

Datasets:
  - "smoke"      — 6 built-in items, no cache needed
  - "mmlu"       — MMLU (57 subjects, 5-shot)
  - "mmlu_pro"   - MMLU-Pro
  - "bfcl"       — Berkeley Function Calling Leaderboard
  - "mmstar"     — MMStar
  - "ocrbench"   — OCRBench

Each cached dataset lives at ``<cache>/<name>/processed.json`` (plus optional
``dev_examples.json`` and ``sample_<N>_indices.json``).  The cache root is
configurable via the ``LOCALLM_VALET_DATASET_CACHE`` environment variable
(default: ``<project>/dataset_cache/dataset``); point it at wherever the
extracted dataset directory lives on your machine.
"""

from __future__ import annotations

import json
import logging
import os
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
# Local dataset cache (configurable root)
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "dataset_cache" / "dataset"


def _cache_dir() -> Path:
    """Dataset cache root.

    Override with the ``LOCALLM_VALET_DATASET_CACHE`` env var (e.g. point it
    at an extracted dataset directory on another drive / machine).
    """

    override = os.environ.get("LOCALLM_VALET_DATASET_CACHE")
    if override:
        return Path(override)
    return _DEFAULT_CACHE_DIR


def _load_processed(name: str, sample: int | None = None) -> tuple[list[dict], Optional[dict]]:
    """Load processed.json (FULL data) + dev_examples.json if present.

    ``sample``: when set, apply the fixed ``sample_{sample}_indices.json``
    subset written by ``benchmark download`` (reproducible). When no fixed
    indices file exists, fall back to a STRATIFIED random sample by
    subject/category (seed 42), persisted so later runs reuse the subset.
    ``None`` = full dataset.

    IMPORTANT: processed.json always holds the FULL dataset — sampling is
    purely an index into it, so a later ``sample=None`` run uses everything.
    """
    cache = _cache_dir()
    p = cache / name / "processed.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"Dataset '{name}' not cached at {p}. "
            f"Run 'benchmark download --datasets {name}' or put an extracted "
            f"dataset directory under the cache root (LOCALLM_VALET_DATASET_CACHE)."
        )
    with open(p, "r", encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    if sample is not None and 0 < sample < len(items):
        # 1) fixed indices shipped / written by `benchmark download`
        #    (already stratified by the upstream generator where available)
        idx_p = cache / name / f"sample_{sample}_indices.json"
        if idx_p.is_file():
            with open(idx_p, "r", encoding="utf-8") as f:
                indices: list[int] = json.load(f)
            indices = [i for i in indices if 0 <= i < len(items)]
            if indices:
                items = [items[i] for i in indices]
                logger.info("Sampled %d items via %s", len(items), idx_p.name)
                _write_sample_indices(cache / name, name, sample, indices)
                return _finish_load(name, items)

        # 2) no fixed indices → stratified sample by subject/category,
        #    persisted so every later run uses the same subset.
        indices = _stratified_indices(items, sample)
        items = [items[i] for i in indices]
        _write_sample_indices(cache / name, name, sample, indices)
        logger.info("Stratified-sampled %d items (seed=42, by subject/category)", len(items))

    return _finish_load(name, items)


def _finish_load(name: str, items: list[dict]) -> tuple[list[dict], Optional[dict]]:
    dev: Optional[dict] = None
    dev_p = _cache_dir() / name / "dev_examples.json"
    if dev_p.is_file():
        with open(dev_p, "r", encoding="utf-8") as f:
            dev = json.load(f)
    logger.info("Loaded %d items from %s", len(items), name)
    return items, dev


def _stratified_indices(items: list[dict], n: int) -> list[int]:
    """Proportional stratified sample indices by subject (MMLU) or category.

    Every subject/category contributes a share proportional to its size in
    the full set (minimum 1), so the sampled distribution mirrors the full
    dataset instead of a flat random draw over/under-representing subjects.
    Reproducible: seed 42.
    """
    random.seed(42)
    key_of = lambda r: r.get("subject") or r.get("category") or "other"
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(items):
        groups.setdefault(key_of(r), []).append(i)

    chosen: list[int] = []
    total = len(items)
    for idxs in groups.values():
        share = max(1, round(n * len(idxs) / total))
        chosen.extend(random.sample(idxs, min(share, len(idxs))))
    random.shuffle(chosen)
    return chosen[:n]


def _write_sample_indices(cache_dir: Path, name: str, sample: int, indices: list[int]) -> None:
    """Persist sample indices for reproducibility (overwrites stale ones)."""
    try:
        out = cache_dir / f"sample_{sample}_indices.json"
        out.write_text(json.dumps(indices), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - sampling must never break loading
        logger.warning("could not persist sample indices for %s: %s", name, exc)


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

        # Some datasets (e.g. MMLU-Pro) store the question and its options
        # separately. If the question text does not already contain the
        # options, render them as a multiple-choice block with an explicit
        # letter-only instruction — without it, models answer with a long
        # explanation and never state the letter within the token budget.
        if choices_raw and all(c not in question_text for c in choices):
            prompt = (
                f"Question: {question_text}\n"
                + "  ".join(choices)
                + "\nAnswer with the letter only: "
            )
        else:
            prompt = question_text

        lang = "zh" if any("\u4e00" <= c <= "\u9fff" for c in prompt) else "en"
        # Keep the real subject/category (MMLU subject, MMLU-Pro category) so
        # per-category statistics actually mean something.
        category = (row.get("category") or row.get("subject") or "fact")
        out.append(BenchmarkItem(
            item_id=row["item_id"], category=category,
            question=prompt, ground_truth=gt,
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
    meta so the runner can pass tools and the scorer can compare calls.

    Two data layouts are supported:
    - HF layout: ``messages`` / ``tools`` / ``expected_tool_calls`` (exact
      comparison scorer)
    - BFCL v3 layout: ``question`` (text) + ``functions`` (tool schemas);
      correctness is judged by the AST key-argument checker (function name +
      required params + types) — the official leaderboard scores
      simple/parallel/multiple this way without releasing ground truths.
    """
    items_raw, _ = _load_processed("bfcl", sample)
    out = []
    for row in items_raw:
        if row.get("functions"):
            # BFCL v3: question text + tool schemas, AST scoring
            out.append(BenchmarkItem(
                item_id=row["item_id"],
                category=str(row.get("subject") or "bfcl"),
                question=row.get("question", row.get("item_id", "")),
                ground_truth="(AST)",
                meta={
                    # OpenAI-compatible tool schemas for the runner
                    "tools": [{"type": "function", "function": f} for f in row["functions"]],
                    # raw schemas for the AST scorer
                    "functions": row["functions"],
                    "check": "ast",
                },
            ))
            continue
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

_CODE_CACHE = _cache_dir() / "code"


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
# Code benchmarks (HumanEval / MBPP) — downloaded from GitHub, cached locally.
# ---------------------------------------------------------------------------

def _github_get(url: str) -> bytes:
    """Fetch a raw GitHub file, honouring HTTPS_PROXY if set; caches under
    dataset_cache/code so the network is only hit once."""
    import hashlib

    code_cache = _cache_dir() / "code"
    code_cache.mkdir(parents=True, exist_ok=True)
    cache_f = code_cache / hashlib.sha1(url.encode()).hexdigest()
    if cache_f.is_file():
        return cache_f.read_bytes()

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