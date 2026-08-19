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
    """Load processed.json (+ dev_examples.json if present) from cache.

    ``sample``: when set, apply the fixed ``sample_<N>_indices.json`` subset
    written by ``benchmark download`` (fallback: first N rows).
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

    if sample is not None and sample > 0 and sample < len(items):
        idx_p = cache / name / f"sample_{sample}_indices.json"
        if idx_p.is_file():
            with open(idx_p, "r", encoding="utf-8") as f:
                indices: list[int] = json.load(f)
            indices = [i for i in indices if 0 <= i < len(items)]
            items = [items[i] for i in indices]
        else:
            items = items[:sample]

    dev: Optional[dict] = None
    dev_p = cache / name / "dev_examples.json"
    if dev_p.is_file():
        with open(dev_p, "r", encoding="utf-8") as f:
            dev = json.load(f)

    logger.info("Loaded %d items from %s", len(items), p)
    return items, dev


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
            category=row.get("category", "fact"),
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
        # Build question text from messages if available
        msgs = row.get("messages")
        if msgs:
            question_text = "\n".join(m["content"] for m in msgs if m.get("content"))
        else:
            question_text = row.get("question", row.get("description", ""))
        gt = row["ground_truth"]
        choices_raw = row.get("options", [])
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
        out.append(BenchmarkItem(
            item_id=row["item_id"], category=row.get("category", "fact"),
            question=prompt, ground_truth=gt,
            choices=choices, language=lang,
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

def _get_bfcl(sample: int | None = None) -> list[BenchmarkItem]:
    items_raw, dev = _load_processed("bfcl", sample)
    return _convert(items_raw, dev)

def _get_mmstar(sample: int | None = None) -> list[BenchmarkItem]:
    items_raw, dev = _load_processed("mmstar", sample)
    return _convert(items_raw, dev)

def _get_ocrbench(sample: int | None = None) -> list[BenchmarkItem]:
    items_raw, dev = _load_processed("ocrbench", sample)
    return _convert(items_raw, dev)


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
}


def get_dataset(name: str = "mmlu", sample: int | None = None) -> list[BenchmarkItem]:
    builder = _DATASETS.get(name)
    if builder is None:
        raise KeyError(f"Unknown dataset: {name!r}. Available: {', '.join(list_datasets())}")
    return builder(sample)


def list_datasets() -> list[str]:
    return sorted(_DATASETS.keys())