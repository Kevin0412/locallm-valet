# -*- coding: utf-8 -*-
"""Benchmark datasets — one built‑in smoke set (zero deps) + MMLU downloaded
from HuggingFace.

Datasets registered:
- "smoke" — 6 hardcoded items, no download needed, quick sanity check.
- "mmlu" — full MMLU (57 subjects, 5‑shot), requires ``pip install datasets``.
- "mmlu_pro" — MMLU‑Pro (more difficult), requires ``pip install datasets``.

Usage::

    from .dataset import get_dataset, list_datasets
    items = get_dataset("mmlu")         # auto‑downloads on first call
    items = get_dataset("smoke")        # no download, instant
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
# Built‑in smoke dataset (from smoke.json, zero dependencies)
# ---------------------------------------------------------------------------

_SMOKE_JSON = Path(__file__).resolve().parent / "smoke.json"


def _get_smoke() -> list[BenchmarkItem]:
    import json
    with open(_SMOKE_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items: list[BenchmarkItem] = []
    for d in raw:
        lang = "zh" if any("\u4e00" <= c <= "\u9fff" for c in d.get("question", "")) else "en"
        items.append(BenchmarkItem(
            item_id=d["item_id"],
            category=d["category"],
            question=d["question"],
            ground_truth=d["ground_truth"],
            choices=d.get("choices", []),
            language=lang,
        ))
    return items


# ---------------------------------------------------------------------------
# MMLU download helper (requires ``datasets`` package)
# ---------------------------------------------------------------------------

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_computer_science",
    "college_mathematics", "college_medicine", "college_physics",
    "computer_security", "conceptual_physics", "econometrics",
    "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history",
    "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous",
    "moral_disputes", "moral_scenarios", "nutrition", "philosophy",
    "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


_CACHE_DIRNAME = "dataset_cache"


def _cache_dir() -> Path:
    return Path(_CACHE_DIRNAME)


def _mmlu_cache_path() -> Path:
    return _cache_dir() / "mmlu_items.json"


def _dev_examples_path() -> Path:
    return _cache_dir() / "mmlu_dev.json"


def _get_mmlu() -> list[BenchmarkItem]:
    """Load / download MMLU (57 subjects, 5‑shot). Cached after first
    download. Requires ``pip install datasets``."""
    cache = _mmlu_cache_path()
    if cache.is_file():
        with open(cache, "r", encoding="utf-8") as f:
            raw = json.load(f)
        logger.info("Loaded %d items from MMLU cache (%s)", len(raw), cache)
        return _deserialize_items(raw)

    logger.info("MMLU cache not found, downloading from HuggingFace …")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "MMLU benchmark requires the 'datasets' package.\n"
            "Install: pip install datasets"
        )

    all_items: list[dict] = []
    dev_examples_all: dict[str, list[dict]] = {}

    for subject in MMLU_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subject, trust_remote_code=True)
        except Exception as exc:
            logger.warning("Failed to load MMLU subject '%s': %s", subject, exc)
            continue

        # 5‑shot: pick 5 from dev split
        dev = ds.get("dev", [])
        random.shuffle(dev)
        fewshot_text = _format_fewshot(dev[:5], subject)

        # Test items
        test = ds.get("test", [])
        for i, row in enumerate(test):
            question = row["question"]
            choices = row["choices"]
            answer_idx = row["answer"]  # 0‑3 int
            answer_letter = chr(ord("A") + answer_idx)

            # Build prompt: subject header + fewshot + question
            prompt = (
                f"The following are multiple choice questions (with answers) about {subject}.\n\n"
                f"{fewshot_text}\n"
                f"Question: {question}\n"
                f"A. {choices[0]}  B. {choices[1]}  C. {choices[2]}  D. {choices[3]}\n"
                f"Answer:"
            )

            all_items.append({
                "item_id": f"mmlu_{subject}_{i}",
                "category": "fact",
                "question": prompt,
                "ground_truth": answer_letter,
                "choices": [f"A. {choices[0]}", f"B. {choices[1]}",
                            f"C. {choices[2]}", f"D. {choices[3]}"],
            })

    # Save cache
    _cache_dir().mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False)
    logger.info("MMLU cached: %d items across %d subjects (%s)",
                len(all_items), len(set(it["item_id"].split("_")[1] for it in all_items)), cache)

    return _deserialize_items(all_items)


def _format_fewshot(rows: list[dict], subject: str) -> str:
    lines = []
    for row in rows:
        q = row["question"]
        c = row["choices"]
        a = chr(ord("A") + row["answer"])
        lines.append(
            f"Question: {q}\n"
            f"A. {c[0]}  B. {c[1]}  C. {c[2]}  D. {c[3]}\n"
            f"Answer: {a}\n"
        )
    return "\n".join(lines)


def _deserialize_items(raw: list[dict]) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    for d in raw:
        lang = "zh" if any("\u4e00" <= c <= "\u9fff" for c in d.get("question", "")) else "en"
        items.append(BenchmarkItem(
            item_id=d["item_id"],
            category=d.get("category", "fact"),
            question=d["question"],
            ground_truth=d["ground_truth"],
            choices=d.get("choices", []),
            language=lang,
        ))
    return items


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DATASETS: dict[str, callable] = {
    "smoke": _get_smoke,
    "mmlu": _get_mmlu,
}


def get_dataset(name: str = "smoke") -> list[BenchmarkItem]:
    builder = _DATASETS.get(name)
    if builder is None:
        raise KeyError(
            f"Unknown dataset: {name!r}. "
            f"Available: {', '.join(list_datasets())}"
        )
    return builder()


def list_datasets() -> list[str]:
    return sorted(_DATASETS.keys())