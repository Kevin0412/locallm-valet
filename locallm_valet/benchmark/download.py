# -*- coding: utf-8 -*-
"""Download benchmark datasets from HuggingFace and cache them locally.

Produces the cache layout the loader expects (see dataset.py)::

    <cache>/<name>/processed.json
    <cache>/<name>/dev_examples.json     (mmlu only)
    <cache>/<name>/sample_<N>_indices.json

Use ``HF_ENDPOINT=https://hf-mirror.com`` for networks where huggingface.co
is blocked (this machine).  Sources:

  - mmlu      -> cais/mmlu          (HF raw: question/choices/answer)
  - mmlu_pro  -> TIGER-Lab/MMLU-Pro (ref: options/answer)
  - bfcl      -> Shiyu-Ni/bfcl      (ref: messages/ground_truth)
  - mmstar    -> LIUqingqing/MMStar (ref: question/choices/answer)
  - ocrbench  -> echo840/OCRBench   (ref: question/ground_truth)
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

from .dataset import _stratified_indices

logger = logging.getLogger("locallm_valet.benchmark.download")

# dataset name -> HF path
_SOURCES = {
    "mmlu": "cais/mmlu",
    "mmlu_pro": "TIGER-Lab/MMLU-Pro",
    "bfcl": "Shiyu-Ni/bfcl",
    "mmstar": "LIUqingqing/MMStar",
    "ocrbench": "echo840/OCRBench",
}

# ModelScope dataset ids (their own org convention, not the HF paths).
_MODELSCOPE_IDS = {
    "mmlu": "modelscope/mmlu",
    "mmlu_pro": "TIGER-Lab/MMLU-Pro",
    # bfcl / mmstar / ocrbench are not mirrored on ModelScope (checked).
}


# Named mirror presets. ``tuna`` reads the endpoint from the
# LOCALLM_VALET_HF_ENDPOINT env var (fully configurable).
MIRRORS = {
    "hf": "https://huggingface.co",
    "hf-mirror": "https://hf-mirror.com",
}


def _apply_mirror(mirror: str) -> str:
    """Resolve the mirror name to an HF_ENDPOINT; returns the endpoint."""

    if mirror in MIRRORS:
        endpoint = MIRRORS[mirror]
    elif mirror == "tuna":
        endpoint = os.environ.get("LOCALLM_VALET_HF_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "mirror='tuna' requires LOCALLM_VALET_HF_ENDPOINT (a TUNA/any "
                "HF-compatible endpoint) to be set"
            )
    elif mirror == "modelscope":
        return "modelscope"
    else:
        raise ValueError(f"unknown mirror {mirror!r} (hf | hf-mirror | tuna | modelscope)")
    os.environ["HF_ENDPOINT"] = endpoint
    logger.info("dataset mirror: %s -> %s", mirror, endpoint)
    return endpoint


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _pick_indices(total: int, sample: int, rng: random.Random) -> list[int]:
    if sample is None or sample >= total:
        return list(range(total))
    return rng.sample(range(total), sample)


def _download_mmlu(cache: Path, sample: int | None, rng: random.Random) -> None:
    from datasets import load_dataset

    # cais/mmlu has one config per subject + 'all' + 'auxiliary_train'
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = [dict(r) for r in ds]
    idx = _stratified_indices(rows, sample)
    picked = rows

    # dev examples: 5-shot per subject present in the sample (streamed, so the
    # large auxiliary_train split is not fully downloaded)
    subjects = {r.get("subject", "general") for r in picked}
    dev: dict = {}
    try:
        aux = load_dataset("cais/mmlu", "auxiliary_train", split="train", streaming=True)
        for r in aux:
            subj = r.get("subject", "general")
            if subj not in subjects:
                continue
            dev.setdefault(subj, []).append(
                {"question": r["question"], "choices": list(r["choices"]), "answer": r["answer"]}
            )
            if len(dev[subj]) >= 5:
                subjects.discard(subj)
            if not subjects:
                break
    except Exception as exc:  # noqa: BLE001 - few-shot is best-effort
        logger.warning("dev examples unavailable: %s", exc)

    _write_json(cache / "processed.json", picked)
    if dev:
        _write_json(cache / "dev_examples.json", dev)
    _write_json(cache / f"sample_{sample}_indices.json", idx) if sample else None
    logger.info("mmlu: %d items cached", len(picked))


def _download_mmlu_pro(cache: Path, sample: int | None, rng: random.Random) -> None:
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = [dict(r) for r in ds]
    idx = _stratified_indices(rows, sample)
    picked = []
    for r in rows:
        options = list(r.get("options") or [])
        picked.append({
            "item_id": r.get("question_id", f"mmpro_{len(picked)}"),
            "subject": r.get("category", "general"),
            "question": r["question"],
            "ground_truth": r.get("answer", ""),
            "options": options,
        })
    _write_json(cache / "processed.json", picked)
    _write_json(cache / f"sample_{sample}_indices.json", idx) if sample else None
    logger.info("mmlu_pro: %d items cached", len(picked))


def _download_bfcl(cache: Path, sample: int | None, rng: random.Random) -> None:
    from datasets import load_dataset

    ds = load_dataset("Shiyu-Ni/bfcl", split="test")
    rows = [dict(r) for r in ds]
    idx = _stratified_indices(rows, sample)
    picked = []
    for r in rows:
        msgs = r.get("messages") or []
        gt = r.get("ground_truth", r.get("answer", ""))
        if isinstance(gt, list):
            gt = json.dumps(gt, ensure_ascii=False)
        picked.append({
            "item_id": r.get("id", r.get("question_id", f"bfcl_{len(picked)}")),
            "subject": r.get("category", r.get("function", "bfcl")),
            "question": str(r.get("question", r.get("prompt", ""))),
            "ground_truth": gt,
            "options": [],
            "messages": msgs,
        })
    _write_json(cache / "processed.json", picked)
    _write_json(cache / f"sample_{sample}_indices.json", idx) if sample else None
    logger.info("bfcl: %d items cached", len(picked))


def _download_mmstar(cache: Path, sample: int | None, rng: random.Random) -> None:
    from datasets import load_dataset

    ds = load_dataset("LIUqingqing/MMStar", split="test")
    rows = [dict(r) for r in ds]
    idx = _stratified_indices(rows, sample)
    picked = []
    for r in rows:
        choices = list(r.get("choices") or [])
        answer = r.get("answer", "")
        # answer may be a letter ("A") or an index
        if isinstance(answer, int) and choices:
            answer = chr(ord("A") + answer)
        picked.append({
            "item_id": r.get("id", f"mmstar_{len(picked)}"),
            "subject": "mmstar",
            "question": r.get("question", ""),
            "ground_truth": answer,
            "options": choices,
        })
    _write_json(cache / "processed.json", picked)
    _write_json(cache / f"sample_{sample}_indices.json", idx) if sample else None
    logger.info("mmstar: %d items cached", len(picked))


def _download_ocrbench(cache: Path, sample: int | None, rng: random.Random) -> None:
    from datasets import load_dataset

    ds = load_dataset("echo840/OCRBench", split="test")
    rows = [dict(r) for r in ds]
    idx = _stratified_indices(rows, sample)
    picked = []
    for r in rows:
        q = r.get("question") or r.get("query") or "Recognize the text in the image."
        picked.append({
            "item_id": r.get("id", f"ocr_{len(picked)}"),
            "subject": "ocrbench",
            "question": q,
            "ground_truth": r.get("answers", r.get("answer", "")),
            "options": [],
        })
    _write_json(cache / "processed.json", picked)
    _write_json(cache / f"sample_{sample}_indices.json", idx) if sample else None
    logger.info("ocrbench: %d items cached", len(picked))


_DOWNLOADERS = {
    "mmlu": _download_mmlu,
    "mmlu_pro": _download_mmlu_pro,
    "bfcl": _download_bfcl,
    "mmstar": _download_mmstar,
    "ocrbench": _download_ocrbench,
}


def download_datasets(
    names: list[str],
    cache_dir: Path,
    sample: int | None = None,
    seed: int = 42,
    mirror: str = "hf",
) -> dict[str, str]:
    """Download the requested datasets into ``cache_dir``.

    ``mirror`` selects the source: ``hf`` (huggingface.co, default),
    ``hf-mirror`` (community mirror), ``tuna`` (endpoint from
    ``LOCALLM_VALET_HF_ENDPOINT``), or ``modelscope`` (ModelScope dataset
    mirror).  Returns {dataset_name: status} where status is "ok" or an
    error message.
    """

    rng = random.Random(seed)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    if mirror == "modelscope":
        return _download_via_modelscope(names, cache, sample, rng)
    _apply_mirror(mirror)
    status: dict[str, str] = {}
    for name in names:
        if name not in _DOWNLOADERS:
            status[name] = "unknown dataset"
            continue
        try:
            _DOWNLOADERS[name](cache, sample, rng)
            status[name] = "ok"
        except Exception as exc:  # noqa: BLE001 - one bad dataset must not block the rest
            logger.exception("download of %s failed", name)
            status[name] = f"failed: {exc}"
    return status


def _download_via_modelscope(
    names: list[str], cache: Path, sample: int | None, rng: random.Random
) -> dict[str, str]:
    """ModelScope path: fetch the dataset snapshot with the modelscope CLI
    (fast in CN networks), then convert per-dataset into the cache layout."""

    import shutil
    import subprocess
    import tempfile

    status: dict[str, str] = {}
    for name in names:
        if name not in _DOWNLOADERS:
            status[name] = "unknown dataset"
            continue
        src = _MODELSCOPE_IDS.get(name, _SOURCES[name])
        try:
            with tempfile.TemporaryDirectory() as td:
                cmd = ["modelscope", "download", "--dataset", src, "--local_dir", td]
                logger.info("modelscope download: %s", " ".join(cmd))
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                # extract data.tar if present
                tar = Path(td) / "data.tar"
                if tar.is_file():
                    subprocess.run(["tar", "xf", str(tar), "-C", td], check=True)
                if name == "mmlu":
                    _ms_mmlu(td, cache, sample, rng)
                elif name == "mmlu_pro":
                    _ms_mmlu_pro(td, cache, sample, rng)
                else:
                    raise ValueError(
                        f"dataset {name!r} is not available on ModelScope "
                        "(only mmlu / mmlu_pro are mirrored); try --mirror hf"
                    )
                status[name] = "ok (modelscope)"
        except Exception as exc:  # noqa: BLE001
            logger.exception("modelscope download of %s failed", name)
            status[name] = f"failed: {exc}"
    return status


def _ms_mmlu(td: str, cache: Path, sample: int | None, rng: random.Random) -> None:
    """MMLU from ModelScope: per-subject CSV files (question,A,B,C,D,answer)."""

    import csv

    def read_csvs(path: Path) -> list[dict]:
        # raw MMLU csv: no header — columns are question,A,B,C,D,answer
        rows = []
        for f in sorted(path.glob("*.csv")):
            subject = f.name.removesuffix(".csv").removesuffix("_test").removesuffix("_dev").replace("_", " ")
            with open(f, newline="", encoding="utf-8") as fh:
                for row in csv.reader(fh):
                    if len(row) < 6:
                        continue
                    question, a, b, c, d, answer = row[:6]
                    rows.append({
                        "question": question,
                        "choices": [a, b, c, d],
                        "answer": {"A": 0, "B": 1, "C": 2, "D": 3}.get(
                            str(answer).strip().upper(), 0),
                        "subject": subject,
                    })
        return rows

    data = Path(td) / "data"
    tests = read_csvs(data / "test")
    idx = _stratified_indices(tests, sample)
    picked = tests

    dev: dict = {}
    try:
        for ex in read_csvs(data / "dev"):
            dev.setdefault(ex["subject"], []).append(
                {"question": ex["question"], "choices": ex["choices"], "answer": ex["answer"]}
            )
    except Exception:  # noqa: BLE001 - dev optional
        pass

    _write_json(cache / "mmlu" / "processed.json", picked)
    if dev:
        _write_json(cache / "mmlu" / "dev_examples.json", dev)
    if sample:
        _write_json(cache / "mmlu" / f"sample_{sample}_indices.json", idx)
    logger.info("mmlu(modelscope): %d items cached", len(picked))


def _ms_mmlu_pro(td: str, cache: Path, sample: int | None, rng: random.Random) -> None:
    """MMLU-Pro from ModelScope: data/test-*.parquet with question/options/answer."""

    from datasets import load_dataset

    root = Path(td)
    data_dir = root / "data" if (root / "data").is_dir() else root
    ds = load_dataset("parquet", data_dir=str(data_dir), split="test")
    rows = [dict(r) for r in ds]
    picked = []
    for r in rows:
        picked.append({
            "item_id": r.get("question_id", f"mmpro_{len(picked)}"),
            "subject": r.get("category", "general"),
            "question": r.get("question", ""),
            "ground_truth": r.get("answer", ""),
            "options": list(r.get("options") or []),
        })
    idx = _stratified_indices(picked, sample)
    _write_json(cache / "mmlu_pro" / "processed.json", picked)
    if sample:
        _write_json(cache / "mmlu_pro" / f"sample_{sample}_indices.json", idx)
    logger.info("mmlu_pro(modelscope): %d items cached", len(picked))


def _ms_generic(td: str, name: str, cache: Path, sample: int | None, rng: random.Random) -> None:
    """Generic ModelScope dataset: locate parquet/csv/jsonl files and convert."""

    from datasets import load_dataset

    root = Path(td)
    data_dir = root / "data" if (root / "data").is_dir() else root
    try:
        ds = load_dataset("parquet", data_dir=str(data_dir), split="train")
    except Exception:  # noqa: BLE001 - try json/csv fallbacks
        ds = load_dataset("json", data_dir=str(data_dir), split="train")
    rows = [dict(r) for r in ds]
    idx = _stratified_indices(rows, sample)
    picked = rows
    _write_json(cache / name / "processed.json", picked)
    if sample:
        _write_json(cache / name / f"sample_{sample}_indices.json", idx)
    logger.info("%s(modelscope): %d items cached", name, len(picked))
