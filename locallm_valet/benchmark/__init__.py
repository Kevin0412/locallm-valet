# -*- coding: utf-8 -*-
"""locallm-valet benchmark — evaluate model capability & detect quantisation degradation."""

from __future__ import annotations

from .schema import BenchmarkItem, BenchmarkResult, BenchmarkReport, SuiteStats
from .dataset import get_dataset, list_datasets
from .runner import run_benchmark
from .scorer import score_result
from .report import render_report
from .store import BenchmarkStore, backfill_jsonl

__all__ = [
    "BenchmarkItem", "BenchmarkResult", "BenchmarkReport", "SuiteStats",
    "get_dataset", "list_datasets",
    "run_benchmark", "score_result", "render_report",
    "BenchmarkStore", "backfill_jsonl",
]