# -*- coding: utf-8 -*-
"""Benchmark schema — item, result, report dataclasses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Dataset item
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkItem:
    """One question in a benchmark dataset."""

    item_id: str
    """Unique identifier."""
    category: str
    """Question category: fact / reasoning / math / chinese / instruction / coding."""
    question: str
    """The question text, in chat-message format (user turn)."""
    ground_truth: str
    """Expected answer string. For multiple choice this is the letter (A/B/C/D)."""
    choices: list[str] = field(default_factory=list)
    """Multiple-choice options if applicable, e.g. ["A. Paris", "B. London", …]."""
    language: str = "en"
    """Question language: en / zh."""
    meta: dict = field(default_factory=dict)
    """Extra per-item data for specialised scorers, e.g.
    {"entry_point": "has_close_elements", "test": "def check(candidate): ..."}
    for code-execution benchmarks (HumanEval / MBPP)."""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-item result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result of running one benchmark item through a model."""

    item: BenchmarkItem
    """The question that was asked."""
    model_name: str
    """Registry name of the model under test."""
    raw_response: str = ""
    """Full response text from the model."""
    extracted_answer: Optional[str] = None
    """Answer extracted from raw_response by the scorer."""
    is_correct: Optional[bool] = None
    """Whether extracted answer matches ground_truth."""
    score_detail: str = ""
    """Verbose explanation from the scorer (e.g. 'matched letter B')."""
    thinking: bool = False
    """Whether this request ran in thinking mode (reasoning on)."""
    ttft_ms: Optional[float] = None
    """Time to first token in milliseconds (backend timings when available)."""
    decode_ms: Optional[float] = None
    """Decode (generation) duration in milliseconds (backend timings)."""
    tps: Optional[float] = None
    """Tokens per second generation throughput."""
    latency_ms: Optional[float] = None
    """Total request round-trip time in milliseconds."""
    prompt_tokens: Optional[int] = None
    """Input token count from usage."""
    completion_tokens: Optional[int] = None
    """Output token count from usage (includes reasoning tokens when thinking)."""
    reasoning_tokens: int = 0
    """Length of the reasoning_content snippet (proxy for reasoning usage)."""

    def to_dict(self) -> dict:
        return {
            "item_id": self.item.item_id,
            "category": self.item.category,
            "question": self.item.question,
            "choices": self.item.choices,
            "ground_truth": self.item.ground_truth,
            "model_name": self.model_name,
            "thinking": self.thinking,
            "raw_response": self.raw_response,
            "extracted_answer": self.extracted_answer,
            "is_correct": self.is_correct,
            "score_detail": self.score_detail,
            "ttft_ms": self.ttft_ms,
            "decode_ms": self.decode_ms,
            "tps": self.tps,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


# ---------------------------------------------------------------------------
# Dataset / suite stats
# ---------------------------------------------------------------------------

@dataclass
class SuiteStats:
    """Aggregated accuracy stats for one model on a dataset suite."""

    model_name: str
    dataset_name: str
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    per_category: dict[str, dict] = field(default_factory=dict)
    """{category: {total, correct, accuracy}}"""

    def add_result(self, cat: str, correct: bool) -> None:
        self.total += 1
        if correct:
            self.correct += 1
        if cat not in self.per_category:
            self.per_category[cat] = {"total": 0, "correct": 0, "accuracy": 0.0}
        self.per_category[cat]["total"] += 1
        if correct:
            self.per_category[cat]["correct"] += 1

    def finalize(self) -> None:
        self.accuracy = round(self.correct / self.total, 4) if self.total else 0.0
        for v in self.per_category.values():
            v["accuracy"] = round(v["correct"] / v["total"], 4) if v["total"] else 0.0


# ---------------------------------------------------------------------------
# Full report (one model on one dataset, or comparison across models)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    """A complete benchmark report that can be rendered to Markdown."""

    dataset_name: str
    """Name of the dataset used."""
    timestamp: str = ""
    """ISO-8601 timestamp when the benchmark was run."""
    stats: list[SuiteStats] = field(default_factory=list)
    """One per model tested (or a single entry for a single-model run)."""
    results: list[BenchmarkResult] = field(default_factory=list)
    """All per-item results (for detailed export)."""
    model_params: dict[str, dict] = field(default_factory=dict)
    """Optional per-model metadata: {model_name: {required_ram, path, …}}."""

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "stats": [asdict(s) for s in self.stats],
            "results": [r.to_dict() for r in self.results],
            "model_params": self.model_params,
        }

    def to_jsonl(self, path: str) -> None:
        """Write all results to a JSONL file (one JSON object per line)."""
        with open(path, "w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: str, dataset_name: str = "") -> BenchmarkReport:
        """Reconstruct from JSONL + an optional dataset name."""
        results: list[BenchmarkResult] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                item = BenchmarkItem(
                    item_id=d["item_id"],
                    category=d["category"],
                    question=d["question"],
                    ground_truth=d["ground_truth"],
                    choices=d.get("choices", []),
                    language="zh" if any("\u4e00" <= c <= "\u9fff" for c in d["question"]) else "en",
                )
                br = BenchmarkResult(
                    item=item,
                    model_name=d["model_name"],
                    raw_response=d.get("raw_response", ""),
                    extracted_answer=d.get("extracted_answer"),
                    is_correct=d.get("is_correct"),
                    score_detail=d.get("score_detail", ""),
                    ttft_ms=d.get("ttft_ms"),
                    tps=d.get("tps"),
                    latency_ms=d.get("latency_ms"),
                )
                results.append(br)
        return cls(dataset_name=dataset_name, results=results)