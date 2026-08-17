# -*- coding: utf-8 -*-
"""Benchmark report generator — Markdown with optional comparison table."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import BenchmarkReport, BenchmarkResult, SuiteStats

logger = logging.getLogger("locallm_valet.benchmark.report")


def render_report(
    *,
    results: list[BenchmarkResult],
    dataset_name: str = "builtin",
    output_dir: str = "benchmark_results",
    report_name: str = "benchmark_report.md",
    save_jsonl: bool = True,
    compare_labels: Optional[list[str]] = None,
) -> Path:
    """Run scoring on results, aggregate stats, and render a Markdown report.

    Args:
        results: All results from one or more benchmark runs.
        dataset_name: Name of the dataset used.
        output_dir: Directory for output files.
        report_name: Filename for the Markdown report.
        save_jsonl: Also save scored JSONL alongside the Markdown.
        compare_labels: If multiple models are in results, labels for the
                        comparison table (in model-alphabetical order).

    Returns:
        Path to the generated Markdown report.
    """

    from .scorer import score_result

    # Score all results
    for r in results:
        if r.is_correct is None:
            score_result(r)

    # Group by model
    from collections import OrderedDict

    model_results: dict[str, list[BenchmarkResult]] = OrderedDict()
    for r in results:
        model_results.setdefault(r.model_name, []).append(r)

    # Build SuiteStats per model
    stats_list: list[SuiteStats] = []
    for model_name, rlist in model_results.items():
        ss = SuiteStats(model_name=model_name, dataset_name=dataset_name)
        for r in rlist:
            cat = r.item.category
            correct = r.is_correct if r.is_correct is not None else False
            ss.add_result(cat, correct)
        ss.finalize()
        stats_list.append(ss)

    report = BenchmarkReport(
        dataset_name=dataset_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        stats=stats_list,
        results=results,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_jsonl:
        jsonl_path = out_dir / f"{dataset_name}_results.jsonl"
        report.to_jsonl(str(jsonl_path))
        logger.info("Scored JSONL saved to %s", jsonl_path)

    # Render Markdown
    md = _render_markdown(report, compare_labels)
    md_path = out_dir / report_name
    md_path.write_text(md, encoding="utf-8")
    logger.info("Report saved to %s", md_path)

    return md_path


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_markdown(report: BenchmarkReport, labels: Optional[list[str]] = None) -> str:
    lines: list[str] = []

    lines.append(f"# Benchmark Report — {report.dataset_name}")
    lines.append("")
    lines.append(f"- **Run at:** {report.timestamp}")
    lines.append(f"- **Total items:** {len(report.results)}")
    if report.stats:
        models_str = ", ".join(s.model_name for s in report.stats)
        lines.append(f"- **Models tested:** {models_str}")
    lines.append("")

    # ------------------------------
    # Overall accuracy comparison
    # ------------------------------
    if len(report.stats) > 1:
        lines.append("## Overall Accuracy Comparison")
        lines.append("")
        lines.append("| Model | Accuracy | Correct / Total |")
        lines.append("|---|---|---|")
        for s in sorted(report.stats, key=lambda x: x.accuracy, reverse=True):
            acc_pct = round(s.accuracy * 100, 1)
            label = _label_for(s.model_name, labels) if labels else s.model_name
            lines.append(f"| {label} | **{acc_pct}%** | {s.correct}/{s.total} |")
        lines.append("")

    # ------------------------------
    # Per-model category breakdown
    # ------------------------------
    for s in report.stats:
        lines.append(f"## {s.model_name}")
        lines.append("")
        lines.append(f"- **Accuracy:** {s.accuracy:.1%} ({s.correct}/{s.total})")
        lines.append("")

        lines.append("### Per-Category")
        lines.append("")
        lines.append("| Category | Accuracy | Correct / Total |")
        lines.append("|---|---|---|")
        for cat in ["fact", "reasoning", "math", "chinese", "instruction", "coding"]:
            if cat in s.per_category:
                v = s.per_category[cat]
                acc_pct = round(v["accuracy"] * 100, 1)
                lines.append(f"| {cat} | **{acc_pct}%** | {v['correct']}/{v['total']} |")
        lines.append("")

        # Show error cases
        model_items = [r for r in report.results if r.model_name == s.model_name]
        errors = [r for r in model_items if r.is_correct is not True]
        if errors:
            lines.append("### Failed Items")
            lines.append("")
            lines.append("| ID | Category | Extracted | Ground Truth |")
            lines.append("|---|---|---|---|")
            for e in errors[:20]:  # limit to 20
                extracted = e.extracted_answer or "(none)"
                if len(extracted) > 40:
                    extracted = extracted[:40] + "..."
                lines.append(f"| {e.item.item_id} | {e.item.category} | {extracted} | {e.item.ground_truth} |")
            if len(errors) > 20:
                lines.append(f"| … | ({len(errors) - 20} more) | | |")
            lines.append("")

    # ------------------------------
    # Latency / throughput
    # ------------------------------
    has_latency = any(r.latency_ms is not None for r in report.results)
    has_tps = any(r.tps is not None for r in report.results)

    if has_latency or has_tps:
        lines.append("## Latency & Throughput")
        lines.append("")
        for model_name in sorted({r.model_name for r in report.results}):
            model_items = [r for r in report.results if r.model_name == model_name]
            latencies = [r.latency_ms for r in model_items if r.latency_ms is not None]
            tps_list = [r.tps for r in model_items if r.tps is not None]
            if latencies:
                avg_lat = round(sum(latencies) / len(latencies), 1)
                lines.append(f"- **{model_name}:** avg latency = {avg_lat} ms")
            if tps_list:
                avg_tps = round(sum(tps_list) / len(tps_list), 2)
                lines.append(f"  avg throughput = {avg_tps} tok/s")
        lines.append("")

    # ------------------------------
    # Comparison summary table if multi-model
    # ------------------------------
    if len(report.stats) > 1:
        lines.append("## Cross-Model Category Comparison")
        lines.append("")
        # Header row
        cats = ["fact", "reasoning", "math", "chinese", "instruction", "coding"]
        header = "| Model | " + " | ".join(c.capitalize() for c in cats) + " | **Overall** |"
        sep = "|" + "|".join("---" for _ in range(len(cats) + 2)) + "|"
        lines.append(header)
        lines.append(sep)
        for s in sorted(report.stats, key=lambda x: x.accuracy, reverse=True):
            label = _label_for(s.model_name, labels) if labels else s.model_name
            row = f"| {label} |"
            for cat in cats:
                if cat in s.per_category:
                    v = s.per_category[cat]
                    row += f" {round(v['accuracy']*100,1)}% |"
                else:
                    row += " N/A |"
            row += f" **{round(s.accuracy*100,1)}%** |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def _label_for(model_name: str, labels: list[str]) -> str:
    """Map model name to human-readable label if available."""
    # labels are ordered by caller in compare mode; fallback to model_name
    return model_name