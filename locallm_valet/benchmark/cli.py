# -*- coding: utf-8 -*-
"""locallm-valet benchmark CLI — subcommand entry point.

Usage via ``python -m locallm_valet benchmark``:

    run           Run benchmark on one model through the valet's API
    compare       Run on multiple models and generate comparison report
    list-datasets Show available built-in datasets
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .dataset import get_dataset, list_datasets
from .runner import run_benchmark
from .report import render_report


def build_subparser(sub: argparse._SubParsersAction) -> None:
    """Add the ``benchmark`` command and its subcommands.

    Usage::

        python -m locallm_valet benchmark run --model Qwen3-1.7B-Q8_0
        python -m locallm_valet benchmark compare --models A B
        python -m locallm_valet benchmark list-datasets
    """
    parser = sub.add_parser("benchmark", help="Run model benchmark / quality comparison")
    bench_sub = parser.add_subparsers(dest="bench_cmd", required=True)

    # -- run --
    run_p = bench_sub.add_parser("run", help="Run benchmark on one model")
    run_p.add_argument("--model", required=True,
                       help="Model registry name (as the valet knows it).")
    run_p.add_argument("--dataset", default="mmlu",
                       help=f"Dataset name (default: mmlu). Available: {', '.join(list_datasets())}")
    run_p.add_argument("--sample", type=int, default=None,
                       help="Sample N items (uses sample_N_indices.json when available; default: full).")
    run_p.add_argument("--concurrency", type=int, default=4,
                       help="Parallel requests (batch). Backends like SGLang/vLLM benefit; default: 4.")
    run_p.add_argument("--thinking", action="store_true",
                       help="Enable thinking mode (Qwen3 reasoning). Default: non-thinking (fast).")
    run_p.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                       help="Valet OpenAI-compatible base URL (default: http://127.0.0.1:8000/v1).")
    run_p.add_argument("--output-dir", default="benchmark_results",
                       help="Output directory for results / report (default: benchmark_results/).")
    run_p.add_argument("--max-tokens", type=int, default=256,
                       help="Max generation tokens per question (default: 256).")
    run_p.add_argument("--timeout", type=int, default=180,
                       help="Per-request timeout in seconds (default: 180).")

    # -- compare --
    cmp_p = bench_sub.add_parser("compare",
                                 help="Run benchmark on multiple models and generate comparison report")
    cmp_p.add_argument("--models", nargs="+", required=True,
                       help="Two or more model registry names to compare.")
    cmp_p.add_argument("--labels", nargs="*", default=None,
                       help="Human-readable labels for each model (same order).")
    cmp_p.add_argument("--dataset", default="mmlu",
                       help=f"Dataset name (default: mmlu).")
    cmp_p.add_argument("--sample", type=int, default=None,
                       help="Sample N items.")
    cmp_p.add_argument("--concurrency", type=int, default=4,
                       help="Parallel requests (batch); default: 4.")
    cmp_p.add_argument("--thinking", action="store_true",
                       help="Enable thinking mode (Qwen3 reasoning). Default: non-thinking.")
    cmp_p.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                       help="Valet OpenAI-compatible base URL.")
    cmp_p.add_argument("--output-dir", default="benchmark_results",
                       help="Output directory.")
    cmp_p.add_argument("--max-tokens", type=int, default=256,
                       help="Max generation tokens.")
    cmp_p.add_argument("--timeout", type=int, default=180,
                       help="Per-request timeout in seconds.")

    # -- all --
    all_p = bench_sub.add_parser("all", help="Run benchmark on EVERY registered model and produce cross-model comparison")
    all_p.add_argument("--dataset", default="mmlu",
                       help=f"Dataset name (default: mmlu). Available: {', '.join(list_datasets())}")
    all_p.add_argument("--sample", type=int, default=None,
                       help="Sample N items per model (uses sample_N_indices.json when available).")
    all_p.add_argument("--concurrency", type=int, default=4,
                       help="Parallel requests (batch); default: 4.")
    all_p.add_argument("--thinking", action="store_true",
                       help="Enable thinking mode (Qwen3 reasoning). Default: non-thinking.")
    all_p.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                       help="Valet OpenAI-compatible base URL (default: http://127.0.0.1:8000/v1).")
    all_p.add_argument("--output-dir", default="benchmark_results",
                       help="Output directory (default: benchmark_results/).")
    all_p.add_argument("--max-tokens", type=int, default=256,
                       help="Max generation tokens per question (default: 256).")
    all_p.add_argument("--timeout", type=int, default=180,
                       help="Per-request timeout in seconds (default: 180).")
    all_p.add_argument("--skip-models", nargs="*", default=[],
                       help="Optional list of model names to skip (e.g. 'llama-3.2-1b').")

    # -- list-datasets --
    bench_sub.add_parser("list-datasets", help="Show available datasets")

    # -- probe --
    probe_p = bench_sub.add_parser("probe", help="Speed probe: TTFT / prefill / decode / cold start / concurrency")
    probe_p.add_argument("--model", required=True, help="Model registry name.")
    probe_p.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                         help="Valet OpenAI-compatible base URL.")
    probe_p.add_argument("--no-cold-start", action="store_true",
                         help="Skip the cold-start (first request incl. load) probe.")
    probe_p.add_argument("--output-dir", default="benchmark_results",
                         help="Directory for the probe JSONL.")


def main(args: argparse.Namespace) -> int:
    """Entry point for the ``benchmark`` subcommand."""

    logger = logging.getLogger("locallm_valet.benchmark")

    if args.bench_cmd == "probe":
        from .probe import probe_speed, probe_to_jsonl
        from pathlib import Path
        print(f"Speed probe: {args.model} @ {args.base_url}")
        result = probe_speed(
            model_name=args.model,
            base_url=args.base_url,
            include_cold_start=not args.no_cold_start,
        )
        import json
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{args.model}_speed_probe.jsonl"
        probe_to_jsonl(result, str(path))
        print(f"\nProbe saved to: {path}")
        return 0

    if args.bench_cmd == "list-datasets":
        print("Available datasets:")
        for name in list_datasets():
            print(f"  {name}")
        return 0

    # Load the dataset
    try:
        items = get_dataset(args.dataset, getattr(args, "sample", None))
    except KeyError as exc:
        print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
        print(f"Available: {', '.join(list_datasets())}", file=sys.stderr)
        return 1
    logger.info("Loaded %d items from dataset '%s' (sample=%s)",
                len(items), args.dataset, getattr(args, "sample", None))

    if args.bench_cmd == "run":
        logger.info("Benchmark run — model=%s  dataset=%s  base_url=%s",
                    args.model, args.dataset, args.base_url)
        results = run_benchmark(
            items=items,
            model_name=args.model,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
            concurrency=args.concurrency,
            enable_thinking=getattr(args, "thinking", False),
        )
        out_path = render_report(
            results=results,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            report_name=f"benchmark_{args.model}.md",
        )
        print(f"\nReport saved to: {out_path}")
        return 0

    if args.bench_cmd == "all":
        import httpx
        # Fetch model list from the valet's /v1/models
        try:
            resp = httpx.get(f"{args.base_url}/models", timeout=10)
            resp.raise_for_status()
        except Exception as exc:
            print(f"Failed to fetch model list from {args.base_url}/models: {exc}", file=sys.stderr)
            return 1
        models_in_registry = [m["id"] for m in resp.json().get("data", [])]
        skip = set(args.skip_models or [])
        models_to_run = [m for m in models_in_registry if m not in skip]
        if not models_to_run:
            print("No models to benchmark (all skipped or registry empty).", file=sys.stderr)
            return 1
        print(f"Found {len(models_in_registry)} models in registry, will benchmark {len(models_to_run)}:\n  {', '.join(models_to_run)}\n")
        all_results = []
        for model_name in models_to_run:
            logger.info("Benchmarking model=%s  (%d items)", model_name, len(items))
            results = run_benchmark(
                items=items,
                model_name=model_name,
                base_url=args.base_url,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
                concurrency=args.concurrency,
                enable_thinking=getattr(args, "thinking", False),
            )
            all_results.extend(results)
        out_path = render_report(
            results=all_results,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            report_name="benchmark_comparison.md",
            compare_labels=models_to_run,
        )
        print(f"\nComparison report saved to: {out_path}")
        return 0

    if args.bench_cmd == "compare":
        models = args.models
        all_results = []
        for model_name in models:
            logger.info("Running benchmark — model=%s", model_name)
            results = run_benchmark(
                items=items,
                model_name=model_name,
                base_url=args.base_url,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
                concurrency=args.concurrency,
                enable_thinking=getattr(args, "thinking", False),
            )
            all_results.extend(results)
        out_path = render_report(
            results=all_results,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            report_name="benchmark_comparison.md",
            compare_labels=args.labels,
        )
        print(f"\nComparison report saved to: {out_path}")
        return 0

    print(f"Unknown benchmark command: {args.bench_cmd}", file=sys.stderr)
    return 1