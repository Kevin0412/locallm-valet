# -*- coding: utf-8 -*-
"""locallm-valet benchmark CLI — subcommand entry point.

Usage via ``python -m locallm_valet benchmark``:

    run           Run benchmark on one model through the valet's API
    compare       Run on multiple models and generate comparison report
    list-datasets Show available built-in datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .dataset import get_dataset, list_datasets
from .runner import probe_single_request_stats, run_benchmark, save_speed
from .report import render_report

logger = logging.getLogger("locallm_valet.benchmark")


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
    run_p.add_argument("--concurrency", type=int, default=4,
                       help="Parallel requests; capped by the backend's max_concurrency "
                            "when the model declares one (default 4).")
    run_p.add_argument("--retries", type=int, default=2,
                       help="Extra attempts per item on timeout/5xx (default: 2).")
    run_p.add_argument("--api-key", default=None,
                       help="Valet API key. Defaults to the key in config.yaml (server.api_key).")

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
    cmp_p.add_argument("--concurrency", type=int, default=4,
                       help="Parallel requests per model; capped by the backend's "
                            "max_concurrency when declared (default 4).")
    cmp_p.add_argument("--retries", type=int, default=2,
                       help="Extra attempts per item on timeout/5xx (default: 2).")
    cmp_p.add_argument("--api-key", default=None,
                       help="Valet API key. Defaults to the key in config.yaml (server.api_key).")

    # -- all --
    all_p = bench_sub.add_parser("all", help="Run benchmark on EVERY registered model and produce cross-model comparison")
    all_p.add_argument("--dataset", default="mmlu",
                       help=f"Dataset name (default: mmlu). Available: {', '.join(list_datasets())}")
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
    all_p.add_argument("--sample", type=int, default=None,
                       help="Sample N items (uses sample_N_indices.json when available; default: full).")
    all_p.add_argument("--concurrency", type=int, default=4,
                       help="Parallel requests per model; capped by each backend's "
                            "max_concurrency when declared (default 4).")
    all_p.add_argument("--retries", type=int, default=2,
                       help="Extra attempts per item on timeout/5xx (default: 2).")
    all_p.add_argument("--thinking", action="store_true",
                       help="Enable thinking mode (Qwen3 reasoning). Default: non-thinking (fast).")
    all_p.add_argument("--api-key", default=None,
                       help="Valet API key. Defaults to the key in config.yaml (server.api_key).")

    # -- download --
    dl_p = bench_sub.add_parser("download",
                                help="Download benchmark datasets into the local cache")
    dl_p.add_argument("--datasets", nargs="*", default=None,
                      help="Dataset names to fetch (default: all cached datasets).")
    dl_p.add_argument("--sample", type=int, default=500,
                      help="Also write a fixed sample_<N>_indices.json (default: 500).")
    dl_p.add_argument("--cache-dir", default=None,
                      help="Cache root (default: LOCALLM_VALET_DATASET_CACHE or ./dataset_cache/dataset).")
    dl_p.add_argument("--mirror", default="hf",
                      help="Source: hf (default) | hf-mirror | tuna (needs LOCALLM_VALET_HF_ENDPOINT) | modelscope")

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
    probe_p.add_argument("--api-key", default=None,
                         help="Valet API key. Defaults to the key in config.yaml (server.api_key).")


def _resolve_api_key(args: argparse.Namespace) -> str:
    """Explicit --api-key wins; otherwise read the valet's config.yaml."""

    if getattr(args, "api_key", None):
        return args.api_key
    try:
        from ..config import load_config

        cfg = load_config()
        if cfg.server.api_keys:
            return cfg.server.api_keys[0]
    except Exception:  # noqa: BLE001 - key is optional; benchmark may run keyless
        pass
    return ""


def _fetch_concurrency_caps(base_url: str, api_key: str) -> dict[str, int]:
    """Fetch per-model backend ``max_concurrency`` from the valet's gateway.

    The gateway (not the benchmark) knows each model's backend — llama.cpp
    can only serve ``n_slots`` requests truly in parallel, SGLang/vLLM batch.
    Returns ``{model: cap}`` for models that declare a cap; ``{}`` when the
    gateway is unreachable (the CLI then falls back to ``--concurrency``).
    """
    import httpx

    gw = base_url.rstrip("/")
    if gw.endswith("/v1"):
        gw = gw[: -len("/v1")]
    url = f"{gw}/gateway/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        caps: dict[str, int] = {}
        for m in resp.json().get("models", []):
            cap = m.get("max_concurrency")
            if isinstance(cap, int) and cap >= 1:
                caps[m["name"]] = cap
        return caps
    except Exception as exc:  # noqa: BLE001 - caps are an optimization, never fatal
        logger.warning("Could not fetch backend concurrency caps from %s: %s", url, exc)
        return {}


def _capped_concurrency(requested: int, caps: dict[str, int], model: str) -> int:
    """Cap ``requested`` concurrency to the backend's declared limit."""
    cap = caps.get(model)
    if not cap:
        return requested
    return min(requested, cap)


def main(args: argparse.Namespace) -> int:
    """Entry point for the ``benchmark`` subcommand."""

    if args.bench_cmd == "probe":
        from .probe import probe_speed, probe_to_jsonl
        from pathlib import Path
        print(f"Speed probe: {args.model} @ {args.base_url}")
        result = probe_speed(model_name=args.model, base_url=args.base_url,
                             api_key=_resolve_api_key(args),
                             include_cold_start=not args.no_cold_start)
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{args.model}_speed_probe.jsonl"
        probe_to_jsonl(result, str(path))
        print(f"\nProbe saved to: {path}")
        return 0

    api_key = _resolve_api_key(args)

    if args.bench_cmd == "list-datasets":
        print("Available datasets:")
        for name in list_datasets():
            print(f"  {name}")
        return 0

    if args.bench_cmd == "download":
        from pathlib import Path

        from .dataset import _cache_dir
        from .download import download_datasets, _SOURCES

        names = args.datasets or sorted(_SOURCES.keys())
        cache = Path(args.cache_dir) if args.cache_dir else _cache_dir()
        print(f"Downloading {len(names)} dataset(s) -> {cache}  (mirror={args.mirror}, sample={args.sample})")
        status = download_datasets(
            names=names, cache_dir=cache, sample=args.sample, mirror=args.mirror,
        )
        for name, st in status.items():
            print(f"  {name:10s} {st}")
        failed = [n for n, st in status.items() if not st.startswith("ok")]
        if failed:
            print(f"Failed: {', '.join(failed)}", file=sys.stderr)
            return 1
        return 0

    # Load the dataset
    try:
        items = get_dataset(args.dataset, sample=getattr(args, "sample", None))
    except KeyError as exc:
        print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
        print(f"Available: {', '.join(list_datasets())}", file=sys.stderr)
        return 1
    logger.info("Loaded %d items from dataset '%s' (sample=%s)",
                len(items), args.dataset, getattr(args, "sample", None))

    if args.bench_cmd == "run":
        logger.info("Benchmark run — model=%s  dataset=%s  base_url=%s",
                    args.model, args.dataset, args.base_url)
        caps = _fetch_concurrency_caps(args.base_url, api_key)
        concurrency = _capped_concurrency(args.concurrency, caps, args.model)
        if concurrency != args.concurrency:
            logger.info("model %s: concurrency capped %d -> %d (backend limit)",
                        args.model, args.concurrency, concurrency)
        results = run_benchmark(
            items=items,
            model_name=args.model,
            base_url=args.base_url,
            api_key=api_key,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
            concurrency=concurrency,
            retries=args.retries,
            enable_thinking=getattr(args, "thinking", False),
        )
        stats = probe_single_request_stats(model_name=args.model, base_url=args.base_url, api_key=api_key)
        save_speed(args.model, stats)
        if stats:
            logger.info("single-request throughput %s: prefill=%.0f decode=%.0f tok/s",
                        args.model, stats["prefill_tps"], stats["decode_tps"])
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
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = httpx.get(f"{args.base_url}/models", headers=headers, timeout=10)
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
        caps = _fetch_concurrency_caps(args.base_url, api_key)
        all_results = []
        for model_name in models_to_run:
            concurrency = _capped_concurrency(args.concurrency, caps, model_name)
            if concurrency != args.concurrency:
                logger.info("model %s: concurrency capped %d -> %d (backend limit)",
                            model_name, args.concurrency, concurrency)
            logger.info("Benchmarking model=%s  (%d items, concurrency=%d)",
                        model_name, len(items), concurrency)
            results = run_benchmark(
                items=items,
                model_name=model_name,
                base_url=args.base_url,
                api_key=api_key,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
                concurrency=concurrency,
                retries=args.retries,
                enable_thinking=getattr(args, "thinking", False),
            )
            all_results.extend(results)
            stats = probe_single_request_stats(model_name=model_name, base_url=args.base_url, api_key=api_key)
            save_speed(model_name, stats)
            if stats:
                logger.info("single-request throughput %s: prefill=%.0f decode=%.0f tok/s",
                            model_name, stats["prefill_tps"], stats["decode_tps"])
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
        caps = _fetch_concurrency_caps(args.base_url, api_key)
        all_results = []
        for model_name in models:
            concurrency = _capped_concurrency(args.concurrency, caps, model_name)
            if concurrency != args.concurrency:
                logger.info("model %s: concurrency capped %d -> %d (backend limit)",
                            model_name, args.concurrency, concurrency)
            logger.info("Running benchmark — model=%s (concurrency=%d)", model_name, concurrency)
            results = run_benchmark(
                items=items,
                model_name=model_name,
                base_url=args.base_url,
                api_key=api_key,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
                concurrency=concurrency,
                retries=args.retries,
                enable_thinking=getattr(args, "thinking", False),
            )
            all_results.extend(results)
            stats = probe_single_request_stats(model_name=model_name, base_url=args.base_url, api_key=api_key)
            save_speed(model_name, stats)
            if stats:
                logger.info("single-request throughput %s: prefill=%.0f decode=%.0f tok/s",
                            model_name, stats["prefill_tps"], stats["decode_tps"])
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