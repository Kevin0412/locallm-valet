"""CLI entry point: ``python -m locallm_valet --config config.yaml`` (serve)
or ``python -m locallm_valet benchmark run --model X`` (benchmark).
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .api import create_app
from .config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="locallm-valet",
        description="backend-agnostic LLM lifecycle gateway + model benchmark.",
    )
    sub = parser.add_subparsers(dest="subcommand", help="Subcommand (default: run the valet server).")

    # ---- benchmark subcommand ----
    from .benchmark.cli import build_subparser, main as benchmark_main
    build_subparser(sub)

    # ---- valet server: implicit when no subcommand ----
    parser.add_argument("--config", default=None,
                        help="path to the YAML config (default: $LOCALLM_VALET_CONFIG or ./config.yaml)")
    parser.add_argument("--host", default=None, help="listen host (overrides config)")
    parser.add_argument("--port", type=int, default=None, help="listen port (overrides config)")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # Delegate to benchmark subcommand
    if args.subcommand == "benchmark":
        return benchmark_main(args)

    # Default: run the valet server
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port

    app = create_app(config=config)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
