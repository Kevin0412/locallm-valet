"""CLI entry point: ``python -m llm_gateway --config config.yaml``."""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .api import create_app
from .config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-gateway",
        description="backend-agnostic LLM lifecycle gateway (single device, single active model).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to the YAML config (default: $LLM_GATEWAY_CONFIG or ./config.yaml)",
    )
    parser.add_argument("--host", default=None, help="listen host (overrides config)")
    parser.add_argument("--port", type=int, default=None, help="listen port (overrides config)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

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
