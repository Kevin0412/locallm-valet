"""Local System One service launched and unloaded by locallm-valet."""

import argparse
import json
import logging
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("serve_jev")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument(
        "--model", required=True, help="Provisioned benchmark configuration name"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument(
        "--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1"
    )
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repository.resolve()))
    sys.path.insert(0, str(args.repository.resolve() / "scripts"))
    from serve_backend import load_engine
    from jev_bench.provision import download_environment
    from jev_bench.util import read_json
    from locallm_valet.jev_api import DecisionService, InputError

    os.chdir(args.repository)
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            del os.environ[key]
    os.environ.update(download_environment())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = next(
        model
        for model in read_json("configs/models.json")
        if model["name"] == args.model
    )
    if config.get("official"):
        raise ValueError("Only local open implementations can be served")
    entry = read_json("configs/backends.lock.json")[args.model]
    if args.model_path.resolve() != Path(entry["model_dir"]).resolve():
        raise ValueError("Registry model path differs from the provisioned checkpoint")
    checkout = entry["checkout"]
    revision = subprocess.check_output(
        ["git", "-C", checkout, "rev-parse", "HEAD"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", checkout, "diff", "HEAD", "--"], text=True
    )
    if revision != entry["source_revision"] or changes.strip():
        raise ValueError("Upstream source differs from the pinned clean checkout")
    preflight, predict, max_options, details = load_engine(config, entry)
    service = DecisionService(
        args.served_model_name, config["model"], preflight, predict, max_options
    )
    metadata = {
        "model": args.served_model_name,
        "underlying_model": config["model"],
        "implementation": config["implementation"],
        "source_revision": revision,
        "model_revision": entry["model_revision"],
        "dtype": config["dtype"],
        "max_options": max_options,
        "max_tokens": config["max_tokens"],
        "capabilities": ["typed_decision", "choice", "noul", "score"],
        "upstream_details": details,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, value):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/health":
                self.send(200, {"status": "ready", "model": args.served_model_name})
            elif path == "/bench/metadata":
                self.send(200, metadata)
            elif path == "/v1/models":
                self.send(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": args.served_model_name,
                                "object": "model",
                                "owned_by": "local-jev",
                                "capabilities": metadata["capabilities"],
                            }
                        ],
                    },
                )
            else:
                self.send(404, {"error": {"message": "Unknown endpoint"}})

        def do_POST(self):
            if urlsplit(self.path).path != "/v1/systemone":
                return self.send(404, {"error": {"message": "Use POST /v1/systemone"}})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 4 * 1024 * 1024:
                    raise InputError("Request must be smaller than 4 MiB")
                payload = json.loads(self.rfile.read(length))
                self.send(200, service.evaluate(payload))
            except InputError as error:
                # The caller can fix these, so name the reason and keep a
                # server-side record for the gateway's log.
                logger.warning("rejected typed decision: %s", error)
                self.send(
                    422,
                    {
                        "error": {
                            "type": "invalid_request_error",
                            "message": str(error),
                        }
                    },
                )
            except (ValueError, TypeError, KeyError) as error:
                logger.warning(
                    "rejected malformed request: %s: %s", type(error).__name__, error
                )
                self.send(
                    422,
                    {
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Invalid typed decision or unsupported input",
                        }
                    },
                )
            except Exception as error:
                logger.exception("upstream inference failed")
                self.send(
                    500,
                    {
                        "error": {
                            "message": "Upstream inference failed",
                            "type": type(error).__name__,
                        }
                    },
                )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps({"ready": True, "model": args.served_model_name, "port": args.port}),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
