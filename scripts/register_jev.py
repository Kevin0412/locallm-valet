"""Register provisioned local JEV scorers in an existing locallm-valet YAML."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from locallm_valet.config import load_config  # noqa: E402

BUDGETS = {
    "openjev_2b": 6,
    "openjev_9b": 19,
    "semif_4b": 10,
    "minijev_4b": 9,
    "simplejev_qwen3_4b": 10,
    "simplejev_qwen35_4b": 11,
}


def registry_entries(repository: Path, slot: str) -> dict:
    repository = repository.resolve()
    service = Path(__file__).resolve().with_name("serve_jev.py")
    if not service.is_file():
        raise ValueError("Gateway checkout requires scripts/serve_jev.py")
    if not (repository / "scripts/serve_backend.py").is_file():
        raise ValueError("JEV checkout requires scripts/serve_backend.py")
    configs = json.loads((repository / "configs/models.json").read_text())
    lock = json.loads((repository / "configs/backends.lock.json").read_text())
    entries = {}
    for config in configs:
        if config.get("official"):
            continue
        name = config["name"]
        entry = lock[name]
        python, model = Path(entry["python"]), Path(entry["model_dir"])
        if not python.is_file() or not os.access(python, os.X_OK) or not model.is_dir():
            raise ValueError(f"Local interpreter or checkpoint unavailable for {name}")
        if name not in BUDGETS:
            raise ValueError(f"Provide a measured resource budget for {name}")
        command = (
            f"{shlex.quote(str(python))} {shlex.quote(str(service))} "
            f"--repository {shlex.quote(str(repository))} --model {shlex.quote(name)} "
            "--model-path {model_path} --served-model-name {model_name} "
            "--host {host} --port {port} {extra_args}"
        )
        entries[name] = {
            "path": str(model),
            "slot": slot,
            "required_vram_gib": BUDGETS[name],
            "required_ram_gib": 0,
            "backend": {
                "api_protocol": "systemone",
                "command_template": command,
                "health_path": "/health",
                "max_concurrency": 1,
                "env": {
                    "PYTHONNOUSERSITE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_ENDPOINT": "https://hf-mirror.com",
                },
            },
        }
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--jev-repository", type=Path, required=True)
    parser.add_argument("--slot", default="gpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    entries = registry_entries(args.jev_repository, args.slot)
    if args.slot not in cfg.get("slots", {}):
        raise ValueError("Use an existing GPU slot shared with the other GPU models")
    models = cfg.setdefault("models", {})
    for name, entry in entries.items():
        if name in models and models[name] != entry:
            raise ValueError(
                f"Existing model {name} differs; reconcile its configuration first"
            )
        models[name] = entry
    serialized = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".jev-config-", suffix=".yaml", dir=args.config.parent
    )
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(serialized)
        load_config(temporary)
        if not args.dry_run:
            backup_root = args.config.parent / "data/config-backups"
            backup_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = backup_root / f"config-before-jev-{stamp}.yaml"
            shutil.copy2(args.config, backup)
            backup.chmod(0o600)
            Path(temporary).chmod(stat.S_IMODE(args.config.stat().st_mode))
            os.replace(temporary, args.config)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "models": list(entries),
                "slot": args.slot,
                "written": not args.dry_run,
                "config": str(args.config),
            }
        )
    )


if __name__ == "__main__":
    main()
