"""Runtime YAML write-back for locallm-valet configuration.

``config.load_config`` is read-only; the settings endpoints need to persist
changes (credentials, API keys, per-model backend overrides). Strategy:
load the raw YAML dict, mutate the target section, dump back with
``yaml.safe_dump``. Comments are not preserved (they are not representable
in the plain dict round-trip) but the file stays human-readable and every
scalar written here comes from validated values.

Writes are atomic: the new content goes to ``<path>.tmp`` first and is then
``os.replace``d into place so a crash mid-write can never leave a truncated
config behind.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file as a dict (empty dict for an empty file)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def save_yaml(path: str | Path, data: dict) -> None:
    """Atomically write ``data`` as YAML to ``path``."""
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(tmp, target)


def _server_section(data: dict) -> dict:
    section = data.get("server")
    if not isinstance(section, dict):
        section = {}
        data["server"] = section
    return section


def update_server_section(path: str | Path, updates: dict[str, Any]) -> None:
    """Update keys under ``server:`` (e.g. username / password / api_key).

    A value of ``None`` removes the key instead of writing ``null``.
    """
    data = load_yaml(path)
    server = _server_section(data)
    for key, value in updates.items():
        if value is None or value == [] or value == "":
            server.pop(key, None)
        else:
            server[key] = value
    save_yaml(path, data)


def update_model_backend(path: str | Path, model: str, updates: dict[str, Any]) -> None:
    """Update keys under ``models.<name>.backend:``.

    A value of ``None`` clears the override (the model falls back to the
    global backend settings), i.e. the key is removed from the file.
    """
    data = load_yaml(path)
    models = data.get("models")
    if not isinstance(models, dict):
        raise KeyError(f"no models section in {path}")
    entry = models.get(model)
    if not isinstance(entry, dict):
        raise KeyError(f"model {model!r} not found in {path}")
    backend = entry.get("backend")
    if not isinstance(backend, dict):
        backend = {}
        entry["backend"] = backend
    for key, value in updates.items():
        if value is None or value == []:
            backend.pop(key, None)
            if not backend:
                entry.pop("backend", None)
        else:
            backend[key] = value
    save_yaml(path, data)
