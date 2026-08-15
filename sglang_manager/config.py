"""Configuration loading & validation for sglang-manager.

All knobs live in a YAML file (default ``config.yaml``, override with
``--config`` or the ``SGLANG_MANAGER_CONFIG`` env var).  A few high-value
settings can additionally be overridden through environment variables:

- ``SGLANG_MANAGER_IDLE_TIMEOUT_SECONDS`` — idle auto-unload timeout (never
  hardcoded; the default in the YAML is 3600 s).
- ``SGLANG_MANAGER_PORT`` / ``SGLANG_MANAGER_HOST`` — manager listen address.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when the configuration file is missing or invalid."""


@dataclass
class ModelSglangArgs:
    """Extra SGLang launch arguments for one model."""

    mem_fraction_static: float | None = None
    context_length: int | None = None
    extra_args: list[str] = field(default_factory=list)
    # Extra environment variables, merged over the global ``sglang.env``
    # (e.g. ``SGLANG_USE_MODELSCOPE: "true"`` or an ``LD_PRELOAD``).
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelSpec:
    """One entry of the model registry."""

    name: str
    path: str
    required_vram_gib: float
    sglang: ModelSglangArgs = field(default_factory=ModelSglangArgs)


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class SglangConfig:
    host: str = "127.0.0.1"
    port: int = 30000
    health_path: str = "/health"
    startup_timeout_seconds: float = 180.0
    stop_timeout_seconds: float = 60.0
    # Base launch command. Defaults to ``[sys.executable, -m,
    # sglang.launch_server]``; override to point at a specific venv python.
    command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class GpuConfig:
    device: int = 0
    # Extra headroom added to required_vram_gib before allowing a start.
    safety_margin_gib: float = 4.0
    # How long to wait for VRAM to actually come back after stopping SGLang.
    vram_release_timeout_seconds: float = 120.0
    vram_poll_interval_seconds: float = 1.0


@dataclass
class IdleConfig:
    # Unload the model after this much time with zero active requests.
    # CONFIGURABLE on purpose — never hardcode this.
    timeout_seconds: float = 3600.0
    check_interval_seconds: float = 15.0


@dataclass
class UsageConfig:
    # Token usage recording (SQLite) + dashboard.
    enabled: bool = True
    db_path: str = "data/usage.db"  # ":memory:" is fine for tests


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    sglang: SglangConfig = field(default_factory=SglangConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    idle: IdleConfig = field(default_factory=IdleConfig)
    usage: UsageConfig = field(default_factory=UsageConfig)
    models: dict[str, ModelSpec] = field(default_factory=dict)

    def get_model(self, name: str) -> ModelSpec | None:
        return self.models.get(name)


def _require_mapping(raw: Any, section: str, path: str) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")
    return raw


def _parse_model_sglang(raw: Any, name: str) -> ModelSglangArgs:
    out = ModelSglangArgs()
    if not raw:
        return out
    for key, value in _require_mapping(raw, "sglang", f"models.{name}.sglang").items():
        if key == "mem_fraction_static":
            out.mem_fraction_static = float(value)
        elif key == "context_length":
            out.context_length = int(value)
        elif key == "extra_args":
            if not isinstance(value, list) or not all(isinstance(a, str) for a in value):
                raise ConfigError(f"models.{name}.sglang.extra_args: expected a list of strings")
            out.extra_args = list(value)
        elif key == "env":
            if not isinstance(value, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in value.items()
            ):
                raise ConfigError(f"models.{name}.sglang.env: expected a mapping of strings to strings")
            out.env = dict(value)
        else:
            raise ConfigError(f"models.{name}.sglang: unknown key {key!r}")
    return out


def _parse_model(name: str, raw: Any) -> ModelSpec:
    data = _require_mapping(raw, "model", f"models.{name}")
    path = data.get("path")
    if not isinstance(path, str) or not path:
        raise ConfigError(f"models.{name}: missing required string field 'path'")
    required = data.get("required_vram_gib")
    if required is None:
        raise ConfigError(f"models.{name}: missing required field 'required_vram_gib'")
    required = float(required)
    if required <= 0:
        raise ConfigError(f"models.{name}: required_vram_gib must be > 0")
    return ModelSpec(
        name=name,
        path=path,
        required_vram_gib=required,
        sglang=_parse_model_sglang(data.get("sglang"), name),
    )


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        raise ConfigError(f"env {name}: not a number: {value!r}") from None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ConfigError(f"env {name}: not an integer: {value!r}") from None


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate configuration from a YAML file (plus env overrides)."""
    if path is None:
        path = os.environ.get("SGLANG_MANAGER_CONFIG", "config.yaml")
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path} (set SGLANG_MANAGER_CONFIG or pass --config)")

    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file {cfg_path}: invalid YAML: {exc}") from None
    if raw is None:
        raw = {}

    root = _require_mapping(raw, "config", "config")
    cfg = Config()

    server = _require_mapping(root.get("server"), "server", "server")
    cfg.server.host = str(server.get("host", cfg.server.host))
    cfg.server.port = int(server.get("port", cfg.server.port))

    sgl = _require_mapping(root.get("sglang"), "sglang", "sglang")
    cfg.sglang.host = str(sgl.get("host", cfg.sglang.host))
    cfg.sglang.port = int(sgl.get("port", cfg.sglang.port))
    cfg.sglang.health_path = str(sgl.get("health_path", cfg.sglang.health_path))
    cfg.sglang.startup_timeout_seconds = float(sgl.get("startup_timeout_seconds", cfg.sglang.startup_timeout_seconds))
    cfg.sglang.stop_timeout_seconds = float(sgl.get("stop_timeout_seconds", cfg.sglang.stop_timeout_seconds))
    if "command" in sgl:
        cmd = sgl["command"]
        if not isinstance(cmd, list) or not all(isinstance(a, str) for a in cmd):
            raise ConfigError("sglang.command: expected a list of strings")
        cfg.sglang.command = list(cmd)
    if "env" in sgl:
        env = sgl["env"]
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ConfigError("sglang.env: expected a mapping of strings to strings")
        cfg.sglang.env = dict(env)

    gpu = _require_mapping(root.get("gpu"), "gpu", "gpu")
    cfg.gpu.device = int(gpu.get("device", cfg.gpu.device))
    cfg.gpu.safety_margin_gib = float(gpu.get("safety_margin_gib", cfg.gpu.safety_margin_gib))
    cfg.gpu.vram_release_timeout_seconds = float(gpu.get("vram_release_timeout_seconds", cfg.gpu.vram_release_timeout_seconds))
    cfg.gpu.vram_poll_interval_seconds = float(gpu.get("vram_poll_interval_seconds", cfg.gpu.vram_poll_interval_seconds))

    idle = _require_mapping(root.get("idle"), "idle", "idle")
    cfg.idle.timeout_seconds = float(idle.get("timeout_seconds", cfg.idle.timeout_seconds))
    cfg.idle.check_interval_seconds = float(idle.get("check_interval_seconds", cfg.idle.check_interval_seconds))

    usage = _require_mapping(root.get("usage"), "usage", "usage")
    cfg.usage.enabled = bool(usage.get("enabled", cfg.usage.enabled))
    cfg.usage.db_path = str(usage.get("db_path", cfg.usage.db_path))

    models_raw = _require_mapping(root.get("models"), "models", "models")
    if not models_raw:
        raise ConfigError("models: at least one model is required")
    for name, model_raw in models_raw.items():
        cfg.models[name] = _parse_model(str(name), model_raw)

    # Environment overrides win over the YAML file.
    cfg.idle.timeout_seconds = _env_float("SGLANG_MANAGER_IDLE_TIMEOUT_SECONDS", cfg.idle.timeout_seconds)
    cfg.server.port = _env_int("SGLANG_MANAGER_PORT", cfg.server.port)
    cfg.server.host = os.environ.get("SGLANG_MANAGER_HOST", cfg.server.host)

    if cfg.idle.timeout_seconds <= 0:
        raise ConfigError("idle.timeout_seconds must be > 0")
    if cfg.gpu.safety_margin_gib < 0:
        raise ConfigError("gpu.safety_margin_gib must be >= 0")
    if cfg.sglang.startup_timeout_seconds <= 0:
        raise ConfigError("sglang.startup_timeout_seconds must be > 0")
    return cfg
