"""Configuration loading & validation for locallm-valet.

All knobs live in a YAML file (default ``config.yaml``, override with
``--config`` or the ``LOCALLM_VALET_CONFIG`` env var).  High-value settings can
be overridden through environment variables (legacy ``SGLANG_MANAGER_*``
names are still accepted as fallbacks):

- ``LOCALLM_VALET_IDLE_TIMEOUT_SECONDS`` — idle auto-unload timeout (never
  hardcoded; YAML default is 3600 s).
- ``LOCALLM_VALET_PORT`` / ``LOCALLM_VALET_HOST`` — manager listen address.
- ``LOCALLM_VALET_API_KEY`` — comma-separated Bearer API keys.
- ``LOCALLM_VALET_CONFIG`` — config file path.

The managed backend is backend-agnostic: ``backend.command_template`` is a
shell-style template (placeholders ``{python} {model_path} {host} {port}
{device} {model_name} {extra_args}``) so SGLang, vLLM, llama.cpp, OpenVINO or
any other OpenAI-compatible server can be launched with one line.  The legacy
section names ``sglang:`` / ``gpu:`` are still accepted as aliases for
``backend:`` / ``memory:`` (deprecated, logged).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Backwards-compatible defaults: SGLang-style launch (reference backend).
DEFAULT_COMMAND_TEMPLATE = (
    "{python} -m sglang.launch_server --model-path {model_path} "
    "--host {host} --port {port} --tp-size 1 {extra_args}"
)


class ConfigError(Exception):
    """Raised when the configuration file is missing or invalid."""


@dataclass
class ModelBackendArgs:
    """Per-model backend launch arguments (backend-agnostic)."""

    extra_args: list[str] = field(default_factory=list)
    # Extra environment variables, merged over the global ``backend.env``
    # (e.g. ``SGLANG_USE_MODELSCOPE: "true"`` or an ``LD_PRELOAD``).
    env: dict[str, str] = field(default_factory=dict)
    # Per-model overrides; fall back to the global ``backend.*`` settings.
    # This is how one registry mixes backends (e.g. a vLLM model and a
    # llama.cpp model side by side — only one runs at a time, so the shared
    # internal port is safe).
    command_template: str | None = None
    health_path: str | None = None


@dataclass
class PoolConfig:
    """A named, probe-able memory source (resource pool).

    Devices differ wildly in how they get memory — shared system RAM (CPU /
    NPU / iGPU), dedicated VRAM (discrete GPUs), unified memory (Apple
    Silicon), dedicated HBM (some accelerators), etc. Rather than hardcode a
    ram/vram dichotomy, every memory source is a *pool* that a slot may
    consume from. Pools are shared (many slots draw from system_ram) or
    private (each GPU has its own vram pool).

    kind determines how the pool is probed:
    - "ram":    system RAM via psutil (cross-platform)
    - "vram":   VRAM via NVML at ``device_index`` (skip gating when no driver)
    - "static": no standard API (NPU/HBM/etc.) — use ``total_gib`` and treat
                "available" as total (no runtime probe available)
    """

    name: str = ""
    kind: str = "ram"          # ram | vram | static
    device_index: int = 0      # vram only: NVML device index
    total_gib: float = 0.0     # static only: fixed capacity

    def __post_init__(self) -> None:
        if self.kind not in ("ram", "vram", "static"):
            raise ConfigError(f"pool '{self.name}': kind must be ram|vram|static")


@dataclass
class SlotConfig:
    """One execution slot (a device lane that can run a single model at a time).

    Slots are the unit of concurrency: CPU / NPU0 / NPU1 / GPU0 / GPU1 …
    each owns an independent backend port + state machine, so models on
    different slots run in parallel. Same-slot models are mutually exclusive
    (serialized switching).

    ``pools`` maps pool name → reserved GiB this slot always needs beyond the
    active model's own requirement (e.g. runtime overhead). The active model
    declares per-pool needs via ``required_pools``.
    """

    name: str = "cpu"
    port: int = 30000          # this slot's internal backend port
    device: str = "CPU"        # display label: CPU / NPU0 / GPU0 …
    pools: dict[str, float] = field(default_factory=dict)
    """Extra GiB this slot needs from each pool (runtime overhead)."""


@dataclass
class ModelSpec:
    """One entry of the model registry."""

    name: str
    path: str  # local directory or hub id (HF / ModelScope / ...)
    # Which slot this model runs on. Defaults to the first ram slot ("cpu").
    slot: str = "cpu"
    # Memory gates. 0 (default) = that resource is not checked.
    required_vram_gib: float = 0.0  # VRAM (checked only when NVML is available)
    required_ram_gib: float = 0.0   # system RAM (psutil, cross-platform)
    # Per-pool needs, e.g. {"gpu0_vram": 6, "system_ram": 3} — for machines
    # where a model consumes several pools (VRAM + RAM staging, unified
    # memory, dedicated HBM…). Pool names must exist under `pools:`.
    required_pools: dict[str, float] = field(default_factory=dict)
    backend: ModelBackendArgs = field(default_factory=ModelBackendArgs)

    def configured_context_length(self) -> int | None:
        """The declared ``--context-length`` from extra_args, if present."""

        args = self.backend.extra_args
        for i, a in enumerate(args):
            if a == "--context-length" and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except ValueError:
                    return None
        return None


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    # Accepted API keys (``Authorization: Bearer <key>``). Empty = auth
    # disabled (open access). Multiple keys allowed.
    api_keys: list[str] = field(default_factory=list)
    # Optional username/password (``Authorization: Basic base64(user:pass)``).
    # When set, model access (v1/*) and benchmark runs require it — in
    # addition to or instead of api_keys. Env: LOCALLM_VALET_USERNAME /
    # LOCALLM_VALET_PASSWORD.
    username: str = ""
    password: str = ""

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys) or bool(self.username and self.password)


@dataclass
class BackendConfig:
    """The managed inference backend (SGLang / vLLM / llama.cpp / OpenVINO...).

    It is expected to expose an OpenAI-compatible API on ``host:port`` and a
    readiness endpoint at ``health_path``.
    """

    host: str = "127.0.0.1"
    port: int = 30000
    health_path: str = "/health"
    startup_timeout_seconds: float = 180.0
    stop_timeout_seconds: float = 60.0
    # Launch template; see DEFAULT_COMMAND_TEMPLATE for placeholders.
    command_template: str = DEFAULT_COMMAND_TEMPLATE
    env: dict[str, str] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class MemoryConfig:
    """Resource gates. VRAM via NVML (when present), RAM via psutil."""

    device: int = 0  # only meaningful for CUDA backends (CUDA_VISIBLE_DEVICES)
    # Headroom added to required_vram_gib / required_ram_gib before allowing
    # a start.
    safety_margin_gib: float = 4.0
    # How long to wait for VRAM to actually come back after stopping the
    # backend (CUDA context teardown lags process exit).
    release_timeout_seconds: float = 120.0
    poll_interval_seconds: float = 1.0


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
    backend: BackendConfig = field(default_factory=BackendConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    idle: IdleConfig = field(default_factory=IdleConfig)
    usage: UsageConfig = field(default_factory=UsageConfig)
    pools: dict[str, PoolConfig] = field(default_factory=dict)
    slots: dict[str, SlotConfig] = field(default_factory=dict)
    models: dict[str, ModelSpec] = field(default_factory=dict)

    def get_model(self, name: str) -> ModelSpec | None:
        return self.models.get(name)

    def get_slot(self, name: str) -> SlotConfig:
        """Resolve a model to its slot config (defaults to 'cpu')."""
        slot_name = self.models.get(name, ModelSpec(name=name, path="")).slot
        return self.slots.get(slot_name, self.slots.get("cpu", SlotConfig()))


def _require_mapping(raw: Any, section: str, path: str) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")
    return raw


def _parse_model_backend(raw: Any, name: str) -> ModelBackendArgs:
    out = ModelBackendArgs()
    if not raw:
        return out
    for key, value in _require_mapping(raw, "backend", f"models.{name}.backend").items():
        if key == "extra_args":
            if not isinstance(value, list):
                raise ConfigError(f"models.{name}.backend.extra_args: expected a list")
            # YAML scalars like --ctx-size / 8192 arrive as int/float/bool;
            # coerce them so users don't have to quote every number.
            out.extra_args = [
                str(a) if isinstance(a, (int, float, bool)) else a for a in value
            ]
            if not all(isinstance(a, str) for a in out.extra_args):
                raise ConfigError(f"models.{name}.backend.extra_args: expected a list of strings")
        elif key == "env":
            if not isinstance(value, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in value.items()
            ):
                raise ConfigError(f"models.{name}.backend.env: expected a mapping of strings to strings")
            out.env = dict(value)
        elif key == "command_template":
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"models.{name}.backend.command_template: expected a non-empty string")
            out.command_template = value
        elif key == "health_path":
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"models.{name}.backend.health_path: expected a non-empty string")
            out.health_path = value
        else:
            raise ConfigError(f"models.{name}.backend: unknown key {key!r}")
    return out


def _parse_model(name: str, raw: Any) -> ModelSpec:
    data = _require_mapping(raw, "model", f"models.{name}")
    path = data.get("path")
    if not isinstance(path, str) or not path:
        raise ConfigError(f"models.{name}: missing required string field 'path'")
    required_vram = float(data.get("required_vram_gib", 0.0))
    if required_vram < 0:
        raise ConfigError(f"models.{name}: required_vram_gib must be >= 0")
    required_ram = float(data.get("required_ram_gib", 0.0))
    if required_ram < 0:
        raise ConfigError(f"models.{name}: required_ram_gib must be >= 0")

    backend_raw = data.get("backend")
    if backend_raw is None and "sglang" in data:  # legacy alias
        logger.warning("models.%s: 'sglang:' is deprecated, use 'backend:'", name)
        backend_raw = data["sglang"]
    slot = str(data.get("slot", "cpu"))
    required_pools: dict[str, float] = {}
    rp_raw = data.get("required_pools")
    if rp_raw is not None:
        if not isinstance(rp_raw, dict):
            raise ConfigError(f"models.{name}.required_pools: expected a mapping")
        for pn, need in rp_raw.items():
            required_pools[str(pn)] = float(need)
    return ModelSpec(
        name=name,
        path=path,
        slot=slot,
        required_vram_gib=required_vram,
        required_ram_gib=required_ram,
        required_pools=required_pools,
        backend=_parse_model_backend(backend_raw, name),
    )


def _env_float(name: str, legacy_names: tuple[str, ...], default: float) -> float:
    value = _env_first(name, legacy_names)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        raise ConfigError(f"env {name}: not a number: {value!r}") from None


def _env_int(name: str, legacy_names: tuple[str, ...], default: int) -> int:
    value = _env_first(name, legacy_names)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ConfigError(f"env {name}: not an integer: {value!r}") from None


def _env_first(name: str, legacy_names: tuple[str, ...]) -> str | None:
    """Primary env name, then deprecated aliases from earlier project names
    (kept as functional migration aids)."""
    for candidate in (name, *legacy_names):
        value = os.environ.get(candidate)
        if value is not None:
            return value
    return None


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate configuration from a YAML file (plus env overrides)."""
    if path is None:
        path = _env_first("LOCALLM_VALET_CONFIG", ("LLM_GATEWAY_CONFIG", "SGLANG_MANAGER_CONFIG")) or "config.yaml"
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path} (set LOCALLM_VALET_CONFIG or pass --config)")

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
    api_key_raw = server.get("api_key")
    if api_key_raw is not None:
        if isinstance(api_key_raw, str):
            cfg.server.api_keys = [api_key_raw] if api_key_raw.strip() else []
        elif isinstance(api_key_raw, list) and all(isinstance(k, str) and k.strip() for k in api_key_raw):
            cfg.server.api_keys = list(api_key_raw)
        else:
            raise ConfigError("server.api_key: expected a string or a list of strings")

    cfg.server.username = str(server.get("username", "") or "")
    cfg.server.password = str(server.get("password", "") or "")

    backend_raw = root.get("backend")
    if backend_raw is None and "sglang" in root:  # legacy alias
        logger.warning("'sglang:' section is deprecated, use 'backend:'")
        backend_raw = root["sglang"]
    be = _require_mapping(backend_raw, "backend", "backend")
    cfg.backend.host = str(be.get("host", cfg.backend.host))
    cfg.backend.port = int(be.get("port", cfg.backend.port))
    cfg.backend.health_path = str(be.get("health_path", cfg.backend.health_path))
    cfg.backend.startup_timeout_seconds = float(be.get("startup_timeout_seconds", cfg.backend.startup_timeout_seconds))
    cfg.backend.stop_timeout_seconds = float(be.get("stop_timeout_seconds", cfg.backend.stop_timeout_seconds))
    if "command_template" in be:
        tpl = be["command_template"]
        if not isinstance(tpl, str) or not tpl.strip():
            raise ConfigError("backend.command_template: expected a non-empty string")
        cfg.backend.command_template = tpl
    if "command" in be:
        raise ConfigError(
            "backend.command (list) is no longer supported; use "
            "backend.command_template with placeholders {model_path} {host} {port} {extra_args} ..."
        )
    if "env" in be:
        env = be["env"]
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ConfigError("backend.env: expected a mapping of strings to strings")
        cfg.backend.env = dict(env)

    memory_raw = root.get("memory")
    if memory_raw is None and "gpu" in root:  # legacy alias
        logger.warning("'gpu:' section is deprecated, use 'memory:'")
        memory_raw = root["gpu"]
    mem = _require_mapping(memory_raw, "memory", "memory")
    cfg.memory.device = int(mem.get("device", cfg.memory.device))
    cfg.memory.safety_margin_gib = float(mem.get("safety_margin_gib", cfg.memory.safety_margin_gib))
    cfg.memory.release_timeout_seconds = float(
        mem.get("release_timeout_seconds",
                mem.get("vram_release_timeout_seconds", cfg.memory.release_timeout_seconds))
    )
    cfg.memory.poll_interval_seconds = float(
        mem.get("poll_interval_seconds",
                mem.get("vram_poll_interval_seconds", cfg.memory.poll_interval_seconds))
    )

    idle = _require_mapping(root.get("idle"), "idle", "idle")
    cfg.idle.timeout_seconds = float(idle.get("timeout_seconds", cfg.idle.timeout_seconds))
    cfg.idle.check_interval_seconds = float(idle.get("check_interval_seconds", cfg.idle.check_interval_seconds))

    usage = _require_mapping(root.get("usage"), "usage", "usage")
    cfg.usage.enabled = bool(usage.get("enabled", cfg.usage.enabled))
    cfg.usage.db_path = str(usage.get("db_path", cfg.usage.db_path))

    # Pools: named memory sources (system_ram, gpu0_vram, ...). When absent,
    # two defaults are implied: "system_ram" (psutil) and "gpu0_vram" (NVML),
    # so existing configs keep working.
    pools_raw = _require_mapping(root.get("pools"), "pools", "pools")
    if pools_raw:
        for pool_name, pool_raw in pools_raw.items():
            p = _require_mapping(pool_raw, "pool", f"pools.{pool_name}")
            cfg.pools[str(pool_name)] = PoolConfig(
                name=str(pool_name),
                kind=str(p.get("kind", "ram")),
                device_index=int(p.get("device_index", 0)),
                total_gib=float(p.get("total_gib", 0.0)),
            )
    else:
        cfg.pools["system_ram"] = PoolConfig(name="system_ram", kind="ram")
        cfg.pools["gpu0_vram"] = PoolConfig(name="gpu0_vram", kind="vram", device_index=0)

    # Slots: optional; when absent a single "cpu" slot (the global backend
    # port) is implied so the config stays backwards compatible.
    slots_raw = _require_mapping(root.get("slots"), "slots", "slots")
    if slots_raw:
        for slot_name, slot_raw in slots_raw.items():
            s = _require_mapping(slot_raw, "slot", f"slots.{slot_name}")
            pools_map: dict[str, float] = {}
            for pool_name, need in (s.get("pools") or {}).items():
                pools_map[str(pool_name)] = float(need)
            for pn in pools_map:
                if pn not in cfg.pools:
                    raise ConfigError(f"slots.{slot_name}.pools.{pn}: unknown pool (defined: {', '.join(cfg.pools)})")
            cfg.slots[str(slot_name)] = SlotConfig(
                name=str(slot_name),
                port=int(s.get("port", 30000)),
                device=str(s.get("device", str(slot_name).upper())),
                pools=pools_map,
            )
    else:
        cfg.slots["cpu"] = SlotConfig(name="cpu", port=cfg.backend.port)

    models_raw = _require_mapping(root.get("models"), "models", "models")
    if not models_raw:
        raise ConfigError("models: at least one model is required")
    for name, model_raw in models_raw.items():
        spec = _parse_model(str(name), model_raw)
        if spec.slot not in cfg.slots:
            raise ConfigError(
                f"models.{name}: slot '{spec.slot}' is not defined under 'slots:' "
                f"(defined: {', '.join(cfg.slots)})"
            )
        for pn in spec.required_pools:
            if pn not in cfg.pools:
                raise ConfigError(f"models.{name}.required_pools.{pn}: unknown pool (defined: {', '.join(cfg.pools)})")
        cfg.models[name] = spec

    # Environment overrides win over the YAML file (legacy names fall back).
    cfg.idle.timeout_seconds = _env_float(
        "LOCALLM_VALET_IDLE_TIMEOUT_SECONDS",
        ("LLM_GATEWAY_IDLE_TIMEOUT_SECONDS", "SGLANG_MANAGER_IDLE_TIMEOUT_SECONDS"),
        cfg.idle.timeout_seconds,
    )
    cfg.server.port = _env_int(
        "LOCALLM_VALET_PORT", ("LLM_GATEWAY_PORT", "SGLANG_MANAGER_PORT"), cfg.server.port
    )
    cfg.server.host = _env_first(
        "LOCALLM_VALET_HOST", ("LLM_GATEWAY_HOST", "SGLANG_MANAGER_HOST")
    ) or cfg.server.host
    env_api_key = _env_first("LOCALLM_VALET_API_KEY", ("LLM_GATEWAY_API_KEY", "SGLANG_MANAGER_API_KEY"))
    if env_api_key:
        cfg.server.api_keys = [k.strip() for k in env_api_key.split(",") if k.strip()]

    cfg.server.username = _env_first(
        "LOCALLM_VALET_USERNAME", ("LLM_GATEWAY_USERNAME",)
    ) or cfg.server.username
    cfg.server.password = _env_first(
        "LOCALLM_VALET_PASSWORD", ("LLM_GATEWAY_PASSWORD",)
    ) or cfg.server.password

    if cfg.idle.timeout_seconds <= 0:
        raise ConfigError("idle.timeout_seconds must be > 0")
    if cfg.memory.safety_margin_gib < 0:
        raise ConfigError("memory.safety_margin_gib must be >= 0")
    if cfg.backend.startup_timeout_seconds <= 0:
        raise ConfigError("backend.startup_timeout_seconds must be > 0")
    return cfg
