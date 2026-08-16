"""Shared fakes + fixtures for locallm-valet tests (no GPU needed)."""

from __future__ import annotations

import pytest

from locallm_valet.config import Config, ModelBackendArgs, ModelSpec, BackendConfig, MemoryConfig, UsageConfig
from locallm_valet.errors import BackendStartupFailed, BackendStartupTimeout
from locallm_valet.manager import ModelManager


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeMemory:
    """Scriptable memory monitor: VRAM (NVML) + RAM."""

    def __init__(self, total: float = 48.0, free: float = 48.0, nvml: bool = True):
        self.device = 0
        self.total_g = total
        self.free_g = free
        self.ram_total_g = 64.0
        self.ram_free_g = 40.0
        self.nvml_available = nvml
        self.release_waits = 0

    def vram_total_gib(self) -> float:
        return self.total_g

    def vram_free_gib(self) -> float:
        return self.free_g

    def vram_used_gib(self) -> float:
        return self.total_g - self.free_g

    def ram_total_gib(self) -> float:
        return self.ram_total_g

    def ram_available_gib(self) -> float:
        return self.ram_free_g

    async def wait_vram_released(self, timeout_seconds, poll_interval=1.0):
        self.release_waits += 1
        return None


class FakeRunner:
    """Records start/stop; health behavior is scriptable per test."""

    def __init__(self):
        self.starts: list[str] = []
        self.stops = 0
        self.model: str | None = None
        self.health_mode = "ok"  # ok | timeout | die
        self.health_delay = 0.0
        self.on_exit = None
        self.memory: FakeMemory | None = None  # optional: set VRAM after stop
        self.free_after_stop: float | None = None
        self.max_tokens_capacity: int | None = 123456

    @property
    def running(self) -> bool:
        return self.model is not None

    async def start(self, spec: ModelSpec) -> None:
        self.starts.append(spec.name)
        self.model = spec.name

    async def wait_health(self, timeout_seconds: float) -> None:
        if self.health_delay:
            import asyncio

            await asyncio.sleep(self.health_delay)
        if self.health_mode == "timeout":
            raise BackendStartupTimeout(f"fake: not healthy within {timeout_seconds}s")
        if self.health_mode == "die":
            self.model = None
            raise BackendStartupFailed("fake: process exited during startup")

    async def stop(self, timeout_seconds: float = 60.0) -> None:
        self.stops += 1
        self.model = None
        if self.free_after_stop is not None and self.memory is not None:
            self.memory.free_g = self.free_after_stop

    async def get_max_total_num_tokens(self) -> int | None:
        return self.max_tokens_capacity

    async def aclose(self) -> None:
        return None


def make_config(**kwargs) -> Config:
    """A minimal Config with two registry models: qwen (30 GiB VRAM / 20 GiB
    RAM) and gemma (18 GiB VRAM / 10 GiB RAM)."""

    cfg = Config()
    cfg.models["qwen"] = ModelSpec(
        name="qwen", path="/models/qwen", required_vram_gib=30, required_ram_gib=20,
        backend=ModelBackendArgs(extra_args=["--flag-a", "--context-length", "262144"]),
    )
    cfg.models["gemma"] = ModelSpec(
        name="gemma", path="/models/gemma", required_vram_gib=18, required_ram_gib=10,
    )
    cfg.memory = MemoryConfig(device=0, safety_margin_gib=4.0, release_timeout_seconds=10.0)
    cfg.backend = BackendConfig(host="127.0.0.1", port=30000, startup_timeout_seconds=180, stop_timeout_seconds=10)
    cfg.idle.timeout_seconds = kwargs.pop("idle_timeout", 3600.0)
    cfg.idle.check_interval_seconds = kwargs.pop("idle_check_interval", 60.0)
    cfg.usage = UsageConfig(enabled=True, db_path=":memory:")
    if kwargs:
        raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
    return cfg


@pytest.fixture
def memory():
    return FakeMemory()


@pytest.fixture
def runner():
    return FakeRunner()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def manager(memory, runner, clock):
    return ModelManager(make_config(), memory=memory, runner=runner, clock=clock)
