"""Shared fakes + fixtures for sglang-manager tests (no GPU needed)."""

from __future__ import annotations

import pytest

from sglang_manager.config import Config, ModelSpec, SglangConfig, UsageConfig
from sglang_manager.errors import SglangStartupFailed, SglangStartupTimeout
from sglang_manager.manager import ModelManager


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeGpu:
    def __init__(self, total: float = 48.0, free: float = 48.0):
        self.device = 0
        self.total_g = total
        self.free_g = free
        self.release_waits = 0

    def total_gib(self) -> float:
        return self.total_g

    def free_gib(self) -> float:
        return self.free_g

    def used_gib(self) -> float:
        return self.total_g - self.free_g

    async def wait_until_free(self, threshold_gib, timeout_seconds, poll_interval=1.0):
        return None

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
        self.gpu: FakeGpu | None = None  # optional: set free VRAM after stop
        self.free_after_stop: float | None = None

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
            raise SglangStartupTimeout(f"fake: not healthy within {timeout_seconds}s")
        if self.health_mode == "die":
            self.model = None
            raise SglangStartupFailed("fake: process exited during startup")

    async def stop(self, timeout_seconds: float = 60.0) -> None:
        self.stops += 1
        self.model = None
        if self.free_after_stop is not None and self.gpu is not None:
            self.gpu.free_g = self.free_after_stop

    async def aclose(self) -> None:
        return None


def make_config(**kwargs) -> Config:
    """A minimal Config with two registry models: qwen (30 GiB) and gemma (18 GiB)."""

    cfg = Config()
    cfg.models["qwen"] = ModelSpec(name="qwen", path="/models/qwen", required_vram_gib=30)
    cfg.models["gemma"] = ModelSpec(name="gemma", path="/models/gemma", required_vram_gib=18)
    cfg.gpu.safety_margin_gib = 4.0
    cfg.sglang = SglangConfig(host="127.0.0.1", port=30000, startup_timeout_seconds=180, stop_timeout_seconds=10)
    cfg.idle.timeout_seconds = kwargs.pop("idle_timeout", 3600.0)
    cfg.idle.check_interval_seconds = kwargs.pop("idle_check_interval", 60.0)
    cfg.usage = UsageConfig(enabled=True, db_path=":memory:")
    if kwargs:
        raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
    return cfg


@pytest.fixture
def gpu():
    return FakeGpu()


@pytest.fixture
def runner():
    return FakeRunner()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def manager(gpu, runner, clock):
    return ModelManager(make_config(), gpu=gpu, runner=runner, clock=clock)
