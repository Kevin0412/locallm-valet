"""State-machine & lifecycle tests using fake GPU/runner (no GPU required)."""

import asyncio

import pytest

from sglang_manager.errors import (
    GpuUnavailable,
    InsufficientGpuMemory,
    ModelNotFound,
    ModelSwitchBusy,
    SglangStartupFailed,
    SglangStartupTimeout,
)
from sglang_manager.manager import ModelManager
from sglang_manager.state import State

from .conftest import FakeGpu, make_config


async def test_start_on_demand(manager, gpu, runner):
    gpu.free_g = 40  # qwen needs 30 + 4 = 34
    spec = await manager.ensure_loaded("qwen")
    assert spec.name == "qwen"
    assert manager.state is State.RUNNING
    assert manager.current_model == "qwen"
    assert runner.starts == ["qwen"]
    assert gpu.release_waits == 0  # a cold start never waits for VRAM release


async def test_direct_route_when_model_matches(manager, gpu, runner):
    gpu.free_g = 40
    await manager.ensure_loaded("qwen")
    await manager.ensure_loaded("qwen")  # already RUNNING: no restart, no VRAM re-check
    assert runner.starts == ["qwen"]
    assert runner.stops == 0


async def test_insufficient_vram_refuses_start(manager, gpu, runner):
    gpu.free_g = 10  # below 34
    with pytest.raises(InsufficientGpuMemory) as excinfo:
        await manager.ensure_loaded("qwen")
    assert "34.0 GiB" in str(excinfo.value)
    assert manager.state is State.STOPPED
    assert runner.starts == []


async def test_gpu_unavailable_passthrough(manager, gpu, runner):
    """NVML absent: the typed error must pass through unchanged (not be mapped
    to a startup failure)."""

    class BrokenGpu(FakeGpu):
        def free_gib(self):
            raise GpuUnavailable("NVML unavailable for device 0")

    manager.gpu = BrokenGpu()
    with pytest.raises(GpuUnavailable):
        await manager.ensure_loaded("qwen")
    assert manager.state is State.STOPPED
    assert runner.starts == []


async def test_unknown_model(manager):
    with pytest.raises(ModelNotFound):
        await manager.ensure_loaded("nope")


async def test_switch_when_idle(manager, gpu, runner):
    gpu.free_g = 40
    await manager.ensure_loaded("qwen")
    await manager.ensure_loaded("gemma")  # 18 + 4 = 22 <= 40
    assert runner.starts == ["qwen", "gemma"]
    assert runner.stops == 1
    assert manager.state is State.RUNNING
    assert manager.current_model == "gemma"
    assert gpu.release_waits == 1  # VRAM release waited for before the re-check


async def test_switch_decides_on_post_stop_vram(manager, gpu, runner):
    """Free VRAM while qwen is loaded is 8 GiB, but after stopping it is 38 GiB.
    The switch must succeed based on the RE-READ value, not the pre-stop one."""
    gpu.free_g = 40
    await manager.ensure_loaded("qwen")
    gpu.free_g = 8  # the loaded SGLang now holds ~30 GiB
    runner.gpu = gpu
    runner.free_after_stop = 38  # memory comes back once qwen exits
    await manager.ensure_loaded("gemma")
    assert manager.current_model == "gemma"
    assert runner.stops == 1


async def test_switch_refused_after_stop_if_vram_still_low(manager, gpu, runner):
    """External workload grabbed the freed memory: switch fails after the old
    model is already unloaded; state lands in STOPPED (not RUNNING(qwen))."""
    gpu.free_g = 40
    await manager.ensure_loaded("qwen")
    gpu.free_g = 8
    runner.gpu = gpu
    runner.free_after_stop = 10  # someone else took the memory
    with pytest.raises(InsufficientGpuMemory):
        await manager.ensure_loaded("gemma")
    assert manager.state is State.STOPPED
    assert manager.current_model is None


async def test_switch_refused_when_busy(manager, gpu, runner):
    gpu.free_g = 40
    await manager.ensure_loaded("qwen")
    manager.request_started()  # an active request (e.g. streaming)
    with pytest.raises(ModelSwitchBusy):
        await manager.ensure_loaded("gemma")
    assert manager.state is State.RUNNING
    assert manager.current_model == "qwen"
    assert runner.stops == 0  # never killed a serving model
    manager.request_finished()


async def test_startup_timeout(manager, gpu, runner):
    gpu.free_g = 40
    runner.health_mode = "timeout"
    with pytest.raises(SglangStartupTimeout):
        await manager.ensure_loaded("qwen")
    assert manager.state is State.STOPPED
    assert runner.stops == 1  # failed instance cleaned up


async def test_startup_process_dies(manager, gpu, runner):
    gpu.free_g = 40
    runner.health_mode = "die"
    with pytest.raises(SglangStartupFailed):
        await manager.ensure_loaded("qwen")
    assert manager.state is State.STOPPED


async def test_concurrent_same_model_shares_one_start(manager, gpu, runner):
    gpu.free_g = 40
    runner.health_delay = 0.05
    results = await asyncio.gather(
        manager.ensure_loaded("qwen"), manager.ensure_loaded("qwen")
    )
    assert [r.name for r in results] == ["qwen", "qwen"]
    assert runner.starts == ["qwen"]  # one start, both requests forwarded
    assert manager.state is State.RUNNING


async def test_concurrent_different_model_refused_during_start(manager, gpu, runner):
    gpu.free_g = 40
    runner.health_delay = 0.1
    t1 = asyncio.create_task(manager.ensure_loaded("qwen"))
    await asyncio.sleep(0.02)  # let it reach STARTING
    with pytest.raises(ModelSwitchBusy):
        await manager.ensure_loaded("gemma")
    await t1
    assert runner.starts == ["qwen"]


async def test_waiter_receives_startup_error(manager, gpu, runner):
    gpu.free_g = 40
    runner.health_mode = "timeout"
    runner.health_delay = 0.05
    t1 = asyncio.create_task(manager.ensure_loaded("qwen"))
    await asyncio.sleep(0.02)
    with pytest.raises(SglangStartupTimeout):
        await manager.ensure_loaded("qwen")
    with pytest.raises(SglangStartupTimeout):
        await t1


async def test_unexpected_process_exit_forces_stopped(manager, gpu, runner):
    gpu.free_g = 40
    await manager.ensure_loaded("qwen")
    await runner.on_exit(1)  # the exit monitor callback
    assert manager.state is State.STOPPED
    assert manager.current_model is None
    # next request can cold-start again
    await manager.ensure_loaded("qwen")
    assert runner.starts == ["qwen", "qwen"]


async def test_admin_stop_busy_then_ok(manager, gpu, runner):
    gpu.free_g = 40
    await manager.ensure_loaded("qwen")
    manager.request_started()
    with pytest.raises(ModelSwitchBusy):
        await manager.stop()
    manager.request_finished()
    await manager.stop()
    assert manager.state is State.STOPPED
    assert runner.stops == 1


async def test_admin_stop_when_already_stopped(manager):
    await manager.stop()  # no-op
    assert manager.state is State.STOPPED


async def test_request_accounting(manager, gpu, runner):
    gpu.free_g = 40
    await manager.ensure_loaded("qwen")
    assert manager.active_requests == 0
    manager.request_started()
    assert manager.active_requests == 1
    manager.request_finished()
    assert manager.active_requests == 0


async def test_idle_seconds_reporting(manager, gpu, runner, clock):
    gpu.free_g = 40
    await manager.ensure_loaded("qwen")
    assert manager.idle_seconds == 0
    clock.advance(100)
    assert manager.idle_seconds == 100
    manager.request_started()
    assert manager.idle_seconds == 0  # busy → not idle
    manager.request_finished()


async def test_idle_watchdog_unloads(gpu, runner):
    m = ModelManager(make_config(idle_timeout=0.15, idle_check_interval=0.05), gpu=gpu, runner=runner)
    gpu.free_g = 40
    m.start()
    try:
        await m.ensure_loaded("qwen")
        await asyncio.sleep(0.5)
        assert m.state is State.STOPPED
        assert runner.stops == 1
    finally:
        await m.shutdown()


async def test_idle_watchdog_respects_active_requests(gpu, runner):
    m = ModelManager(make_config(idle_timeout=0.15, idle_check_interval=0.05), gpu=gpu, runner=runner)
    gpu.free_g = 40
    m.start()
    try:
        await m.ensure_loaded("qwen")
        m.request_started()
        await asyncio.sleep(0.4)
        assert m.state is State.RUNNING  # busy: never unload mid-generation
        m.request_finished()
        await asyncio.sleep(0.4)
        assert m.state is State.STOPPED  # idle again: unloaded
    finally:
        await m.shutdown()


async def test_shutdown_stops_sglang(gpu, runner):
    m = ModelManager(make_config(), gpu=gpu, runner=runner)
    gpu.free_g = 40
    await m.ensure_loaded("qwen")
    await m.shutdown()
    assert m.state is State.STOPPED
    assert runner.stops == 1


async def test_status_shape(manager, gpu, runner):
    gpu.free_g = 40
    gpu.total_g = 48
    await manager.ensure_loaded("qwen")
    status = manager.status()
    assert status["state"] == "running"
    assert status["model"] == "qwen"
    assert status["active_requests"] == 0
    assert status["idle_timeout_seconds"] == 3600
    assert status["gpu"]["free_gib"] == 40
    assert status["gpu"]["total_gib"] == 48
