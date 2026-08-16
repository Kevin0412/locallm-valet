"""State-machine & lifecycle tests using fake GPU/runner (no GPU required)."""

import asyncio

import pytest

from locallm_valet.errors import (
    MemoryUnavailable,
    InsufficientMemory,
    ModelNotFound,
    ModelSwitchBusy,
    BackendStartupFailed,
    BackendStartupTimeout,
    BackendUnavailable,
)
from locallm_valet.manager import ModelManager
from locallm_valet.state import State

from .conftest import FakeMemory, make_config


async def test_start_on_demand(manager, memory, runner):
    memory.free_g = 40  # qwen needs 30 + 4 = 34
    spec = await manager.ensure_loaded("qwen")
    assert spec.name == "qwen"
    assert manager.state is State.RUNNING
    assert manager.current_model == "qwen"
    assert runner.starts == ["qwen"]
    assert memory.release_waits == 0  # a cold start never waits for VRAM release


async def test_direct_route_when_model_matches(manager, memory, runner):
    memory.free_g = 40
    await manager.ensure_loaded("qwen")
    await manager.ensure_loaded("qwen")  # already RUNNING: no restart, no VRAM re-check
    assert runner.starts == ["qwen"]
    assert runner.stops == 0


async def test_insufficient_vram_refuses_start(manager, memory, runner):
    memory.free_g = 10  # below 34
    with pytest.raises(InsufficientMemory) as excinfo:
        await manager.ensure_loaded("qwen")
    assert "34.0 GiB" in str(excinfo.value)
    assert manager.state is State.STOPPED
    assert runner.starts == []


async def test_vram_gate_skipped_when_nvml_unavailable(manager, memory, runner):
    """CPU/NPU machine: no NVML -> the VRAM gate is skipped, only RAM gates."""
    memory.nvml_available = False
    memory.ram_free_g = 40
    await manager.ensure_loaded("qwen")  # required_vram ignored, RAM 20+4=24 <= 40
    assert manager.state is State.RUNNING
    assert runner.starts == ["qwen"]


async def test_ram_gate_refuses_when_insufficient(manager, memory, runner):
    memory.ram_free_g = 5  # qwen needs 20 + 4 = 24 GiB RAM
    with pytest.raises(InsufficientMemory) as excinfo:
        await manager.ensure_loaded("qwen")
    assert "RAM needs 24.0 GiB" in str(excinfo.value)
    assert "VRAM needs" not in str(excinfo.value)
    assert manager.state is State.STOPPED
    assert runner.starts == []


async def test_ram_gate_on_cpu_only_machine(manager, memory, runner):
    """Windows/Intel machine: no NVML + only required_ram_gib set."""
    memory.nvml_available = False
    cfg = make_config()
    cfg.models["qwen"].required_vram_gib = 0  # CPU-only model: no VRAM gate
    cfg.models["qwen"].required_ram_gib = 20
    m = ModelManager(cfg, memory=memory, runner=runner)
    memory.ram_free_g = 30
    await m.ensure_loaded("qwen")
    assert m.state is State.RUNNING


async def test_unknown_model(manager):
    with pytest.raises(ModelNotFound):
        await manager.ensure_loaded("nope")


async def test_switch_when_idle(manager, memory, runner):
    memory.free_g = 40
    await manager.ensure_loaded("qwen")
    await manager.ensure_loaded("gemma")  # 18 + 4 = 22 <= 40
    assert runner.starts == ["qwen", "gemma"]
    assert runner.stops == 1
    assert manager.state is State.RUNNING
    assert manager.current_model == "gemma"
    assert manager.status()["switch_from"] is None  # cleared after success
    assert manager.status()["switch_to"] is None
    assert memory.release_waits == 1  # VRAM release waited for before the re-check


async def test_switch_decides_on_post_stop_vram(manager, memory, runner):
    """Free VRAM while qwen is loaded is 8 GiB, but after stopping it is 38 GiB.
    The switch must succeed based on the RE-READ value, not the pre-stop one."""
    memory.free_g = 40
    await manager.ensure_loaded("qwen")
    memory.free_g = 8  # the loaded SGLang now holds ~30 GiB
    runner.memory = memory
    runner.free_after_stop = 38  # memory comes back once qwen exits
    await manager.ensure_loaded("gemma")
    assert manager.current_model == "gemma"
    assert runner.stops == 1


async def test_switch_refused_after_stop_if_vram_still_low(manager, memory, runner):
    """External workload grabbed the freed memory: switch fails after the old
    model is already unloaded; state lands in STOPPED (not RUNNING(qwen))."""
    memory.free_g = 40
    await manager.ensure_loaded("qwen")
    memory.free_g = 8
    runner.memory = memory
    runner.free_after_stop = 10  # someone else took the memory
    with pytest.raises(InsufficientMemory):
        await manager.ensure_loaded("gemma")
    assert manager.state is State.STOPPED
    assert manager.current_model is None


async def test_switch_refused_when_busy(manager, memory, runner):
    memory.free_g = 40
    await manager.ensure_loaded("qwen")
    manager.request_started()  # an active request (e.g. streaming)
    with pytest.raises(ModelSwitchBusy):
        await manager.ensure_loaded("gemma")
    assert manager.state is State.RUNNING
    assert manager.current_model == "qwen"
    assert runner.stops == 0  # never killed a serving model
    manager.request_finished()


async def test_startup_timeout(manager, memory, runner):
    memory.free_g = 40
    runner.health_mode = "timeout"
    with pytest.raises(BackendStartupTimeout):
        await manager.ensure_loaded("qwen")
    assert manager.state is State.STOPPED
    assert runner.stops == 1  # failed instance cleaned up


async def test_startup_process_dies(manager, memory, runner):
    memory.free_g = 40
    runner.health_mode = "die"
    with pytest.raises(BackendStartupFailed):
        await manager.ensure_loaded("qwen")
    assert manager.state is State.STOPPED


async def test_concurrent_same_model_shares_one_start(manager, memory, runner):
    memory.free_g = 40
    runner.health_delay = 0.05
    results = await asyncio.gather(
        manager.ensure_loaded("qwen"), manager.ensure_loaded("qwen")
    )
    assert [r.name for r in results] == ["qwen", "qwen"]
    assert runner.starts == ["qwen"]  # one start, both requests forwarded
    assert manager.state is State.RUNNING


async def test_concurrent_different_model_refused_during_start(manager, memory, runner):
    memory.free_g = 40
    runner.health_delay = 0.1
    t1 = asyncio.create_task(manager.ensure_loaded("qwen"))
    await asyncio.sleep(0.02)  # let it reach STARTING
    with pytest.raises(ModelSwitchBusy):
        await manager.ensure_loaded("gemma")
    await t1
    assert runner.starts == ["qwen"]


async def test_waiter_receives_startup_error(manager, memory, runner):
    memory.free_g = 40
    runner.health_mode = "timeout"
    runner.health_delay = 0.05
    t1 = asyncio.create_task(manager.ensure_loaded("qwen"))
    await asyncio.sleep(0.02)
    with pytest.raises(BackendStartupTimeout):
        await manager.ensure_loaded("qwen")
    with pytest.raises(BackendStartupTimeout):
        await t1


async def test_unexpected_process_exit_forces_stopped(manager, memory, runner):
    memory.free_g = 40
    await manager.ensure_loaded("qwen")
    await runner.on_exit(1)  # the exit monitor callback
    assert manager.state is State.STOPPED
    assert manager.current_model is None
    # next request can cold-start again
    await manager.ensure_loaded("qwen")
    assert runner.starts == ["qwen", "qwen"]


async def test_admin_stop_busy_then_ok(manager, memory, runner):
    memory.free_g = 40
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


async def test_stop_cancels_pending_start(manager, memory, runner):
    """Idle stop must be accepted while a model is still STARTING: the
    startup is cancelled and nothing is left running."""
    memory.free_g = 40
    runner.health_delay = 0.2
    t = asyncio.create_task(manager.ensure_loaded("qwen"))
    await asyncio.sleep(0.05)
    assert manager.state is State.STARTING
    await manager.stop()  # accepted while idle (active_requests == 0)
    assert manager.state is State.STOPPED
    assert runner.model is None  # no orphan process
    with pytest.raises(BackendUnavailable):
        await t  # the waiting request gets a clear error


async def test_stop_cancels_pending_switch(manager, memory, runner):
    memory.free_g = 40
    await manager.ensure_loaded("qwen")
    runner.health_delay = 0.2
    t = asyncio.create_task(manager.ensure_loaded("gemma"))
    await asyncio.sleep(0.05)
    assert manager.state is State.SWITCHING
    await manager.stop()
    assert manager.state is State.STOPPED
    assert runner.model is None
    with pytest.raises(BackendUnavailable):
        await t


async def test_stop_idempotent_during_stopping(manager):
    manager.state = State.STOPPING  # an earlier stop is mid-teardown
    await manager.stop()  # no raise, no double teardown
    assert manager.state is State.STOPPING


async def test_force_stop_kills_busy(manager, memory, runner):
    """Force stop works even with active requests (normal stop refuses)."""
    memory.free_g = 40
    await manager.ensure_loaded("qwen")
    manager.request_started()  # in-flight request
    with pytest.raises(ModelSwitchBusy):
        await manager.stop()
    await manager.stop(force=True)
    assert manager.state is State.STOPPED
    assert runner.stops == 1
    manager.request_finished()  # the cut request self-heals the counter


async def test_force_stop_during_starting(manager, memory, runner):
    memory.free_g = 40
    runner.health_delay = 0.2
    t = asyncio.create_task(manager.ensure_loaded("qwen"))
    await asyncio.sleep(0.05)
    await manager.stop(force=True)
    assert manager.state is State.STOPPED
    with pytest.raises(BackendUnavailable):
        await t
    assert runner.model is None


async def test_client_disconnect_during_start_does_not_wedge(manager, memory, runner):
    """The triggering request being cancelled mid-startup (client disconnect)
    must not leave the machine wedged in STARTING: the process is cleaned up,
    the state lands in STOPPED, and a later request can start again."""
    memory.free_g = 40
    runner.health_delay = 0.2
    t = asyncio.create_task(manager.ensure_loaded("qwen"))
    await asyncio.sleep(0.05)
    assert manager.state is State.STARTING
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert manager.state is State.STOPPED
    assert runner.model is None  # no orphan process
    # not wedged: a new request can cold-start again
    await manager.ensure_loaded("qwen")
    assert manager.state is State.RUNNING
    assert runner.starts == ["qwen", "qwen"]


async def test_shutdown_during_start_cancels_cleanly(memory, runner):
    m = ModelManager(make_config(), memory=memory, runner=runner)
    memory.free_g = 40
    runner.health_delay = 0.2
    t = asyncio.create_task(m.ensure_loaded("qwen"))
    await asyncio.sleep(0.05)
    await m.shutdown()
    assert m.state is State.STOPPED
    assert runner.model is None
    with pytest.raises(BackendUnavailable):
        await t  # the waiting request gets a clean error, not a hang


async def test_request_accounting(manager, memory, runner):
    memory.free_g = 40
    await manager.ensure_loaded("qwen")
    assert manager.active_requests == 0
    manager.request_started()
    assert manager.active_requests == 1
    manager.request_finished()
    assert manager.active_requests == 0


async def test_idle_seconds_reporting(manager, memory, runner, clock):
    memory.free_g = 40
    await manager.ensure_loaded("qwen")
    assert manager.idle_seconds == 0
    clock.advance(100)
    assert manager.idle_seconds == 100
    manager.request_started()
    assert manager.idle_seconds == 0  # busy → not idle
    manager.request_finished()


async def test_idle_watchdog_unloads(memory, runner):
    m = ModelManager(make_config(idle_timeout=0.15, idle_check_interval=0.05), memory=memory, runner=runner)
    memory.free_g = 40
    m.start()
    try:
        await m.ensure_loaded("qwen")
        await asyncio.sleep(0.5)
        assert m.state is State.STOPPED
        assert runner.stops == 1
    finally:
        await m.shutdown()


async def test_idle_watchdog_respects_active_requests(memory, runner):
    m = ModelManager(make_config(idle_timeout=0.15, idle_check_interval=0.05), memory=memory, runner=runner)
    memory.free_g = 40
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


async def test_shutdown_stops_sglang(memory, runner):
    m = ModelManager(make_config(), memory=memory, runner=runner)
    memory.free_g = 40
    await m.ensure_loaded("qwen")
    await m.shutdown()
    assert m.state is State.STOPPED
    assert runner.stops == 1


async def test_status_shape(manager, memory, runner):
    memory.free_g = 40
    memory.total_g = 48
    await manager.ensure_loaded("qwen")
    status = manager.status()
    assert status["state"] == "running"
    assert status["model"] == "qwen"
    assert status["starting_model"] is None  # cleared after a successful start
    assert status["switch_from"] is None
    assert status["switch_to"] is None
    assert status["active_requests"] == 0
    assert status["idle_timeout_seconds"] == 3600
    assert status["max_context_tokens"] == 123456  # real KV capacity probed at load
    assert status["memory"]["vram_free_gib"] == 40
    assert status["memory"]["vram_total_gib"] == 48
    # per-model registry view: loaded model carries the number, others None
    ms = {m["name"]: m for m in manager.models_status()}
    assert ms["qwen"]["max_context_tokens"] == 123456
    assert ms["gemma"]["max_context_tokens"] is None  # unknown until loaded
    # unloaded -> None everywhere
    await manager.stop()
    assert manager.status()["max_context_tokens"] is None
    assert all(m["max_context_tokens"] is None for m in manager.models_status())


async def test_context_probe_failure_is_benign(manager, memory, runner):
    """If the backend has no probe endpoint, the load must still succeed."""
    memory.free_g = 40
    runner.max_tokens_capacity = None
    await manager.ensure_loaded("qwen")
    assert manager.state is State.RUNNING
    assert manager.max_context_tokens is None
