"""The core orchestrator: state machine + lifecycle transitions + gating.

Concurrency model
-----------------
- All lifecycle work (start / stop / switch) is serialized behind one
  ``asyncio.Lock`` (``_transition_lock``): at most one SGLang transition runs
  at a time, so two racing requests can never start two SGLang instances.
- Requests whose model already matches ``RUNNING`` never take that lock — they
  are forwarded straight to SGLang.
- Requests that arrive while a transition for *their* model is in flight await
  the shared transition future, then re-evaluate the state machine.
- Requests for a *different* model while a transition is in flight get
  ``model_switch_busy``; while the loaded model is busy (``active_requests >
  0``) switches are refused — no preemption, streaming connections are never
  cut.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol

from .config import Config, ModelSpec
from .errors import (
    GpuUnavailable,
    InsufficientGpuMemory,
    ModelNotFound,
    ModelSwitchBusy,
    SglangStartupFailed,
    SglangStartupTimeout,
    SglangUnavailable,
)
from .gpu import GpuMonitorProtocol
from .runner import SglangRunner
from .state import State

logger = logging.getLogger(__name__)


class Clock(Protocol):
    def __call__(self) -> float: ...


class ModelManager:
    def __init__(
        self,
        config: Config,
        gpu: GpuMonitorProtocol,
        runner: SglangRunner,
        clock: Clock = time.monotonic,
    ):
        self.cfg = config
        self.gpu = gpu
        self.runner = runner
        self._clock = clock

        self._transition_lock = asyncio.Lock()
        self.state = State.STOPPED
        self.current_model: str | None = None
        self._starting_model: str | None = None
        self._switch_from: str | None = None
        self._switch_to: str | None = None
        self._start_future: asyncio.Future | None = None
        self._switch_future: asyncio.Future | None = None

        self.active_requests = 0
        self.last_activity: float = self._clock()
        self.started_at: float = self._clock()
        self._watchdog_task: asyncio.Task | None = None

        self.runner.on_exit = self._on_sglang_exit

    # ------------------------------------------------------------------ util

    def _now(self) -> float:
        return self._clock()

    @property
    def idle_seconds(self) -> float:
        if self.active_requests > 0 or self.state is not State.RUNNING:
            return 0.0
        return max(0.0, self._now() - self.last_activity)

    def _require_running(self) -> None:
        """Fail fast if SGLang is dead but the state machine says RUNNING."""

        if self.state is State.RUNNING and not self.runner.running:
            logger.error("state=RUNNING but SGLang process is gone; forcing STOPPED")
            self._force_stopped()

    def _force_stopped(self) -> None:
        self.state = State.STOPPED
        self.current_model = None
        self._starting_model = None
        self._switch_from = None
        self._switch_to = None
        self.last_activity = self._now()

    # ------------------------------------------------------------- lifecycle

    async def ensure_loaded(self, model_name: str) -> ModelSpec:
        """Gate: make sure SGLang is RUNNING with ``model_name``.

        Returns the model spec when ready; raises a typed error otherwise.
        This is the only place requests and transitions interact.
        """

        spec = self.cfg.get_model(model_name)
        if spec is None:
            raise ModelNotFound(f"model {model_name!r} is not in the registry")

        while True:
            self._require_running()
            st = self.state

            if st is State.RUNNING and self.current_model == model_name:
                return spec

            if st is State.STOPPED:
                await self._start(spec)
                continue

            if st is State.STARTING:
                if self._starting_model == model_name:
                    assert self._start_future is not None
                    await self._start_future  # raises the startup error on failure
                    continue
                raise ModelSwitchBusy(
                    f"model {self._starting_model!r} is starting; switch to {model_name!r} refused"
                )

            if st is State.STOPPING:
                raise SglangUnavailable("SGLang is stopping, retry in a moment")

            if st is State.SWITCHING:
                if self._switch_to == model_name:
                    assert self._switch_future is not None
                    await self._switch_future  # raises the switch error on failure
                    continue
                raise ModelSwitchBusy(
                    f"switching {self._switch_from!r} -> {self._switch_to!r}; "
                    f"request for {model_name!r} refused"
                )

            # RUNNING with a different model.
            if self.active_requests > 0:
                raise ModelSwitchBusy(
                    f"model {self.current_model!r} has {self.active_requests} active "
                    f"request(s); switching to {model_name!r} refused (no preemption)"
                )
            await self._switch(spec)

    def request_started(self) -> None:
        """Call once the model is gated and about to proxy a request."""

        self.last_activity = self._now()
        self.active_requests += 1

    def request_finished(self) -> None:
        """Call when the request truly ends.

        For ``stream=true`` this must be wired to the SSE stream's close /
        full-consumption event, not to the moment the response headers are
        sent — otherwise the idle watchdog could unload the model while a
        generation is still streaming.
        """

        self.active_requests = max(0, self.active_requests - 1)
        self.last_activity = self._now()

    async def _check_vram(self, spec: ModelSpec, context: str) -> None:
        free = self.gpu.free_gib()
        margin = self.cfg.gpu.safety_margin_gib
        needed = spec.required_vram_gib + margin
        if free < needed:
            raise InsufficientGpuMemory(
                f"cannot {context} model {spec.name!r}: needs {needed:.1f} GiB "
                f"({spec.required_vram_gib:.1f} required + {margin:.1f} safety margin), "
                f"only {free:.1f} GiB free on device {self.gpu.device}"
            )

    async def _start(self, spec: ModelSpec) -> None:
        """STOPPED -> STARTING -> RUNNING(spec), or back to STOPPED on failure."""

        async with self._transition_lock:
            if self.state is not State.STOPPED:
                return  # somebody else took care of it; re-evaluate in the gate loop
            self.state = State.STARTING
            self._starting_model = spec.name
            self._start_future = asyncio.get_running_loop().create_future()

        try:
            # VRAM is checked BEFORE launching: never hard-start into a CUDA OOM.
            await self._check_vram(spec, context="start")
            await self.runner.start(spec)
            await self.runner.wait_health(self.cfg.sglang.startup_timeout_seconds)
        except InsufficientGpuMemory as exc:
            logger.warning("start refused: %s", exc)
            self._fail_start(exc)
            raise
        except GpuUnavailable as exc:
            logger.error("start refused: %s", exc)
            self._fail_start(exc)
            raise
        except SglangStartupTimeout as exc:
            logger.error("startup timed out: %s", exc)
            await self._cleanup_after_failed_start()
            self._fail_start(exc)
            raise
        except Exception as exc:  # process died, spawn error, ...
            if isinstance(exc, SglangStartupFailed):
                logger.error("startup failed: %s", exc)
            else:
                logger.exception("startup failed unexpectedly")
            await self._cleanup_after_failed_start()
            mapped = exc if isinstance(exc, SglangStartupFailed) else SglangStartupFailed(str(exc))
            self._fail_start(mapped)
            raise mapped

        self.state = State.RUNNING
        self.current_model = spec.name
        self.last_activity = self._now()
        self._resolve(self._start_future, None)
        logger.info("model %s ready", spec.name)

    def _fail_start(self, exc: Exception) -> None:
        self._force_stopped()
        self._resolve(self._start_future, exc)

    async def _cleanup_after_failed_start(self) -> None:
        try:
            await self.runner.stop(self.cfg.sglang.stop_timeout_seconds)
        except Exception:  # noqa: BLE001 - best effort; original error matters more
            logger.exception("failed to clean up SGLang after startup failure")

    async def _switch(self, spec: ModelSpec) -> None:
        """RUNNING(from) -> SWITCHING -> RUNNING(to), or STOPPED on failure.

        Decision is made on the VRAM re-read *after* the old model has been
        stopped and the memory has actually come back: the pre-stop free VRAM
        is meaningless because the old SGLang is still holding it.
        """

        async with self._transition_lock:
            if self.state is not State.RUNNING or self.current_model == spec.name:
                return
            if self.active_requests > 0:
                raise ModelSwitchBusy(
                    f"model {self.current_model!r} has {self.active_requests} active "
                    f"request(s); switch to {spec.name!r} refused (no preemption)"
                )
            self.state = State.SWITCHING
            self._switch_from = self.current_model
            self._switch_to = spec.name
            self._switch_future = asyncio.get_running_loop().create_future()

        from_model = self._switch_from
        try:
            logger.info("switching %s -> %s", from_model, spec.name)
            await self.runner.stop(self.cfg.sglang.stop_timeout_seconds)
            # Wait for VRAM to genuinely come back before re-measuring.
            await self.gpu.wait_vram_released(
                self.cfg.gpu.vram_release_timeout_seconds,
                self.cfg.gpu.vram_poll_interval_seconds,
            )
            await self._check_vram(spec, context="switch to")
            await self.runner.start(spec)
            await self.runner.wait_health(self.cfg.sglang.startup_timeout_seconds)
        except InsufficientGpuMemory as exc:
            logger.warning("switch refused after unloading %s: %s", from_model, exc)
            self._fail_switch(exc)
            raise
        except GpuUnavailable as exc:
            logger.error("switch aborted: %s", exc)
            self._fail_switch(exc)
            raise
        except Exception as exc:
            logger.exception("switch %s -> %s failed", from_model, spec.name)
            mapped = (
                exc
                if isinstance(exc, (SglangStartupFailed, SglangStartupTimeout))
                else SglangStartupFailed(str(exc))
            )
            self._fail_switch(mapped)
            raise mapped

        self.state = State.RUNNING
        self.current_model = spec.name
        self.last_activity = self._now()
        self._resolve(self._switch_future, None)
        logger.info("switch complete: %s -> %s", from_model, spec.name)

    def _fail_switch(self, exc: Exception) -> None:
        self._force_stopped()
        self._resolve(self._switch_future, exc)

    def _resolve(self, fut: asyncio.Future | None, exc: Exception | None) -> None:
        if fut is None or fut.done():
            return
        if exc is None:
            fut.set_result(None)
        else:
            fut.set_exception(exc)

    async def stop(self, reason: str = "manual") -> None:
        """Admin stop: refuse while busy or mid-transition."""

        async with self._transition_lock:
            if self.state is State.STOPPED:
                return
            if self.state is not State.RUNNING:
                raise ModelSwitchBusy(f"cannot stop while {self.state.value}")
            if self.active_requests > 0:
                raise ModelSwitchBusy(
                    f"cannot stop: {self.active_requests} active request(s)"
                )
            self.state = State.STOPPING
        await self._do_stop(reason)

    async def _do_stop(self, reason: str) -> None:
        try:
            await self.runner.stop(self.cfg.sglang.stop_timeout_seconds)
            await self.gpu.wait_vram_released(
                self.cfg.gpu.vram_release_timeout_seconds,
                self.cfg.gpu.vram_poll_interval_seconds,
            )
        except Exception:  # noqa: BLE001 - stopping must always land in STOPPED
            logger.exception("stop failed while tearing down SGLang")
        finally:
            self._force_stopped()
            logger.info("SGLang stopped: %s", reason)

    # ------------------------------------------------------------ watchdog

    async def _watchdog_loop(self) -> None:
        """Auto-unload after ``idle.timeout_seconds`` of zero activity."""

        while True:
            await asyncio.sleep(self.cfg.idle.check_interval_seconds)
            if self.state is State.RUNNING and self.active_requests == 0:
                idle = self._now() - self.last_activity
                if idle >= self.cfg.idle.timeout_seconds:
                    logger.info("idle watchdog: no requests for %.0fs, unloading model", idle)
                    try:
                        await self.stop(reason="idle timeout")
                    except ModelSwitchBusy:
                        pass  # a request or transition won the race; fine

    async def _on_sglang_exit(self, exit_code: int | None) -> None:
        """The SGLang subprocess died on its own. Only act if we were RUNNING —
        intentional stops happen under STOPPING/SWITCHING and must be ignored."""

        if self.state is State.RUNNING:
            logger.error("SGLang died unexpectedly (code=%s); forcing STOPPED", exit_code)
            self._force_stopped()

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(self._watchdog_loop(), name="idle-watchdog")
            logger.info(
                "manager started: %d models, idle timeout %.0fs, safety margin %.1f GiB",
                len(self.cfg.models), self.cfg.idle.timeout_seconds, self.cfg.gpu.safety_margin_gib,
            )

    async def shutdown(self) -> None:
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            self._watchdog_task = None
        if self.state is not State.STOPPED:
            self.state = State.STOPPING
            await self._do_stop(reason="manager shutdown")
        await self.runner.aclose()

    # -------------------------------------------------------------- status

    def status(self) -> dict:
        gpu_info: dict
        try:
            gpu_info = {
                "available": True,
                "device": self.gpu.device,
                "total_gib": round(self.gpu.total_gib(), 2),
                "free_gib": round(self.gpu.free_gib(), 2),
                "used_gib": round(self.gpu.used_gib(), 2),
            }
        except Exception:  # noqa: BLE001 - NVML may be absent; report it
            gpu_info = {"available": False, "device": self.gpu.device, "error": "NVML unavailable"}

        return {
            "state": self.state.value,
            "model": self.current_model,
            "starting_model": self._starting_model,
            "switch_from": self._switch_from,
            "switch_to": self._switch_to,
            "active_requests": self.active_requests,
            "idle_seconds": round(self.idle_seconds, 1),
            "idle_timeout_seconds": self.cfg.idle.timeout_seconds,
            "uptime_seconds": round(self._now() - self.started_at, 1),
            "gpu": gpu_info,
        }

    def models_status(self) -> list[dict]:
        out = []
        for name, spec in self.cfg.models.items():
            out.append(
                {
                    "name": name,
                    "path": spec.path,
                    "required_vram_gib": spec.required_vram_gib,
                    "mem_fraction_static": spec.sglang.mem_fraction_static,
                    "context_length": spec.sglang.context_length,
                    "loaded": self.state is State.RUNNING and self.current_model == name,
                }
            )
        return out
