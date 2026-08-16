"""The core orchestrator: state machine + lifecycle transitions + gating.

Concurrency model
-----------------
- All lifecycle work (start / stop / switch) is serialized behind one
  ``asyncio.Lock`` (``_transition_lock``): at most one backend transition runs
  at a time, so two racing requests can never start two backend instances.
- Requests whose model already matches ``RUNNING`` never take that lock — they
  are forwarded straight to the backend.
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
    MemoryUnavailable,
    InsufficientMemory,
    ModelNotFound,
    ModelSwitchBusy,
    BackendStartupFailed,
    BackendStartupTimeout,
    BackendUnavailable,
)
from .memory import MemoryMonitorProtocol
from .runner import BackendRunner
from .state import State

logger = logging.getLogger(__name__)


class Clock(Protocol):
    def __call__(self) -> float: ...


class ModelManager:
    def __init__(
        self,
        config: Config,
        memory: MemoryMonitorProtocol,
        runner: BackendRunner,
        clock: Clock = time.monotonic,
    ):
        self.cfg = config
        self.memory = memory
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
        # Real KV-pool capacity (tokens) of the loaded backend, probed once
        # after a successful load. None while stopped.
        self.max_context_tokens: int | None = None
        self.last_activity: float = self._clock()
        self.started_at: float = self._clock()
        self._watchdog_task: asyncio.Task | None = None
        # Set when an admin stop cancels an in-flight STARTING/SWITCHING
        # transition; checked at safe points inside _start/_switch.
        self._cancel_requested = False

        self.runner.on_exit = self._on_backend_exit

    # ------------------------------------------------------------------ util

    def _now(self) -> float:
        return self._clock()

    @property
    def idle_seconds(self) -> float:
        if self.active_requests > 0 or self.state is not State.RUNNING:
            return 0.0
        return max(0.0, self._now() - self.last_activity)

    def _require_running(self) -> None:
        """Fail fast if the backend is dead but the state machine says RUNNING."""

        if self.state is State.RUNNING and not self.runner.running:
            logger.error("state=RUNNING but the backend process is gone; forcing STOPPED")
            self._force_stopped()

    def _force_stopped(self) -> None:
        self.state = State.STOPPED
        self.current_model = None
        self.max_context_tokens = None
        self._starting_model = None
        self._switch_from = None
        self._switch_to = None
        # NOTE: _cancel_requested is deliberately NOT reset here — a stop that
        # cancelled an in-flight transition must stay visible until that
        # transition observes it at its next safe point.  It is reset at the
        # start of every new transition (_start/_switch).
        self.last_activity = self._now()

    # ------------------------------------------------------------- lifecycle

    async def ensure_loaded(self, model_name: str) -> ModelSpec:
        """Gate: make sure the backend is RUNNING with ``model_name``.

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
                raise BackendUnavailable("backend is stopping, retry in a moment")

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

    async def _check_memory(self, spec: ModelSpec, context: str) -> None:
        """Gate on VRAM (when NVML is available) and/or system RAM.

        - ``required_vram_gib > 0`` → checked only if NVML works; on CPU/NPU
          machines (no NVIDIA driver) it is skipped with a warning.
        - ``required_ram_gib > 0`` → checked via psutil (cross-platform).

        Never hard-start into an OOM: refuse BEFORE launching.
        """

        margin = self.cfg.memory.safety_margin_gib
        problems: list[str] = []

        if spec.required_vram_gib > 0:
            if self.memory.nvml_available:
                free = self.memory.vram_free_gib()
                needed = spec.required_vram_gib + margin
                if free < needed:
                    problems.append(
                        f"VRAM needs {needed:.1f} GiB ({spec.required_vram_gib:.1f} "
                        f"required + {margin:.1f} margin), only {free:.1f} GiB free "
                        f"on device {self.memory.device}"
                    )
            else:
                logger.warning(
                    "model %s requires VRAM but NVML is unavailable; VRAM gate skipped",
                    spec.name,
                )

        if spec.required_ram_gib > 0:
            free = self.memory.ram_available_gib()
            needed = spec.required_ram_gib + margin
            if free < needed:
                problems.append(
                    f"RAM needs {needed:.1f} GiB ({spec.required_ram_gib:.1f} "
                    f"required + {margin:.1f} margin), only {free:.1f} GiB available"
                )

        if problems:
            raise InsufficientMemory(f"cannot {context} model {spec.name!r}: " + "; ".join(problems))

    async def _start(self, spec: ModelSpec) -> None:
        """STOPPED -> STARTING -> RUNNING(spec), or back to STOPPED on failure."""

        async with self._transition_lock:
            if self.state is not State.STOPPED:
                return  # somebody else took care of it; re-evaluate in the gate loop
            self._cancel_requested = False
            self.state = State.STARTING
            self._starting_model = spec.name
            self._start_future = asyncio.get_running_loop().create_future()
            self._start_future.add_done_callback(self._silence_unretrieved)

        try:
            # VRAM is checked BEFORE launching: never hard-start into a CUDA OOM.
            await self._check_memory(spec, context="start")
            if self._cancel_requested:
                raise BackendUnavailable("start cancelled by stop request")
            await self.runner.start(spec)
            if self._cancel_requested:
                raise BackendUnavailable("start cancelled by stop request")
            await self.runner.wait_health(self.cfg.backend.startup_timeout_seconds)
            if self._cancel_requested:
                raise BackendUnavailable("start cancelled by stop request")
        except InsufficientMemory as exc:
            logger.warning("start refused: %s", exc)
            self._fail_start(exc)
            raise
        except MemoryUnavailable as exc:
            logger.error("start refused: %s", exc)
            self._fail_start(exc)
            raise
        except BackendUnavailable as exc:
            logger.info("start aborted: %s", exc)
            await self._cleanup_after_failed_start()
            self._fail_start(exc)
            raise
        except BackendStartupTimeout as exc:
            logger.error("startup timed out: %s", exc)
            await self._cleanup_after_failed_start()
            self._fail_start(exc)
            raise
        except Exception as exc:  # process died, spawn error, ...
            if isinstance(exc, BackendStartupFailed):
                logger.error("startup failed: %s", exc)
            else:
                logger.exception("startup failed unexpectedly")
            await self._cleanup_after_failed_start()
            mapped = exc if isinstance(exc, BackendStartupFailed) else BackendStartupFailed(str(exc))
            self._fail_start(mapped)
            raise mapped
        except BaseException as exc:  # CancelledError: client disconnected mid-start
            logger.warning("start interrupted by %s; aborting cleanly", type(exc).__name__)
            try:
                await self._cleanup_after_failed_start()
            except BaseException:
                pass
            self._fail_start(BackendUnavailable("start cancelled before completion"))
            raise

        self.state = State.RUNNING
        self.current_model = spec.name
        self._starting_model = None
        self.last_activity = self._now()
        try:
            self.max_context_tokens = await self.runner.get_max_total_num_tokens()
            if self.max_context_tokens:
                logger.info("model %s ready, real max context = %d tokens", spec.name, self.max_context_tokens)
        except Exception:  # noqa: BLE001 - probing must never break the load
            self.max_context_tokens = None
        self._resolve(self._start_future, None)
        logger.info("model %s ready", spec.name)

    def _fail_start(self, exc: Exception) -> None:
        self._force_stopped()
        self._resolve(self._start_future, exc)

    async def _cleanup_after_failed_start(self) -> None:
        try:
            await self.runner.stop(self.cfg.backend.stop_timeout_seconds)
        except Exception:  # noqa: BLE001 - best effort; original error matters more
            logger.exception("failed to clean up the backend after startup failure")

    async def _switch(self, spec: ModelSpec) -> None:
        """RUNNING(from) -> SWITCHING -> RUNNING(to), or STOPPED on failure.

        Decision is made on the VRAM re-read *after* the old model has been
        stopped and the memory has actually come back: the pre-stop free VRAM
        is meaningless because the old backend is still holding it.
        """

        async with self._transition_lock:
            if self.state is not State.RUNNING or self.current_model == spec.name:
                return
            if self.active_requests > 0:
                raise ModelSwitchBusy(
                    f"model {self.current_model!r} has {self.active_requests} active "
                    f"request(s); switch to {spec.name!r} refused (no preemption)"
                )
            self._cancel_requested = False
            self.state = State.SWITCHING
            self._switch_from = self.current_model
            self._switch_to = spec.name
            self._switch_future = asyncio.get_running_loop().create_future()
            self._switch_future.add_done_callback(self._silence_unretrieved)

        from_model = self._switch_from
        try:
            logger.info("switching %s -> %s", from_model, spec.name)
            await self.runner.stop(self.cfg.backend.stop_timeout_seconds)
            # Wait for VRAM to genuinely come back before re-measuring.
            await self.memory.wait_vram_released(
                self.cfg.memory.release_timeout_seconds,
                self.cfg.memory.poll_interval_seconds,
            )
            if self._cancel_requested:
                raise BackendUnavailable("switch cancelled by stop request")
            await self._check_memory(spec, context="switch to")
            if self._cancel_requested:
                raise BackendUnavailable("switch cancelled by stop request")
            await self.runner.start(spec)
            if self._cancel_requested:
                raise BackendUnavailable("switch cancelled by stop request")
            await self.runner.wait_health(self.cfg.backend.startup_timeout_seconds)
            if self._cancel_requested:
                raise BackendUnavailable("switch cancelled by stop request")
        except InsufficientMemory as exc:
            logger.warning("switch refused after unloading %s: %s", from_model, exc)
            self._fail_switch(exc)
            raise
        except MemoryUnavailable as exc:
            logger.error("switch aborted: %s", exc)
            self._fail_switch(exc)
            raise
        except BackendUnavailable as exc:
            logger.info("switch aborted: %s", exc)
            await self._cleanup_after_failed_start()
            self._fail_switch(exc)
            raise
        except Exception as exc:
            logger.exception("switch %s -> %s failed", from_model, spec.name)
            mapped = (
                exc
                if isinstance(exc, (BackendStartupFailed, BackendStartupTimeout))
                else BackendStartupFailed(str(exc))
            )
            self._fail_switch(mapped)
            raise mapped
        except BaseException as exc:  # CancelledError: client disconnected mid-switch
            logger.warning("switch interrupted by %s; aborting cleanly", type(exc).__name__)
            try:
                await self._cleanup_after_failed_start()
            except BaseException:
                pass
            self._fail_switch(BackendUnavailable("switch cancelled before completion"))
            raise

        self.state = State.RUNNING
        self.current_model = spec.name
        self._switch_from = None
        self._switch_to = None
        self.last_activity = self._now()
        try:
            self.max_context_tokens = await self.runner.get_max_total_num_tokens()
            if self.max_context_tokens:
                logger.info("switch complete: %s -> %s, real max context = %d tokens",
                            from_model, spec.name, self.max_context_tokens)
        except Exception:  # noqa: BLE001 - probing must never break the load
            self.max_context_tokens = None
        self._resolve(self._switch_future, None)
        logger.info("switch complete: %s -> %s", from_model, spec.name)

    def _fail_switch(self, exc: Exception) -> None:
        self._force_stopped()
        self._resolve(self._switch_future, exc)

    @staticmethod
    def _silence_unretrieved(fut: asyncio.Future) -> None:
        """Retrieve a cancelled-transition exception if no request ever awaits
        the future — otherwise asyncio logs 'Future exception was never
        retrieved' when the future is garbage collected."""

        if not fut.cancelled():
            fut.exception()

    def _resolve(self, fut: asyncio.Future | None, exc: Exception | None) -> None:
        if fut is None or fut.done():
            return
        if exc is None:
            fut.set_result(None)
        else:
            fut.set_exception(exc)

    async def stop(self, reason: str = "manual", force: bool = False) -> None:
        """Admin stop: unload the backend and release memory.

        Normal stop (``force=False``) is accepted whenever the system is
        idle (``active_requests == 0``), in any state: a RUNNING model is
        unloaded, an in-flight STARTING/SWITCHING transition is cancelled,
        STOPPING is idempotent, STOPPED is a no-op.  Refused only while
        requests are being served (no preemption).

        Force stop (``force=True``) tears down unconditionally, even with
        active requests — in-flight streaming connections are cut.  It exists
        to reclaim VRAM from a stuck or unwanted workload.
        """

        async with self._transition_lock:
            if self.state is State.STOPPED:
                return
            if not force and self.active_requests > 0:
                raise ModelSwitchBusy(
                    f"cannot stop: {self.active_requests} active request(s); "
                    f"use force-stop to override"
                )
            if self.state is State.STOPPING:
                return  # an earlier stop is already tearing down (idempotent)
            if self.state is State.STARTING:
                logger.info("stop cancels in-flight start of %s", self._starting_model)
                self._cancel_transition(BackendUnavailable("start cancelled by stop request"))
            elif self.state is State.SWITCHING:
                logger.info(
                    "stop cancels in-flight switch %s -> %s",
                    self._switch_from, self._switch_to,
                )
                self._cancel_transition(BackendUnavailable("switch cancelled by stop request"))
            self.state = State.STOPPING
        await self._do_stop(reason)

    def _cancel_transition(self, exc: Exception) -> None:
        """Fail the in-flight transition futures and flag cancellation so
        ``_start``/``_switch`` abort at their next safe point — no new
        backend process may be spawned after the stop decision."""

        self._cancel_requested = True
        for fut in (self._start_future, self._switch_future):
            if fut is not None and not fut.done():
                fut.set_exception(exc)
        self._starting_model = None
        self._switch_from = None
        self._switch_to = None

    async def _do_stop(self, reason: str) -> None:
        try:
            await self.runner.stop(self.cfg.backend.stop_timeout_seconds)
            await self.memory.wait_vram_released(
                self.cfg.memory.release_timeout_seconds,
                self.cfg.memory.poll_interval_seconds,
            )
        except Exception:  # noqa: BLE001 - stopping must always land in STOPPED
            logger.exception("stop failed while tearing down the backend")
        finally:
            self._force_stopped()
            logger.info("backend stopped: %s", reason)

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

    async def _on_backend_exit(self, exit_code: int | None) -> None:
        """The backend subprocess died on its own. Only act if we were RUNNING —
        intentional stops happen under STOPPING/SWITCHING and must be ignored."""

        if self.state is State.RUNNING:
            logger.error("backend died unexpectedly (code=%s); forcing STOPPED", exit_code)
            self._force_stopped()

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(self._watchdog_loop(), name="idle-watchdog")
            logger.info(
                "manager started: %d models, idle timeout %.0fs, safety margin %.1f GiB",
                len(self.cfg.models), self.cfg.idle.timeout_seconds, self.cfg.memory.safety_margin_gib,
            )

    async def shutdown(self) -> None:
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            self._watchdog_task = None
        if self.state is not State.STOPPED:
            async with self._transition_lock:
                if self.state in (State.STARTING, State.SWITCHING):
                    self._cancel_transition(BackendUnavailable("manager shutdown"))
                self.state = State.STOPPING
            await self._do_stop(reason="manager shutdown")
        await self.runner.aclose()

    # -------------------------------------------------------------- status

    def status(self) -> dict:
        memory_info: dict
        if self.memory.nvml_available:
            memory_info = {
                "nvml_available": True,
                "device": self.memory.device,
                "vram_total_gib": round(self.memory.vram_total_gib(), 2),
                "vram_free_gib": round(self.memory.vram_free_gib(), 2),
                "vram_used_gib": round(self.memory.vram_used_gib(), 2),
            }
        else:
            memory_info = {
                "nvml_available": False,
                "device": self.memory.device,
                "vram_total_gib": None,
                "vram_free_gib": None,
                "vram_used_gib": None,
            }
        try:
            memory_info["ram_total_gib"] = round(self.memory.ram_total_gib(), 2)
            memory_info["ram_available_gib"] = round(self.memory.ram_available_gib(), 2)
        except Exception:  # noqa: BLE001 - psutil failure must not break status
            memory_info["ram_total_gib"] = None
            memory_info["ram_available_gib"] = None

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
            "max_context_tokens": self.max_context_tokens,
            "memory": memory_info,
        }

    def models_status(self) -> list[dict]:
        out = []
        for name, spec in self.cfg.models.items():
            out.append(
                {
                    "name": name,
                    "path": spec.path,
                    "required_vram_gib": spec.required_vram_gib,
                    "required_ram_gib": spec.required_ram_gib,
                    "extra_args": spec.backend.extra_args,
                    "loaded": self.state is State.RUNNING and self.current_model == name,
                    "max_context_tokens": (
                        self.max_context_tokens
                        if self.state is State.RUNNING and self.current_model == name
                        else None
                    ),
                }
            )
        return out
