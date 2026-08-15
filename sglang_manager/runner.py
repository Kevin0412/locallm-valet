"""SGLang subprocess lifecycle: launch, health-poll, terminate.

The manager spawns SGLang itself (on demand) instead of a systemd unit, which
keeps the whole lifecycle under one owner: start, health-gate, stop, restart,
and automatic unload are all driven here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Awaitable, Callable

import httpx

from .config import ModelSpec, SglangConfig
from .errors import SglangStartupFailed, SglangStartupTimeout, SglangUnavailable

logger = logging.getLogger(__name__)

OnExit = Callable[[int | None], Awaitable[None]]


def build_launch_command(cfg: SglangConfig, spec: ModelSpec, gpu_device: int) -> list[str]:
    """Assemble the SGLang launch command line.

    Base command (default: the running interpreter's ``-m
    sglang.launch_server``, overridable via ``sglang.command`` in the config —
    e.g. point it at the conda env that has SGLang installed).
    """

    base = list(cfg.command) or [sys.executable, "-m", "sglang.launch_server"]
    cmd: list[str] = [
        *base,
        "--model-path", spec.path,
        "--host", cfg.host,
        "--port", str(cfg.port),
        "--tp-size", "1",
    ]
    if spec.sglang.mem_fraction_static is not None:
        cmd += ["--mem-fraction-static", str(spec.sglang.mem_fraction_static)]
    if spec.sglang.context_length is not None:
        cmd += ["--context-length", str(spec.sglang.context_length)]
    cmd += spec.sglang.extra_args
    logger.info("launching SGLang: %s (device %d)", " ".join(cmd), gpu_device)
    return cmd


class SglangRunner:
    """Owns the SGLang subprocess. Only one instance at a time (single GPU,
    single active model — a core V1 constraint enforced by the manager)."""

    def __init__(
        self,
        cfg: SglangConfig,
        gpu_device: int = 0,
        on_exit: OnExit | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.cfg = cfg
        self.gpu_device = gpu_device
        self.on_exit = on_exit
        self._http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        self.proc: asyncio.subprocess.Process | None = None
        self.model_name: str | None = None
        self._monitor_task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self, spec: ModelSpec) -> None:
        if self.proc is not None and self.proc.returncode is None:
            raise SglangUnavailable(f"SGLang already running (pid {self.proc.pid})")
        cmd = build_launch_command(self.cfg, spec, self.gpu_device)
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(self.gpu_device),
            **self.cfg.env,
            **spec.sglang.env,
        }
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.model_name = spec.name
        logger.info("SGLang started: pid=%d model=%s", self.proc.pid, spec.name)
        self._monitor_task = asyncio.create_task(self._monitor(), name="sglang-exit-monitor")
        # Drain the pipes into the manager log, otherwise a chatty SGLang can
        # fill the OS pipe buffer and block.
        asyncio.create_task(self._drain(self.proc.stdout, logging.INFO, "sglang"), name="sglang-stdout-drain")
        asyncio.create_task(self._drain(self.proc.stderr, logging.WARNING, "sglang"), name="sglang-stderr-drain")

    async def _drain(self, stream: asyncio.StreamReader | None, level: int, tag: str) -> None:
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    return
                logger.log(level, "[%s] %s", tag, line.decode(errors="replace").rstrip())
        except Exception:  # noqa: BLE001 - best-effort log draining
            pass

    async def _monitor(self) -> None:
        proc = self.proc
        if proc is None:
            return
        rc = await proc.wait()
        logger.warning("SGLang process exited: pid=%d model=%s code=%s", proc.pid, self.model_name, rc)
        if self.on_exit is not None:
            try:
                await self.on_exit(rc)
            except Exception:  # noqa: BLE001 - never let the monitor die noisily
                logger.exception("on_exit callback failed")

    async def wait_health(self, timeout_seconds: float) -> None:
        """Poll the health endpoint until 200, the process dies, or timeout."""

        proc = self.proc
        if proc is None:
            raise SglangStartupFailed("SGLang process was never started")
        url = f"{self.cfg.base_url}{self.cfg.health_path}"
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        while True:
            if proc.returncode is not None:
                raise SglangStartupFailed(
                    f"SGLang process exited during startup (code {proc.returncode}); "
                    "check the manager log for the SGLang traceback"
                )
            try:
                resp = await self._http.get(url)
                if resp.status_code == 200:
                    logger.info("SGLang healthy after %.1fs: %s", time.monotonic() - deadline + timeout_seconds, url)
                    return
                last_error = RuntimeError(f"health returned HTTP {resp.status_code}")
            except httpx.HTTPError as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                raise SglangStartupTimeout(
                    f"SGLang did not become healthy within {timeout_seconds:.0f}s "
                    f"(last probe error: {last_error})"
                )
            await asyncio.sleep(1.0)

    async def stop(self, timeout_seconds: float = 60.0) -> None:
        """Terminate SGLang: SIGTERM, escalate to SIGKILL after timeout."""

        proc = self.proc
        if proc is None:
            return
        self.proc = None
        self.model_name = None
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            self._monitor_task = None
        if proc.returncode is None:
            logger.info("stopping SGLang pid=%d (SIGTERM, %ss grace)", proc.pid, timeout_seconds)
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning("SGLang pid=%d ignored SIGTERM, sending SIGKILL", proc.pid)
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
        logger.info("SGLang stopped (pid=%d)", proc.pid)

    async def aclose(self) -> None:
        await self._http.aclose()
