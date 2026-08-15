"""Backend subprocess lifecycle: launch (command template), health-poll,
terminate.

The manager spawns the inference backend itself (on demand) instead of a
systemd unit, which keeps the whole lifecycle under one owner: start,
health-gate, stop, restart, and automatic unload are all driven here.

The backend is deliberately backend-agnostic — SGLang, vLLM, llama.cpp,
OpenVINO or any OpenAI-compatible server — the launch line comes from
``backend.command_template`` (see :func:`build_backend_command`).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import sys
import time
from typing import Awaitable, Callable

import httpx

from .config import ModelSpec, BackendConfig
from .errors import BackendStartupFailed, BackendStartupTimeout, BackendUnavailable

logger = logging.getLogger(__name__)

OnExit = Callable[[int | None], Awaitable[None]]

# Placeholders available in backend.command_template
TEMPLATE_PLACEHOLDERS = (
    "{python} {model_path} {model_name} {host} {port} {device} {extra_args}"
)


def build_backend_command(cfg: BackendConfig, spec: ModelSpec, device: int) -> list[str]:
    """Render ``backend.command_template`` into a launch argv list.

    Placeholders:

    - ``{python}``      → the current interpreter (quoted)
    - ``{model_path}``  → the model's path / hub id (quoted)
    - ``{model_name}``  → the registry name (quoted)
    - ``{host}`` / ``{port}`` → backend listen address (port unquoted)
    - ``{device}``      → ``memory.device`` (CUDA_VISIBLE_DEVICES value)
    - ``{extra_args}``  → per-model ``backend.extra_args``, shell-quoted;
                          appended at the end when the template lacks it

    Examples::

        # SGLang (default template)
        {python} -m sglang.launch_server --model-path {model_path} \\
                 --host {host} --port {port} --tp-size 1 {extra_args}

        # llama.cpp (Linux / macOS)
        llama-server -m {model_path} --host {host} --port {port} {extra_args}

        # llama.cpp (Windows, .exe with a quoted path)
        C:\\llama.cpp\\llama-server.exe -m {model_path} --host {host} --port {port} {extra_args}

        # vLLM
        {python} -m vllm.entrypoints.openai.api_server --model {model_path} \\
                 --host {host} --port {port} {extra_args}

        # OpenVINO GenAI (custom OpenAI-compatible server)
        {python} -m my_openvino_server --model {model_path} --port {port} {extra_args}
    """

    extra = " ".join(shlex.quote(a) for a in spec.backend.extra_args)
    rendered = cfg.command_template.format(
        python=shlex.quote(sys.executable),
        model_path=shlex.quote(spec.path),
        model_name=shlex.quote(spec.name),
        host=shlex.quote(cfg.host),
        port=str(cfg.port),
        device=str(device),
        extra_args=extra,
    )
    if "{extra_args}" not in cfg.command_template and extra:
        rendered += " " + extra
    cmd = shlex.split(rendered)
    logger.info("launching backend: %s (device %s)", " ".join(cmd), device)
    return cmd


def build_process_env(
    cfg: BackendConfig,
    spec: ModelSpec,
    device: int,
    base_env: dict | None = None,
) -> dict[str, str]:
    """Environment for the backend child process.

    The manager's own ``PYTHONPATH`` (e.g. pointing at its local deps dir)
    must NOT leak into the child — it must use its interpreter's own
    site-packages (a leaked path has caused real breakage, e.g. a wrong
    pydantic).  ``PYTHONPATH`` is only passed through when the config or the
    model explicitly sets one.  ``CUDA_VISIBLE_DEVICES`` is only meaningful
    for CUDA backends and harmless elsewhere.
    """

    env = {
        **(os.environ if base_env is None else base_env),
        "CUDA_VISIBLE_DEVICES": str(device),
        **cfg.env,
        **spec.backend.env,
    }
    if "PYTHONPATH" not in cfg.env and "PYTHONPATH" not in spec.backend.env:
        env.pop("PYTHONPATH", None)
    return env


class BackendRunner:
    """Owns the backend subprocess. Only one instance at a time (single
    device, single active model — a core V1 constraint enforced by the
    manager)."""

    def __init__(
        self,
        cfg: BackendConfig,
        device: int = 0,
        on_exit: OnExit | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.cfg = cfg
        self.device = device
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
            raise BackendUnavailable(f"backend already running (pid {self.proc.pid})")
        cmd = build_backend_command(self.cfg, spec, self.device)
        env = build_process_env(self.cfg, spec, self.device)
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.model_name = spec.name
        logger.info("backend started: pid=%d model=%s", self.proc.pid, spec.name)
        self._monitor_task = asyncio.create_task(self._monitor(), name="backend-exit-monitor")
        # Drain the pipes into the manager log, otherwise a chatty backend can
        # fill the OS pipe buffer and block.
        asyncio.create_task(self._drain(self.proc.stdout, logging.INFO, "backend"), name="backend-stdout-drain")
        asyncio.create_task(self._drain(self.proc.stderr, logging.WARNING, "backend"), name="backend-stderr-drain")

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
        logger.warning("backend process exited: pid=%d model=%s code=%s", proc.pid, self.model_name, rc)
        if self.on_exit is not None:
            try:
                await self.on_exit(rc)
            except Exception:  # noqa: BLE001 - never let the monitor die noisily
                logger.exception("on_exit callback failed")

    async def wait_health(self, timeout_seconds: float) -> None:
        """Poll the health endpoint until 200, the process dies, or timeout."""

        proc = self.proc
        if proc is None:
            raise BackendStartupFailed("backend process was never started")
        url = f"{self.cfg.base_url}{self.cfg.health_path}"
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        while True:
            if proc.returncode is not None:
                raise BackendStartupFailed(
                    f"backend process exited during startup (code {proc.returncode}); "
                    "check the manager log for the backend traceback"
                )
            try:
                resp = await self._http.get(url)
                if resp.status_code == 200:
                    logger.info("backend healthy after %.1fs: %s", time.monotonic() - deadline + timeout_seconds, url)
                    return
                last_error = RuntimeError(f"health returned HTTP {resp.status_code}")
            except httpx.HTTPError as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                raise BackendStartupTimeout(
                    f"backend did not become healthy within {timeout_seconds:.0f}s "
                    f"(last probe error: {last_error})"
                )
            await asyncio.sleep(1.0)

    async def stop(self, timeout_seconds: float = 60.0) -> None:
        """Terminate the backend: SIGTERM, escalate to SIGKILL after timeout.
        (On Windows terminate() is already a hard kill; the flow still works.)"""

        proc = self.proc
        if proc is None:
            return
        self.proc = None
        self.model_name = None
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            self._monitor_task = None
        if proc.returncode is None:
            logger.info("stopping backend pid=%d (SIGTERM, %ss grace)", proc.pid, timeout_seconds)
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
            except BaseException:
                # Timeout OR cancellation (e.g. the requesting client
                # disconnected) — never leave a half-dead child holding memory.
                logger.warning("backend pid=%d did not exit in time, sending SIGKILL", proc.pid)
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.shield(proc.wait())
                except BaseException:
                    pass
                raise
        logger.info("backend stopped (pid=%d)", proc.pid)

    async def aclose(self) -> None:
        await self._http.aclose()
