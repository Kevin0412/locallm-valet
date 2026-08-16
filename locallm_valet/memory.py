"""Resource monitoring: GPU VRAM (NVML, optional) + system RAM (psutil).

Backend-agnostic gates:

- VRAM: read through NVML; only enforced when NVML is available (a CUDA
  machine).  On CPU/NPU machines (no NVIDIA driver) the VRAM check is
  skipped automatically and only the RAM gate applies.
- RAM: read through psutil (``virtual_memory().available``), works on
  Linux, Windows and macOS alike.

Start feasibility is computed as::

    required_vram_gib + safety_margin_gib <= free VRAM   (if NVML available)
    required_ram_gib  + safety_margin_gib <= free RAM    (if required > 0)

``free`` is always re-read *after* the previous backend has exited and memory
has been observed to come back, so decisions reflect reality, not estimates.
locallm-valet only reads these numbers — it never touches processes it does not
own.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol

from .errors import MemoryUnavailable

logger = logging.getLogger(__name__)

GIB = 1024**3


class MemoryMonitorProtocol(Protocol):
    device: int

    @property
    def nvml_available(self) -> bool: ...

    def vram_total_gib(self) -> float: ...
    def vram_free_gib(self) -> float: ...
    def vram_used_gib(self) -> float: ...
    def ram_total_gib(self) -> float: ...
    def ram_available_gib(self) -> float: ...

    async def wait_vram_released(self, timeout_seconds: float, poll_interval: float = 1.0) -> None: ...


class MemoryMonitor:
    """VRAM via NVML (lazy init, optional) + RAM via psutil."""

    def __init__(self, device: int = 0):
        self.device = device
        self._nvml = None
        self._handle = None
        self._nvml_failed = False

    # ------------------------------------------------------------- NVML

    @property
    def nvml_available(self) -> bool:
        """Lazy NVML probe. Never raises; a missing driver just disables the
        VRAM gate (CPU/NPU machines are a first-class target)."""

        if self._handle is not None:
            return True
        if self._nvml_failed:
            return False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
            return True
        except Exception:  # noqa: BLE001 - any NVML failure means "no VRAM visibility"
            self._nvml_failed = True
            logger.warning(
                "NVML unavailable for device %d; VRAM checks disabled "
                "(CPU/NPU machine or missing NVIDIA driver)", self.device
            )
            return False

    def _mem(self):
        if not self.nvml_available:
            raise MemoryUnavailable(
                f"NVML unavailable for device {self.device}; cannot read VRAM"
            )
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle)

    def vram_total_gib(self) -> float:
        return self._mem().total / GIB

    def vram_free_gib(self) -> float:
        return self._mem().free / GIB

    def vram_used_gib(self) -> float:
        return self._mem().used / GIB

    # -------------------------------------------------------------- RAM

    def ram_total_gib(self) -> float:
        import psutil

        return psutil.virtual_memory().total / GIB

    def ram_available_gib(self) -> float:
        import psutil

        return psutil.virtual_memory().available / GIB

    # ------------------------------------------------------------ waits

    async def wait_vram_released(self, timeout_seconds: float, poll_interval: float = 1.0) -> None:
        """Wait until the freed VRAM settles.

        After killing a model the OS/driver need a moment to reclaim memory;
        simply waiting for the process to exit is not enough.  We poll until
        the free-VRAM delta between consecutive samples is below 256 MiB, or
        the timeout expires (then we proceed anyway with whatever is free).
        No-op when NVML is unavailable (CPU/NPU: process exit releases RAM
        synchronously).
        """

        if not self.nvml_available:
            return
        deadline = time.monotonic() + timeout_seconds
        prev = self.vram_free_gib()
        while True:
            await asyncio.sleep(poll_interval)
            cur = self.vram_free_gib()
            if abs(cur - prev) < 0.25:
                return
            prev = cur
            if time.monotonic() >= deadline:
                logger.warning("VRAM did not fully settle after %.0fs; proceeding with %.1f GiB free",
                               timeout_seconds, cur)
                return
