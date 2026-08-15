"""NVML-based GPU VRAM inspection.

Only *reading* free/used/total VRAM — sglang-manager never touches processes it
does not own.  The manager computes start feasibility as::

    required_vram_gib (per model) + safety_margin_gib <= free VRAM

``free`` is always re-read *after* the old SGLang has exited and VRAM has been
observed to come back, so decisions reflect reality, not estimates.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol

from .errors import GpuUnavailable

logger = logging.getLogger(__name__)

GIB = 1024**3


class GpuMonitorProtocol(Protocol):
    device: int

    def total_gib(self) -> float: ...
    def free_gib(self) -> float: ...
    def used_gib(self) -> float: ...

    async def wait_until_free(self, threshold_gib: float, timeout_seconds: float) -> None:
        """Wait until free VRAM stays above ``threshold_gib`` (or timeout)."""
        ...

    async def wait_vram_released(self, timeout_seconds: float, poll_interval: float = 1.0) -> None:
        """Wait until free VRAM stops growing (release has settled)."""
        ...


class GpuMonitor:
    """Reads VRAM through NVML. Lazy-initializes so the manager can boot on
    machines where NVML is unavailable; start decisions then fail with
    :class:`GpuUnavailable` instead of crashing the manager."""

    def __init__(self, device: int = 0):
        self.device = device
        self._handle = None

    def _ensure(self):
        if self._handle is not None:
            return self._handle
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
            self._nvml = pynvml
            return self._handle
        except Exception as exc:  # noqa: BLE001 - any NVML failure is fatal here
            raise GpuUnavailable(
                f"NVML unavailable for device {self.device}: {exc}. "
                "Cannot inspect VRAM; refusing model start."
            ) from exc

    def total_gib(self) -> float:
        return self._mem().total / GIB

    def free_gib(self) -> float:
        return self._mem().free / GIB

    def used_gib(self) -> float:
        return self._mem().used / GIB

    def _mem(self):
        h = self._ensure()
        info = self._nvml.nvmlDeviceGetMemoryInfo(h)
        return info

    async def wait_until_free(self, threshold_gib: float, timeout_seconds: float, poll_interval: float = 1.0) -> None:
        """Wait until free VRAM >= threshold_gib; raise GpuUnavailable on timeout."""

        deadline = time.monotonic() + timeout_seconds
        while True:
            free = self.free_gib()
            if free >= threshold_gib:
                return
            if time.monotonic() >= deadline:
                raise GpuUnavailable(
                    f"free VRAM did not reach {threshold_gib:.1f} GiB within {timeout_seconds:.0f}s "
                    f"(last seen {free:.1f} GiB free)"
                )
            await asyncio.sleep(poll_interval)

    async def wait_vram_released(self, timeout_seconds: float, poll_interval: float = 1.0) -> None:
        """Wait until the freed VRAM settles.

        After killing a model the OS/driver need a moment to reclaim memory;
        simply waiting for the process to exit is not enough.  We poll until
        the free-VRAM delta between consecutive samples is below 256 MiB, or
        the timeout expires (then we proceed anyway with whatever is free).
        """

        deadline = time.monotonic() + timeout_seconds
        prev = self.free_gib()
        while True:
            await asyncio.sleep(poll_interval)
            cur = self.free_gib()
            if abs(cur - prev) < 0.25:
                return
            prev = cur
            if time.monotonic() >= deadline:
                logger.warning("VRAM did not fully settle after %.0fs; proceeding with %.1f GiB free",
                               timeout_seconds, cur)
                return
