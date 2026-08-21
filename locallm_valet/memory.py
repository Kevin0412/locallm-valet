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


# ---------------------------------------------------------------------------
# Multi-pool memory monitor (slots architecture)
# ---------------------------------------------------------------------------

class PoolMonitor:
    """Probes named resource pools (system_ram / gpu0_vram / npu0_hbm / …).

    Pool kinds:
    - "ram":    system RAM via psutil (shared by all ram-consuming slots)
    - "vram":   NVML VRAM at a device index (skip gating when no driver)
    - "static": no standard API (NPU/HBM/etc.) — configured capacity

    This replaces the single-device MemoryMonitor for the slots architecture:
    every slot's memory gate is expressed against pools, and shared pools
    (system_ram) are naturally accounted across all running slots.
    """

    def __init__(self, pools: dict):
        from .config import PoolConfig
        self.pools: dict[str, PoolConfig] = dict(pools)
        self._nvml = None
        self._nvml_failed = False
        self._handles: dict[int, object] = {}

    # ---------------------------------------------------------- probing

    def _nvml_ready(self) -> bool:
        if self._nvml is not None:
            return True
        if self._nvml_failed:
            return False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            return True
        except Exception:  # noqa: BLE001
            self._nvml_failed = True
            logger.warning("NVML unavailable; vram pools are not gated")
            return False

    def _handle(self, index: int):
        if index not in self._handles:
            if not self._nvml_ready():
                return None
            try:
                self._handles[index] = self._nvml.nvmlDeviceGetHandleByIndex(index)
            except Exception:  # noqa: BLE001
                logger.warning("NVML device %d not found; its vram pool is not gated", index)
                self._handles[index] = None
        return self._handles.get(index)

    def pool_total_gib(self, name: str) -> float:
        pool = self.pools.get(name)
        if pool is None:
            return 0.0
        if pool.kind == "ram":
            import psutil
            return psutil.virtual_memory().total / GIB
        if pool.kind == "vram":
            handle = self._handle(pool.device_index)
            if handle is None:
                return 0.0
            return self._nvml.nvmlDeviceGetMemoryInfo(handle).total / GIB
        return pool.total_gib  # static

    def pool_available_gib(self, name: str) -> float:
        pool = self.pools.get(name)
        if pool is None:
            return 0.0
        if pool.kind == "ram":
            import psutil
            return psutil.virtual_memory().available / GIB
        if pool.kind == "vram":
            handle = self._handle(pool.device_index)
            if handle is None:
                return 0.0
            return self._nvml.nvmlDeviceGetMemoryInfo(handle).free / GIB
        return pool.total_gib  # static: no runtime probe

    def pool_probeable(self, name: str) -> bool:
        """False when the pool cannot be read (e.g. vram without NVML) — the
        gate is then skipped rather than hard-failing, matching single-slot
        behaviour."""
        pool = self.pools.get(name)
        if pool is None:
            return False
        if pool.kind == "ram":
            return True
        if pool.kind == "vram":
            return self._handle(pool.device_index) is not None
        return pool.total_gib > 0  # static

    def status(self) -> dict:
        out = {}
        for name in self.pools:
            out[name] = {
                "total_gib": round(self.pool_total_gib(name), 2),
                "available_gib": round(self.pool_available_gib(name), 2),
                "probeable": self.pool_probeable(name),
            }
        return out
