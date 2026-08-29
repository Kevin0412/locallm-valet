"""SlotManager — aggregates one ModelManager per execution slot.

Slots are the unit of concurrency: CPU / NPU0 / NPU1 / GPU0 / GPU1 … each
owns an independent backend port, state machine and ModelManager, so models
on different slots run in parallel. Same-slot models stay mutually exclusive
(serialized switching) — that is exactly the single-slot valet behaviour,
scoped per slot.

Resource accounting happens through shared pools (PoolMonitor): system_ram is
one pool that every ram-consuming slot draws from, while each vram slot has
its own private pool. A start is refused if ANY pool the model needs is short,
which naturally accounts for CPU/NPU/iGPU sharing RAM and discrete GPUs having
their own VRAM (plus RAM staging when the machine loads through RAM).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Optional

from .config import Config, ModelSpec
from .errors import ModelNotFound, ManagerError
from .manager import ModelManager
from .memory import PoolMonitor, MemoryMonitor
from .runner import BackendRunner
from .state import State

logger = logging.getLogger(__name__)


class SlotManager:
    """Owns one ModelManager per slot; routes requests by model → slot."""

    def __init__(self, config: Config, pool_monitor: PoolMonitor | None = None):
        self.cfg = config
        self.pools = pool_monitor or PoolMonitor(config.pools)
        self.slots: dict[str, ModelManager] = {}

        for slot_name, slot_cfg in config.slots.items():
            backend_cfg = replace(config.backend, port=slot_cfg.port)
            runner = BackendRunner(backend_cfg, device=config.memory.device)
            # MemoryMonitor is still constructed for the legacy status fields,
            # but the actual gating goes through pool_monitor.
            memory = MemoryMonitor(config.memory.device)
            self.slots[slot_name] = ModelManager(
                config,
                memory,
                runner,
                pool_monitor=self.pools,
                slot_name=slot_name,
            )

    # ------------------------------------------------------------ routing

    def _manager_for(self, model_name: str) -> ModelManager:
        spec = self.cfg.get_model(model_name)
        if spec is None:
            raise ModelNotFound(f"model {model_name!r} is not in the registry")
        slot_name = spec.slot
        manager = self.slots.get(slot_name)
        if manager is None:
            raise ModelNotFound(
                f"model {model_name!r} is on slot '{slot_name}' which is not configured"
            )
        return manager

    async def ensure_loaded(self, model_name: str) -> ModelSpec:
        """Make the model's slot RUNNING with it. Different slots run in
        parallel; the same slot serializes like the single-slot valet."""
        return await self._manager_for(model_name).ensure_loaded(model_name)

    async def admit_request(self, model_name: str) -> ModelSpec:
        """Atomically gate + admit a request on the model's slot.

        See ModelManager.admit_request — the confirm-and-count happens in one
        critical section, closing the race where a request could be proxied
        to a backend that switched/stopped between ensure_loaded and the
        active-request bump.
        """
        return await self._manager_for(model_name).admit_request(model_name)

    def get_slot_manager(self, model_name: str) -> ModelManager:
        """Public: the ModelManager for the model's slot."""
        return self._manager_for(model_name)

    def base_url_for(self, model_name: str) -> str:
        """The backend base URL (no /v1 suffix — the proxy appends the path
        which already starts with /v1) of the slot this model runs on."""
        spec = self.cfg.get_model(model_name)
        if spec is None:
            raise ModelNotFound(f"model {model_name!r} is not in the registry")
        slot = self.cfg.slots.get(spec.slot)
        port = slot.port if slot else self.cfg.backend.port
        return f"http://{self.cfg.backend.host}:{port}"

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        for m in self.slots.values():
            m.start()
        logger.info("slot manager started: %d slots, %d models",
                    len(self.slots), len(self.cfg.models))

    async def shutdown(self) -> None:
        for m in self.slots.values():
            try:
                await m.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("slot %s shutdown failed", m.slot_name)

    async def stop(self, reason: str = "manual", force: bool = False) -> dict:
        """Stop all slots. Returns an aggregate view."""
        results = {}
        for name, m in self.slots.items():
            try:
                await m.stop(reason=reason, force=force)
                results[name] = {"state": m.state.value, "model": m.current_model}
            except ManagerError as exc:
                results[name] = {"error": exc.to_payload()}
        return results

    # -------------------------------------------------------------- status

    def status(self) -> dict:
        slot_states = {}
        for name, m in self.slots.items():
            st = m.status()
            slot_states[name] = {
                "state": st["state"],
                "model": st["model"],
                "active_requests": st["active_requests"],
                "idle_seconds": st["idle_seconds"],
                "max_context_tokens": st["max_context_tokens"],
            }
        return {
            "slots": slot_states,
            "pools": self.pools.status(),
            "uptime_seconds": round(
                max((m.status()["uptime_seconds"] for m in self.slots.values()), default=0), 1
            ),
        }

    def models_status(self) -> list[dict]:
        """Per-model status including slot assignment and load state.

        Every entry also carries the *read-only* start feasibility of its
        slot manager (``state``: running / startable / switchable / blocked,
        ``startable`` boolean, ``start_reason`` human explanation) so the
        dashboard can answer "can this model be started, or is it running"
        without spawning anything.
        """
        out = []
        for name, spec in self.cfg.models.items():
            m = self.slots.get(spec.slot)
            loaded = m is not None and m.state is State.RUNNING and m.current_model == name
            if m is not None:
                st = m.start_status(spec)
                state, startable, reason = st["state"], st["state"] != "blocked", st["reason"]
            else:
                state, startable, reason = "blocked", False, f"slot '{spec.slot}' is not configured"
            out.append(
                {
                    "name": name,
                    "slot": spec.slot,
                    "path": spec.path,
                    "required_vram_gib": spec.required_vram_gib,
                    "required_ram_gib": spec.required_ram_gib,
                    "required_pools": spec.required_pools,
                    "extra_args": spec.backend.extra_args,
                    "loaded": loaded,
                    "state": state,
                    "startable": startable,
                    "start_reason": reason,
                    "max_context_tokens": (
                        m.max_context_tokens if loaded else None
                    ),
                }
            )
        return out

    # ------------------------------------------------------------- helpers

    @property
    def current_model(self) -> Optional[str]:
        loaded = [m.current_model for m in self.slots.values() if m.current_model]
        return loaded[0] if len(loaded) == 1 else (loaded if loaded else None)

    @property
    def state(self):
        return State.RUNNING if any(m.state is State.RUNNING for m in self.slots.values()) else State.STOPPED
