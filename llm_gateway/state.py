"""The formal model lifecycle state machine."""

from __future__ import annotations

from enum import Enum


class State(Enum):
    STOPPED = "stopped"          # no backend instance under management
    STARTING = "starting"        # launching the backend, waiting for health
    RUNNING = "running"          # backend serving ``current_model``
    STOPPING = "stopping"        # tearing the backend down
    SWITCHING = "switching"      # stopping ``switch_from``, starting ``switch_to``

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value
