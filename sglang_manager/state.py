"""The formal model lifecycle state machine."""

from __future__ import annotations

from enum import Enum


class State(Enum):
    STOPPED = "stopped"          # no SGLang instance under management
    STARTING = "starting"        # launching SGLang, waiting for health
    RUNNING = "running"          # SGLang serving ``current_model``
    STOPPING = "stopping"        # tearing SGLang down
    SWITCHING = "switching"      # stopping ``switch_from``, starting ``switch_to``

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value
