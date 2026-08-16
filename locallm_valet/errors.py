"""Typed error hierarchy for locallm-valet.

Every failure surfaced to the client is a :class:`ManagerError` subclass with a
stable ``error_type`` string, rendered in the OpenAI-style shape::

    {"error": {"type": ..., "message": ..., "code": ...}}
"""

from __future__ import annotations


class ManagerError(Exception):
    http_status: int = 503
    error_type: str = "manager_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def to_payload(self) -> dict:
        return {"type": self.error_type, "message": self.message, "code": self.error_type}


class ModelNotFound(ManagerError):
    """Requested model is not in the registry."""

    http_status = 404
    error_type = "model_not_found"


class InsufficientMemory(ManagerError):
    """Free VRAM (after any stop + release wait) is below required + margin."""

    http_status = 503
    error_type = "insufficient_memory"


class ModelSwitchBusy(ManagerError):
    """A switch/stop is refused because the model is busy or a transition is in flight."""

    http_status = 503
    error_type = "model_switch_busy"


class BackendUnavailable(ManagerError):
    """The backend is not reachable (stopping, dead, or connection refused)."""

    http_status = 503
    error_type = "backend_unavailable"


class BackendStartupFailed(ManagerError):
    """The backend process died before becoming healthy."""

    http_status = 503
    error_type = "backend_startup_failed"


class BackendStartupTimeout(ManagerError):
    """The backend did not become healthy within startup_timeout_seconds."""

    http_status = 503
    error_type = "backend_startup_timeout"


class MemoryUnavailable(ManagerError):
    """NVML could not be initialized; VRAM cannot be inspected."""

    http_status = 503
    error_type = "memory_unavailable"


class InvalidRequest(ManagerError):
    """Malformed request (missing/invalid ``model`` field, bad JSON...)."""

    http_status = 400
    error_type = "invalid_request"


class AuthenticationFailed(ManagerError):
    """Missing or wrong API key."""

    http_status = 401
    error_type = "authentication_error"
