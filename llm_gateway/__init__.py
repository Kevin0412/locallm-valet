"""llm-gateway: a backend-agnostic LLM lifecycle gateway.

It exposes one fixed OpenAI-compatible endpoint, starts / stops / switches the
single managed inference backend (SGLang / vLLM / llama.cpp / OpenVINO...) based on
the request ``model`` field, refuses to
start when VRAM is insufficient, and unloads the model after a configurable
idle period.
"""

__version__ = "0.4.0"
