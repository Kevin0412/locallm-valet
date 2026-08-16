"""locallm-valet: a local LLM valet.

It parks (unloads) your model when idle and brings it back on demand: one
fixed OpenAI-compatible endpoint that starts / stops / switches the single
managed inference backend (SGLang / vLLM / llama.cpp / OpenVINO ...) based on
the request ``model`` field, gates on VRAM + RAM before starting, refuses
switches while the model is busy, and auto-unloads after a configurable idle
period to release your GPU / CPU / NPU.
"""

__version__ = "0.5.0"
