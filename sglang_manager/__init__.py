"""sglang-manager: a GPU-aware SGLang lifecycle manager for a single GPU.

It exposes one fixed OpenAI-compatible endpoint, starts / stops / switches the
single managed SGLang instance based on the request ``model`` field, refuses to
start when VRAM is insufficient, and unloads the model after a configurable
idle period.
"""

__version__ = "0.1.0"
