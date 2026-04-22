"""Ollama providers - pass-through to Anthropic-compatible endpoint."""

from .cloud import OllamaCloudProvider
from .local import OllamaLocalProvider

__all__ = ["OllamaCloudProvider", "OllamaLocalProvider"]