"""Providers package - implement your own provider by extending BaseProvider."""

from .base import BaseProvider, ProviderConfig
from .exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    OverloadedError,
    ProviderError,
    RateLimitError,
)
from .llamacpp import LlamaCppProvider
from .lmstudio import LMStudioProvider
from .nvidia_nim import NvidiaNimProvider
from .ollama import OllamaCloudProvider, OllamaLocalProvider
from .open_router import OpenRouterProvider
from .openai_compat_generic import OpenAICompatProvider

__all__ = [
    "APIError",
    "AuthenticationError",
    "BaseProvider",
    "InvalidRequestError",
    "LMStudioProvider",
    "LlamaCppProvider",
    "NvidiaNimProvider",
    "OllamaCloudProvider",
    "OllamaLocalProvider",
    "OpenAICompatProvider",
    "OpenRouterProvider",
    "OverloadedError",
    "ProviderConfig",
    "ProviderError",
    "RateLimitError",
]
