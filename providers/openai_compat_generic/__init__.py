"""OpenAI-compatible generic provider — bring-your-own-endpoint.

For any endpoint that accepts OpenAI /v1/chat/completions format:
vLLM, LiteLLM proxy, TGI, SGLang, Together, Fireworks, Groq, etc.
Uses the existing Anthropic→OpenAI translation layer.
"""

from .client import OpenAICompatProvider

__all__ = ["OpenAICompatProvider"]