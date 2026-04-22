"""OpenAI-compatible generic provider.

Routes Anthropic-format requests through the translation layer
to any OpenAI-compatible endpoint. No provider-specific extras —
just clean OpenAI format conversion via build_base_request_body().
"""

from typing import Any

from providers.base import ProviderConfig
from providers.common.message_converter import build_base_request_body
from providers.openai_compat import OpenAICompatibleProvider

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatProvider(OpenAICompatibleProvider):
    """Generic OpenAI-compatible provider.

    For endpoints that accept /v1/chat/completions format.
    User provides OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_API_KEY.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="OPENAI_COMPAT",
            base_url=config.base_url or DEFAULT_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Build request body using standard Anthropic→OpenAI conversion.

        No provider-specific extras — just clean OpenAI format.
        """
        return build_base_request_body(request)