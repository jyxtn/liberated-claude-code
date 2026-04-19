"""Ollama Local provider - pass-through to local ollama instance.

Connects to ollama running locally (default: localhost:11434).
Uses Anthropic-compatible /v1/messages endpoint.
No authentication required.
"""

from providers.ollama.cloud import OllamaCloudProvider
from providers.base import ProviderConfig


OLLAMA_LOCAL_BASE_URL = "http://localhost:11434/v1"


class OllamaLocalProvider(OllamaCloudProvider):
    """Pass-through provider for local Ollama instance.

    Inherits from OllamaCloudProvider - same Anthropic pass-through logic.
    Only difference: default base URL and no auth requirement.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        model_map: dict[str, str] | None = None,
    ):
        # Override base_url if not provided
        if not config.base_url:
            config = config.model_copy(update={"base_url": OLLAMA_LOCAL_BASE_URL})
        # Local ollama doesn't require auth, use dummy key if not provided
        if config.api_key is None:
            config = config.model_copy(update={"api_key": "ollama"})
        super().__init__(config, model_map=model_map)