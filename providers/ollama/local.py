"""Ollama Local provider - placeholder for future implementation."""

from providers.base import BaseProvider, ProviderConfig


class OllamaLocalProvider(BaseProvider):
    """Placeholder for local ollama provider - not yet implemented."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        raise NotImplementedError("OllamaLocalProvider is not yet implemented")

    async def cleanup(self) -> None:
        """Release resources."""
        pass

    async def stream_response(self, request, input_tokens: int = 0, *, request_id: str | None = None):
        """Stream response - not implemented."""
        raise NotImplementedError("OllamaLocalProvider is not yet implemented")