"""Ollama Cloud provider - pass-through to Anthropic-compatible endpoint.

Ollama v0.14.0+ supports Anthropic's native /v1/messages format.
This provider passes requests through with minimal transformation:
- Model name substitution based on tier mapping
- Direct streaming of Anthropic SSE format

No OpenAI format translation needed.
"""

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any
from uuid import uuid4

import httpx
from loguru import logger

from providers.base import BaseProvider, ProviderConfig
from providers.common import SSEBuilder, map_error, get_user_facing_error_message


OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"


class OllamaCloudProvider(BaseProvider):
    """Pass-through provider for Ollama Cloud's Anthropic-compatible endpoint."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        model_map: dict[str, str] | None = None,
    ):
        super().__init__(config)
        self._base_url = (config.base_url or OLLAMA_CLOUD_BASE_URL).rstrip("/")
        self._api_key = config.api_key
        self._model_map = model_map or {}
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
            ),
        )

    async def cleanup(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()

    def _resolve_model(self, requested_model: str) -> str:
        """Resolve model name using tier mapping.

        Maps claude-sonnet-4-5 -> configured sonnet model, etc.
        Falls back to requested model if no mapping found.
        """
        name_lower = requested_model.lower()
        if "opus" in name_lower and "opus" in self._model_map:
            return self._model_map["opus"]
        if "haiku" in name_lower and "haiku" in self._model_map:
            return self._model_map["haiku"]
        if "sonnet" in name_lower and "sonnet" in self._model_map:
            return self._model_map["sonnet"]
        return self._model_map.get("model", requested_model)

    def _build_request_body(self, request: Any) -> dict:
        """Build request body for ollama.

        ollama's Anthropic-compatible endpoint accepts Anthropic format directly.
        We only substitute the model name.
        """
        body = {
            "model": self._resolve_model(request.model),
            "messages": [],
            "max_tokens": getattr(request, "max_tokens", 4096),
            "stream": True,
        }

        # Convert messages
        for msg in request.messages:
            msg_dict = {"role": msg.role}
            if isinstance(msg.content, str):
                msg_dict["content"] = msg.content
            else:
                # Content blocks
                msg_dict["content"] = [
                    {"type": block.type, **{k: v for k, v in block.model_dump().items() if k != "type"}}
                    for block in msg.content
                ]
            body["messages"].append(msg_dict)

        # Optional fields
        if request.system:
            body["system"] = request.system
        if request.tools:
            body["tools"] = [t.model_dump() for t in request.tools]
        if request.tool_choice:
            body["tool_choice"] = request.tool_choice

        return body

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream response from ollama in Anthropic SSE format."""
        message_id = f"msg_{uuid4().hex[:24]}"
        sse = SSEBuilder(message_id, request.model, input_tokens)

        body = self._build_request_body(request)
        req_tag = f" request_id={request_id}" if request_id else ""
        logger.info(
            f"OLLAMA_CLOUD_STREAM:{req_tag} model={body['model']} msgs={len(body['messages'])}"
        )

        yield sse.message_start()

        try:
            async with self._client.stream(
                "POST",
                "/v1/messages",
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or line == "\n":
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        # Pass through SSE events from ollama
                        yield data + "\n"

        except Exception as e:
            logger.error(f"OLLAMA_CLOUD_ERROR:{req_tag} {type(e).__name__}: {e}")
            mapped_e = map_error(e)
            error_message = get_user_facing_error_message(
                mapped_e, read_timeout_s=self._config.http_read_timeout
            )
            for event in sse.emit_error(error_message):
                yield event

        yield sse.message_stop()