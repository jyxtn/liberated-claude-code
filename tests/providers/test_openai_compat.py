"""Tests for OpenAI-compatible generic provider."""

import pytest
from unittest.mock import MagicMock

from providers.openai_compat_generic import OpenAICompatProvider
from providers.openai_compat_generic.client import DEFAULT_BASE_URL
from providers.base import ProviderConfig


class TestOpenAICompatProvider:
    """Test OpenAI-compatible provider initialization and configuration."""

    def test_openai_compat_provider_uses_config_base_url(self):
        """OpenAI-compat provider should use the base URL from config."""
        config = ProviderConfig(
            api_key="test-key",
            base_url="https://employer-litellm.internal/v1",
            rate_limit=10,
            rate_window=60,
            max_concurrency=5,
            http_read_timeout=300.0,
            http_write_timeout=10.0,
            http_connect_timeout=2.0,
        )
        provider = OpenAICompatProvider(config)

        assert provider._base_url == "https://employer-litellm.internal/v1"

    def test_openai_compat_provider_default_base_url(self):
        """OpenAI-compat provider should default to api.openai.com."""
        config = ProviderConfig(
            api_key="test-key",
            base_url=None,
            rate_limit=10,
            rate_window=60,
            max_concurrency=5,
            http_read_timeout=300.0,
            http_write_timeout=10.0,
            http_connect_timeout=2.0,
        )
        provider = OpenAICompatProvider(config)

        assert provider._base_url == DEFAULT_BASE_URL
        assert provider._base_url == "https://api.openai.com/v1"

    def test_openai_compat_provider_strips_trailing_slash(self):
        """OpenAI-compat provider should strip trailing slash from base URL."""
        config = ProviderConfig(
            api_key="test-key",
            base_url="https://employer-litellm.internal/v1/",
            rate_limit=10,
            rate_window=60,
            max_concurrency=5,
            http_read_timeout=300.0,
            http_write_timeout=10.0,
            http_connect_timeout=2.0,
        )
        provider = OpenAICompatProvider(config)

        assert provider._base_url == "https://employer-litellm.internal/v1"

    def test_openai_compat_provider_api_key(self):
        """OpenAI-compat provider should store API key."""
        config = ProviderConfig(
            api_key="sk-test-key-123",
            base_url="https://employer-litellm.internal/v1",
            rate_limit=10,
            rate_window=60,
            max_concurrency=5,
            http_read_timeout=300.0,
            http_write_timeout=10.0,
            http_connect_timeout=2.0,
        )
        provider = OpenAICompatProvider(config)

        assert provider._api_key == "sk-test-key-123"

    def test_openai_compat_provider_name(self):
        """OpenAI-compat provider should have correct provider name."""
        config = ProviderConfig(
            api_key="test-key",
            base_url="https://employer-litellm.internal/v1",
        )
        provider = OpenAICompatProvider(config)

        assert provider._provider_name == "OPENAI_COMPAT"

    def test_openai_compat_build_request_body(self):
        """OpenAI-compat provider should build request body via base converter."""
        config = ProviderConfig(
            api_key="test-key",
            base_url="https://employer-litellm.internal/v1",
        )
        provider = OpenAICompatProvider(config)

        mock_request = MagicMock()
        mock_request.model = "claude-sonnet-4-5"
        mock_request.messages = [MagicMock()]
        mock_request.max_tokens = 100

        body = provider._build_request_body(mock_request)

        assert "messages" in body
        assert "max_tokens" in body
        assert body["max_tokens"] == 100