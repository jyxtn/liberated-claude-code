"""Tests for ollama cloud provider."""

import pytest
from providers.ollama import OllamaCloudProvider
from providers.base import ProviderConfig


# Mock data classes
class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "test-model"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.system = None
        self.tools = []
        self.tool_choice = None
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_ollama_cloud_provider_init():
    """Provider initializes with correct base URL and API key."""
    config = ProviderConfig(
        api_key="test-ollama-key",
        base_url="https://ollama.com/v1",
    )
    provider = OllamaCloudProvider(config)
    assert provider._base_url == "https://ollama.com/v1"
    assert provider._api_key == "test-ollama-key"


def test_ollama_cloud_provider_passes_through_anthropic_format():
    """Provider passes Anthropic format directly to ollama."""
    config = ProviderConfig(
        api_key="test-ollama-key",
        base_url="https://ollama.com/v1",
    )
    provider = OllamaCloudProvider(config)
    # The provider should NOT transform the request body
    # It only replaces the model name based on tier mapping
    request = MockRequest(
        model="claude-sonnet-4-5",
        messages=[MockMessage("user", "Hello")],
        max_tokens=100,
    )
    request_body = provider._build_request_body(request)
    # Request should be in Anthropic format, not transformed to OpenAI format
    assert request_body["messages"] == [{"role": "user", "content": "Hello"}]
    assert request_body["max_tokens"] == 100