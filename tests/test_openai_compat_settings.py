"""Tests for openai-compat provider settings."""

import pytest
from config.settings import Settings


def test_openai_compat_api_key_setting(monkeypatch):
    """Settings recognizes OPENAI_COMPAT_API_KEY env var."""
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-test-key")
    settings = Settings()
    assert settings.openai_compat_api_key == "sk-test-key"


def test_openai_compat_base_url_default():
    """Settings provides default base URL for openai-compat."""
    settings = Settings()
    assert settings.openai_compat_base_url == "https://api.openai.com/v1"


def test_openai_compat_base_url_from_env(monkeypatch):
    """OPENAI_COMPAT_BASE_URL env var is loaded into settings."""
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://employer-litellm.internal/v1")
    settings = Settings()
    assert settings.openai_compat_base_url == "https://employer-litellm.internal/v1"


def test_model_validation_accepts_openai_compatible_prefix(monkeypatch):
    """Model validation accepts openai_compatible prefix."""
    monkeypatch.setenv("MODEL", "openai_compatible/employer-model-large")
    settings = Settings()
    assert settings.provider_type == "openai_compatible"
    assert settings.model_name == "employer-model-large"