"""Tests for ollama provider settings."""

import pytest
from config.settings import Settings


def test_ollama_api_key_setting(monkeypatch):
    """Settings recognizes OLLAMA_API_KEY env var."""
    monkeypatch.setenv("OLLAMA_API_KEY", "test-ollama-key")
    settings = Settings()
    assert settings.ollama_api_key == "test-ollama-key"


def test_ollama_base_url_default():
    """Settings provides default base URL for ollama cloud."""
    settings = Settings()
    assert settings.ollama_cloud_base_url == "https://ollama.com/v1"


def test_ollama_local_base_url_default():
    """Settings provides default base URL for local ollama."""
    settings = Settings()
    assert settings.ollama_local_base_url == "http://localhost:11434/v1"


def test_ollama_cloud_base_url_from_env(monkeypatch):
    """OLLAMA_CLOUD_BASE_URL env var is loaded into settings."""
    monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "https://custom.ollama.com/v1")
    settings = Settings()
    assert settings.ollama_cloud_base_url == "https://custom.ollama.com/v1"


def test_ollama_local_base_url_from_env(monkeypatch):
    """OLLAMA_LOCAL_BASE_URL env var is loaded into settings."""
    monkeypatch.setenv("OLLAMA_LOCAL_BASE_URL", "http://custom:1234/v1")
    settings = Settings()
    assert settings.ollama_local_base_url == "http://custom:1234/v1"


def test_model_validation_accepts_ollama_cloud_prefix(monkeypatch):
    """Model validation accepts ollama_cloud prefix."""
    monkeypatch.setenv("MODEL", "ollama_cloud/glm5")
    settings = Settings()
    assert settings.provider_type == "ollama_cloud"
    assert settings.model_name == "glm5"


def test_model_validation_accepts_ollama_local_prefix(monkeypatch):
    """Model validation accepts ollama_local prefix."""
    monkeypatch.setenv("MODEL", "ollama_local/qwen3-14b")
    settings = Settings()
    assert settings.provider_type == "ollama_local"
    assert settings.model_name == "qwen3-14b"