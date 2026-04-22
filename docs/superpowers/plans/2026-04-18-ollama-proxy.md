# Ollama Cloud Proxy + Ephemeral Switching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend liberated-claude-code with ollama cloud/local providers and add claude-with CLI for ephemeral switching with multi-tier model routing.

**Architecture:** Ollama providers use Anthropic-compatible `/v1/messages` endpoint (pass-through, no format translation). CLI reads config from central + project-local TOML files, sets env vars, and launches commands. Proxy maps tier-based model names to configured models.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, tomli (TOML parsing), click (CLI)

---

## File Structure

**New files in `liberated-claude-code/`:**
```
providers/
├── ollama/__init__.py           # NEW: exports OllamaCloudProvider, OllamaLocalProvider
├── ollama/cloud.py              # NEW: ollama cloud pass-through provider
└── ollama/local.py              # NEW: local ollama pass-through provider

claude_with/                     # NEW: CLI package (underscore for Python import)
├── __init__.py
├── __main__.py                  # Entry point: python -m claude_with
├── cli.py                       # Click CLI commands
├── config.py                    # TOML config parsing
├── init.py                      # Project init helper
├── providers.py                 # Provider configuration
├── keys.py                      # API key management (env + keychain)
└── pyproject.toml               # Package definition
```

**Modified files in `liberated-claude-code/`:**
```
config/settings.py               # MODIFY: add ollama_api_key, ollama_base_url settings
providers/__init__.py            # MODIFY: export ollama providers
api/dependencies.py              # MODIFY: register ollama providers
pyproject.toml                   # MODIFY: add tomli dependency
```

---

## Task 1: Add Ollama Cloud Provider

**Files:**
- Create: `liberated-claude-code/providers/ollama/__init__.py`
- Create: `liberated-claude-code/providers/ollama/cloud.py`
- Modify: `liberated-claude-code/providers/__init__.py`

- [ ] **Step 1: Write failing test for ollama cloud provider**

Create test file at `liberated-claude-code/tests/providers/test_ollama_cloud.py`:

```python
"""Tests for ollama cloud provider."""

import pytest
from providers.ollama import OllamaCloudProvider
from providers.base import ProviderConfig


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
    request_body = provider._build_request_body(
        type("Request", (), {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        })
    )
    # Request should be in Anthropic format, not transformed to OpenAI format
    assert request_body["messages"] == [{"role": "user", "content": "Hello"}]
    assert request_body["max_tokens"] == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/providers/test_ollama_cloud.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'providers.ollama'"

- [ ] **Step 3: Create ollama package directory**

Run: `mkdir -p /Users/justincantrall/Projects/liberated-claude-code/providers/ollama`

Create `liberated-claude-code/providers/ollama/__init__.py`:

```python
"""Ollama providers - pass-through to Anthropic-compatible endpoint."""

from .cloud import OllamaCloudProvider
from .local import OllamaLocalProvider

__all__ = ["OllamaCloudProvider", "OllamaLocalProvider"]
```

- [ ] **Step 4: Create ollama cloud provider**

Create `liberated-claude-code/providers/ollama/cloud.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/providers/test_ollama_cloud.py -v`
Expected: PASS

- [ ] **Step 6: Update providers __init__.py to export ollama**

Modify `liberated-claude-code/providers/__init__.py`:

```python
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
    "OpenRouterProvider",
    "OverloadedError",
    "ProviderConfig",
    "ProviderError",
    "RateLimitError",
]
```

- [ ] **Step 7: Commit ollama cloud provider**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add providers/ollama/ tests/providers/test_ollama_cloud.py providers/__init__.py
git commit -m "feat: add ollama cloud provider with Anthropic-compatible pass-through"
```

---

## Task 2: Add Ollama Local Provider

**Files:**
- Create: `liberated-claude-code/providers/ollama/local.py`
- Modify: `liberated-claude-code/tests/providers/test_ollama_cloud.py` (add local tests)

- [ ] **Step 1: Write failing test for ollama local provider**

Add to `liberated-claude-code/tests/providers/test_ollama_cloud.py`:

```python
def test_ollama_local_provider_init():
    """Local provider initializes with localhost URL."""
    config = ProviderConfig(
        api_key="not-needed",
        base_url="http://localhost:11434/v1",
    )
    provider = OllamaLocalProvider(config)
    assert provider._base_url == "http://localhost:11434/v1"


def test_ollama_local_provider_no_auth():
    """Local provider does not require API key."""
    config = ProviderConfig(
        api_key="",  # Empty key is fine for local
        base_url="http://localhost:11434/v1",
    )
    provider = OllamaLocalProvider(config)
    # Should not raise any errors
    assert provider._api_key == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/providers/test_ollama_cloud.py::test_ollama_local_provider_init -v`
Expected: FAIL with "ImportError: cannot import name 'OllamaLocalProvider'"

- [ ] **Step 3: Create ollama local provider**

Create `liberated-claude-code/providers/ollama/local.py`:

```python
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
        # Local ollama doesn't require auth, use dummy key
        if not config.api_key:
            config = config.model_copy(update={"api_key": "ollama"})
        super().__init__(config, model_map=model_map)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/providers/test_ollama_cloud.py -v`
Expected: PASS

- [ ] **Step 5: Commit ollama local provider**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add providers/ollama/local.py tests/providers/test_ollama_cloud.py
git commit -m "feat: add ollama local provider for localhost ollama instances"
```

---

## Task 3: Register Ollama Providers in Settings

**Files:**
- Modify: `liberated-claude-code/config/settings.py`
- Modify: `liberated-claude-code/api/dependencies.py`

- [ ] **Step 1: Write failing test for ollama settings**

Create `liberated-claude-code/tests/test_ollama_settings.py`:

```python
"""Tests for ollama provider settings."""

import pytest
from config.settings import Settings


def test_ollama_api_key_setting():
    """Settings recognizes OLLAMA_API_KEY env var."""
    settings = Settings(ollama_api_key="test-ollama-key")
    assert settings.ollama_api_key == "test-ollama-key"


def test_ollama_base_url_default():
    """Settings provides default base URL for ollama cloud."""
    settings = Settings()
    assert settings.ollama_cloud_base_url == "https://ollama.com/v1"


def test_ollama_local_base_url_default():
    """Settings provides default base URL for local ollama."""
    settings = Settings()
    assert settings.ollama_local_base_url == "http://localhost:11434/v1"


def test_model_validation_accepts_ollama_prefix():
    """Model validation accepts ollama_cloud and ollama_local prefixes."""
    settings = Settings(model="ollama_cloud/glm5")
    assert settings.provider_type == "ollama_cloud"
    assert settings.model_name == "glm5"

    settings = Settings(model="ollama_local/qwen3-14b")
    assert settings.provider_type == "ollama_local"
    assert settings.model_name == "qwen3-14b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/test_ollama_settings.py -v`
Expected: FAIL with "pydantic_core._pydantic_core.ValidationError" or "ValidationError"

- [ ] **Step 3: Add ollama settings to config/settings.py**

Modify `liberated-claude-code/config/settings.py`, add after line 43 (after modal settings):

```python
    # ==================== Ollama Cloud Config ====================
    ollama_api_key: str = Field(default="", validation_alias="OLLAMA_API_KEY")
    ollama_cloud_base_url: str = Field(
        default="https://ollama.com/v1",
        validation_alias="OLLAMA_CLOUD_BASE_URL",
    )

    # ==================== Ollama Local Config ====================
    ollama_local_base_url: str = Field(
        default="http://localhost:11434/v1",
        validation_alias="OLLAMA_LOCAL_BASE_URL",
    )
```

Then modify the `validate_model_format` method to include ollama providers (around line 166):

```python
    @field_validator("model", "model_opus", "model_sonnet", "model_haiku")
    @classmethod
    def validate_model_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        valid_providers = (
            "nvidia_nim", "open_router", "lmstudio", "llamacpp", "modal",
            "ollama_cloud", "ollama_local",
        )
        if "/" not in v:
            raise ValueError(
                f"Model must be prefixed with provider type. "
                f"Valid providers: {', '.join(valid_providers)}. "
                f"Format: provider_type/model/name"
            )
        provider = v.split("/", 1)[0]
        if provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: '{provider}'. "
                f"Supported: 'nvidia_nim', 'open_router', 'lmstudio', 'llamacpp', 'modal', 'ollama_cloud', 'ollama_local'"
            )
        return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/test_ollama_settings.py -v`
Expected: PASS

- [ ] **Step 5: Add ollama providers to dependencies.py**

Modify `liberated-claude-code/api/dependencies.py`, add import at top:

```python
from providers.ollama import OllamaCloudProvider, OllamaLocalProvider
```

Then add provider creation functions in `_create_provider_for_type` (after modal, around line 100):

```python
    if provider_type == "ollama_cloud":
        if not settings.ollama_api_key or not settings.ollama_api_key.strip():
            raise AuthenticationError(
                "OLLAMA_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://ollama.com/settings/keys"
            )
        config = ProviderConfig(
            api_key=settings.ollama_api_key,
            base_url=settings.ollama_cloud_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return OllamaCloudProvider(config)
    if provider_type == "ollama_local":
        config = ProviderConfig(
            api_key="ollama",  # Local doesn't need auth
            base_url=settings.ollama_local_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return OllamaLocalProvider(config)
```

Update the error message in the else branch (around line 110):

```python
    logger.error(
        "Unknown provider_type: '{}'. Supported: 'nvidia_nim', 'open_router', 'lmstudio', 'llamacpp', 'modal', 'ollama_cloud', 'ollama_local'",
        provider_type,
    )
    raise ValueError(
        f"Unknown provider_type: '{provider_type}'. "
        f"Supported: 'nvidia_nim', 'open_router', 'lmstudio', 'llamacpp', 'modal', 'ollama_cloud', 'ollama_local'"
    )
```

- [ ] **Step 6: Run all provider tests to verify no regression**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 7: Commit settings and dependencies changes**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add config/settings.py api/dependencies.py tests/test_ollama_settings.py
git commit -m "feat: register ollama_cloud and ollama_local providers in settings"
```

---

## Task 4: Create claude-with CLI Package

**Files:**
- Create: `liberated-claude-code/claude_with/__init__.py`
- Create: `liberated-claude-code/claude_with/__main__.py`
- Create: `liberated-claude-code/claude_with/cli.py`
- Create: `liberated-claude-code/claude_with/config.py`
- Create: `liberated-claude-code/claude_with/providers.py`
- Create: `liberated-claude-code/claude_with/keys.py`
- Create: `liberated-claude-code/claude_with/pyproject.toml`

- [ ] **Step 1: Create claude_with package structure**

Run:
```bash
mkdir -p /Users/justincantrall/Projects/liberated-claude-code/claude_with
```

Create `liberated-claude-code/claude_with/__init__.py`:

```python
"""claude-with CLI - ephemeral switching for Claude Code."""

__version__ = "0.1.0"
```

Create `liberated-claude-code/claude_with/__main__.py`:

```python
"""Entry point for python -m claude_with."""

from .cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create providers.py with provider definitions**

Create `liberated-claude-code/claude_with/providers.py`:

```python
"""Provider configuration for claude-with."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Self


class Provider(StrEnum):
    """Supported providers."""
    OLLAMA_CLOUD = "ollama"
    OLLAMA_LOCAL = "ollama-local"
    NVIDIA_NIM = "nvidia"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    name: str
    env_var: str | None
    base_url: str | None
    requires_proxy: bool

    @classmethod
    def get(cls, provider: Provider) -> Self:
        """Get configuration for a provider."""
        configs = {
            Provider.OLLAMA_CLOUD: cls(
                name="ollama_cloud",
                env_var="OLLAMA_API_KEY",
                base_url="https://ollama.com/v1",
                requires_proxy=True,
            ),
            Provider.OLLAMA_LOCAL: cls(
                name="ollama_local",
                env_var=None,
                base_url="http://localhost:11434/v1",
                requires_proxy=True,
            ),
            Provider.NVIDIA_NIM: cls(
                name="nvidia_nim",
                env_var="NVIDIA_NIM_API_KEY",
                base_url="https://integrate.api.nvidia.com/v1",
                requires_proxy=True,
            ),
            Provider.OPENROUTER: cls(
                name="open_router",
                env_var="OPENROUTER_API_KEY",
                base_url="https://openrouter.ai/api/v1",
                requires_proxy=True,
            ),
            Provider.ANTHROPIC: cls(
                name="anthropic",
                env_var="ANTHROPIC_API_KEY",
                base_url=None,  # Native Anthropic, no proxy
                requires_proxy=False,
            ),
        }
        return configs[provider]


# Provider prefix mapping for mixed profiles
PROVIDER_PREFIXES = {
    "ollama_cloud": "ollama_cloud",
    "ollama_local": "ollama_local",
    "nvidia_nim": "nvidia_nim",
    "open_router": "open_router",
    "lmstudio": "lmstudio",
    "llamacpp": "llamacpp",
}
```

- [ ] **Step 3: Create config.py for TOML parsing**

Create `liberated-claude-code/claude_with/config.py`:

```python
"""Configuration file parsing for claude-with."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli


@dataclass
class ModelTier:
    """Model configuration for a single tier."""
    large: str | None = None
    medium: str | None = None
    small: str | None = None

    # Alias support
    opus: str | None = None  # alias for large
    sonnet: str | None = None  # alias for medium
    haiku: str | None = None  # alias for small

    def get_large(self) -> str | None:
        return self.large or self.opus

    def get_medium(self) -> str | None:
        return self.medium or self.sonnet

    def get_small(self) -> str | None:
        return self.small or self.haiku


@dataclass
class Profile:
    """A named configuration profile."""
    name: str
    provider: str
    models: ModelTier = field(default_factory=ModelTier)


@dataclass
class Config:
    """Complete configuration from all sources."""
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    defaults: dict[str, ModelTier] = field(default_factory=dict)
    profiles: dict[str, Profile] = field(default_factory=dict)
    proxy_port: int = 8082
    proxy_host: str = "127.0.0.1"

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from all sources.

        Priority: project-local > central config > defaults
        """
        config = cls()
        config._load_central()
        config._load_project_local()
        return config

    def _load_central(self) -> None:
        """Load central configuration from ~/.config/claude-with/config.toml."""
        central_path = Path.home() / ".config" / "claude-with" / "config.toml"
        if central_path.exists():
            self._merge_toml(central_path)

    def _load_project_local(self) -> None:
        """Load project-local configuration from .claude-with.toml."""
        local_path = Path.cwd() / ".claude-with.toml"
        if local_path.exists():
            self._merge_toml(local_path)

    def _merge_toml(self, path: Path) -> None:
        """Merge TOML file into configuration."""
        with open(path, "rb") as f:
            data = tomli.load(f)

        # Parse providers
        for name, attrs in data.get("providers", {}).items():
            self.providers[name] = attrs

        # Parse defaults
        for name, attrs in data.get("defaults", {}).items():
            self.defaults[name] = self._parse_model_tier(attrs)

        # Parse profiles
        for name, attrs in data.get("profiles", {}).items():
            provider = attrs.get("provider", "ollama_cloud")
            models = self._parse_model_tier(attrs)
            self.profiles[name] = Profile(
                name=name,
                provider=provider,
                models=models,
            )

        # Parse proxy settings
        if "proxy" in data:
            self.proxy_port = data["proxy"].get("port", self.proxy_port)
            self.proxy_host = data["proxy"].get("host", self.proxy_host)

    def _parse_model_tier(self, data: dict[str, Any]) -> ModelTier:
        """Parse model tier configuration."""
        return ModelTier(
            large=data.get("large"),
            medium=data.get("medium"),
            small=data.get("small"),
            opus=data.get("opus"),
            sonnet=data.get("sonnet"),
            haiku=data.get("haiku"),
        )

    def get_profile(self, name: str | None = None) -> Profile | None:
        """Get a profile by name, or the default profile.

        Priority:
        1. Named profile (if name provided)
        2. Project-local profile reference
        3. Default profile for provider
        """
        if name and name in self.profiles:
            return self.profiles[name]

        # Check for project-local profile reference
        local_path = Path.cwd() / ".claude-with.toml"
        if local_path.exists():
            with open(local_path, "rb") as f:
                data = tomli.load(f)
                if "profile" in data:
                    profile_name = data["profile"]
                    if profile_name in self.profiles:
                        return self.profiles[profile_name]

        return None

    def get_default_tier(self, provider: str) -> ModelTier:
        """Get default model tier for a provider."""
        return self.defaults.get(provider, ModelTier())
```

- [ ] **Step 4: Create keys.py for API key management**

Create `liberated-claude-code/claude_with/keys.py`:

```python
"""API key management for claude-with."""

import os
from pathlib import Path


def get_api_key(env_var: str) -> str | None:
    """Get API key from environment or keychain.

    Priority:
    1. Environment variable
    2. macOS Keychain (if available)
    3. ~/.config/claude-with/.env file

    Returns:
        API key string or None if not found.
    """
    # 1. Check environment variable
    key = os.environ.get(env_var)
    if key:
        return key

    # 2. Check macOS Keychain
    key = _get_from_keychain(env_var)
    if key:
        return key

    # 3. Check .env file
    key = _get_from_env_file(env_var)
    if key:
        return key

    return None


def _get_from_keychain(env_var: str) -> str | None:
    """Get API key from macOS Keychain."""
    import subprocess

    # Map env var to keychain service name
    service_name = f"claude-with-{env_var.lower().replace('_', '-')}"

    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", env_var, "-s", service_name, "-w"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        # security command not available (not macOS)
        pass

    return None


def _get_from_env_file(env_var: str) -> str | None:
    """Get API key from ~/.config/claude-with/.env file."""
    env_path = Path.home() / ".config" / "claude-with" / ".env"

    if not env_path.exists():
        return None

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{env_var}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    return None


def require_api_key(env_var: str, provider_name: str) -> str:
    """Require an API key, raising an error if not found.

    Args:
        env_var: Environment variable name (e.g., "OLLAMA_API_KEY")
        provider_name: Human-readable provider name for error message

    Returns:
        API key string

    Raises:
        SystemExit: If API key not found
    """
    key = get_api_key(env_var)
    if not key:
        print(f"✗ {env_var} not set")
        print(f"  Set it in your shell: export {env_var}=\"your-key\"")
        print(f"  Or add to ~/.config/claude-with/.env: {env_var}=your-key")
        raise SystemExit(1)
    return key
```

- [ ] **Step 5: Create cli.py with Click commands**

Create `liberated-claude-code/claude_with/cli.py`:

```python
"""CLI for claude-with - ephemeral switching for Claude Code."""

import os
import subprocess
import sys
from pathlib import Path

import click

from .config import Config, ModelTier
from .keys import get_api_key, require_api_key
from .providers import Provider, ProviderConfig, PROVIDER_PREFIXES


@click.group()
@click.version_option()
def main():
    """claude-with - Ephemeral switching for Claude Code.

    Launch Claude Code with different providers and model configurations.
    Each invocation sets environment variables for that session only.
    """
    pass


@main.command()
@click.argument("provider", type=click.Choice(["ollama", "ollama-local", "nvidia", "openrouter", "anthropic"]))
@click.option("--large", "--opus", "large", help="Large/Opus tier model")
@click.option("--medium", "--sonnet", "medium", help="Medium/Sonnet tier model")
@click.option("--small", "--haiku", "small", help="Small/Haiku tier model")
@click.option("--profile", "-p", "profile_name", help="Named profile from config")
@click.option("--command", "-c", "command", help="Command to run (default: claude)")
@click.argument("args", nargs=-1)
def run(provider: str, large: str | None, medium: str | None, small: str | None,
        profile_name: str | None, command: str | None, args: tuple[str, ...]):
    """Launch Claude Code with the specified provider and models.

    Examples:
        claude-with ollama --large glm5 --medium kimi-k2.5 --small step-3.5-flash
        claude-with ollama --profile oculus
        claude-with ollama --profile oculus -- code .
    """
    config = Config.load()

    # Get provider config
    provider_enum = {
        "ollama": Provider.OLLAMA_CLOUD,
        "ollama-local": Provider.OLLAMA_LOCAL,
        "nvidia": Provider.NVIDIA_NIM,
        "openrouter": Provider.OPENROUTER,
        "anthropic": Provider.ANTHROPIC,
    }[provider]

    provider_config = ProviderConfig.get(provider_enum)

    # Build environment
    env = os.environ.copy()

    # Set proxy URL if needed
    if provider_config.requires_proxy:
        proxy_url = f"http://{config.proxy_host}:{config.proxy_port}"
        env["ANTHROPIC_BASE_URL"] = proxy_url
        env["ANTHROPIC_API_KEY"] = "freecc"  # Proxy uses its own auth
    else:
        # Native Anthropic - remove proxy settings
        env.pop("ANTHROPIC_BASE_URL", None)
        # Use Anthropic API key
        api_key = get_api_key("ANTHROPIC_API_KEY")
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key

    # Set model tier env vars
    models = _resolve_models(config, provider_config, profile_name, large, medium, small)

    if models.get_large():
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = models.get_large()
    if models.get_medium():
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = models.get_medium()
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = models.get_medium()  # Sonnet for subagents
    if models.get_small():
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = models.get_small()

    # Resolve command
    if command:
        cmd = command
    else:
        cmd = "claude"

    # Run command
    cmd_list = [cmd] + list(args)
    print(f"⚡ Provider: {provider}")
    if models.get_large():
        print(f"  large: {models.get_large()}")
    if models.get_medium():
        print(f"  medium: {models.get_medium()}")
    if models.get_small():
        print(f"  small: {models.get_small()}")

    result = subprocess.run(cmd_list, env=env)
    sys.exit(result.returncode)


def _resolve_models(
    config: Config,
    provider_config: ProviderConfig,
    profile_name: str | None,
    large: str | None,
    medium: str | None,
    small: str | None,
) -> ModelTier:
    """Resolve model configuration from CLI flags, profile, and defaults.

    Priority: CLI flags > profile > defaults
    """
    # Start with defaults
    models = config.get_default_tier(provider_config.name)

    # Override with profile if specified
    if profile_name:
        profile = config.get_profile(profile_name)
        if profile:
            if profile.models.get_large():
                models.large = profile.models.get_large()
            if profile.models.get_medium():
                models.medium = profile.models.get_medium()
            if profile.models.get_small():
                models.small = profile.models.get_small()
        else:
            print(f"✗ Profile '{profile_name}' not found in config")
            print(f"  Available profiles: {', '.join(config.profiles.keys())}")
            raise SystemExit(1)

    # Override with CLI flags
    if large:
        models.large = large
    if medium:
        models.medium = medium
    if small:
        models.small = small

    return models


@main.command("init")
@click.option("--large", "--opus", "large", help="Large/Opus tier model")
@click.option("--medium", "--sonnet", "medium", help="Medium/Sonnet tier model")
@click.option("--small", "--haiku", "small", help="Small/Haiku tier model")
@click.option("--profile", "-p", "profile_name", help="Use existing profile")
@click.option("--save-profile", "save_profile", help="Save as new named profile")
def init_command(large: str | None, medium: str | None, small: str | None,
                 profile_name: str | None, save_profile: str | None):
    """Initialize a new project with claude-with configuration.

    Creates .claude-with.toml in the current directory.
    """
    config = Config.load()

    # Interactive mode if no options
    if not any([large, medium, small, profile_name]):
        large = click.prompt("Large (Opus) model", default="glm5.1")
        medium = click.prompt("Medium (Sonnet) model", default="kimi-k2.5")
        small = click.prompt("Small (Haiku) model", default="step-3.5-flash")

        if click.confirm("Save as named profile?", default=False):
            save_profile = click.prompt("Profile name")

    # Use existing profile
    if profile_name:
        profile = config.get_profile(profile_name)
        if not profile:
            print(f"✗ Profile '{profile_name}' not found")
            raise SystemExit(1)
        large = large or profile.models.get_large()
        medium = medium or profile.models.get_medium()
        small = small or profile.models.get_small()

    # Create .claude-with.toml
    local_path = Path.cwd() / ".claude-with.toml"

    if save_profile:
        # Create profile reference
        content = f'profile = "{save_profile}"\n'
        # Add to central config
        central_path = Path.home() / ".config" / "claude-with" / "config.toml"
        _append_profile_to_config(central_path, save_profile, large, medium, small)
    else:
        # Create inline config
        content = "[ollama_cloud]\n"
        if large:
            content += f'large = "{large}"\n'
        if medium:
            content += f'medium = "{medium}"\n'
        if small:
            content += f'small = "{small}"\n'

    with open(local_path, "w") as f:
        f.write(content)

    print("✓ Created .claude-with.toml")
    if save_profile:
        print(f"✓ Added profile '{save_profile}' to ~/.config/claude-with/config.toml")


def _append_profile_to_config(
    path: Path,
    name: str,
    large: str | None,
    medium: str | None,
    small: str | None,
) -> None:
    """Append a new profile to the central config file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    content = f'\n[profiles.{name}]\nprovider = "ollama_cloud"\n'
    if large:
        content += f'large = "{large}"\n'
    if medium:
        content += f'medium = "{medium}"\n'
    if small:
        content += f'small = "{small}"\n'

    with open(path, "a") as f:
        f.write(content)


@main.command("list")
def list_command():
    """List available profiles and default models."""
    config = Config.load()

    print("Available profiles:")
    for name, profile in config.profiles.items():
        print(f"  {name}:")
        print(f"    provider: {profile.provider}")
        if profile.models.get_large():
            print(f"    large: {profile.models.get_large()}")
        if profile.models.get_medium():
            print(f"    medium: {profile.models.get_medium()}")
        if profile.models.get_small():
            print(f"    small: {profile.models.get_small()}")

    print("\nDefault models by provider:")
    for provider, tier in config.defaults.items():
        print(f"  {provider}:")
        if tier.get_large():
            print(f"    large: {tier.get_large()}")
        if tier.get_medium():
            print(f"    medium: {tier.get_medium()}")
        if tier.get_small():
            print(f"    small: {tier.get_small()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Create pyproject.toml for claude_with package**

Create `liberated-claude-code/claude_with/pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "claude-with"
version = "0.1.0"
description = "Ephemeral switching for Claude Code with multi-provider support"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "click>=8.0.0",
    "tomli>=2.0.0",
]

[project.scripts]
claude-with = "claude_with.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["claude_with"]
```

- [ ] **Step 7: Run CLI to verify it works**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run python -m claude_with --help`
Expected: Shows help text with commands: run, init, list

- [ ] **Step 8: Commit CLI package**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add claude_with/
git commit -m "feat: add claude-with CLI for ephemeral provider switching"
```

---

## Task 5: Create Default Configuration File

**Files:**
- Create: `~/.config/claude-with/config.toml` template
- Create: `~/.config/claude-with/.env.example`

- [ ] **Step 1: Create default config.toml template**

Create `liberated-claude-code/claude_with/templates/config.toml`:

```toml
# claude-with configuration
# Location: ~/.config/claude-with/config.toml

# Provider credentials (env var references, NOT actual keys)
[providers.ollama_cloud]
api_key_env = "OLLAMA_API_KEY"
base_url = "https://ollama.com/v1"

[providers.ollama_local]
base_url = "http://localhost:11434/v1"

[providers.nvidia_nim]
api_key_env = "NVIDIA_NIM_API_KEY"
base_url = "https://integrate.api.nvidia.com/v1"

[providers.open_router]
api_key_env = "OPENROUTER_API_KEY"
base_url = "https://openrouter.ai/api/v1"

# Default model tiers by provider
[defaults.ollama_cloud]
large = "glm5.1"
medium = "kimi-k2.5"
small = "step-3.5-flash"

[defaults.ollama_local]
large = "qwen3-32b"
medium = "qwen3-14b"
small = "qwen3-4b"

[defaults.nvidia_nim]
large = "nvidia_nim/z-ai/glm5"
medium = "nvidia_nim/meta/llama3-70b"
small = "nvidia_nim/stepfun-ai/step-3.5-flash"

[defaults.open_router]
large = "open_router/deepseek/deepseek-r1-0528:free"
medium = "open_router/openai/gpt-oss-120b:free"
small = "open_router/stepfun/step-3.5-flash:free"

# Named profiles
[profiles.oculus]
provider = "ollama_cloud"
large = "glm5.1"
medium = "kimi-k2.5"
small = "step-3.5-flash"

# Proxy settings
[proxy]
port = 8082
host = "127.0.0.1"
```

- [ ] **Step 2: Create .env.example**

Create `liberated-claude-code/claude_with/templates/.env.example`:

```bash
# API keys for claude-with
# Copy to ~/.config/claude-with/.env and fill in your keys

# Ollama Cloud
OLLAMA_API_KEY=your-ollama-cloud-key-here

# NVIDIA NIM
NVIDIA_NIM_API_KEY=your-nvidia-nim-key-here

# OpenRouter
OPENROUTER_API_KEY=your-openrouter-key-here

# Anthropic (for native Anthropic usage)
ANTHROPIC_API_KEY=your-anthropic-key-here
```

- [ ] **Step 3: Create init script to set up config**

Create `liberated-claude-code/claude_with/templates/setup.sh`:

```bash
#!/bin/bash
# Initialize claude-with configuration

CONFIG_DIR="$HOME/.config/claude-with"
TEMPLATES_DIR="$(dirname "$0")"

mkdir -p "$CONFIG_DIR"

if [ ! -f "$CONFIG_DIR/config.toml" ]; then
    cp "$TEMPLATES_DIR/config.toml" "$CONFIG_DIR/config.toml"
    echo "✓ Created $CONFIG_DIR/config.toml"
fi

if [ ! -f "$CONFIG_DIR/.env" ]; then
    cp "$TEMPLATES_DIR/.env.example" "$CONFIG_DIR/.env"
    echo "✓ Created $CONFIG_DIR/.env"
    echo "  Edit this file to add your API keys"
fi

echo ""
echo "Configuration initialized. Next steps:"
echo "  1. Edit ~/.config/claude-with/.env to add your API keys"
echo "  2. Run: claude-with ollama --profile oculus"
```

- [ ] **Step 4: Commit templates**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add claude_with/templates/
git commit -m "feat: add default configuration templates for claude-with"
```

---

## Task 6: Add Integration Tests

**Files:**
- Create: `liberated-claude-code/tests/integration/test_claude_with.py`

- [ ] **Step 1: Write integration test for config loading**

Create `liberated-claude-code/tests/integration/test_claude_with.py`:

```python
"""Integration tests for claude-with CLI."""

import os
import tempfile
from pathlib import Path

import pytest

from claude_with.config import Config, ModelTier, Profile


def test_config_loads_defaults():
    """Config loads without any files."""
    config = Config.load()
    assert isinstance(config.defaults, dict)
    assert isinstance(config.profiles, dict)


def test_config_merges_toml():
    """Config correctly merges TOML files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.toml"
        config_path.write_text("""
[defaults.ollama_cloud]
large = "test-large"
medium = "test-medium"
small = "test-small"

[profiles.test-profile]
provider = "ollama_cloud"
large = "profile-large"
medium = "profile-medium"
small = "profile-small"
""")
        config = Config()
        config._merge_toml(config_path)

        assert "ollama_cloud" in config.defaults
        assert config.defaults["ollama_cloud"].get_large() == "test-large"
        assert "test-profile" in config.profiles
        assert config.profiles["test-profile"].models.get_large() == "profile-large"


def test_model_tier_aliases():
    """ModelTier supports both large/opus, medium/sonnet, small/haiku."""
    tier = ModelTier(opus="opus-model", medium="sonnet-model", haiku="haiku-model")

    assert tier.get_large() == "opus-model"
    assert tier.get_medium() == "sonnet-model"
    assert tier.get_small() == "haiku-model"

    # Explicit takes precedence
    tier2 = ModelTier(large="explicit-large", opus="alias-large")
    assert tier2.get_large() == "explicit-large"


def test_profile_resolution():
    """Profile resolution follows priority: CLI > profile > defaults."""
    config = Config()
    config.defaults["ollama_cloud"] = ModelTier(large="default-large")
    config.profiles["test"] = Profile(
        name="test",
        provider="ollama_cloud",
        models=ModelTier(large="profile-large"),
    )

    # Without profile, use defaults
    tier = config.get_default_tier("ollama_cloud")
    assert tier.get_large() == "default-large"

    # With profile, use profile
    profile = config.get_profile("test")
    assert profile.models.get_large() == "profile-large"
```

- [ ] **Step 2: Run integration tests**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/integration/test_claude_with.py -v`
Expected: All tests pass

- [ ] **Step 3: Commit integration tests**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add tests/integration/test_claude_with.py
git commit -m "test: add integration tests for claude-with config and profile resolution"
```

---

## Task 7: Update Documentation

**Files:**
- Modify: `liberated-claude-code/README.md`
- Create: `liberated-claude-code/claude_with/README.md`

- [ ] **Step 1: Add ollama providers to main README**

Add to `liberated-claude-code/README.md` in the Providers section:

```markdown
| **Ollama Cloud** | Free (rate limits) | Varies | Cloud models via Anthropic-compatible API |
| **Ollama Local** | Free (local) | Unlimited | Local ollama serve instances |

Models use a prefix format: `provider_prefix/model/name`. An invalid prefix causes an error.

| Provider | `MODEL` prefix | API Key Variable | Default Base URL |
| `ollama_cloud/...` | `OLLAMA_API_KEY` | `https://ollama.com/v1` |
| `ollama_local/...` | (none) | `http://localhost:11434/v1` |
```

- [ ] **Step 2: Create claude-with README**

Create `liberated-claude-code/claude_with/README.md`:

```markdown
# claude-with

Ephemeral switching for Claude Code with multi-provider support.

## Installation

```bash
cd liberated-claude-code
uv sync
uv pip install -e ./claude_with
```

## Quick Start

```bash
# Initialize configuration
claude-with init

# Use ollama cloud with default models
claude-with ollama

# Use a named profile
claude-with ollama --profile oculus

# Launch VS Code with ollama config
claude-with ollama --profile oculus -- code .

# Specify models inline
claude-with ollama --large glm5 --medium kimi-k2.5 --small step-3.5-flash
```

## Configuration

### Central Config

`~/.config/claude-with/config.toml`:

```toml
[defaults.ollama_cloud]
large = "glm5.1"
medium = "kimi-k2.5"
small = "step-3.5-flash"

[profiles.oculus]
provider = "ollama_cloud"
large = "glm5.1"
medium = "kimi-k2.5"
small = "step-3.5-flash"
```

### API Keys

Store in `~/.config/claude-with/.env`:

```bash
OLLAMA_API_KEY=your-key-here
NVIDIA_NIM_API_KEY=your-key-here
OPENROUTER_API_KEY=your-key-here
```

### Project-Local Config

`.claude-with.toml` in your project root:

```toml
# Reference a named profile
profile = "oculus"

# Or define inline
[ollama_cloud]
large = "glm5.1"
medium = "kimi-k2.5"
small = "step-3.5-flash"
```

## Commands

- `claude-with <provider>` - Launch Claude Code with a provider
- `claude-with init` - Initialize project configuration
- `claude-with list` - List available profiles

## Providers

| Provider | Command | Requires API Key |
|----------|---------|-------------------|
| Ollama Cloud | `claude-with ollama` | `OLLAMA_API_KEY` |
| Ollama Local | `claude-with ollama-local` | No |
| NVIDIA NIM | `claude-with nvidia` | `NVIDIA_NIM_API_KEY` |
| OpenRouter | `claude-with openrouter` | `OPENROUTER_API_KEY` |
| Anthropic | `claude-with anthropic` | `ANTHROPIC_API_KEY` |
```

- [ ] **Step 3: Commit documentation**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add README.md claude_with/README.md
git commit -m "docs: add documentation for ollama providers and claude-with CLI"
```

---

## Task 8: Update Existing Switch Script

**Files:**
- Modify: `liberated-claude-code/claude-switch.sh`

- [ ] **Step 1: Add ollama provider to claude-switch**

Add new command in `liberated-claude-code/claude-switch.sh` after the modal command (around line 220):

```bash
# ── Ollama Cloud ──────────────────────────────────────────────────────────────

_vscode_set_ollama() {
    python3 -c "
import json, sys
path = sys.argv[1]
with open(path) as f:
    s = json.load(f)
s['claudeCode.environmentVariables'] = [
    {'name': 'ANTHROPIC_BASE_URL', 'value': 'http://localhost:$PROXY_PORT'},
    {'name': 'ANTHROPIC_API_KEY', 'value': 'freecc'},
]
with open(path, 'w') as f:
    json.dump(s, f, indent=4)
print('  ✓ VS Code settings updated — reload the Claude extension to take effect')
" "$VSCODE_SETTINGS"
}

_verify_ollama() {
    echo ""
    echo "  Verifying Ollama Cloud routing..."
    local response
    response=$(curl -sf "$PROXY_URL/health") || { echo "  ✗ Proxy not responding"; exit 1; }
    echo "  ✓ Proxy health: $response"

    if [[ -z "$OLLAMA_API_KEY" ]]; then
        echo "  ✗ OLLAMA_API_KEY not set"
        echo "  Set it in your shell: export OLLAMA_API_KEY=\"your-key\""
        exit 1
    fi
    echo "  ✓ Auth: OLLAMA_API_KEY set"

    echo ""
    echo "  ⚡ Active provider: Ollama Cloud"
    echo "  Models: large→glm5.1, medium→kimi-k2.5, small→step-3.5-flash"
}

cmd_ollama() {
    echo ""
    echo "━━━ Switching to Ollama Cloud ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    if [[ -z "$OLLAMA_API_KEY" ]]; then
        echo "  ✗ OLLAMA_API_KEY not set"
        echo "  Get a key at: https://ollama.com/settings/keys"
        echo "  Set it: export OLLAMA_API_KEY=\"your-key\""
        exit 1
    fi
    _start_proxy
    echo "  Logging out of Anthropic OAuth..."
    claude auth logout 2>/dev/null && echo "  ✓ Logged out" || echo "  ✓ Already logged out"
    _vscode_set_ollama
    _verify_ollama
    echo ""
    echo "  Launch CLI: claude-with ollama --profile oculus"
    echo "  Or: ANTHROPIC_BASE_URL=$PROXY_URL ANTHROPIC_API_KEY=freecc claude"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}
```

Then add `ollama` to the case statement (around line 227):

```bash
case "${1:-}" in
    nim) cmd_nim ;;
    modal) cmd_modal ;;
    ollama) cmd_ollama ;;
    anthropic) cmd_anthropic ;;
    status)
        # ... existing status code ...
    ;;
    *)
        echo "Usage: claude-switch [nim|modal|ollama|anthropic|status]"
        exit 1
        ;;
esac
```

- [ ] **Step 2: Commit switch script update**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add claude-switch.sh
git commit -m "feat: add ollama provider to claude-switch"
```

---

## Verification

- [ ] **Final verification: Run all tests**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
uv run pytest tests/ -v
```

Expected: All tests pass

- [ ] **Final verification: Test CLI**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
uv run claude-with --help
uv run claude-with list
```

Expected: Shows help and lists profiles

- [ ] **Final verification: Test proxy with ollama provider**

```bash
# Start proxy
cd /Users/justincantrall/Projects/liberated-claude-code
uv run uvicorn server:app --host 127.0.0.1 --port 8082 &

# Set ollama key
export OLLAMA_API_KEY="your-ollama-key"

# Test request
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: freecc" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model": "claude-sonnet-4-5", "max_tokens": 50, "messages": [{"role": "user", "content": "Hello"}]}'
```

Expected: Response from ollama cloud

---

## Completion Checklist

- [ ] All tests pass
- [ ] CLI works: `claude-with ollama --help`
- [ ] Config loading works: `claude-with list`
- [ ] Proxy routes to ollama cloud
- [ ] Documentation updated
- [ ] All files committed