# OpenAI-Compat Provider + Anthropic Naming + Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add openai-compat provider, Anthropic naming split, translation layer docs, and getting-started guide to liberated-claude-code.

**Architecture:** OpenAICompatProvider subclasses the existing OpenAICompatibleProvider, using the existing Anthropic→OpenAI translation layer. Anthropic-api provider routes through the proxy for tier routing. Documentation is standalone markdown files.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, Click, httpx, openai SDK

---

## File Structure

**New files in `liberated-claude-code/`:**
```
providers/openai_compat_generic/__init__.py  # NEW: exports OpenAICompatProvider
providers/openai_compat_generic/client.py     # NEW: OpenAICompatProvider subclass
tests/providers/test_openai_compat.py          # NEW: unit tests for openai-compat provider
tests/test_openai_compat_settings.py          # NEW: settings tests
docs/translation-layer.md                     # NEW: translation layer documentation
docs/getting-started.md                        # NEW: fresh environment setup guide
```

**Modified files in `liberated-claude-code/`:**
```
providers/__init__.py                      # MODIFY: add OpenAICompatProvider export
config/settings.py                         # MODIFY: add openai_compat settings, add "openai_compatible" to validator
api/dependencies.py                        # MODIFY: register openai_compatible provider
claude_with/providers.py                   # MODIFY: add OPENAI_COMPAT and ANTHROPIC_API enum values + configs
claude_with/cli.py                         # MODIFY: add openai-compat and anthropic-api choices
claude-switch.sh                           # MODIFY: add openai-compat and anthropic-api commands
```

---

## Task 1: Add OpenAI-Compat Provider

**Files:**
- Create: `liberated-claude-code/providers/openai_compat_generic/__init__.py`
- Create: `liberated-claude-code/providers/openai_compat_generic/client.py`
- Create: `liberated-claude-code/tests/providers/test_openai_compat.py`

- [ ] **Step 1: Write failing test for OpenAICompatProvider**

Create `liberated-claude-code/tests/providers/test_openai_compat.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/providers/test_openai_compat.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'providers.openai_compat_generic'"

- [ ] **Step 3: Create openai-compat package directory**

Run: `mkdir -p /Users/justincantrall/Projects/liberated-claude-code/providers/openai_compat_generic`

Create `liberated-claude-code/providers/openai_compat_generic/__init__.py`:

```python
"""OpenAI-compatible generic provider — bring-your-own-endpoint.

For any endpoint that accepts OpenAI /v1/chat/completions format:
vLLM, LiteLLM proxy, TGI, SGLang, Together, Fireworks, Groq, etc.
Uses the existing Anthropic→OpenAI translation layer.
"""

from .client import OpenAICompatProvider

__all__ = ["OpenAICompatProvider"]
```

- [ ] **Step 4: Create OpenAICompatProvider**

Create `liberated-claude-code/providers/openai_compat_generic/client.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/providers/test_openai_compat.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Update providers/__init__.py to export OpenAICompatProvider**

Modify `liberated-claude-code/providers/__init__.py`. Add import and export:

```python
from .openai_compat_generic import OpenAICompatProvider
```

Add `"OpenAICompatProvider"` to the `__all__` list.

The full updated `__init__.py` should look like:

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
from .openai_compat_generic import OpenAICompatProvider

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
    "OpenAICompatProvider",
    "OpenRouterProvider",
    "OverloadedError",
    "ProviderConfig",
    "ProviderError",
    "RateLimitError",
]
```

- [ ] **Step 7: Run all provider tests to verify no regression**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/providers/ -v`
Expected: All tests pass

- [ ] **Step 8: Commit openai-compat provider**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add providers/openai_compat_generic/ tests/providers/test_openai_compat.py providers/__init__.py
git commit -m "feat: add openai-compat generic provider for any OpenAI-compatible endpoint"
```

---

## Task 2: Register OpenAI-Compat Provider in Settings and Dependencies

**Files:**
- Create: `liberated-claude-code/tests/test_openai_compat_settings.py`
- Modify: `liberated-claude-code/config/settings.py`
- Modify: `liberated-claude-code/api/dependencies.py`

- [ ] **Step 1: Write failing test for openai-compat settings**

Create `liberated-claude-code/tests/test_openai_compat_settings.py`:

```python
"""Tests for openai-compat provider settings."""

from config.settings import Settings


def test_openai_compat_api_key_setting():
    """Settings recognizes OPENAI_COMPAT_API_KEY."""
    settings = Settings(openai_compat_api_key="sk-test-key")
    assert settings.openai_compat_api_key == "sk-test-key"


def test_openai_compat_base_url_default():
    """Settings provides default base URL for openai-compat."""
    settings = Settings()
    assert settings.openai_compat_base_url == "https://api.openai.com/v1"


def test_openai_compat_base_url_override():
    """Settings allows overriding openai-compat base URL."""
    settings = Settings(openai_compat_base_url="https://employer-litellm.internal/v1")
    assert settings.openai_compat_base_url == "https://employer-litellm.internal/v1"


def test_model_validation_accepts_openai_compatible_prefix():
    """Model validation accepts openai_compatible prefix."""
    settings = Settings(model="openai_compatible/employer-model-large")
    assert settings.provider_type == "openai_compatible"
    assert settings.model_name == "employer-model-large"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/test_openai_compat_settings.py -v`
Expected: FAIL with "ValidationError" or "AttributeError"

- [ ] **Step 3: Add openai-compat settings to config/settings.py**

Modify `liberated-claude-code/config/settings.py`. Add after the ollama settings section:

```python
    # ==================== OpenAI-Compatible Config ====================
    openai_compat_api_key: str = Field(default="", validation_alias="OPENAI_COMPAT_API_KEY")
    openai_compat_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="OPENAI_COMPAT_BASE_URL",
    )
```

Then modify the `validate_model_format` method's `valid_providers` tuple to include `"openai_compatible"`:

```python
    valid_providers = (
        "nvidia_nim", "open_router", "lmstudio", "llamacpp", "modal",
        "ollama_cloud", "ollama_local", "openai_compatible",
    )
```

Also update the error message in that validator to include `"openai_compatible"` in the supported list.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/test_openai_compat_settings.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Add openai_compatible provider to dependencies.py**

Modify `liberated-claude-code/api/dependencies.py`. Add import at top:

```python
from providers.openai_compat_generic import OpenAICompatProvider
```

Then add provider creation in `_create_provider_for_type`, after the `ollama_local` branch:

```python
    if provider_type == "openai_compatible":
        if not settings.openai_compat_api_key or not settings.openai_compat_api_key.strip():
            raise AuthenticationError(
                "OPENAI_COMPAT_API_KEY is not set. Add it to your .env file. "
                "Set OPENAI_COMPAT_BASE_URL to your endpoint URL."
            )
        config = ProviderConfig(
            api_key=settings.openai_compat_api_key,
            base_url=settings.openai_compat_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return OpenAICompatProvider(config)
```

Update the error message in the else branch to include `"openai_compatible"` in the supported providers list.

- [ ] **Step 6: Run all tests to verify no regression**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 7: Commit settings and dependencies changes**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add config/settings.py api/dependencies.py tests/test_openai_compat_settings.py
git commit -m "feat: register openai_compatible provider in settings and dependencies"
```

---

## Task 3: Update claude-with CLI for openai-compat and anthropic-api

**Files:**
- Modify: `liberated-claude-code/claude_with/providers.py`
- Modify: `liberated-claude-code/claude_with/cli.py`

- [ ] **Step 1: Update Provider enum and ProviderConfig in providers.py**

Modify `liberated-claude-code/claude_with/providers.py`. Add two new enum values to `Provider`:

```python
class Provider(StrEnum):
    OLLAMA_CLOUD = "ollama"
    OLLAMA_LOCAL = "ollama-local"
    NVIDIA_NIM = "nvidia"
    OPENROUTER = "openrouter"
    OPENAI_COMPAT = "openai-compat"
    ANTHROPIC = "anthropic"
    ANTHROPIC_API = "anthropic-api"
```

Add two new entries to `ProviderConfig.get()`:

```python
    Provider.OPENAI_COMPAT: cls(
        name="openai_compatible",
        env_var="OPENAI_COMPAT_API_KEY",
        base_url="https://api.openai.com/v1",
        requires_proxy=True,
    ),
    Provider.ANTHROPIC_API: cls(
        name="anthropic",
        env_var="ANTHROPIC_API_KEY",
        base_url=None,
        requires_proxy=True,
    ),
```

Add `"openai_compatible"` to `PROVIDER_PREFIXES`:

```python
PROVIDER_PREFIXES = {
    "ollama_cloud": "ollama_cloud",
    "ollama_local": "ollama_local",
    "nvidia_nim": "nvidia_nim",
    "open_router": "open_router",
    "lmstudio": "lmstudio",
    "llamacpp": "llamacpp",
    "openai_compatible": "openai_compatible",
}
```

- [ ] **Step 2: Update CLI commands in cli.py**

Modify `liberated-claude-code/claude_with/cli.py`. Update the `run` command's provider choice list:

```python
@click.argument("provider", type=click.Choice([
    "ollama", "ollama-local", "nvidia", "openrouter",
    "openai-compat", "anthropic", "anthropic-api",
]))
```

Update the provider-to-enum mapping:

```python
    provider_enum = {
        "ollama": Provider.OLLAMA_CLOUD,
        "ollama-local": Provider.OLLAMA_LOCAL,
        "nvidia": Provider.NVIDIA_NIM,
        "openrouter": Provider.OPENROUTER,
        "openai-compat": Provider.OPENAI_COMPAT,
        "anthropic": Provider.ANTHROPIC,
        "anthropic-api": Provider.ANTHROPIC_API,
    }[provider]
```

Add handling for openai-compat base URL in the environment setup section of `run()`. After the existing `if provider_config.requires_proxy:` block, add handling for openai-compat's base URL:

```python
    # Set provider-specific base URL override for openai-compat
    if provider_enum == Provider.OPENAI_COMPAT:
        base_url = os.environ.get("OPENAI_COMPAT_BASE_URL", provider_config.base_url)
        if base_url:
            env["OPENAI_COMPAT_BASE_URL"] = base_url
```

- [ ] **Step 3: Test CLI changes**

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run python -m claude_with --help`
Expected: Shows help with updated provider choices including `openai-compat` and `anthropic-api`

Run: `cd /Users/justincantrall/Projects/liberated-claude-code && uv run python -m claude_with list`
Expected: Shows profiles and defaults

- [ ] **Step 4: Commit CLI changes**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add claude_with/providers.py claude_with/cli.py
git commit -m "feat: add openai-compat and anthropic-api to claude-with CLI"
```

---

## Task 4: Add openai-compat and anthropic-api to claude-switch.sh

**Files:**
- Modify: `liberated-claude-code/claude-switch.sh`

- [ ] **Step 1: Add openai-compat command to claude-switch.sh**

Add after the `cmd_ollama` function (around line 282), before the entry point section:

```bash
# ── OpenAI-Compatible ─────────────────────────────────────────────────────────

_vscode_set_openai_compat() {
    python3 -c "
import json, sys
path = sys.argv[1]
with open(path) as f:
    s = json.load(f)
s['claudeCode.environmentVariables'] = [
    {'name': 'ANTHROPIC_BASE_URL', 'value': 'http://localhost:$PROXY_PORT'},
    {'name': 'ANTHROPIC_API_KEY',  'value': 'freecc'},
]
with open(path, 'w') as f:
    json.dump(s, f, indent=4)
print('  ✓ VS Code settings updated — reload the Claude extension to take effect')
" "$VSCODE_SETTINGS"
}

_verify_openai_compat() {
    echo ""
    echo "  Verifying OpenAI-compatible endpoint routing..."
    local response
    response=$(curl -sf "$PROXY_URL/health") || { echo "  ✗ Proxy not responding"; exit 1; }
    echo "  ✓ Proxy health: $response"

    if [[ -z "$OPENAI_COMPAT_API_KEY" ]]; then
        echo "  ⚠ OPENAI_COMPAT_API_KEY not set (some endpoints don't require auth)"
    else
        echo "  ✓ Auth: OPENAI_COMPAT_API_KEY set"
    fi

    if [[ -z "$OPENAI_COMPAT_BASE_URL" ]]; then
        echo "  ✗ OPENAI_COMPAT_BASE_URL not set"
        echo "  Set it: export OPENAI_COMPAT_BASE_URL=\"https://your-endpoint/v1\""
        exit 1
    fi
    echo "  ✓ Endpoint: $OPENAI_COMPAT_BASE_URL"

    echo ""
    echo "  ⚡ Active provider: OpenAI-Compatible"
    echo "  Endpoint: $OPENAI_COMPAT_BASE_URL"
}

cmd_openai_compat() {
    echo ""
    echo "━━━ Switching to OpenAI-Compatible Endpoint ━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    _start_proxy
    echo "  Logging out of Anthropic OAuth..."
    claude auth logout 2>/dev/null && echo "  ✓ Logged out" || echo "  ✓ Already logged out"
    _vscode_set_openai_compat
    _verify_openai_compat
    echo ""
    echo "  Launch CLI: claude-with openai-compat --large <model>"
    echo "  Or: ANTHROPIC_BASE_URL=$PROXY_URL ANTHROPIC_API_KEY=freecc claude"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# ── Anthropic API ─────────────────────────────────────────────────────────────

_vscode_set_anthropic_api() {
    python3 -c "
import json, sys
path = sys.argv[1]
with open(path) as f:
    s = json.load(f)
s['claudeCode.environmentVariables'] = [
    {'name': 'ANTHROPIC_BASE_URL', 'value': 'http://localhost:$PROXY_PORT'},
    {'name': 'ANTHROPIC_API_KEY',  'value': 'freecc'},
]
with open(path, 'w') as f:
    json.dump(s, f, indent=4)
print('  ✓ VS Code settings updated — reload the Claude extension to take effect')
" "$VSCODE_SETTINGS"
}

_verify_anthropic_api() {
    echo ""
    echo "  Verifying Anthropic API routing..."
    local response
    response=$(curl -sf "$PROXY_URL/health") || { echo "  ✗ Proxy not responding"; exit 1; }
    echo "  ✓ Proxy health: $response"

    if [[ -z "$ANTHROPIC_API_KEY" ]]; then
        echo "  ✗ ANTHROPIC_API_KEY not set"
        echo "  Set it: export ANTHROPIC_API_KEY=\"your-key\""
        exit 1
    fi
    echo "  ✓ Auth: ANTHROPIC_API_KEY set"

    echo ""
    echo "  ⚡ Active provider: Anthropic API (through proxy)"
    echo "  Models: tier routing through proxy"
}

cmd_anthropic_api() {
    echo ""
    echo "━━━ Switching to Anthropic API (proxy) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    if [[ -z "$ANTHROPIC_API_KEY" ]]; then
        echo "  ✗ ANTHROPIC_API_KEY not set"
        echo "  Set it: export ANTHROPIC_API_KEY=\"your-key\""
        exit 1
    fi
    _start_proxy
    echo "  Logging out of Anthropic OAuth..."
    claude auth logout 2>/dev/null && echo "  ✓ Logged out" || echo "  ✓ Already logged out"
    _vscode_set_anthropic_api
    _verify_anthropic_api
    echo ""
    echo "  Launch CLI: claude-with anthropic-api --large claude-opus-4-7"
    echo "  Or: ANTHROPIC_BASE_URL=$PROXY_URL ANTHROPIC_API_KEY=freecc claude"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}
```

- [ ] **Step 2: Update the case statement**

Modify the entry point `case` statement to include both new commands:

```bash
case "${1:-}" in
    nim) cmd_nim ;;
    modal) cmd_modal ;;
    ollama) cmd_ollama ;;
    openai-compat) cmd_openai_compat ;;
    anthropic-api) cmd_anthropic_api ;;
    anthropic) cmd_anthropic ;;
    status)
        # ... existing status code ...
    ;;
    *)
        echo "Usage: claude-switch [nim|modal|ollama|openai-compat|anthropic-api|anthropic|status]"
        exit 1
        ;;
esac
```

Note: `anthropic` stays as-is (OAuth, no proxy). `anthropic-api` is the new one (API key, through proxy).

- [ ] **Step 3: Commit switch script changes**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add claude-switch.sh
git commit -m "feat: add openai-compat and anthropic-api commands to claude-switch"
```

---

## Task 5: Write Translation Layer Documentation

**Files:**
- Create: `liberated-claude-code/docs/translation-layer.md`

- [ ] **Step 1: Write translation layer documentation**

Create `liberated-claude-code/docs/translation-layer.md` with content covering:

1. **Provider categories** — pass-through vs. translation, with tables listing each provider and which category it falls into
2. **Translation flow diagrams** — request arrives as Anthropic format, either forwarded as-is (pass-through) or converted via `build_base_request_body()` + `SSEBuilder` (translation)
3. **Key files** — `providers/common/message_converter.py`, `providers/common/sse_builder.py`, `providers/openai_compat.py` (base class)
4. **Decision flowchart** — "Does your endpoint accept /v1/messages?" → Yes: subclass BaseProvider → No: subclass OpenAICompatibleProvider
5. **How to add a new provider** — step-by-step for both categories
6. **Translation details** — message conversion table, tool definition conversion, stop reason mapping

The full content matches what was approved in the design spec (Section 2).

- [ ] **Step 2: Commit translation layer docs**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add docs/translation-layer.md
git commit -m "docs: add translation layer documentation"
```

---

## Task 6: Write Getting Started Guide

**Files:**
- Create: `liberated-claude-code/docs/getting-started.md`

- [ ] **Step 1: Write getting started guide**

Create `liberated-claude-code/docs/getting-started.md` with content covering:

1. **What is claude-with** — session-scoped env var wrapper, not a shell profile modification
2. **Installation** — `uv sync` + `uv pip install -e ./claude_with`
3. **`claude-with init`** — what it does, three modes (interactive, inline, profile reference)
4. **`claude-with <provider>`** — how it works, priority chain, all provider examples including openai-compat and anthropic-api
5. **Providers table** — all 7 providers with command, API key, and format
6. **Anthropic naming note** — `anthropic` vs `anthropic-api` distinction
7. **Central config** — `~/.config/claude-with/config.toml` full example with all providers
8. **API keys** — `~/.config/claude-with/.env` full example with key lookup order
9. **Full walkthrough** — new machine, zero to running, with all commands
10. **Troubleshooting** — common issues and fixes

The full content matches what was approved in the design spec (Section 4, revised).

- [ ] **Step 2: Commit getting started guide**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add docs/getting-started.md
git commit -m "docs: add getting started guide for claude-with"
```

---

## Task 7: Update Main README

**Files:**
- Modify: `liberated-claude-code/README.md`

- [ ] **Step 1: Add openai-compat and anthropic-api to README provider table**

Find the providers table in `liberated-claude-code/README.md` and add two new rows:

| **OpenAI-Compat** | Varies | Varies | Any OpenAI-compatible endpoint (vLLM, LiteLLM, TGI, etc.) |
| **Anthropic API** | Pay-per-token | Varies | Anthropic API through proxy (tier routing) |

Add to the model prefix format table:

| `openai_compatible/...` | `OPENAI_COMPAT_API_KEY` + `OPENAI_COMPAT_BASE_URL` | User-configured |

- [ ] **Step 2: Add links to new documentation**

Add links section in README pointing to new docs:

```markdown
## Documentation

- [Getting Started Guide](docs/getting-started.md) — Setup, configuration, and usage
- [Translation Layer](docs/translation-layer.md) — How the proxy translates between Anthropic and OpenAI formats
```

- [ ] **Step 3: Commit README changes**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
git add README.md
git commit -m "docs: add openai-compat and anthropic-api to README"
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
uv run python -m claude_with --help
uv run python -m claude_with list
```

Expected: Shows help with `openai-compat` and `anthropic-api` in provider choices. Shows profiles and defaults.

- [ ] **Final verification: Test claude-switch**

```bash
cd /Users/justincantrall/Projects/liberated-claude-code
bash claude-switch.sh
```

Expected: Shows usage with `openai-compat` and `anthropic-api` in the command list.

---

## Completion Checklist

- [ ] OpenAI-compat provider implemented and tested
- [ ] Settings and dependencies updated with openai_compatible registration
- [ ] claude-with CLI updated with openai-compat and anthropic-api
- [ ] claude-switch.sh updated with openai-compat and anthropic-api commands
- [ ] Translation layer documentation written
- [ ] Getting started guide written
- [ ] README updated
- [ ] All tests pass
- [ ] All files committed