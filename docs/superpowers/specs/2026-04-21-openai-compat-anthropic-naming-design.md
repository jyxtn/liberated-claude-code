# OpenAI-Compat Provider, Anthropic Naming, and Setup Guide

**Date:** 2026-04-21
**Status:** Approved
**Related:** 2026-04-18-ollama-proxy-design.md (existing ollama provider spec)

## Problem

Three gaps in the current design:

1. **No generic OpenAI-compatible provider.** Users with self-hosted endpoints (vLLM, LiteLLM proxy, TGI, SGLang, Together, Fireworks, Groq) that only accept OpenAI-format `/v1/chat/completions` requests have no way to route through the proxy. Each would need a dedicated provider subclass.

2. **Anthropic naming is ambiguous.** `claude-with anthropic` currently means "use your Anthropic OAuth subscription directly, no proxy." Users with an Anthropic API key who want per-tier model routing through the proxy have no option. The naming doesn't distinguish between subscription (OAuth, no proxy) and API key (proxy, tier routing).

3. **No setup documentation.** New users don't know how to install, configure, or invoke `claude-with` from scratch. The existing spec assumes familiarity.

4. **No translation layer documentation.** Contributors don't know which providers use translation vs. pass-through, or how to decide for a new provider.

## Solution Overview

Four additions:

1. **`openai_compatible` provider** — thin wrapper around `OpenAICompatibleProvider` using the existing translation layer. User provides `OPENAI_COMPAT_API_KEY` and `OPENAI_COMPAT_BASE_URL`. Model string format: `openai_compatible/your-model-name`.

2. **`anthropic-api` provider** — new CLI option that routes Anthropic API key traffic through the proxy for tier routing and optimizations. Existing `anthropic` remains unchanged (OAuth subscription, no proxy).

3. **Translation layer documentation** — new `docs/translation-layer.md` explaining which providers translate, which pass through, and how to decide for new providers.

4. **Fresh environment setup guide** — new `docs/getting-started.md` with full walkthrough from zero to running, including what each command does, how installation persists, and per-project config.

## Section 1: openai-compat Provider

### Architecture

```
Claude Code → Proxy (/v1/messages, Anthropic format)
                    │
                    ▼
         OpenAICompatProvider._build_request_body()
                    │
                    ▼
         build_base_request_body() (AnthropicToOpenAIConverter)
                    │
                    ▼
         POST /v1/chat/completions → OpenAI-compatible endpoint
                    │
                    ▼
         OpenAI SSE chunks → SSEBuilder → Anthropic SSE events
```

The provider subclasses `OpenAICompatibleProvider` and overrides `_build_request_body()` to call the existing `build_base_request_body()` converter. No provider-specific extras (no NIM thinking, no OpenRouter reasoning). All streaming, rate limiting, and SSE translation is inherited.

### Files

- **Create:** `liberated-claude-code/providers/openai_compat/client.py`
- **Create:** `liberated-claude-code/tests/providers/test_openai_compat.py`
- **Modify:** `liberated-claude-code/providers/__init__.py` (add export)
- **Modify:** `liberated-claude-code/config/settings.py` (add `OPENAI_COMPAT_API_KEY`, `OPENAI_COMPAT_BASE_URL`)
- **Modify:** `liberated-claude-code/api/dependencies.py` (register provider)
- **Modify:** `liberated-claude-code/claude_with/providers.py` (add `Provider.OPENAI_COMPAT` enum, `ProviderConfig` entry)
- **Modify:** `liberated-claude-code/claude_with/cli.py` (add `openai-compat` to CLI choices)

### Provider Implementation

```python
"""OpenAI-compatible provider — bring-your-own-endpoint.

For any endpoint that accepts OpenAI /v1/chat/completions format:
vLLM, LiteLLM proxy, TGI, SGLang, Together, Fireworks, Groq, etc.
Uses the existing Anthropic→OpenAI translation layer.
"""

from providers.openai_compat import OpenAICompatibleProvider
from providers.base import ProviderConfig


DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatProvider(OpenAICompatibleProvider):
    """Generic OpenAI-compatible provider.

    Routes Anthropic-format requests through the translation layer
    to any OpenAI-compatible endpoint.
    """

    provider_name = "openai_compatible"

    def __init__(self, config: ProviderConfig):
        # Use configured base_url or default to OpenAI's API
        base_url = config.base_url or DEFAULT_BASE_URL
        api_key = config.api_key or ""
        super().__init__(
            provider_name=self.provider_name,
            base_url=base_url,
            api_key=api_key,
        )

    def _build_request_body(self, request) -> dict:
        from providers.common.message_converter import build_base_request_body
        return build_base_request_body(request)
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_COMPAT_API_KEY` | Yes | — | API key for the endpoint |
| `OPENAI_COMPAT_BASE_URL` | Yes | `https://api.openai.com/v1` | Base URL of the OpenAI-compatible endpoint |

### Model String Format

```
openai_compatible/<model-name>
```

The part after `/` is sent as-is to the endpoint's `model` field.

Examples:

| Use Case | Model String | Base URL |
|---|---|---|
| Employer LiteLLM proxy | `openai_compatible/employer-model-large` | `https://employer-litellm.internal/v1` |
| Self-hosted vLLM | `openai_compatible/llama3-70b` | `http://gpu-rig:8000/v1` |
| Together AI | `openai_compatible/meta-llama/Llama-3-70b` | `https://api.together.xyz/v1` |
| Fireworks AI | `openai_compatible/llama-v3-70b` | `https://api.fireworks.ai/inference/v1` |
| Groq | `openai_compatible/llama3-70b-8192` | `https://api.groq.com/openai/v1` |

### Settings Changes

Add to `config/settings.py`:

```python
# ==================== OpenAI-Compatible Config ====================
openai_compat_api_key: str = Field(default="", validation_alias="OPENAI_COMPAT_API_KEY")
openai_compat_base_url: str = Field(
    default="https://api.openai.com/v1",
    validation_alias="OPENAI_COMPAT_BASE_URL",
)
```

Add `"openai_compatible"` to `validate_model_format`'s valid providers list.

### Dependencies Changes

Add to `api/dependencies.py`:

```python
if provider_type == "openai_compatible":
    if not settings.openai_compat_api_key or not settings.openai_compat_api_key.strip():
        raise AuthenticationError(
            "OPENAI_COMPAT_API_KEY is not set. Add it to your .env file."
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

### CLI Changes

Add to `claude_with/providers.py`:

```python
class Provider(StrEnum):
    OLLAMA_CLOUD = "ollama"
    OLLAMA_LOCAL = "ollama-local"
    NVIDIA_NIM = "nvidia"
    OPENROUTER = "openrouter"
    OPENAI_COMPAT = "openai-compat"  # NEW
    ANTHROPIC = "anthropic"
    ANTHROPIC_API = "anthropic-api"  # NEW
```

Add to `claude_with/cli.py` Click choices and provider mapping.

### What This Provider Does NOT Do

- No health checks or endpoint validation at startup
- No model listing (`/v1/models`)
- No provider-specific request tweaks
- No special error messages beyond what `OpenAICompatibleProvider` provides

These can be added later without breaking changes.

## Section 2: Translation Layer Documentation

### New File: `docs/translation-layer.md`

Explains how the proxy handles format translation:

**Anthropic→OpenAI translation providers** (need format conversion):
- NvidiaNIM, OpenRouter, Modal, openai-compat
- Request: `AnthropicToOpenAIConverter` + `build_base_request_body()` converts Anthropic `/v1/messages` format to OpenAI `/v1/chat/completions` format
- Response: `SSEBuilder` converts OpenAI SSE chunks to Anthropic SSE events

**Anthropic pass-through providers** (no translation):
- LM Studio, llama.cpp, Ollama Cloud, Ollama Local
- Request body forwarded as-is to provider's `/v1/messages` endpoint
- Response SSE events yielded directly

**Key files:**
- `providers/common/message_converter.py` — request translation (`AnthropicToOpenAIConverter`, `build_base_request_body()`)
- `providers/common/sse_builder.py` — response translation (`SSEBuilder`)
- `providers/openai_compat.py` — `OpenAICompatibleProvider` base class (streaming loop, tool parsing, rate limiting)

**Provider decision flowchart:**

```
Does your endpoint accept /v1/messages (Anthropic format)?
├── Yes → Subclass BaseProvider, pass through
│          Examples: LM Studio, llama.cpp, Ollama
└── No  → Subclass OpenAICompatibleProvider, get translation for free
           Examples: NvidiaNIM, OpenRouter, Modal, openai-compat
```

**Translation details:**

The `AnthropicToOpenAIConverter` handles:
- Messages: Anthropic content blocks → OpenAI message roles (`tool_result` → `role: "tool"`, `thinking` → inline `<think>` tags, `tool_use` → `tool_calls` arrays)
- Tools: Anthropic tool definitions → OpenAI `type: "function"` format
- System prompt: Anthropic `system` field → OpenAI `{"role": "system"}` message
- Stop reasons: OpenAI → Anthropic mapping (`stop` → `end_turn`, `tool_calls` → `tool_use`, etc.)

The `SSEBuilder` handles:
- OpenAI streaming chunks → Anthropic SSE events (`message_start`, `content_block_start/delta/stop`, `message_delta`, `message_stop`)
- Think tag parsing (`<think>`/`</think>`)
- Heuristic tool call parsing (text-emitted `<function=Name>` patterns)
- Native tool call parsing (OpenAI `delta.tool_calls`)

## Section 3: Anthropic Naming Split

### Current Behavior

| CLI command | What it does | Proxy |
|---|---|---|
| `claude-with anthropic` | Uses Anthropic OAuth subscription | No proxy needed |

### New Behavior

| CLI command | What it does | Proxy |
|---|---|---|
| `claude-with anthropic` | Uses Anthropic OAuth subscription (unchanged) | No |
| `claude-with anthropic-api` | Uses `ANTHROPIC_API_KEY`, routes through proxy for tier routing and optimizations | Yes |

### Implementation

**`claude_with/providers.py`:**

```python
class Provider(StrEnum):
    # ... existing ...
    ANTHROPIC = "anthropic"          # OAuth subscription, no proxy
    ANTHROPIC_API = "anthropic-api"   # API key, through proxy
```

```python
ProviderConfig.get() additions:
    Provider.ANTHROPIC_API: cls(
        name="anthropic",
        env_var="ANTHROPIC_API_KEY",
        base_url=None,  # Uses proxy URL
        requires_proxy=True,
    ),
```

**`claude_with/cli.py`:**

The `anthropic-api` provider:
1. Sets `ANTHROPIC_BASE_URL` to the proxy URL (`http://127.0.0.1:8082`)
2. Sets `ANTHROPIC_API_KEY` from env/keychain
3. Launches `claude`

The `anthropic` provider (unchanged):
1. Removes `ANTHROPIC_BASE_URL` from env (no proxy)
2. Uses `ANTHROPIC_API_KEY` from env/keychain if available, otherwise relies on OAuth
3. Launches `claude`

**`claude-switch.sh`:**

Add `anthropic-api` command that:
1. Starts the proxy on port 8082
2. Sets `ANTHROPIC_API_KEY` in VS Code settings
3. Sets `ANTHROPIC_BASE_URL` to proxy URL in VS Code settings
4. Verifies connectivity

**Help text update:**

```
Providers:
  ollama          Ollama Cloud (OLLAMA_API_KEY required)
  ollama-local    Local Ollama instance (no auth)
  nvidia          NVIDIA NIM
  openrouter      OpenRouter
  openai-compat   Any OpenAI-compatible endpoint (OPENAI_COMPAT_API_KEY + OPENAI_COMPAT_BASE_URL)
  anthropic       Anthropic OAuth subscription (no proxy)
  anthropic-api   Anthropic API key (through proxy, ANTHROPIC_API_KEY required)
```

## Section 4: Fresh Environment Setup Guide

### New File: `docs/getting-started.md`

#### What is claude-with?

`claude-with` is a CLI tool that launches Claude Code with provider-specific environment variables set for that session only. It does not modify your shell profile or global config. It's a wrapper — you run it instead of `claude`, and it sets up the right env vars, then launches `claude` (or any other command) with those vars.

#### Installation

```bash
cd liberated-claude-code
uv sync                        # installs the proxy + all dependencies
uv pip install -e ./claude_with  # installs the claude-with CLI
```

After this, `claude-with` is available in your terminal from any directory. It's a Python package installed into your virtual environment — as long as that venv is active, the command works everywhere. (If you use `uv`, it's available whenever `uv run` can find it.)

#### `claude-with init` — what it does

`claude-with init` creates a `.claude-with.toml` file in your current project directory. This file tells `claude-with` which models to use when you run it in that project. It can work three ways:

**Interactive mode** (no flags) — prompts you for model names:
```bash
cd my-project
claude-with init
# Prompts:
#   Large (Opus) model [glm5.1]:
#   Medium (Sonnet) model [kimi-k2.5]:
#   Small (Haiku) model [step-3.5-flash]:
#   Save as named profile? [y/N]:
```

**Inline mode** (with flags) — no prompts, writes directly:
```bash
cd my-project
claude-with init --large employer-model-large --medium employer-model-medium --small employer-model-small
```

**Profile reference mode** — point to a named profile you've already saved:
```bash
cd my-project
claude-with init --profile oculus
# Writes: profile = "oculus" to .claude-with.toml
```

After running `init`, any `claude-with` invocation in that project directory will pick up the config automatically. No need to specify `--large`/`--medium`/`--small` every time.

#### `claude-with <provider>` — launching sessions

Run from any directory. The command:
1. Reads config (CLI flags → `.claude-with.toml` → `~/.config/claude-with/config.toml` → defaults)
2. Sets environment variables for that process only (they don't persist after exit)
3. Launches `claude` (or whatever command you specify after `--`)

Two terminals can run two different providers simultaneously.

```bash
# With inline model names:
claude-with openai-compat --large employer-model-large --medium employer-model-medium --small employer-model-small

# With a profile (defined in ~/.config/claude-with/config.toml):
claude-with ollama --profile oculus

# Launch VS Code instead of Claude CLI:
claude-with ollama --profile oculus -- code .

# Just the defaults (uses config.toml defaults):
claude-with ollama

# Anthropic subscription (no proxy):
claude-with anthropic

# Anthropic API key through proxy:
claude-with anthropic-api --large claude-opus-4-7 --medium claude-sonnet-4-6 --small claude-haiku-4-5
```

#### `~/.config/claude-with/config.toml` — central config

This is where profiles and defaults live. Edit it manually or use `claude-with init --save-profile <name>` to append.

```toml
[defaults.openai_compatible]
large = "employer-model-large"
medium = "employer-model-medium"
small = "employer-model-small"

[defaults.ollama_cloud]
large = "glm5.1"
medium = "kimi-k2.5"
small = "step-3.5-flash"

[defaults.ollama_local]
large = "qwen3-32b"
medium = "qwen3-14b"
small = "qwen3-4b"

[profiles.employer]
provider = "openai_compatible"
large = "employer-model-large"
medium = "employer-model-medium"
small = "employer-model-small"

[profiles.oculus]
provider = "ollama_cloud"
large = "glm5.1"
medium = "kimi-k2.5"
small = "step-3.5-flash"

[proxy]
port = 8082
host = "127.0.0.1"
```

#### `~/.config/claude-with/.env` — API keys

```bash
OPENAI_COMPAT_API_KEY=sk-your-employer-key
OPENAI_COMPAT_BASE_URL=https://employer-litellm.internal/v1
OLLAMA_API_KEY=your-ollama-key
NVIDIA_NIM_API_KEY=your-nvidia-key
OPENROUTER_API_KEY=your-openrouter-key
ANTHROPIC_API_KEY=your-anthropic-key
```

Key lookup order: shell env var → macOS Keychain → this `.env` file.

#### `claude-with list` — see what's configured

```bash
claude-with list
# Shows all profiles, defaults, and provider settings
```

#### Full walkthrough: new machine, zero to running

```bash
# 1. Clone and install
git clone <repo-url> && cd liberated-claude-code
uv sync
uv pip install -e ./claude_with

# 2. Set up API keys
mkdir -p ~/.config/claude-with
cat > ~/.config/claude-with/.env << 'EOF'
OPENAI_COMPAT_API_KEY=sk-your-employer-key
OPENAI_COMPAT_BASE_URL=https://employer-litellm.internal/v1
OLLAMA_API_KEY=your-ollama-key
EOF

# 3. Create profiles
cat > ~/.config/claude-with/config.toml << 'EOF'
[profiles.employer]
provider = "openai_compatible"
large = "employer-model-large"
medium = "employer-model-medium"
small = "employer-model-small"

[profiles.oculus]
provider = "ollama_cloud"
large = "glm5.1"
medium = "kimi-k2.5"
small = "step-3.5-flash"

[proxy]
port = 8082
EOF

# 4. Start the proxy (in a separate terminal)
uv run uvicorn server:app --host 127.0.0.1 --port 8082

# 5. Launch Claude Code with employer's models
claude-with openai-compat --profile employer

# Or with Ollama Cloud
claude-with ollama --profile oculus

# 6. (Optional) Per-project config
cd my-project
claude-with init --profile employer
# Now just `claude-with openai-compat` in this directory uses employer models
```

#### Troubleshooting

- **"command not found: claude-with"** → Run `uv pip install -e ./claude_with` from the repo root, or make sure your venv is active.
- **"Proxy not responding on port 8082"** → Start it: `uv run uvicorn server:app --host 127.0.0.1 --port 8082`
- **"OPENAI_COMPAT_API_KEY not set"** → Add it to `~/.config/claude-with/.env` or export it in your shell.
- **"Model not found at endpoint"** → Verify the model name matches what your endpoint serves. Check your endpoint's `/v1/models` endpoint.

## Completion Checklist

- [ ] OpenAI-compat provider implemented and tested
- [ ] Anthropic naming split implemented in CLI and switch script
- [ ] Translation layer documentation written
- [ ] Getting started guide written
- [ ] All existing tests still pass
- [ ] README updated with links to new docs