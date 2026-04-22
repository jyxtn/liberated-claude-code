# Ollama Cloud Proxy + Ephemeral Switching Design

**Date:** 2026-04-18
**Status:** Approved
**Related:** ollama/ollama#15677

## Problem

`ollama launch claude` wrapper doesn't register MCP tools properly. MCP tools work fine when launching `claude` directly. Bug report filed at ollama/ollama#15677.

**Workaround:** Use native `claude` with a proxy that exposes Anthropic-compatible endpoints for ollama cloud models, with multi-tier model routing.

## Solution Overview

Extend `liberated-claude-code` with ollama cloud/local providers, add `claude-with` CLI for ephemeral switching, and support per-tier model routing.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              claude-with                                      │
│  (ephemeral env var setter + command launcher)                              │
│                                                                              │
│  Sets: ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY,                               │
│        ANTHROPIC_DEFAULT_OPUS_MODEL, SONNET_MODEL, HAIKU_MODEL              │
│                                                                              │
│  Reads: CLI flags → project config → named profile → defaults               │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                              │
                    ▼                              ▼
        ┌───────────────────┐          ┌───────────────────┐
        │  claude (TUI)     │          │  code . (VS Code) │
        └───────────────────┘          └───────────────────┘
                    │                              │
                    ▼                              ▼
        ┌───────────────────────────────────────────────────┐
        │           liberated-claude-code proxy             │
        │  (port 8082, extended with ollama providers)     │
│
│  For ollama providers: pass-through to Anthropic-compatible endpoint
│  For other providers: translate Anthropic → OpenAI format
        └───────────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┬───────────────┐
        ▼           ▼           ▼               ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐   ┌─────────────┐
   │ NVIDIA  │ │OpenRouter│ │LM Studio│   │ Ollama Cloud│
   │   NIM   │ │         │ │ (local)  │   │ (NEW)       │
   └─────────┘ └─────────┘ └─────────┘   └─────────────┘
```

## Components

### 1. Ollama Providers

**Files:**
- `providers/ollama_cloud.py` - ollama cloud API (https://ollama.com/v1)
- `providers/ollama_local.py` - local ollama (http://localhost:11434/v1)

**Implementation:**
Ollama v0.14.0+ supports Anthropic's native `/v1/messages` format. No translation needed - just pass-through with model name substitution based on tier.

```python
class OllamaCloudProvider(AnthropicPassthroughProvider):
    def stream_response(self, request: AnthropicRequest) -> Iterator[str]:
        # Replace model name based on tier mapping
        request.model = self.model_map.get(request.model, request.model)
        # Pass through to ollama's Anthropic-compatible endpoint
        yield from self._stream_passthrough(request)
```

### 2. claude-with CLI

**Location:** New directory `liberated-claude-code/claude-with/`, installed alongside the proxy. Single repo for simplicity.

**Command syntax:**
```bash
claude-with <provider> [options] [-- command]

# Provider (required)
ollama          # ollama cloud (OLLAMA_API_KEY required)
ollama-local    # local ollama (localhost:11434)
nvidia          # NVIDIA NIM
openrouter      # OpenRouter
anthropic       # native Anthropic (no proxy)

# Model tier flags
--large MODEL   # opus tier (alias: --opus)
--medium MODEL  # sonnet tier (alias: --sonnet)
--small MODEL   # haiku tier (alias: --haiku)

# Profile flags
--profile NAME  # use named profile from config

# Command to run (optional)
-- command      # run command with env vars set
```

**Examples:**
```bash
# One-off with specific models
claude-with ollama --large glm5 --medium kimi-k2.5 --small step-3.5-flash

# Named profile
claude-with ollama --profile oculus

# Launch VS Code with config
claude-with ollama --profile oculus -- code .

# Run claude directly (default)
claude-with ollama
```

### 3. Config System

**Central config:** `~/.config/claude-with/config.toml`

```toml
# Provider credentials (env var references, NOT actual keys)
[providers.ollama_cloud]
api_key_env = "OLLAMA_API_KEY"
base_url = "https://ollama.com/v1"

# Default model tiers
[defaults.ollama_cloud]
large = "glm5.1"
medium = "kimi-k2.5"
small = "step-3.5-flash"

# Named profiles
[profiles.oculus]
provider = "ollama_cloud"
large = "glm5.1"
medium = "kimi-k2.5"
small = "step-3.5-flash"

# Mixed provider profile
[profiles.mixed]
provider = "mixed"
large = "ollama_cloud/glm5.1"
medium = "ollama_local/qwen3-14b"
small = "open_router/stepfun/step-3.5-flash:free"
```

**Project-local config:** `.claude-with.toml`

```toml
# Minimal: reference named profile
profile = "oculus"

# Or: full override
[ollama_cloud]
large = "glm5.1"
medium = "kimi-k2.5"
small = "step-3.5-flash"
```

**Priority chain:**
```
CLI flags → project-local config → named profile → defaults
```

### 4. Init Helper

```bash
claude-with init [options]

# Interactive mode
claude-with init

# Point to existing profile
claude-with init --profile oculus

# Define models inline
claude-with init --large glm5 --medium kimi-k2.5 --small step-3.5-flash

# Save as new named profile
claude-with init --save-profile new-project-name
```

### 5. Key Storage

**Location:** `~/.config/claude-with/.env`

```bash
OLLAMA_API_KEY=your-key-here
NVIDIA_NIM_API_KEY=your-key-here
OPENROUTER_API_KEY=your-key-here
```

**Precedence:**
```
CLI flags → shell env vars → keychain → ~/.config/claude-with/.env → error
```

## Model Mappings

### Default Tiers

| Tier | Anthropic | Ollama Cloud Default | Purpose |
|------|-----------|----------------------|---------|
| **Large** | Opus | `glm5.1` | Primary conversation, complex reasoning |
| **Medium** | Sonnet | `kimi-k2.5` | Subagents, daily work |
| **Small** | Haiku | `step-3.5-flash` | Explore agent, quick searches |

### Provider Prefixes

| Prefix | Base URL | Auth |
|--------|----------|------|
| `ollama_cloud/...` | `https://ollama.com/v1` | `OLLAMA_API_KEY` |
| `ollama_local/...` | `http://localhost:11434/v1` | None |
| `nvidia_nim/...` | `https://integrate.api.nvidia.com/v1` | `NVIDIA_NIM_API_KEY` |
| `open_router/...` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `lmstudio/...` | `http://localhost:1234/v1` | None |

## Switching Behavior

### Global vs Ephemeral

| Tool | Scope | Use Case |
|------|-------|----------|
| `claude-switch` | Global | VS Code GUI, persistent default |
| `claude-with` | Ephemeral | TUI sessions, terminal-launched VS Code |

### Simultaneous Sessions

```bash
# VS Code (global default)
claude-switch nim              # VS Code uses NIM

# Terminal 1 (ephemeral)
claude-with ollama --profile oculus
claude                         # Uses ollama cloud

# Terminal 2 (ephemeral)
claude-with anthropic
claude                         # Uses native Anthropic

# Terminal 3 (ephemeral, launches VS Code)
claude-with ollama --profile oculus -- code .
# VS Code window uses ollama config
```

## Error Handling

```bash
# Missing API key
✗ OLLAMA_API_KEY not set
  Set it in your shell: export OLLAMA_API_KEY="your-key"
  Or add to ~/.zshrc: export OLLAMA_API_KEY="your-key"

# Model not found
✗ Model 'nonexistent-model' not available on ollama cloud
  Available: glm5.1, kimi-k2.5, step-3.5-flash, ...
  Run: claude-with ollama --list-models

# Proxy not running
✗ Proxy not running on port 8082
  Start it: claude-with proxy start

# Profile not found
✗ Profile 'nonexistent' not found in ~/.config/claude-with/config.toml
  Available profiles: oculus, agentic_ai_eng, mixed

# Mixed provider with missing credential
✗ Profile 'mixed' requires OPENROUTER_API_KEY (for haiku tier)
  Set the missing credential or modify the profile
```

## Implementation Phases

| Phase | Scope | Files |
|-------|-------|-------|
| **1. Ollama providers** | Add pass-through providers to proxy | `providers/ollama_*.py` |
| **2. Config system** | Central + project-local config parsing | `claude-with/config.py` |
| **3. CLI tool** | `claude-with` command with flags | `claude-with/cli.py` |
| **4. Init helper** | `claude-with init` for project setup | `claude-with/init.py` |
| **5. Key storage** | `.env` + keychain support | `claude-with/config.py` |

## Dependencies

- `liberated-claude-code` already has: FastAPI, httpx, loguru
- `claude-with` needs: tomli (TOML parsing), click or argparse (CLI)
- Both can share the same virtual environment

## References

- [Anthropic compatibility - Ollama](https://docs.ollama.com/api/anthropic-compatibility)
- [OpenAI compatibility - Ollama](https://docs.ollama.com/api/openai-compatibility)
- [Tool calling - Ollama](https://docs.ollama.com/capabilities/tool-calling)
- [Claude Code with Anthropic API compatibility - Ollama Blog](http://ollama.com/blog/claude)