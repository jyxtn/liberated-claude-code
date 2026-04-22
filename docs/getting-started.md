# Getting Started with claude-with

## What is claude-with?

`claude-with` is a CLI tool that launches Claude Code with provider-specific environment variables set for that session only. It does not modify your shell profile or global config. It's a wrapper — you run it instead of `claude`, and it sets up the right env vars, then launches `claude` (or any other command) with those vars.

The env vars don't persist after the command exits. Two terminals can run two different providers simultaneously.

## Installation

```bash
cd liberated-claude-code
uv sync                        # installs the proxy + all dependencies
uv pip install -e ./claude_with  # installs the claude-with CLI
```

After this, `claude-with` is available in your terminal from any directory. It's a Python package installed into your virtual environment — as long as that venv is active, the command works everywhere. (If you use `uv`, it's available whenever `uv run` can find it.)

## `claude-with init` — Project Configuration

`claude-with init` creates a `.claude-with.toml` file in your current project directory. This file tells `claude-with` which models to use when you run it in that project. It can work three ways:

### Interactive mode (no flags)

Prompts you for model names:

```bash
cd my-project
claude-with init
# Prompts:
#   Large (Opus) model [glm5.1]:
#   Medium (Sonnet) model [kimi-k2.5]:
#   Small (Haiku) model [step-3.5-flash]:
#   Save as named profile? [y/N]:
```

### Inline mode (with flags)

No prompts, writes directly:

```bash
cd my-project
claude-with init --large employer-model-large --medium employer-model-medium --small employer-model-small
```

### Profile reference mode

Point to a named profile you've already saved:

```bash
cd my-project
claude-with init --profile oculus
# Writes: profile = "oculus" to .claude-with.toml
```

After running `init`, any `claude-with` invocation in that project directory will pick up the config automatically. No need to specify `--large`/`--medium`/`--small` every time.

## `claude-with <provider>` — Launching Sessions

Run from any directory. The command:

1. Reads config (CLI flags → `.claude-with.toml` → `~/.config/claude-with/config.toml` → defaults)
2. Sets environment variables for that process only
3. Launches `claude` (or whatever command you specify after `--`)

```bash
# With inline model names:
claude-with openai-compat --large employer-model-large --medium employer-model-medium --small employer-model-small

# With a profile (defined in ~/.config/claude-with/config.toml):
claude-with ollama --profile oculus

# Launch VS Code instead of Claude CLI:
claude-with ollama --profile oculus -- code .

# Just the defaults (uses config.toml defaults):
claude-with ollama

# Anthropic OAuth subscription (no proxy):
claude-with anthropic

# Anthropic API key through proxy (tier routing):
claude-with anthropic-api --large claude-opus-4-7 --medium claude-sonnet-4-6 --small claude-haiku-4-5

# Local Ollama:
claude-with ollama-local --large qwen3-32b --medium qwen3-14b --small qwen3-4b
```

## Providers

| Provider | Command | Requires API Key | Format |
|---|---|---|---|
| Ollama Cloud | `claude-with ollama` | `OLLAMA_API_KEY` | Anthropic pass-through |
| Ollama Local | `claude-with ollama-local` | No | Anthropic pass-through |
| NVIDIA NIM | `claude-with nvidia` | `NVIDIA_NIM_API_KEY` | Anthropic→OpenAI |
| OpenRouter | `claude-with openrouter` | `OPENROUTER_API_KEY` | Anthropic→OpenAI |
| OpenAI-Compat | `claude-with openai-compat` | `OPENAI_COMPAT_API_KEY` + `OPENAI_COMPAT_BASE_URL` | Anthropic→OpenAI |
| Anthropic (sub) | `claude-with anthropic` | No (OAuth) | Direct (no proxy) |
| Anthropic (API) | `claude-with anthropic-api` | `ANTHROPIC_API_KEY` | Through proxy |

**Anthropic naming note:** `claude-with anthropic` uses your Anthropic OAuth subscription directly — no proxy needed. `claude-with anthropic-api` routes through the proxy using an API key, enabling per-tier model routing and optimizations.

## `~/.config/claude-with/config.toml` — Central Config

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

## `~/.config/claude-with/.env` — API Keys

```bash
OPENAI_COMPAT_API_KEY=sk-your-employer-key
OPENAI_COMPAT_BASE_URL=https://employer-litellm.internal/v1
OLLAMA_API_KEY=your-ollama-key
NVIDIA_NIM_API_KEY=your-nvidia-key
OPENROUTER_API_KEY=your-openrouter-key
ANTHROPIC_API_KEY=your-anthropic-key
```

Key lookup order: shell env var → macOS Keychain → this `.env` file.

## `claude-with list` — See What's Configured

```bash
claude-with list
# Shows all profiles, defaults, and provider settings
```

## Full Walkthrough: New Machine, Zero to Running

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

## Troubleshooting

- **"command not found: claude-with"** — Run `uv pip install -e ./claude_with` from the repo root, or make sure your venv is active.
- **"Proxy not responding on port 8082"** — Start it: `uv run uvicorn server:app --host 127.0.0.1 --port 8082`
- **"OPENAI_COMPAT_API_KEY not set"** — Add it to `~/.config/claude-with/.env` or export it in your shell.
- **"OPENAI_COMPAT_BASE_URL not set"** — Set it in your `.env` file or shell: `export OPENAI_COMPAT_BASE_URL="https://your-endpoint/v1"`
- **"Model not found at endpoint"** — Verify the model name matches what your endpoint serves. Check your endpoint's `/v1/models` endpoint.
- **"Provider 'openai_compatible' not found"** — Make sure you're running the latest version of the proxy. `git pull && uv sync`.