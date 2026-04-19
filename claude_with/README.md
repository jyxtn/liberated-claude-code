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