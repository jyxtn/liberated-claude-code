"""CLI for claude-with - ephemeral switching for Claude Code."""

import os
import subprocess
import sys
from pathlib import Path

import click

from .config import Config, ModelTier
from .keys import get_api_key
from .providers import Provider, ProviderConfig


PROVIDERS = {
    "ollama": Provider.OLLAMA_CLOUD,
    "ollama-local": Provider.OLLAMA_LOCAL,
    "nvidia": Provider.NVIDIA_NIM,
    "openrouter": Provider.OPENROUTER,
    "openai-compat": Provider.OPENAI_COMPAT,
    "anthropic": Provider.ANTHROPIC,
    "anthropic-api": Provider.ANTHROPIC_API,
}


@click.group()
@click.version_option()
def main():
    """claude-with - Ephemeral switching for Claude Code.

    Launch Claude Code with different providers and model configurations.
    Each invocation sets environment variables for that session only.

    \b
    Examples:
        claude-with ollama --large glm5 --medium kimi-k2.5 --small step-3.5-flash
        claude-with ollama --profile oculus
        claude-with ollama --profile oculus -- code .
        claude-with anthropic-api --large claude-opus-4-7
    """
    pass


def _create_provider_command(name, provider_enum):
    """Create a Click command for a provider."""
    @click.command(name=name)
    @click.option("--large", "--opus", "large", help="Large/Opus tier model")
    @click.option("--medium", "--sonnet", "medium", help="Medium/Sonnet tier model")
    @click.option("--small", "--haiku", "small", help="Small/Haiku tier model")
    @click.option("--profile", "-p", "profile_name", help="Named profile from config")
    @click.option("--command", "-c", "command", help="Command to run (default: claude)")
    @click.argument("args", nargs=-1)
    def provider_cmd(large, medium, small, profile_name, command, args):
        _launch(name, provider_enum, large, medium, small, profile_name, command, args)
    return provider_cmd


# Register each provider as a subcommand
for _name, _enum in PROVIDERS.items():
    main.add_command(_create_provider_command(_name, _enum))


def _launch(provider_name, provider_enum, large, medium, small, profile_name, command, args):
    """Launch Claude Code with the specified provider and models."""
    config = Config.load()
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

    # Set provider-specific base URL override for openai-compat
    if provider_enum == Provider.OPENAI_COMPAT:
        base_url = os.environ.get("OPENAI_COMPAT_BASE_URL", provider_config.base_url)
        if base_url:
            env["OPENAI_COMPAT_BASE_URL"] = base_url

    if models.get_large():
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = models.get_large()
    if models.get_medium():
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = models.get_medium()
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = models.get_medium()  # Sonnet for subagents
    if models.get_small():
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = models.get_small()

    # Resolve command
    cmd = command or "claude"

    # Run command
    cmd_list = [cmd, *args]
    print(f"⚡ Provider: {provider_name}")
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
