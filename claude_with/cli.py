"""CLI for claude-with - ephemeral switching for Claude Code."""

import atexit
import importlib.util
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import click

from .config import Config, ModelTier
from .keys import get_api_key
from .providers import Provider, ProviderConfig


def _resolve_proxy_dir() -> Path | None:
    """Find the directory containing the installed api/ package.

    Returns the parent directory of api/ (typically site-packages),
    or None if the api package cannot be located.
    """
    spec = importlib.util.find_spec("api")
    if spec and spec.submodule_search_locations:
        return Path(spec.submodule_search_locations[0]).parent
    return None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_proxy(port: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.25)
    return False


def _start_ephemeral_proxy(
    proxy_project_path: Path,
    port: int,
    proxy_env: dict[str, str],
) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "--directory",
            str(proxy_project_path),
            "uvicorn",
            "server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=proxy_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def _cleanup() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(_cleanup)
    return proc


def _with_provider_prefix(model: str, provider_name: str) -> str:
    """Prepend provider name to model if not already prefixed."""
    if "/" in model:
        return model
    return f"{provider_name}/{model}"


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


def _launch(
    provider_name, provider_enum, large, medium, small, profile_name, command, args
):
    """Launch Claude Code with the specified provider and models."""
    config = Config.load()
    provider_config = ProviderConfig.get(provider_enum)

    # Auto-resolve profile from local .claude-with.toml if no --profile given
    if not profile_name:
        local_profile = config.get_profile()
        if local_profile:
            profile_name = local_profile.name

    # Load project-local [env] overrides
    project_env = config.get_project_env()

    # Build Claude Code environment
    env = os.environ.copy()

    # Apply project-local env overrides (lowest priority among explicit sources)
    for key, value in project_env.items():
        env.setdefault(key, value)

    # Resolve models before building proxy env
    models = _resolve_models(
        config, provider_config, profile_name, large, medium, small
    )

    if provider_config.requires_proxy:
        # Collect provider credentials
        provider_api_key = None
        if provider_config.env_var:
            provider_api_key = get_api_key(provider_config.env_var)
            if provider_api_key:
                env[provider_config.env_var] = provider_api_key

        openai_compat_base_url = None
        if provider_enum == Provider.OPENAI_COMPAT:
            openai_compat_base_url = (
                os.environ.get("OPENAI_COMPAT_BASE_URL")
                or project_env.get("OPENAI_COMPAT_BASE_URL")
                or get_api_key("OPENAI_COMPAT_BASE_URL")
                or provider_config.base_url
            )
            if openai_compat_base_url:
                env["OPENAI_COMPAT_BASE_URL"] = openai_compat_base_url

        if config.proxy_project_path:
            # Ephemeral proxy: spin up a fresh proxy process for this session
            port = _find_free_port()
            proxy_env = _build_proxy_env(
                os.environ.copy(),
                provider_config,
                models,
                provider_api_key,
                openai_compat_base_url,
            )
            print(f"  Starting proxy on port {port}...")
            _start_ephemeral_proxy(
                Path(config.proxy_project_path).expanduser(), port, proxy_env
            )
            if not _wait_for_proxy(port):
                print("✗ Proxy failed to start")
                sys.exit(1)
            proxy_url = f"http://127.0.0.1:{port}"
        else:
            proxy_url = f"http://{config.proxy_host}:{config.proxy_port}"

        env["ANTHROPIC_BASE_URL"] = proxy_url
        env["ANTHROPIC_API_KEY"] = "freecc"
    else:
        # Native Anthropic — remove any proxy settings
        env.pop("ANTHROPIC_BASE_URL", None)
        api_key = get_api_key("ANTHROPIC_API_KEY")
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key

    # Set model env vars for Claude Code, with provider prefix so the proxy
    # can route by provider type without needing MODEL_* vars configured.
    prefix = provider_config.name
    if large_model := models.get_large():
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = _with_provider_prefix(large_model, prefix)
    if medium_model := models.get_medium():
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = _with_provider_prefix(medium_model, prefix)
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = _with_provider_prefix(medium_model, prefix)
    if small_model := models.get_small():
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = _with_provider_prefix(small_model, prefix)

    # Resolve command
    cmd = command or "claude"

    # Proxy-based providers need --bare so Claude Code uses ANTHROPIC_API_KEY
    # instead of OAuth (which would bypass the proxy)
    cmd_list = [cmd]
    if provider_config.requires_proxy and cmd == "claude":
        cmd_list.append("--bare")
    cmd_list.extend(args)

    print(f"⚡ Provider: {provider_name}")
    if models.get_large():
        print(f"  large: {models.get_large()}")
    if models.get_medium():
        print(f"  medium: {models.get_medium()}")
    if models.get_small():
        print(f"  small: {models.get_small()}")

    result = subprocess.run(cmd_list, env=env)
    sys.exit(result.returncode)


def _build_proxy_env(
    base_env: dict[str, str],
    provider_config: ProviderConfig,
    models: ModelTier,
    provider_api_key: str | None,
    openai_compat_base_url: str | None,
) -> dict[str, str]:
    """Build the environment dict for the ephemeral proxy subprocess."""
    env = base_env

    # Clear settings that should not bleed through from the parent process
    for key in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "MODEL",
        "MODEL_OPUS",
        "MODEL_SONNET",
        "MODEL_HAIKU",
    ):
        env.pop(key, None)

    # Provider credentials
    if provider_config.env_var and provider_api_key:
        env[provider_config.env_var] = provider_api_key
    if openai_compat_base_url:
        env["OPENAI_COMPAT_BASE_URL"] = openai_compat_base_url

    # Model routing: proxy maps these to provider/model strings.
    # Setting them means the proxy doesn't need its own MODEL_* config.
    prefix = provider_config.name
    fallback: str | None = None
    if medium_model := models.get_medium():
        m = _with_provider_prefix(medium_model, prefix)
        env["MODEL_SONNET"] = m
        fallback = m
    if large_model := models.get_large():
        m = _with_provider_prefix(large_model, prefix)
        env["MODEL_OPUS"] = m
        fallback = fallback or m
    if small_model := models.get_small():
        m = _with_provider_prefix(small_model, prefix)
        env["MODEL_HAIKU"] = m
        fallback = fallback or m
    if fallback:
        env["MODEL"] = fallback

    return env


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
def init_command(
    large: str | None,
    medium: str | None,
    small: str | None,
    profile_name: str | None,
    save_profile: str | None,
):
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
