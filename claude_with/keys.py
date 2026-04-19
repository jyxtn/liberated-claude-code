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
