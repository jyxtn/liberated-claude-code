"""Configuration file parsing for claude-with."""

from __future__ import annotations

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
    proxy_project_path: str | None = None

    @classmethod
    def load(cls) -> Config:
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
            if "project_path" in data["proxy"]:
                self.proxy_project_path = data["proxy"]["project_path"]

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

    def get_project_env(self) -> dict[str, str]:
        """Get project-local [env] overrides from .claude-with.toml."""
        local_path = Path.cwd() / ".claude-with.toml"
        if not local_path.exists():
            return {}
        with open(local_path, "rb") as f:
            data = tomli.load(f)
        return dict(data.get("env", {}))
