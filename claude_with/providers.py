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
