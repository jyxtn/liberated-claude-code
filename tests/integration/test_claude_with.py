"""Integration tests for claude-with CLI."""

import os
import tempfile
from pathlib import Path

import pytest

from claude_with.config import Config, ModelTier, Profile


def test_config_loads_defaults():
    """Config loads without any files."""
    config = Config.load()
    assert isinstance(config.defaults, dict)
    assert isinstance(config.profiles, dict)


def test_config_merges_toml():
    """Config correctly merges TOML files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.toml"
        config_path.write_text("""
[defaults.ollama_cloud]
large = "test-large"
medium = "test-medium"
small = "test-small"

[profiles.test-profile]
provider = "ollama_cloud"
large = "profile-large"
medium = "profile-medium"
small = "profile-small"
""")
        config = Config()
        config._merge_toml(config_path)

        assert "ollama_cloud" in config.defaults
        assert config.defaults["ollama_cloud"].get_large() == "test-large"
        assert "test-profile" in config.profiles
        assert config.profiles["test-profile"].models.get_large() == "profile-large"


def test_model_tier_aliases():
    """ModelTier supports both large/opus, medium/sonnet, small/haiku."""
    tier = ModelTier(opus="opus-model", medium="sonnet-model", haiku="haiku-model")

    assert tier.get_large() == "opus-model"
    assert tier.get_medium() == "sonnet-model"
    assert tier.get_small() == "haiku-model"

    # Explicit takes precedence
    tier2 = ModelTier(large="explicit-large", opus="alias-large")
    assert tier2.get_large() == "explicit-large"


def test_profile_resolution():
    """Profile resolution follows priority: CLI > profile > defaults."""
    config = Config()
    config.defaults["ollama_cloud"] = ModelTier(large="default-large")
    config.profiles["test"] = Profile(
        name="test",
        provider="ollama_cloud",
        models=ModelTier(large="profile-large"),
    )

    # Without profile, use defaults
    tier = config.get_default_tier("ollama_cloud")
    assert tier.get_large() == "default-large"

    # With profile, use profile
    profile = config.get_profile("test")
    assert profile.models.get_large() == "profile-large"