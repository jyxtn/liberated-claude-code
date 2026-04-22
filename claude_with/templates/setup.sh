#!/bin/bash
# Initialize claude-with configuration

CONFIG_DIR="$HOME/.config/claude-with"
TEMPLATES_DIR="$(dirname "$0")"

mkdir -p "$CONFIG_DIR"

if [ ! -f "$CONFIG_DIR/config.toml" ]; then
    cp "$TEMPLATES_DIR/config.toml" "$CONFIG_DIR/config.toml"
    echo "✓ Created $CONFIG_DIR/config.toml"
fi

if [ ! -f "$CONFIG_DIR/.env" ]; then
    cp "$TEMPLATES_DIR/.env.example" "$CONFIG_DIR/.env"
    echo "✓ Created $CONFIG_DIR/.env"
    echo "  Edit this file to add your API keys"
fi

echo ""
echo "Configuration initialized. Next steps:"
echo "  1. Edit ~/.config/claude-with/.env to add your API keys"
echo "  2. Run: claude-with ollama --profile oculus"