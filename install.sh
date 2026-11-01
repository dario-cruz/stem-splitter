#!/usr/bin/env bash
# Cross-platform installer for Stem Splitter (Linux / macOS / WSL)
# Usage: ./install.sh [--local] [--skip-models]

set -euo pipefail

MODE="global"
SKIP_MODELS=false

for arg in "$@"; do
    case "$arg" in
        --local|-l) MODE="local" ;;
        --global|-g) MODE="global" ;;
        --skip-models) SKIP_MODELS=true ;;
        -h|--help)
            echo "Usage: ./install.sh [--global | --local] [--skip-models]"
            exit 0
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo ""
echo "======================================================"
echo "  Stem Splitter Installer ($MODE mode)"
echo "======================================================"
echo ""

# 1. Check / Install uv
if ! command -v uv &>/dev/null; then
    echo "[1/4] 'uv' not found. Installing Astral uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/4] Found uv: $(uv --version)"
fi

# 2. Check / Install FFmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "[2/4] 'ffmpeg' not found. Attempting to install..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y ffmpeg
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm ffmpeg
    elif command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "Warning: Could not auto-install ffmpeg. Please install ffmpeg via your system package manager."
    fi
else
    echo "[2/4] Found FFmpeg on PATH"
fi

# 3. Install stem-splitter
if [[ "$MODE" == "global" ]]; then
    echo "[3/4] Installing 'stems' & 'stem-splitter' globally via uv..."
    uv sync
    uv tool install --editable . --force
    uv tool update-shell || true
else
    echo "[3/4] Installing local virtual environment (.venv) via uv..."
    uv sync
fi

# 4. Pre-cache offline Demucs weights
if [[ "$SKIP_MODELS" == false ]]; then
    echo "[4/4] Ensuring local offline Demucs v4 weights exist in ./models..."
    uv run stems --download-models
else
    echo "[4/4] Skipping model weight download (--skip-models specified)."
fi

echo ""
echo "======================================================"
echo "  Setup Complete!"
echo "======================================================"
if [[ "$MODE" == "global" ]]; then
    echo "  Restart your shell (or run 'source ~/.bashrc' / 'source ~/.zshrc') and run:"
    echo "    stems song.mp3"
else
    echo "  Run from inside this repository:"
    echo "    uv run stems song.mp3"
fi
echo ""
