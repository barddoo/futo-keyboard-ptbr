#!/usr/bin/env bash
# One-time environment setup for macOS Apple Silicon
set -euo pipefail

echo "=== pt-br-futo setup ==="

# Ensure uv is available
if ! command -v uv &>/dev/null; then
  echo "Installing uv …"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
fi

# Create venv + install deps
echo "[1/3] Installing Python packages …"
uv venv .venv
uv pip install -r requirements.txt

# Verify PyTorch MPS
echo "[2/3] Verifying PyTorch MPS …"
uv run python3 -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  MPS available: {torch.backends.mps.is_available()}')
print(f'  CUDA available: {torch.cuda.is_available()}')
"

# cmake for llama.cpp quantize build
echo "[3/3] Checking cmake …"
if ! command -v cmake &>/dev/null; then
  echo "  Installing cmake via brew …"
  brew install cmake
else
  echo "  cmake: $(cmake --version | head -1)"
fi

echo ""
echo "Setup complete. Activate venv then run:"
echo "  source .venv/bin/activate"
echo "  make"
