#!/usr/bin/env bash
# Quantize the F32 GGUF to Q6_K using llama.cpp
# Run this AFTER export_to_gguf.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

F32_GGUF="$SCRIPT_DIR/pt_br_keyboard_f32.gguf"
Q6K_GGUF="$SCRIPT_DIR/pt_br_keyboard_q6k.gguf"
LLAMACPP_DIR="$PROJECT_DIR/llama.cpp"

# Clone llama.cpp if not present
if [ ! -d "$LLAMACPP_DIR" ]; then
  echo "Cloning llama.cpp …"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMACPP_DIR"
fi

# Build quantize binary with Metal (macOS Apple Silicon)
if [ ! -f "$LLAMACPP_DIR/build/bin/llama-quantize" ]; then
  echo "Building llama.cpp (Metal) …"
  cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" \
    -DGGML_METAL=ON \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMACPP_DIR/build" --target llama-quantize -j"$(sysctl -n hw.logicalcpu)"
fi

if [ ! -f "$F32_GGUF" ]; then
  echo "ERROR: $F32_GGUF not found — run export_to_gguf.py first"
  exit 1
fi

echo "Quantizing to Q6_K …"
# Run from SCRIPT_DIR and use ./-prefixed paths: llama-quantize parses the
# last positional arg as the ftype; the ./ prefix stops a leading "03_..."
# from ever being misread as an integer.
cd "$SCRIPT_DIR"
"$LLAMACPP_DIR/build/bin/llama-quantize" \
  "./pt_br_keyboard_f32.gguf" "./pt_br_keyboard_q6k.gguf" Q6_K

echo "Done: $Q6K_GGUF"
ls -lh "$Q6K_GGUF"
