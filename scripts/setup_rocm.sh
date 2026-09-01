#!/usr/bin/env bash
# Run this ON the AMD GPU machine to set up the training environment.
# Copy it there with: scp scripts/setup_rocm.sh user@amd-host:~/
set -euo pipefail

echo "=== ROCm training setup ==="

# ROCm installs to /opt/rocm* and is usually not on PATH
for d in /opt/rocm /opt/rocm-* /opt/rocm*/bin; do
  if [ -d "$d" ] && [[ ":$PATH:" != *":$d:"* ]]; then
    export PATH="$d:$PATH"
  fi
done

# Verify ROCm is available
echo "[1/4] Checking ROCm …"
if ! command -v rocm-smi &>/dev/null; then
  echo "ERROR: rocm-smi not found — is ROCm installed?"
  exit 1
fi
rocm-smi --version
rocm-smi --showid

# Python venv
echo "[2/4] Creating venv …"
python3 -m venv .venv
source .venv/bin/activate

# Install ROCm PyTorch — set ROCM_VER to match (or exceed) your installed ROCm
# (this machine reports ROCm 7.2.4). torch wheels bundle their own ROCm
# runtime, so rocm7.x wheels are the safest match.
ROCM_VER="${ROCM_VER:-rocm7.2}"
echo "[3/4] Installing PyTorch for $ROCM_VER …"
pip install torch torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/${ROCM_VER}"

# Remaining deps (standard PyPI)
pip install sentencepiece protobuf datasets tokenizers gguf numpy tqdm

# Verify
echo "[4/4] Verifying …"
python3 -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  CUDA/ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Setup complete. Run training with:"
echo "  source .venv/bin/activate && python3 02_train/train.py"
