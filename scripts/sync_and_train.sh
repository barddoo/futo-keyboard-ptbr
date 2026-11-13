#!/usr/bin/env bash
# Sync project to AMD GPU machine and launch training.
# Usage: AMD_HOST=user@192.168.x.x bash scripts/sync_and_train.sh
set -euo pipefail

AMD_HOST="${AMD_HOST:-}"
REMOTE_DIR="${REMOTE_DIR:-~/pt-br-futo}"

if [[ -z "$AMD_HOST" ]]; then
  echo "ERROR: set AMD_HOST=user@<ip> before running"
  echo "  AMD_HOST=user@192.168.0.x bash scripts/sync_and_train.sh"
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Language to train (langs/<code>.py); paths are relative to the project root
export KEYBOARD_LANG="${KEYBOARD_LANG:-pt}"
SP_MODEL="$(cd "$PROJECT_DIR" && python3 -m langs sp_model)"
CORRECTION="$(cd "$PROJECT_DIR" && python3 -m langs processed_dir)/correction_train.txt"

echo "=== Syncing to $AMD_HOST:$REMOTE_DIR ==="

# Sync everything except venv, generated GGUF files, llama.cpp and the raw
# corpus downloads (the VM regenerates tokenizer + correction data from
# 01_data/processed, which IS synced).
rsync -avz --progress \
  --exclude='.venv/' \
  --exclude='03_convert/*.gguf' \
  --exclude='04_inject/*.gguf' \
  --exclude='llama.cpp/' \
  --exclude='01_data/raw/' \
  --exclude='__pycache__/' \
  "$PROJECT_DIR/" "$AMD_HOST:$REMOTE_DIR/"

echo ""
echo "=== Launching training on $AMD_HOST ==="

# Write the remote script to a temp file and ship it over
cat > /tmp/remote_train.sh << REMOTESCRIPT
set -e
cd $REMOTE_DIR

if [ ! -d .venv ]; then
  echo "Setting up environment …"
  bash scripts/setup_rocm.sh
fi

source .venv/bin/activate
export KEYBOARD_LANG=$KEYBOARD_LANG

python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"

# Generate the FUTO tokenizer if not present (or if it's stale/missing)
if [ ! -f $SP_MODEL ]; then
  echo "Training tokenizer …"
  python3 01_data/train_tokenizer.py
fi

# Generate correction/swipe training data if not synced over
if [ ! -f $CORRECTION ]; then
  echo "Generating correction training data …"
  python3 01_data/make_correction_data.py
fi

nohup python3 02_train/train.py > train.log 2>&1 &
echo "Training started (PID \$!)"
REMOTESCRIPT

scp /tmp/remote_train.sh "$AMD_HOST:/tmp/remote_train.sh"
ssh "$AMD_HOST" bash /tmp/remote_train.sh

echo ""
echo "Training running in background on $AMD_HOST"
echo "Monitor: ssh $AMD_HOST tail -f $REMOTE_DIR/train.log"
echo "Sync back checkpoints (and the tokenizer, if the remote generated it) when done:"
echo "  rsync -avz $AMD_HOST:$REMOTE_DIR/02_train/checkpoints/ 02_train/checkpoints/"
echo "  rsync -avz $AMD_HOST:$REMOTE_DIR/01_data/tokenizer/ 01_data/tokenizer/"
