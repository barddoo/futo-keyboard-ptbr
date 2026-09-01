# futo-keyboard-ptbr

A privacy-respecting, on-device keyboard language model for Brazilian Portuguese
(pt-BR), built for [FUTO Keyboard](https://keyboard.futo.tech/).

This repo trains a small GPT (decoder-only) language model on a pt-BR corpus,
exports it to the GGUF format with the exact metadata FUTO Keyboard's loader
expects, and validates the result so it can be dropped straight into the app as
a keyboard model. Everything runs locally or on your own hardware — no cloud
inference, no telemetry.

## Overview

FUTO Keyboard supports pluggable on-device language models for next-word
prediction, autocorrect, and swipe (gesture) typing. This project reproduces
that model from scratch in Brazilian Portuguese:

- **~42M-parameter** decoder-only transformer — same scale as FUTO's official
  ml4 model (512 embedding dim, 8 layers).
- **Standard `llama` architecture** (RMSNorm, RoPE, SiLU-gated FFN, tied
  embeddings) so it runs on FUTO's bundled llama.cpp unchanged.
- **FUTO-compatible SentencePiece tokenizer**: inverted-space (whitespace is a
  suffix), BOS hardcoded to token id 1, and the `<XBU>/<XBC>/<XEC>/<XC0>/<CHAR_A..Z>`
  control tokens that drive autocorrect and swipe at runtime.
- **Correction-format training data**: prompts with simulated mistyped keys and
  gesture (swipe) input teach the model to correct words, not just predict the
  next one.
- **Validated GGUF**: a post-quantization checker verifies every key, tensor,
  and vocab invariant FUTO's loader depends on before you ship it.

## Model specs

| Property          | Value                                                         |
| ----------------- | ------------------------------------------------------------ |
| Architecture      | `llama` (RMSNorm, RoPE θ=10000, SiLU-gated FFN, tied lm_head) |
| Parameters        | ~42M                                                          |
| Embedding dim     | 512                                                           |
| Layers / heads    | 8 / 8                                                         |
| Context length    | 256 tokens                                                    |
| Vocabulary        | 16,000 (SentencePiece BPE, inverted space, byte fallback)     |
| Tokenizer type    | `sentencepiece` (`keyboardlm.ext_tokenizer_data`)             |
| Training data     | pt-BR corpora → cleaned sentences → correction/swipe samples  |
| Final artifact    | `pt_br_keyboard_q6k.gguf` (Q6_K, ~40 MB)                      |

## Pipeline

The `Makefile` drives the whole pipeline; each numbered stage owns one step.

```
01_data/   corpus download + cleanup + tokenizer + correction/swipe data
02_train/  model definition + PyTorch training (CUDA / ROCm / MPS)
03_convert/ PyTorch → GGUF export + llama.cpp quantization (F32 → Q6_K)
04_inject/  validate the final GGUF against FUTO's loader requirements
```

### 1. Data (`01_data/`)

- `download.py` — streams pt-BR corpora (curated pt-BR, Wikipedia pt,
  conversational/instruct) as plain text, ~500M-token target.
- `prepare.py` — sentence-splits, filters out non-keyboard-like prose, and
  deduplicates into `train.txt` / `valid.txt`.
- `train_tokenizer.py` — trains the SentencePiece BPE with FUTO's conventions
  and splices the control tokens into the vocab after `<unk>`.
- `make_correction_data.py` — a faithful port of FUTO's
  `TrainingDataGenerator.kt`, producing the `<XBU><CHAR_..><XBC>word <XEC>`
  autocorrect samples and `<XC0>` swipe samples mixed into plain next-word lines.

### 2. Training (`02_train/`)

- `model.py` — `KeyboardLM`, a GPT matching FUTO's llama inference graph
  exactly (no extra heads: autocorrect/swipe are learned behaviors over the
  control tokens).
- `dataset.py` — each sample mirrors FUTO's runtime: `[BOS] + encode(line + " ")`
  so every word carries its whitespace-suffix token.
- `train.py` — AdamW + cosine LR with warmup, bf16 autocast, `torch.compile`,
  periodic checkpoints, and validation on `valid.txt`. Runs on NVIDIA CUDA,
  AMD ROCm, or Apple MPS.

### 3. Conversion (`03_convert/`)

- `export_to_gguf.py` — writes a standard `llama`-arch GGUF with the
  `keyboardlm.*` metadata (languages, features, embedded SentencePiece model)
  and FUTO's required tensor layout.
- `quantize.sh` — clones and builds llama.cpp (Metal), then quantizes the F32
  export to Q6_K (~40 MB).

### 4. Validation (`04_inject/`)

- `inject_metadata.py` — verifies every KV key/type, tensor prefix, block
  tensor, vocab convention (BOS=1, `<CHAR_A..Z>` consecutive, valid
  SentencePiece blob, etc.) that FUTO's `ModelMeta.cpp` and llama.cpp rely on.
  Exit code 0 means the file should load in FUTO Keyboard.

## Getting started

### macOS (Apple Silicon)

```sh
make setup        # installs uv, Python deps, verifies MPS, installs cmake
make              # runs the full pipeline: data → train → convert → validate
```

The first `make` run downloads the corpora and trains the model, which takes a
while. Individual stages can be run directly:

```sh
make 01_data/download.py       # or any of the file targets in the Makefile
```

### Remote training on an AMD GPU

Training runs locally by default, but the heavy lifting can be pushed to a
ROCm box over SSH:

```sh
cp .envrc.example .envrc       # set AMD_HOST=user@192.168.x.x
direnv allow                   # or: export AMD_HOST=user@192.168.x.x

make train-remote              # rsyncs the project, sets up ROCm, trains in background
```

Monitor with `ssh $AMD_HOST tail -f ~/pt-br-futo/train.log` and sync the
checkpoints back when done:

```sh
rsync -avz "$AMD_HOST:~/pt-br-futo/02_train/checkpoints/" 02_train/checkpoints/
rsync -avz "$AMD_HOST:~/pt-br-futo/01_data/tokenizer/" 01_data/tokenizer/
```

Then finish the pipeline locally:

```sh
make 03_convert/pt_br_keyboard_q6k.gguf
make validate
```

### Custom hardware

- **ROCm**: run `scripts/setup_rocm.sh` on the target machine, or install with
  `pip install torch --index-url https://download.pytorch.org/whl/rocm7.2`
  (set `ROCM_VER` to match your ROCm release).
- **Apple Silicon**: `make setup` handles everything; training uses MPS.

## Installing in FUTO Keyboard

1. Build the model: `make` (or the conversion + validate targets above).
2. Copy `03_convert/pt_br_keyboard_q6k.gguf` to your device.
3. In FUTO Keyboard, go to **Settings → Keyboard models → Import** and select
   the `.gguf` file.

## Configuration

Stage behavior is overridable via environment variables:

| Variable                     | Default                              | Purpose                                    |
| ---------------------------- | ------------------------------------ | ------------------------------------------ |
| `AMD_HOST`                   | *(unset)*                            | SSH target for `make train-remote`         |
| `KEYBOARD_MODEL`             | `03_convert/pt_br_keyboard_q6k.gguf` | GGUF to validate                           |
| `KEYBOARD_CKPT`              | `02_train/checkpoints/best.pt`       | Checkpoint to export                       |
| `KEYBOARD_SP_MODEL`          | `01_data/tokenizer/pt_br_keyboard.model` | SentencePiece model to embed           |
| `KEYBOARD_OUT`               | `03_convert/pt_br_keyboard_f32.gguf` | F32 GGUF output path                       |
| `KEYBOARD_FEATURES`          | `base_v1 inverted_space`             | Features declared in `keyboardlm.features` |
| `KEYBOARD_TOKENIZER_VOCAB_SIZE` | `16000`                           | Tokenizer vocab size                       |
| `COMPILE`                    | `1`                                  | Set `0` to disable `torch.compile` (flaky on some ROCm setups) |

### Enabling autocorrect & swipe features

The base model ships with `base_v1 inverted_space`. To teach the keyboard's
autocorrect and swipe behaviors, train with the correction-format data
(`make_correction_data.py` already generates it) and export with:

```sh
KEYBOARD_FEATURES="base_v1 inverted_space xbu_char_autocorrect_v1 xc0_swipe_typing_v1 char_embed_mixing_v1" \
  python3 03_convert/export_to_gguf.py
```

`xbu_char_autocorrect_v1` and `xc0_swipe_typing_v1` must be declared together
with `char_embed_mixing_v1` — FUTO only allocates its embedding-mix buffer when
that feature is present.

## Repository layout

```
Makefile                      pipeline driver (data → train → convert → validate)
scripts/                      environment setup + remote training helper
01_data/                      corpus + tokenizer + correction data
02_train/                     model, dataset, training loop
03_convert/                   GGUF export + quantization
04_inject/                    GGUF validation
```

## License

Model weights and code are provided as-is. The corpora used are each subject
to their own licenses (Wikipedia, MADRAS1 corpus-ptbr, Pt-Corpus-Instruct) —
review them before redistributing the training data or derived models.
