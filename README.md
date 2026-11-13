# futo-keyboard-ptbr

A privacy-respecting, on-device keyboard language model for Brazilian Portuguese
(pt-BR), built for [FUTO Keyboard](https://keyboard.futo.tech/).

**Model downloads:** [huggingface.co/barddoo/futo-keyboard-ptbr](https://huggingface.co/barddoo/futo-keyboard-ptbr)

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

## Languages

The pipeline is language-agnostic; each language is one config file in
`langs/` (corpus sources, artifact names, `keyboardlm.languages` code, GGUF
name/description, key layout for simulated typos). Pick one with
`KEYBOARD_LANG` (default `pt`, so plain `make` builds the pt-BR model exactly
as before):

```sh
make                      # pt-BR  → 03_convert/pt_br_keyboard_q6k.gguf
make KEYBOARD_LANG=fr     # French → 03_convert/fr_keyboard_q6k.gguf
```

| `KEYBOARD_LANG` | Config        | Corpus                                                                 | Typo layout | Artifact                  |
| --------------- | ------------- | ---------------------------------------------------------------------- | ----------- | ------------------------- |
| `pt` (default)  | `langs/pt.py` | curated pt-BR, Wikipedia pt, conversational/instruct                   | QWERTY      | `pt_br_keyboard_q6k.gguf` |
| `fr`            | `langs/fr.py` | OpenSubtitles ~50 %, FineWeb-2 forums/comments ~35 %, Claire dialogues ~10 %, Wikipedia 5 % | AZERTY | `fr_keyboard_q6k.gguf` |

pt keeps the original flat directories (`01_data/raw`, `01_data/processed`,
`02_train/checkpoints`); other languages use a sub-directory
(`01_data/raw/fr`, …), so several languages can live side by side.

French specifics:

- "Messaging" register: mostly subtitles, forum/comment text and transcribed
  conversations, with Wikipedia capped at 5 %.
- `prepare.py` maps typographic apostrophes (`’` → `'`) before anything else —
  NFKC does not, and French has one every few words (`j'ai`, `l'ami`).
- `make_correction_data.py` simulates typos on an AZERTY layout and types
  accented words with base letters (`ete` → `été`, `francais` → `français`),
  because only `<CHAR_A..Z>` exist in the vocabulary.
- `OpenLLM-France/Claire-Dialogue-French-0.1` is gated (accept its terms on
  the Hub, then `hf auth login`) and CC BY-NC-SA 4.0. Without access the
  download logs an error for that source and continues with the others.

To add a language, copy `langs/fr.py` to `langs/<code>.py` and adjust it.
`KEYBOARD_DOWNLOAD_SCALE=0.002 KEYBOARD_BATCH_SIZE=8 KEYBOARD_MAX_STEPS=300 COMPILE=0 make KEYBOARD_LANG=fr`
runs the whole pipeline on a tiny corpus as a smoke test (a few minutes on
Apple Silicon).

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
| Features          | `base_v1 inverted_space xbu_char_autocorrect_v1 xc0_swipe_typing_v1 char_embed_mixing_v1` |
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

The pipeline is the same on every platform — `make` drives data → train →
convert → validate — only the PyTorch backend differs. `train.py` auto-detects
your GPU: NVIDIA CUDA → `cuda`, AMD ROCm → `cuda` (ROCm wheels expose the same
API), Apple Silicon → `mps`, fallback → `cpu`.

> On Linux/macOS, `torch` from PyPI installs CPU-only wheels by default. To use
> a GPU you must install the build that matches your hardware first (see below).

### macOS (Apple Silicon, MPS)

```sh
make setup        # installs uv, Python deps, verifies MPS, installs cmake
make              # runs the full pipeline: data → train → convert → validate
```

`setup.sh` installs PyTorch with Metal support, so training runs on the MPS
GPU. MPS is single-threaded (DataLoader workers are forced to 0) and only
supports fp32, so this is the slowest path — fine for sanity runs, slow for
full training.

### NVIDIA GPU (CUDA)

On a Linux box with an NVIDIA GPU:

```sh
# Python deps with a CUDA-enabled torch build (match cuNNNN to your driver)
uv venv .venv
uv pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124

# macOS/laptop: prepare data locally, then copy the repo over and train there
make 01_data/processed/train.txt 01_data/tokenizer/pt_br_keyboard.model

# train (auto-detects cuda), then finish conversion + validation
make 03_convert/pt_br_keyboard_q6k.gguf
make validate
```

The CUDA path enables bf16 autocast, `torch.compile`, and 2 DataLoader workers,
so it is significantly faster than MPS. Any recent PyTorch CUDA wheel
(cu118/cu121/cu124/...) works — pick the one matching your driver.

### AMD GPU (ROCm)

PyTorch-on-ROCm uses the same `cuda` device API, so nothing in the pipeline
changes. Two ways to run:

**1. Locally on the ROCm box** — copy the project over, then:

```sh
bash scripts/setup_rocm.sh     # checks ROCm, creates .venv, installs ROCm torch
source .venv/bin/activate
python3 01_data/train_tokenizer.py    # tokenizer not synced, regenerate remotely
python3 01_data/make_correction_data.py
python3 02_train/train.py
```

`setup_rocm.sh` sets `ROCM_VER` (default `rocm7.2`) to match your installed
ROCm — check `rocm-smi --version` and adjust if needed. If `torch.compile` is
flaky on your ROCm build, set `COMPILE=0`.

**2. Remote from macOS (recommended)** — keep development on your laptop, push
training to an AMD box over SSH:

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

## Installing in FUTO Keyboard

1. Build the model: `make` (or the conversion + validate targets above).
2. Copy `03_convert/pt_br_keyboard_q6k.gguf` to your device — or download it
   directly from
   [`huggingface.co/barddoo/futo-keyboard-ptbr`](https://huggingface.co/barddoo/futo-keyboard-ptbr).
3. In FUTO Keyboard, go to **Settings → Keyboard models → Import** and select
   the `.gguf` file.

## Configuration

Stage behavior is overridable via environment variables:

| Variable                     | Default                              | Purpose                                    |
| ---------------------------- | ------------------------------------ | ------------------------------------------ |
| `KEYBOARD_LANG`              | `pt`                                 | Language config to build (`langs/<code>.py`); the defaults below follow it |
| `KEYBOARD_DOWNLOAD_SCALE`    | `1`                                  | Scales every source's download budget (e.g. `0.001` for a smoke test) |
| `AMD_HOST`                   | *(unset)*                            | SSH target for `make train-remote`         |
| `KEYBOARD_MODEL`             | `03_convert/pt_br_keyboard_q6k.gguf` | GGUF to validate                           |
| `KEYBOARD_CKPT`              | `02_train/checkpoints/best.pt`       | Checkpoint to export                       |
| `KEYBOARD_SP_MODEL`          | `01_data/tokenizer/pt_br_keyboard.model` | SentencePiece model to embed           |
| `KEYBOARD_OUT`               | `03_convert/pt_br_keyboard_f32.gguf` | F32 GGUF output path                       |
| `KEYBOARD_FEATURES`          | `base_v1 inverted_space xbu_char_autocorrect_v1 xc0_swipe_typing_v1 char_embed_mixing_v1` | Features declared in `keyboardlm.features` |
| `KEYBOARD_TOKENIZER_VOCAB_SIZE` | `16000`                           | Tokenizer vocab size                       |
| `KEYBOARD_BATCH_SIZE`        | `64`                                 | Training batch size (lower it on small-memory machines) |
| `KEYBOARD_MAX_STEPS`         | `0` (no limit)                       | Stop training after N steps (smoke tests)  |
| `COMPILE`                    | `1`                                  | Set `0` to disable `torch.compile` (flaky on some ROCm setups) |

### Features

The model declares the full feature set FUTO uses — next-word prediction
(`base_v1`), inverted-space tokenization (`inverted_space`), autocorrect
(`xbu_char_autocorrect_v1`), and swipe/gesture typing (`xc0_swipe_typing_v1`
+ `char_embed_mixing_v1`) — because the training data
(`make_correction_data.py`) already contains correction and swipe samples.

`xbu_char_autocorrect_v1` and `xc0_swipe_typing_v1` must be declared together
with `char_embed_mixing_v1` — FUTO only allocates its embedding-mix buffer when
that feature is present. To export a plain next-word-only model (no autocorrect
or swipe), override:

```sh
KEYBOARD_FEATURES="base_v1 inverted_space" \
  python3 03_convert/export_to_gguf.py
```

## Repository layout

```
Makefile                      pipeline driver (data → train → convert → validate)
langs/                        per-language configs (pt, fr)
scripts/                      environment setup + remote training helper
01_data/                      corpus + tokenizer + correction data
02_train/                     model, dataset, training loop
03_convert/                   GGUF export + quantization
04_inject/                    GGUF validation
```

## License

Model weights and code are provided as-is. The corpora used are each subject
to their own licenses (pt: Wikipedia, MADRAS1 corpus-ptbr, Pt-Corpus-Instruct;
fr: OpenSubtitles, FineWeb-2, Claire-Dialogue-French [CC BY-NC-SA 4.0],
Wikipedia) — review them before redistributing the training data or derived models.

The French configuration uses the Claire French Dialogue Dataset (CFDD). When
using it, please cite:

> Julie Hunter, Jérôme Louradour, Virgile Rennard, Ismaïl Harrando, Guokan
> Shang, Jean-Pierre Lorré (2023).
> [The Claire French Dialogue Dataset](https://arxiv.org/abs/2311.16840).

The dataset card also asks to cite each original sub-corpus; see
[OpenLLM-France/Claire-Dialogue-French-0.1](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1).
