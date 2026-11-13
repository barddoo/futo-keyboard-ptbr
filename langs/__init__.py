"""
Per-language configuration for the keyboard LM pipeline.

Every stage (download → prepare → tokenizer → correction data → train →
export → quantize → validate) reads the active language from the
KEYBOARD_LANG environment variable (default: "pt") and loads
langs/<code>.py, which must define CONFIG = LangConfig(...).

The Makefile exports KEYBOARD_LANG, so `make KEYBOARD_LANG=fr` runs the
whole pipeline for French. (The variable is deliberately not called LANG:
that name is the POSIX locale variable, make would import it from the
environment and export the override to every child process.)

This package is stdlib-only so the Makefile and shell scripts can query it
without the project venv:

    python3 -m langs name          # → pt_br_keyboard
    python3 -m langs q6k_gguf      # → 03_convert/pt_br_keyboard_q6k.gguf
"""
from __future__ import annotations  # keep importable by the system python3 (3.9)

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LANG = "pt"


@dataclass(frozen=True)
class LangConfig:
    # Language code written to keyboardlm.languages (what FUTO matches
    # against the active keyboard locale).
    code: str
    # File stem for the tokenizer and GGUF artifacts (<name>.model,
    # <name>_f32.gguf, <name>_q6k.gguf).
    name: str
    # general.name / general.description in the GGUF.
    gguf_name: str
    gguf_description: str
    # Corpus sources, see 01_data/download.py for the dict schema.
    sources: list = field(default_factory=list)
    # Sub-directory under 01_data/{raw,processed} and 02_train/checkpoints.
    # "" keeps the original flat layout (used by pt, the first language).
    subdir: str = ""
    # Key layout used to simulate mistyped words (make_correction_data.py).
    layout: str = "qwerty"
    # prepare.py: map typographic apostrophes (’ ‘ ʼ …) to ' before anything
    # else. NFKC (the tokenizer's normalisation) does not do it.
    normalize_apostrophes: bool = False
    # make_correction_data.py: accept words with accented letters and fold
    # them to the base letter in the *typed* side (é → <CHAR_E>), since only
    # <CHAR_A>..<CHAR_Z> exist in the vocabulary. The corrected word keeps
    # its accents.
    fold_accents: bool = False
    # prepare.py: shuffle sentences (fixed seed) before the train/valid
    # split, so valid.txt is not just the tail of the last raw file.
    shuffle_before_split: bool = False

    # ---- derived paths -------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return ROOT / "01_data" / "raw" / self.subdir

    @property
    def processed_dir(self) -> Path:
        return ROOT / "01_data" / "processed" / self.subdir

    @property
    def tokenizer_dir(self) -> Path:
        return ROOT / "01_data" / "tokenizer"

    @property
    def sp_model(self) -> Path:
        return self.tokenizer_dir / f"{self.name}.model"

    @property
    def checkpoint_dir(self) -> Path:
        return ROOT / "02_train" / "checkpoints" / self.subdir

    @property
    def f32_gguf(self) -> Path:
        return ROOT / "03_convert" / f"{self.name}_f32.gguf"

    @property
    def q6k_gguf(self) -> Path:
        return ROOT / "03_convert" / f"{self.name}_q6k.gguf"


def load(code: str | None = None) -> LangConfig:
    code = code or os.environ.get("KEYBOARD_LANG") or DEFAULT_LANG
    if not code.isalnum():
        raise SystemExit(f"invalid KEYBOARD_LANG {code!r}")
    try:
        module = importlib.import_module(f"langs.{code}")
    except ModuleNotFoundError as e:
        if e.name != f"langs.{code}":
            raise
        available = sorted(p.stem for p in Path(__file__).parent.glob("*.py")
                           if not p.stem.startswith("_"))
        raise SystemExit(
            f"unknown KEYBOARD_LANG {code!r} — available: {', '.join(available)}")
    return module.CONFIG
