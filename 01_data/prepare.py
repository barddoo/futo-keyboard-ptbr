"""
Clean and merge raw corpora into a single keyboard-optimised training file.

Keyboard LM training differs from general LM:
- Sentence-level chunking (each line = one typing context)
- Strip academic/technical prose that never appears in keyboard use
- Deduplicate near-identical lines
"""
import re
import hashlib
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langs import load as load_lang  # noqa: E402

LANG = load_lang()
RAW_DIR = LANG.raw_dir
OUT_DIR = LANG.processed_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = OUT_DIR / "train.txt"
VALID_OUT = OUT_DIR / "valid.txt"

VALID_RATIO = 0.005  # 0.5% for validation
MAX_LINE_LEN = 300   # chars; longer than this rarely appears in mobile typing


def is_keyboard_like(line: str) -> bool:
    """Keep lines that look like things a human would type on a phone."""
    # Too many numbers / punctuation = likely structured data
    punct_ratio = sum(1 for c in line if not c.isalpha() and not c.isspace()) / max(len(line), 1)
    if punct_ratio > 0.25:
        return False
    # Non-latin scripts (should be mostly the target language)
    non_latin = sum(1 for c in line if c.isalpha() and ord(c) > 0x024F)
    if non_latin / max(len(line), 1) > 0.05:
        return False
    return True


def sentence_split(text: str) -> list[str]:
    """Split a doc into sentence-sized chunks."""
    # Split on common sentence endings, keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if 10 < len(s.strip()) <= MAX_LINE_LEN]


def fingerprint(line: str) -> str:
    normalised = re.sub(r'\s+', ' ', line.lower().strip())
    return hashlib.md5(normalised.encode()).hexdigest()


def run() -> None:
    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        print("No raw files found — run 01_data/download.py first")
        return

    seen: set[str] = set()
    all_lines: list[str] = []

    for fpath in raw_files:
        print(f"Processing {fpath.name} …")
        with open(fpath, encoding="utf-8") as fin:
            for raw_line in tqdm(fin):
                for sent in sentence_split(raw_line):
                    if not is_keyboard_like(sent):
                        continue
                    fp = fingerprint(sent)
                    if fp in seen:
                        continue
                    seen.add(fp)
                    all_lines.append(sent)

    print(f"Total unique sentences: {len(all_lines):,}")

    split_idx = int(len(all_lines) * (1 - VALID_RATIO))
    train_lines = all_lines[:split_idx]
    valid_lines = all_lines[split_idx:]

    TRAIN_OUT.write_text("\n".join(train_lines), encoding="utf-8")
    VALID_OUT.write_text("\n".join(valid_lines), encoding="utf-8")

    print(f"Train: {len(train_lines):,} lines → {TRAIN_OUT}")
    print(f"Valid: {len(valid_lines):,} lines → {VALID_OUT}")


if __name__ == "__main__":
    run()
