"""
Download pt-BR corpus slices for keyboard LM training.

Uses streaming to avoid downloading the full dataset upfront.
Targets everyday mobile writing: short sentences, conversational text.
"""
import os
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = Path(__file__).parent / "raw"
OUT_DIR.mkdir(exist_ok=True)

# Target: ~500M tokens is enough for a small keyboard LM.
SOURCES = [
    {
        # Large, clean, curated pt-BR corpus — real subset only (no synthetic)
        "id": "Madras1/corpus-ptbr-v1",
        "config": None,
        "split": "train",
        "text_col": "text",
        "filter": lambda doc: doc.get("subset") == "real",
        "max_docs": 500_000,
        "out": "corpus_ptbr.txt",
    },
    {
        # Wikipedia pt — grammatically clean, good vocabulary coverage
        "id": "wikimedia/wikipedia",
        "config": "20231101.pt",
        "split": "train",
        "text_col": "text",
        "filter": None,
        "max_docs": 200_000,
        "out": "wikipedia_pt.txt",
    },
    {
        # Conversational pt-BR pairs — closes the casual/messaging style gap
        "id": "nicholasKluge/Pt-Corpus-Instruct",
        "config": None,
        "split": "train",
        "text_col": "text",
        "filter": None,
        "max_docs": 200_000,
        "out": "pt_instruct.txt",
    },
]


def clean(text: str) -> str | None:
    text = text.strip()
    # Drop very short or very long docs (not useful for keyboard context)
    if len(text) < 20 or len(text) > 2000:
        return None
    # Drop lines heavy on URLs / HTML
    if text.count("http") > 2 or text.count("<") > 4:
        return None
    return text


def download_source(src: dict) -> None:
    out_path = OUT_DIR / src["out"]
    if out_path.exists():
        print(f"[skip] {src['out']} already exists")
        return

    print(f"[fetch] {src['id']} …")
    kwargs = {"split": src["split"], "streaming": True}
    if src["config"]:
        kwargs["name"] = src["config"]

    dataset = load_dataset(src["id"], **kwargs)

    row_filter = src.get("filter")
    written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for doc in tqdm(dataset, total=src["max_docs"], unit="doc"):
            if written >= src["max_docs"]:
                break
            if row_filter and not row_filter(doc):
                continue
            text = doc.get(src["text_col"], "")
            cleaned = clean(text)
            if cleaned:
                fout.write(cleaned + "\n")
                written += 1

    print(f"[done] {written:,} docs → {out_path}")


if __name__ == "__main__":
    for src in SOURCES:
        try:
            download_source(src)
        except Exception as e:
            print(f"[error] {src['id']}: {e}", file=sys.stderr)
