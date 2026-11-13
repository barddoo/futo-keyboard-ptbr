"""
Download corpus slices for keyboard LM training.

Uses streaming to avoid downloading the full dataset upfront.
Targets everyday mobile writing: short sentences, conversational text.

The sources come from the active language config (langs/<KEYBOARD_LANG>.py).
Each source is a dict:

  id, config, split   passed to datasets.load_dataset (streaming)
  load_kwargs         optional extra load_dataset kwargs (data_files, …)
  text_col            column holding the text (ignored if "extract" is set)
  filter              optional doc -> bool row filter
  extract             optional doc -> iterable of text units; lets one row
                      yield several lines (dialogue turns, forum posts) or
                      pick a nested field
  min_len             optional minimum unit length in chars (default 20)
  max_docs            stop after this many units written
  max_chars           optional: also stop after this many chars written
  out                 output file name under the language's raw dir

KEYBOARD_DOWNLOAD_SCALE (float, default 1) scales max_docs / max_chars —
e.g. 0.001 for a quick end-to-end smoke test.
"""
import os
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langs import load as load_lang  # noqa: E402

LANG = load_lang()
OUT_DIR = LANG.raw_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = LANG.sources
SCALE = float(os.environ.get("KEYBOARD_DOWNLOAD_SCALE", "1"))


def clean(text: str, min_len: int = 20) -> str | None:
    text = text.strip()
    # Drop very short or very long docs (not useful for keyboard context)
    if len(text) < min_len or len(text) > 2000:
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
    kwargs.update(src.get("load_kwargs") or {})

    dataset = load_dataset(src["id"], **kwargs)

    max_docs = max(1, int(src["max_docs"] * SCALE))
    max_chars = src.get("max_chars")
    if max_chars is not None:
        max_chars = max(1, int(max_chars * SCALE))
    row_filter = src.get("filter")
    extract = src.get("extract")
    min_len = src.get("min_len", 20)
    written = 0
    chars = 0

    def full() -> bool:
        return written >= max_docs or (max_chars is not None and chars >= max_chars)

    with open(out_path, "w", encoding="utf-8") as fout:
        total = max_docs if max_chars is None else None
        for doc in tqdm(dataset, total=total, unit="doc"):
            if full():
                break
            if row_filter and not row_filter(doc):
                continue
            units = extract(doc) if extract else (doc.get(src["text_col"], ""),)
            for text in units:
                cleaned = clean(text, min_len)
                if cleaned:
                    fout.write(cleaned + "\n")
                    written += 1
                    chars += len(cleaned) + 1
                    if full():
                        break

    print(f"[done] {written:,} docs, {chars / 1e6:,.1f}M chars → {out_path}")


if __name__ == "__main__":
    for src in SOURCES:
        try:
            download_source(src)
        except Exception as e:
            print(f"[error] {src['id']} ({src['out']}): {e}", file=sys.stderr)
            # Don't leave a truncated file behind: the next run would [skip]
            # it (e.g. after being granted access to a gated dataset).
            (OUT_DIR / src["out"]).unlink(missing_ok=True)
    # Abandoned streaming iterators can leave non-daemon threads that block
    # interpreter shutdown forever (seen with datasets 4.x); everything is
    # written and closed at this point, so exit without waiting for them.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
