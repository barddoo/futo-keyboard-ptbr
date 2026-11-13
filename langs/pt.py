"""Brazilian Portuguese — the original configuration of this repo."""
from . import LangConfig

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

CONFIG = LangConfig(
    code="pt",
    name="pt_br_keyboard",
    gguf_name="KeyboardLM-pt-BR",
    gguf_description="pt-BR keyboard language model (FUTO Keyboard)",
    sources=SOURCES,
    subdir="",          # original flat layout: 01_data/raw, 01_data/processed, …
    layout="qwerty",
)
