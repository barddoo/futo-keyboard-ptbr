"""French — "messaging" register.

A keyboard LM should sound like what people type on a phone, so the mix is
weighted towards dialogue and informal writing rather than edited prose:

    OpenSubtitles (fr side of en-fr)        ~50 %   spoken-style dialogue
    FineWeb-2 fra_Latn, forums / comments   ~35 %   informal written French
    Claire-Dialogue-French (conversational) ~10 %   transcribed conversations
    Wikipedia fr                              5 %   vocabulary / proper nouns

Shares are character budgets at download time (max_chars); prepare.py prints
what is left of each source after filtering and deduplication.

Notes on the sources:
  - OpenLLM-France/Claire-Dialogue-French-0.1 is gated: accept its terms on
    the dataset page and `hf auth login` first, otherwise download.py logs
    an error for it and carries on with the other sources. It is CC BY-NC-SA
    4.0, which matters if you redistribute a model trained on it. ~90 % of
    Claire is Assemblée nationale debates — the wrong register — so only the
    conversational sub-corpora are listed below; together they are smaller
    than the 10 % budget at full scale.
    When using it, cite: Julie Hunter, Jérôme Louradour, Virgile Rennard,
    Ismaïl Harrando, Guokan Shang, Jean-Pierre Lorré (2023), "The Claire
    French Dialogue Dataset", https://arxiv.org/abs/2311.16840 — and the
    original sub-corpora listed on the dataset card.
  - OpenSubtitles has no explicit license on the Hub ("unknown").
"""
import re

from . import LangConfig

# Nominal corpus size in characters before prepare.py (~250M tokens).
TOTAL_CHARS = 1_000_000_000
NO_DOC_LIMIT = 10**12  # sources below are bounded by max_chars instead


# ---- OpenSubtitles ---------------------------------------------------------
def opensubtitles_fr(doc: dict):
    pair = doc["translation"]
    fr = pair["fr"].strip()
    # Untranslated lines (song lyrics, English left as-is) are identical on
    # both sides of the alignment.
    if fr == pair["en"].strip():
        return
    # Dialogue dashes: "- Tu viens ?" / "- Oui. - Non."
    for part in re.split(r"(?:^|\s)-\s+", fr):
        yield part


# ---- FineWeb-2: keep forum threads and comment sections --------------------
_FORUM_URL = re.compile(
    r"forum|viewtopic|showthread|/topics?[/-]|/threads?/|/discussions?/"
    r"|/sujet|commentaire|[#/]comments?\b",
    re.IGNORECASE,
)


def is_forum_or_comments(doc: dict) -> bool:
    return bool(_FORUM_URL.search(doc.get("url") or ""))


def web_lines(doc: dict):
    # A page is far longer than a typing context; posts are separated by
    # newlines, and prepare.py splits them further into sentences.
    return doc["text"].split("\n")


# ---- Claire: one conversation per row, one "[speaker:] turn" per line -----
_CLAIRE_REPO = "hf://datasets/OpenLLM-France/Claire-Dialogue-French-0.1/FR"
# Conversational sub-corpora only. Left out: AssembleeNationale_* (133M of
# the 150M words, parliamentary), TheatreClassique (17th-century verse),
# conferences / formal addresses / political debates, and OTG + UBS
# ("contact before usage").
_CLAIRE_SUBSETS = [
    "ACSYNT", "CFPP", "CID", "CLAPI", "ESLO_assistance", "ESLO_free",
    "ESLO_interview", "LINAGORA_free", "LINAGORA_meetings", "OFROM",
    "ORFEO_cfpb", "ORFEO_coralrom", "ORFEO_crfp", "ORFEO_fleuron",
    "ORFEO_reunions-de-travail", "ORFEO_valibel_interview", "PFC_free",
    "PFC_guided", "ParisStories", "Rhapsodie", "SUMM-RE", "TCOF_adults",
    "TheatreGratuit",
]
_SPEAKER = re.compile(r"^\[[^\]]*:\]\s*")
_ANNOTATION = re.compile(r"\[[^\]]*\]")   # [NOISE], [LAUGHTER], …


def claire_turns(doc: dict):
    for line in doc["text"].split("\n"):
        turn = _SPEAKER.sub("", line.strip())
        if "[PII]" in turn:       # anonymised names leave a hole in the turn
            continue
        yield re.sub(r"\s+", " ", _ANNOTATION.sub(" ", turn))


SOURCES = [
    {
        "id": "alamios/OpenSubtitles-v2018",
        "config": "en-fr",
        "split": "train",
        "extract": opensubtitles_fr,
        "min_len": 12,     # short replies are the point of this source
        "max_docs": NO_DOC_LIMIT,
        "max_chars": int(TOTAL_CHARS * 0.50),
        "out": "opensubtitles_fr.txt",
    },
    {
        "id": "HuggingFaceFW/fineweb-2",
        "config": "fra_Latn",
        "split": "train",
        "filter": is_forum_or_comments,
        "extract": web_lines,
        "max_docs": NO_DOC_LIMIT,
        "max_chars": int(TOTAL_CHARS * 0.35),
        "out": "fineweb2_forums_fr.txt",
    },
    {
        "id": "text",
        "config": None,
        "split": "train",
        "load_kwargs": {
            "data_files": [f"{_CLAIRE_REPO}/{s}/train.txt" for s in _CLAIRE_SUBSETS],
            "sample_by": "paragraph",
        },
        "extract": claire_turns,
        "min_len": 12,
        "max_docs": NO_DOC_LIMIT,
        "max_chars": int(TOTAL_CHARS * 0.10),
        "out": "claire_dialogue_fr.txt",
    },
    {
        "id": "wikimedia/wikipedia",
        "config": "20231101.fr",
        "split": "train",
        "text_col": "text",
        "filter": None,
        "extract": web_lines,   # articles are > 2000 chars: take paragraphs
        "max_docs": NO_DOC_LIMIT,
        "max_chars": int(TOTAL_CHARS * 0.05),
        "out": "wikipedia_fr.txt",
    },
]

CONFIG = LangConfig(
    code="fr",
    name="fr_keyboard",
    gguf_name="KeyboardLM-fr",
    gguf_description="French keyboard language model (FUTO Keyboard)",
    sources=SOURCES,
    subdir="fr",
    layout="azerty",
    normalize_apostrophes=True,
    fold_accents=True,
    shuffle_before_split=True,
)
