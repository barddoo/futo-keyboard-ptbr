"""
Generate FUTO correction-format training data for the active language.

Faithful Python port of FUTO's TrainingDataGenerator.kt
(java/src/org/futo/inputmethod/latin/xlm/TrainingDataGenerator.kt) plus the
swipe variant used in org_futo_inputmethod_latin_xlm_LanguageModel.cpp.

Output lines look like:

  plain next-word sample:
      o presidente da república

  correction sample (a word was mistyped, model must emit the fix):
      o presidente <XBU><CHAR_R><CHAR_E><CHAR_P><CHAR_U><CHAR_B><CHAR_L><CHAR_I><CHAR_C><CHAR_A><XBC>república <XEC>

  swipe sample (same, but <XC0> marks gesture input):
      o presidente <XBU><XC0><CHAR_R><CHAR_E><CHAR_P><CHAR_U><CHAR_B><CHAR_L><CHAR_I><CHAR_C><CHAR_A><XBC>república <XEC>

Consumed by 02_train/train.py via the final SentencePiece tokenizer (the
<XBU>/<XBC>/<XEC>/<XC0>/<CHAR_*> tokens are in the vocab). The model learns
that after <XBU><CHAR_...><XBC> it must emit the correct word + trailing
space, and after <XC0> the same for a gesture-decoded string.

Language-specific behaviour comes from langs/<KEYBOARD_LANG>.py:
  - layout: which key positions simulate tap noise (QWERTY or AZERTY);
  - fold_accents: the vocabulary only has <CHAR_A>..<CHAR_Z>, so the *typed*
    side of a sample cannot carry accents. With fold_accents, accented words
    are eligible for misspelling and are typed with base letters, as a user
    does on a phone ("ete" → "été", "francais" → "français", "coeur" →
    "cœur"); the corrected word keeps its accents.
"""
import math
import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langs import load as load_lang  # noqa: E402

LANG = load_lang()
PROCESSED_DIR = LANG.processed_dir
PLAIN_FILE = PROCESSED_DIR / "train.txt"
OUT_FILE = PROCESSED_DIR / "correction_train.txt"

# ---- FUTO tokens ----
BEGIN_USER_INPUT = "<XBU>"
BEGIN_CORRECTION = "<XBC>"
END_CORRECTION = "<XEC>"
SWIPE_MODE = "<XC0>"

LETTERS = "abcdefghijklmnopqrstuvwxyz"
CHAR_TOKEN = {c: f"<CHAR_{c.upper()}>" for c in LETTERS}

# ---- QWERTY key positions (same rough layout as FUTO) ----
QWERTY = {
    "q": (75.0, 106.0), "w": (214.0, 106.0), "e": (363.0, 106.0),
    "r": (499.0, 106.0), "t": (645.0, 106.0), "y": (789.0, 106.0),
    "u": (928.0, 106.0), "i": (1073.0, 106.0), "o": (1216.0, 106.0),
    "p": (1357.0, 106.0),
    "a": (150.0, 312.0), "s": (291.0, 312.0), "d": (434.0, 312.0),
    "f": (574.0, 312.0), "g": (717.0, 312.0), "h": (859.0, 312.0),
    "j": (1005.0, 312.0), "k": (1140.0, 312.0), "l": (1288.0, 312.0),
    "z": (287.0, 515.0), "x": (434.0, 515.0), "c": (576.0, 515.0),
    "v": (718.0, 515.0), "b": (860.0, 515.0), "n": (1003.0, 515.0),
    "m": (1145.0, 515.0),
}
# ---- AZERTY key positions (same grid as above; FUTO's French layout) ----
#   a z e r t y u i o p
#   q s d f g h j k l m      (10 keys: no half-key offset on the home row)
#     w x c v b n '
_ROW1_X = [QWERTY[c][0] for c in "qwertyuiop"]
_ROW3_X = [QWERTY[c][0] for c in "zxcvbn"]
AZERTY = {
    **{c: (x, 106.0) for c, x in zip("azertyuiop", _ROW1_X)},
    **{c: (x, 312.0) for c, x in zip("qsdfghjklm", _ROW1_X)},
    **{c: (x, 515.0) for c, x in zip("wxcvbn", _ROW3_X)},
}
LAYOUTS = {"qwerty": QWERTY, "azerty": AZERTY}
KEYS = LAYOUTS[LANG.layout]
assert set(KEYS) == set(LETTERS), f"layout {LANG.layout} must cover a-z"

TAP_SIZE = (80.0, 80.0)
PERMITTED = set("abcdefghijklmnopqrstuvwxyz'-")

# Ligatures have no canonical decomposition; everything else (é è ê ë à â ç
# î ï ô ù û ü ÿ …) folds via NFD + dropping the combining marks.
_LIGATURES = str.maketrans({"œ": "oe", "æ": "ae"})


def fold_accents(word: str) -> str:
    """été → ete, français → francais, cœur → coeur (expects lowercase)."""
    decomposed = unicodedata.normalize("NFD", word.translate(_LIGATURES))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def typed_form(word: str) -> str:
    """Lowercase letters the user would hit for word, before any typo."""
    w = word.strip().lower()
    return fold_accents(w) if LANG.fold_accents else w


def _normal(mean: float, std: float) -> float:
    return random.gauss(mean, std)


def _closest_key(x: float, y: float) -> str:
    return min(KEYS, key=lambda c: (KEYS[c][0] - x) ** 2 + (KEYS[c][1] - y) ** 2)


def substitute_keyboard_letters(word: str, temperature: float = 0.6) -> str:
    """Add Gaussian tap noise around each key and snap to the closest key."""
    out = []
    for ch in word:
        pos = KEYS.get(ch)
        if pos is None:
            out.append(ch)
            continue
        new_pos = (
            _normal(pos[0], temperature * TAP_SIZE[0]),
            _normal(pos[1], temperature * TAP_SIZE[1]),
        )
        out.append(_closest_key(*new_pos))
    return "".join(out)


def misspell_word(word: str, correctness: float = 0.8) -> str:
    """Simulate a realistic mistyping of word (FUTO WordMisspelling port)."""
    w = typed_form(word).replace("'", "")

    def rnd() -> float:
        return random.random() ** correctness

    if rnd() > 0.5 and len(w) >= 2:
        i, j = random.sample(range(len(w)), 2)
        w = list(w)
        w[i], w[j] = w[j], w[i]
        w = "".join(w)

    if rnd() > 0.5 and len(w) >= 2:
        i = random.randrange(len(w) - 1)
        w = w[:i] + w[i + 1] + w[i] + w[i + 2:]

    if rnd() > 0.5 and w:
        i = random.randrange(len(w))
        w = w[:i] + w[i + 1:]

    w = substitute_keyboard_letters(w, temperature=1.0 * rnd())

    # Trim as if the user hasn't finished typing the word yet (teaches
    # completion of partial words).
    if rnd() > 0.33 and len(w) >= 2:
        new_len = int(math.ceil((1.0 - rnd() * rnd()) * len(w)))
        w = w[:max(2, min(new_len, len(w)))]

    return w


def format_word_correction(misspelled: str, truth: str, swipe: bool = False) -> str:
    """<XBU>[<XC0>]<CHAR_*...><XBC>word <XEC>"""
    chars = "".join(CHAR_TOKEN[c] for c in misspelled if c in CHAR_TOKEN)
    if not chars or not truth.strip():
        return ""
    swipe_tok = SWIPE_MODE if swipe else ""
    return (BEGIN_USER_INPUT + swipe_tok + chars + BEGIN_CORRECTION
            + truth.strip() + " " + END_CORRECTION)


def word_correction(word: str, correctness: float = 0.8, swipe: bool = False) -> str:
    return format_word_correction(misspell_word(word, correctness), word, swipe)


def concat_word_correction(context: str, word: str, correctness: float = 0.8,
                           swipe: bool = False) -> str:
    s = word_correction(word, correctness, swipe)
    return f"{context.strip()} {s}" if s else ""


def randomly_misspell_words(text: str, proportion: float = 0.333,
                            correctness: float = 0.8) -> str:
    """Replace ~proportion of words in a sentence with correction blocks,
    keeping the rest as plain text (mirrors FUTO's randomlyMisspellWords)."""
    words = text.split()
    n = int(len(words) * proportion)
    targets = set()
    attempts = 0
    while len(targets) < n and attempts < n * 10:
        i = random.randrange(len(words))
        if i not in targets and all(c in PERMITTED for c in typed_form(words[i])):
            targets.add(i)
        attempts += 1
    for i in targets:
        w = word_correction(words[i], correctness)
        if w:
            words[i] = w
    return (" ".join(words)
            .replace("  ", " ")
            # Do not put spaces after these tokens, as it messes up
            # tokenization (the corrected word already carries its space).
            .replace(BEGIN_CORRECTION + " ", BEGIN_CORRECTION)
            .replace(END_CORRECTION + " ", END_CORRECTION))


def main() -> None:
    if not PLAIN_FILE.exists():
        print("ERROR: run 01_data/prepare.py first")
        sys.exit(1)

    random.seed(1337)  # reproducible
    plain = 0
    correction = 0
    swipe = 0
    with open(PLAIN_FILE, encoding="utf-8") as fin, \
            open(OUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            roll = random.random()
            if roll < 0.35:
                # plain next-word line (keeps base prediction)
                fout.write(line + "\n")
                plain += 1
            elif roll < 0.80:
                # one or two mistyped words in the sentence
                fout.write(randomly_misspell_words(line) + "\n")
                correction += 1
            else:
                # swipe: same correction format but marked gesture input
                words = line.split()
                i = random.randrange(len(words)) if words else -1
                words[i] = word_correction(words[i], 0.8, swipe=True) \
                    if 0 <= i < len(words) and words[i] else words[i]
                fout.write(" ".join(words)
                           .replace("  ", " ")
                           .replace(BEGIN_CORRECTION + " ", BEGIN_CORRECTION)
                           .replace(END_CORRECTION + " ", END_CORRECTION) + "\n")
                swipe += 1

    print(f"correction training data written to {OUT_FILE}")
    print(f"  plain: {plain:,}  correction: {correction:,}  swipe: {swipe:,}")
    print("Sample:")
    with open(OUT_FILE, encoding="utf-8") as f:
        for _ in range(3):
            print("  " + f.readline().rstrip())


if __name__ == "__main__":
    main()
