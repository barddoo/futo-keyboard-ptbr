"""
Train the SentencePiece tokenizer for pt-BR with FUTO Keyboard's conventions.

FUTO's runtime (native/jni/src/ggml/LanguageModel.cpp and the JNI in
org_futo_inputmethod_latin_xlm_LanguageModel.cpp) makes hard assumptions
about the vocabulary, and they must be replicated exactly:

  - whitespace is a SUFFIX marker: pieces look like "olá▁" instead of "▁olá"
    (this is the "inverted_space" feature). Enabled via
    treat_whitespace_as_suffix=True.
  - token ids 0..3 are <pad>, <s>, </s>, <unk>. FUTO hardcodes BOS to token
    id 1 (next_context.insert(begin, 1)), so <s> MUST be at id 1.
  - the autocorrect / swipe / char-mixing features are driven by special
    vocabulary tokens: <XBU>, <XBC>, <XEC>, <XC0>, and <CHAR_A>..<CHAR_Z>
    (which must be consecutive ids). A block of reserved <FUTO0>..<FUTO11>
    tokens mirrors the official ml4 model.
  - byte_fallback=True adds <0xNN> tokens so any byte can round-trip.

SentencePiece can't be told to insert arbitrary tokens, so we train a base
BPE model with vocab_size = TARGET - len(specials) and then post-process the
ModelProto to splice the control tokens in right after <unk> before
re-serializing. Because ids are just indices into the pieces array, all
subsequent piece ids shift by the same amount — so training data MUST be
re-tokenized with the final model (dataset.py does this), keeping token ids
consistent with the model's embedding rows.
"""
import os
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent / "processed"
OUT_DIR = Path(os.environ.get("KEYBOARD_TOKENIZER_OUT_DIR", Path(__file__).parent / "tokenizer"))
OUT_DIR.mkdir(exist_ok=True)

TRAIN_FILE = Path(os.environ.get("KEYBOARD_TOKENIZER_INPUT", PROCESSED_DIR / "train.txt"))
BASE_PREFIX = str(OUT_DIR / "pt_br_keyboard_base")
MODEL_PATH = OUT_DIR / "pt_br_keyboard.model"
VOCAB_PATH = OUT_DIR / "pt_br_keyboard.vocab"

VOCAB_SIZE = int(os.environ.get("KEYBOARD_TOKENIZER_VOCAB_SIZE", "16000"))
# small vocab suits a keyboard LM (fast lookup, less memory)

# Control tokens FUTO's runtime looks up by piece name. <CHAR_A..Z> must be
# consecutive ids; everything else can live anywhere.
SPECIAL_TOKENS = (
    [f"<FUTO{i}>" for i in range(12)]
    + ["<XBU>", "<XBC>", "<XEC>", "<XC0>"]
    + [f"<CHAR_{chr(ord('A') + i)}>" for i in range(26)]
)


def _validate(final: sp_pb2.ModelProto) -> None:
    """Sanity check the final vocab against FUTO's runtime expectations."""
    sp = spm.SentencePieceProcessor()
    status = sp.Load(str(MODEL_PATH))
    assert status, "final tokenizer failed to load"

    n = sp.GetPieceSize()
    assert n == VOCAB_SIZE, f"vocab size {n} != {VOCAB_SIZE}"

    assert sp.IdToPiece(0) == "<pad>", "id 0 must be <pad>"
    assert sp.IdToPiece(1) == "<s>", "id 1 must be <s> (FUTO hardcodes BOS=1)"
    assert sp.IdToPiece(2) == "</s>", "id 2 must be </s>"
    assert sp.IdToPiece(3) == "<unk>", "id 3 must be <unk>"

    for tok in SPECIAL_TOKENS:
        assert sp.PieceToId(tok) > 0, f"control token {tok} missing from vocab"

    char_a = sp.PieceToId("<CHAR_A>")
    char_z = sp.PieceToId("<CHAR_Z>")
    assert char_z == char_a + 25, "<CHAR_A..Z> must be consecutive ids"

    assert sp.PieceToId("▁") > 0, "space piece ▁ missing"
    print(f"  vocab size: {n}")
    print(f"  <CHAR_A>= {char_a}  <CHAR_Z>= {char_z}")
    print(f"  ▁ = {sp.PieceToId('▁')}")


def train() -> None:
    if not TRAIN_FILE.exists():
        print("ERROR: run 01_data/prepare.py first")
        return

    base_vocab = VOCAB_SIZE - len(SPECIAL_TOKENS)
    print(f"Training SentencePiece BPE (base vocab={base_vocab} + "
          f"{len(SPECIAL_TOKENS)} control tokens = {VOCAB_SIZE}) …")

    spm.SentencePieceTrainer.train(
        input=str(TRAIN_FILE),
        model_prefix=BASE_PREFIX,
        vocab_size=base_vocab,
        character_coverage=0.9999,     # byte_fallback covers the rest
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        # Normalisation rules suitable for pt-BR
        normalization_rule_name="nfkc",
        remove_extra_whitespaces=True,
        add_dummy_prefix=True,
        treat_whitespace_as_suffix=True,  # inverted_space: pieces end in ▁
        byte_fallback=True,
        num_threads=8,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
    )

    # ---- Post-process: splice control tokens in after <unk> ----
    base = sp_pb2.ModelProto()
    base.ParseFromString(Path(BASE_PREFIX + ".model").read_bytes())
    assert base.pieces[0].piece == "<pad>", "base vocab layout unexpected"

    final = sp_pb2.ModelProto()
    final.CopyFrom(base)
    del final.pieces[:]

    # (piece, score, type) — must preserve the SentencePiece.Type field
    # (UNKNOWN/CONTROL/BYTE/…); dropping it makes SP reject the model
    # ("unk is not defined").
    head = [(p.piece, p.score, p.type) for p in base.pieces[:4]]   # <pad> <s> </s> <unk>
    rest = [(p.piece, p.score, p.type) for p in base.pieces[4:]]   # learned pieces

    # Same convention as the official ml4 model: special tokens are
    # USER_DEFINED pieces AND listed in trainer_spec.user_defined_symbols,
    # which is what makes EncodeAsIds keep "<XBU>" atomic instead of splitting
    # it into chars.
    USER_DEFINED = sp_pb2.ModelProto.SentencePiece.Type.USER_DEFINED

    for name, score, ptype in head:
        p = final.pieces.add()
        p.piece = name
        p.score = score
        p.type = ptype
    for name in SPECIAL_TOKENS:
        p = final.pieces.add()
        p.piece = name
        p.score = 0.0
        p.type = USER_DEFINED
    for name, score, ptype in rest:
        p = final.pieces.add()
        p.piece = name
        p.score = score
        p.type = ptype

    final.trainer_spec.vocab_size = len(final.pieces)
    final.trainer_spec.user_defined_symbols.extend(SPECIAL_TOKENS)
    assert len(final.pieces) == VOCAB_SIZE

    MODEL_PATH.write_bytes(final.SerializeToString())

    # Human-readable vocab dump
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        for p in final.pieces:
            f.write(p.piece.replace("\n", "<NEWLINE>") + "\n")

    # Clean up the intermediate base model
    for suffix in (".model", ".vocab"):
        p = Path(BASE_PREFIX + suffix)
        if p.exists():
            p.unlink()

    print(f"Tokenizer saved to {OUT_DIR}/")
    _validate(final)
    print("NOTE: whitespace is a suffix (inverted_space); training data is "
          "tokenized with this final model so ids match the GGUF embedding rows.")


if __name__ == "__main__":
    train()
