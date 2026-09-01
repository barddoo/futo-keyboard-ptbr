"""
Validate a FUTO-compatible keyboard GGUF after quantization.

Checks every key / tensor / vocab invariant that FUTO Keyboard's loader
(ModelMeta.cpp, ggml/LanguageModel.cpp and the llama-arch GGUF reader) depends
on. Exit code 0 = the file should load in FUTO Keyboard.

The metadata is now written directly by 03_convert/export_to_gguf.py and is
preserved through quantization, so this script only verifies — it does not
rewrite the model (the previous copy-and-re-emit approach corrupted arrays).
"""
import os
import sys
from pathlib import Path

from gguf import GGUFReader, GGUFValueType

MODEL = Path(os.environ.get(
    "KEYBOARD_MODEL",
    Path(__file__).parent.parent / "03_convert" / "pt_br_keyboard_q6k.gguf",
))

REQUIRED_KV = {
    "general.architecture": GGUFValueType.STRING,
    "keyboardlm.languages": GGUFValueType.STRING,
    "keyboardlm.finetuning_count": GGUFValueType.UINT32,
    "keyboardlm.history": GGUFValueType.STRING,
    "keyboardlm.features": GGUFValueType.STRING,
    "keyboardlm.ext_tokenizer_type": GGUFValueType.STRING,
    "keyboardlm.ext_tokenizer_data": GGUFValueType.ARRAY,
    "tokenizer.ggml.model": GGUFValueType.STRING,
    "tokenizer.ggml.tokens": GGUFValueType.ARRAY,
    "tokenizer.ggml.scores": GGUFValueType.ARRAY,
    "tokenizer.ggml.token_type": GGUFValueType.ARRAY,
    "llama.context_length": GGUFValueType.UINT32,
    "llama.embedding_length": GGUFValueType.UINT32,
    "llama.block_count": GGUFValueType.UINT32,
    "llama.feed_forward_length": GGUFValueType.UINT32,
    "llama.attention.head_count": GGUFValueType.UINT32,
}

REQUIRED_TENSOR_PREFIXES = [
    "token_embd.weight",
    "output_norm.weight",
    "output.weight",
]
# per block, checked for however many blocks the metadata declares
BLOCK_TENSORS = [
    "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "attn_output.weight", "ffn_norm.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
]


def val_str(field) -> str:
    return field.parts[-1].tobytes().decode("utf-8")


def main() -> int:
    if not MODEL.exists():
        print(f"ERROR: {MODEL} not found — run 03_convert/quantize.sh first")
        return 1

    r = GGUFReader(str(MODEL))
    errors = []

    # ---- Metadata presence + type ----
    for key, vtype in REQUIRED_KV.items():
        f = r.fields.get(key)
        if f is None:
            errors.append(f"missing KV {key}")
            continue
        if vtype == GGUFValueType.ARRAY:
            if f.types[0] != GGUFValueType.ARRAY:
                errors.append(f"{key} is {f.types[0].name}, expected ARRAY")
        elif f.types[0] != vtype:
            errors.append(f"{key} is {f.types[0].name}, expected {vtype.name}")

    # keyboardlm.ext_tokenizer_data must be UINT8 array
    f = r.fields.get("keyboardlm.ext_tokenizer_data")
    if f is not None and len(f.types) < 2:
        errors.append("keyboardlm.ext_tokenizer_data must be an ARRAY")
    elif f is not None and f.types[1] != GGUFValueType.UINT8:
        errors.append("keyboardlm.ext_tokenizer_data must be ARRAY of UINT8")

    # keyboardlm.features must be a string (FUTO reads gguf_get_val_str)
    f = r.fields.get("keyboardlm.features")
    if f is not None and f.types[0] == GGUFValueType.STRING:
        feats = val_str(f).split()
        if not feats:
            errors.append("keyboardlm.features is empty")
    elif f is not None:
        errors.append(f"keyboardlm.features is {f.types[0].name}, "
                      "must be a space-separated STRING")

    # tokenizer arrays consistent length
    n_vocab = None
    for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.scores",
                "tokenizer.ggml.token_type"):
        f = r.fields.get(key)
        if f is None or f.types[0] != GGUFValueType.ARRAY:
            errors.append(f"{key} missing or not an array")
            continue
        n = int(f.parts[4].view("Q")[0])
        if n_vocab is None:
            n_vocab = n
        elif n != n_vocab:
            errors.append(f"{key} length {n} != {n_vocab}")
    if n_vocab is None:
        n_vocab = 0

    tok_emb = next((t for t in r.tensors if t.name == "token_embd.weight"), None)
    if tok_emb is not None and tok_emb.shape[1] != n_vocab:
        errors.append(
            f"token_embd.weight second dim {tok_emb.shape[1]} != vocab {n_vocab}")
    out = next((t for t in r.tensors if t.name == "output.weight"), None)
    if out is not None and out.shape[1] != n_vocab:
        errors.append(
            f"output.weight second dim {out.shape[1]} != vocab {n_vocab}")

    # ---- Tensors ----
    names = {t.name for t in r.tensors}
    for prefix in REQUIRED_TENSOR_PREFIXES:
        if prefix not in names:
            errors.append(f"missing tensor {prefix}")

    blk = r.fields.get("llama.block_count")
    n_blocks = int(blk.parts[-1][0]) if blk is not None else -1
    for i in range(max(n_blocks, 0)):
        for tname in BLOCK_TENSORS:
            if f"blk.{i}.{tname}" not in names:
                errors.append(f"missing tensor blk.{i}.{tname}")

    # no learned position embeddings (llama arch uses RoPE)
    if any("position_embd" in n for n in names):
        errors.append("position_embd.weight present but llama arch uses RoPE")

    # ---- FUTO runtime vocab lookups ----
    from sentencepiece import SentencePieceProcessor
    f = r.fields.get("keyboardlm.ext_tokenizer_data")
    if f is not None:
        blob = bytes(int(p[0]) for p in f.parts[5:])
        sp = SentencePieceProcessor()
        if not sp.LoadFromSerializedProto(blob):
            errors.append("keyboardlm.ext_tokenizer_data is not a valid "
                          "SentencePiece model")
        else:
            for tok in ("<XBU>", "<XBC>", "<XEC>", "<XC0>", "<CHAR_A>", "▁"):
                if sp.PieceToId(tok) <= 0:
                    errors.append(f"vocab missing token {tok!r}")
            if sp.PieceToId("<CHAR_Z>") != sp.PieceToId("<CHAR_A>") + 25:
                errors.append("<CHAR_A..Z> not consecutive in vocab")
            if sp.IdToPiece(1) != "<s>":
                errors.append("token id 1 is not <s> (FUTO hardcodes BOS=1)")

    # ---- Report ----
    if errors:
        print(f"FAIL: {MODEL}")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"OK: {MODEL} is a valid FUTO keyboard model "
          f"(vocab={n_vocab}, blocks={n_blocks})")
    print(f"    features: {val_str(r.fields['keyboardlm.features'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
