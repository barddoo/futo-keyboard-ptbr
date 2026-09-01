"""
Export the trained KeyboardLM checkpoint to a FUTO-compatible GGUF.

Writes a standard "llama"-arch GGUF (RMSNorm, RoPE, SiLU-gated FFN) plus the
keyboardlm.* metadata that FUTO Keyboard's loader requires
(native/jni/src/ggml/ModelMeta.cpp). The tokenizer metadata embeds the final
SentencePiece model bytes in keyboardlm.ext_tokenizer_data; the
tokenizer.ggml.* arrays let llama.cpp size its vocab consistently.

There are deliberately NO extra feature-head tensors: FUTO derives
autocorrect / swipe / char-embed-mixing at runtime from the special vocab
tokens (<XBU> <XBC> <XEC> <XC0> <CHAR_A..Z>) and token_embd.weight.

Note on tensor layout: the gguf writer reverses numpy dims into ggml's
column-major order, so PyTorch weights are passed as-is (their natural
[out, in] shape) and the file ends up with the ggml [in, out] / [n_embd,
vocab] layout that llama.cpp expects.
"""
import datetime
import os
import re
import sys
from pathlib import Path

import sentencepiece as spm
import torch
from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).parent.parent / "02_train"))
from model import KeyboardLM, ModelConfig  # noqa: E402

CKPT = Path(os.environ.get(
    "KEYBOARD_CKPT",
    Path(__file__).parent.parent / "02_train" / "checkpoints" / "best.pt",
))
SP_MODEL = Path(os.environ.get(
    "KEYBOARD_SP_MODEL",
    Path(__file__).parent.parent / "01_data" / "tokenizer" / "pt_br_keyboard.model",
))
OUT_F32 = Path(os.environ.get(
    "KEYBOARD_OUT",
    Path(__file__).parent / "pt_br_keyboard_f32.gguf",
))

# Space-separated string, exactly how FUTO reads keyboardlm.features.
#
# The autocorrect / swipe / char-embed-mixing features are NOT extra weights in
# FUTO — they are behaviors the model learns from correction-format training
# data (prompts with <XBU>/<XBC>/<XEC>/<XC0>/<CHAR_A..Z>). Because the default
# pipeline (01_data/make_correction_data.py) trains on that data, the full set
# is declared by default. To export a plain next-word-only model, override:
#   KEYBOARD_FEATURES="base_v1 inverted_space"
# NOTE: xbu_char_autocorrect_v1 and xc0_swipe_typing_v1 MUST be declared
# together with char_embed_mixing_v1 — FUTO only allocates its embedding-mix
# buffer when that feature is present, and correction decoding reads it.
DEFAULT_FEATURES = ("base_v1 inverted_space xbu_char_autocorrect_v1 "
                    "xc0_swipe_typing_v1 char_embed_mixing_v1")
FUTO_FEATURES = os.environ.get("KEYBOARD_FEATURES", DEFAULT_FEATURES)

# llama.cpp token types (llama_token_type)
T_NORMAL = 1
T_UNKNOWN = 2
T_CONTROL = 3
T_BYTE = 6

_BYTE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
_SPECIAL_SET = {
    "<pad>", "<s>", "</s>", "<unk>",
} | {f"<FUTO{i}>" for i in range(12)} | {
    "<XBU>", "<XBC>", "<XEC>", "<XC0>",
} | {f"<CHAR_{chr(ord('A') + i)}>" for i in range(26)}


def classify(token: str) -> int:
    if token == "<unk>":
        return T_UNKNOWN
    if _BYTE_RE.match(token):
        return T_BYTE
    if token in _SPECIAL_SET:
        return T_CONTROL
    return T_NORMAL


def export(ckpt_path: Path = CKPT, out_path: Path = OUT_F32) -> None:
    print(f"Loading checkpoint {ckpt_path} …")
    torch.serialization.add_safe_globals([ModelConfig])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg: ModelConfig = ckpt["config"]
    state = ckpt["model"]

    # ---- Tokenizer (final model, incl. FUTO control tokens) ----
    sp = spm.SentencePieceProcessor()
    assert sp.Load(str(SP_MODEL)), "failed to load SentencePiece model"
    n_vocab = sp.GetPieceSize()
    assert n_vocab == cfg.vocab_size, (
        f"tokenizer vocab {n_vocab} != model vocab {cfg.vocab_size}")
    tokens = [sp.IdToPiece(i) for i in range(n_vocab)]
    scores = [float(sp.GetScore(i)) for i in range(n_vocab)]
    token_types = [classify(t) for t in tokens]

    writer = GGUFWriter(str(out_path), arch="llama")

    # ---- General metadata ----
    writer.add_name("KeyboardLM-pt-BR")
    writer.add_author("pt-br-futo")
    writer.add_description("pt-BR keyboard language model (FUTO Keyboard)")
    writer.add_license("See model creator")
    writer.add_url("https://keyboard.futo.tech/")

    # ---- llama architecture hparams (read by llama.cpp llm_load_hparams) ----
    head_dim = cfg.n_embd // cfg.n_head
    writer.add_uint32("llama.context_length", cfg.max_seq_len)
    writer.add_uint32("llama.embedding_length", cfg.n_embd)
    writer.add_uint32("llama.block_count", cfg.n_layer)
    writer.add_uint32("llama.feed_forward_length", cfg.n_embd * cfg.ffn_mult)
    writer.add_uint32("llama.rope.dimension_count", head_dim)
    writer.add_uint32("llama.attention.head_count", cfg.n_head)
    writer.add_uint32("llama.attention.head_count_kv", cfg.n_head)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps)
    writer.add_float32("llama.rope.freq_base", cfg.rope_theta)

    # ---- Tokenizer metadata (llama.cpp sizes its vocab from these) ----
    writer.add_string("tokenizer.ggml.model", "llama")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.scores", scores)
    writer.add_array("tokenizer.ggml.token_type", token_types)
    writer.add_uint32("tokenizer.ggml.bos_token_id", 1)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 2)

    # ---- FUTO Keyboard metadata (ModelMeta.cpp) ----
    writer.add_string("keyboardlm.languages", "pt")
    writer.add_uint32("keyboardlm.finetuning_count", 0)
    writer.add_string(
        "keyboardlm.history", f"{datetime.date.today()}: Model created")
    writer.add_string("keyboardlm.features", FUTO_FEATURES)
    writer.add_string("keyboardlm.ext_tokenizer_type", "sentencepiece")
    writer.add_array("keyboardlm.ext_tokenizer_data", SP_MODEL.read_bytes())

    # ---- Tensors (standard llama names; see model.py for state dict) ----
    def w(name: str, t: torch.Tensor) -> None:
        writer.add_tensor(name, t.detach().float().cpu().numpy())

    w("token_embd.weight", state["tok_emb.weight"])
    w("output_norm.weight", state["ln_f.weight"])
    w("output.weight", state["tok_emb.weight"])  # tied lm_head

    for i in range(cfg.n_layer):
        b = f"blk.{i}"
        w(f"{b}.attn_norm.weight", state[f"blocks.{i}.ln1.weight"])
        qkv = state[f"blocks.{i}.attn.qkv.weight"]  # (3*n_embd, n_embd)
        w(f"{b}.attn_q.weight", qkv[:cfg.n_embd])
        w(f"{b}.attn_k.weight", qkv[cfg.n_embd:2 * cfg.n_embd])
        w(f"{b}.attn_v.weight", qkv[2 * cfg.n_embd:])
        w(f"{b}.attn_output.weight", state[f"blocks.{i}.attn.out_proj.weight"])
        w(f"{b}.ffn_norm.weight", state[f"blocks.{i}.ln2.weight"])
        w(f"{b}.ffn_gate.weight", state[f"blocks.{i}.mlp.gate.weight"])
        w(f"{b}.ffn_up.weight", state[f"blocks.{i}.mlp.up.weight"])
        w(f"{b}.ffn_down.weight", state[f"blocks.{i}.mlp.down.weight"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"F32 GGUF written to {out_path} ({n_vocab} vocab, "
          f"{cfg.n_layer} layers, {cfg.n_embd} embd)")
    print("Next: quantize with llama.cpp")
    print(f"  ./llama.cpp/quantize {out_path} pt_br_keyboard_q6k.gguf Q6_K")


if __name__ == "__main__":
    export()
