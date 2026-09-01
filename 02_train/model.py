"""
KeyboardLM: small GPT decoder that exactly matches the "llama" architecture
FUTO Keyboard's bundled llama.cpp runs at inference
(native/jni/src/ggml/llama.cpp build_llama / llm_build_ffn):

  - RMSNorm (weight-only, no bias) on attn_norm and ffn_norm
  - RoPE (theta=10000, full head dim, consecutive-pair convention) — NO
    learned position embeddings
  - SiLU-gated FFN: down(silu(gate(x)) * up(x))  ->  blk.*.ffn_{gate,up,down}
  - lm_head shares weights with token embeddings (output.weight in GGUF)

Nothing else: FUTO implements autocorrect / swipe / char-embed-mixing at
runtime from the special vocab tokens (<XBU> <XBC> <XEC> <XC0> <CHAR_A..Z>)
and the token_embd.weight table, so there are no extra heads to train or
export.

State dict names are chosen so 03_convert/export_to_gguf.py can write the
standard llama GGUF tensor names directly.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 16_000
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    max_seq_len: int = 256
    ffn_mult: int = 4
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.1


class RMSNorm(nn.Module):
    """Weight-only RMSNorm, matching ggml LLM_NORM_RMS."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().square().mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def precompute_rope(theta: float, dim: int, max_seq_len: int) -> torch.Tensor:
    """Returns (max_seq_len, dim/2) frequencies, same formula as ggml RoPE
    (freq = theta ** (-2i / dim) for each consecutive pair i)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
    return positions * inv_freq


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs over the full head dim.
    x: (B, T, H, D), freqs: (T, D/2). Convention identical to ggml rope mode 0
    (NEOX) and HF LLaMA (pairs (x_0,x_1), (x_2,x_3), ...)."""
    freqs = freqs.to(x.dtype)
    cos = freqs.cos().unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(2)
    x = x.reshape(*x.shape[:-1], -1, 2)          # (B, T, H, D/2, 2)
    x0, x1 = x[..., 0], x[..., 1]
    rot = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return rot.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class MLP(nn.Module):
    """SiLU gated FFN: down(silu(gate(x)) * up(x)), like llama ffn_gate/up/down."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ffn = cfg.n_embd * cfg.ffn_mult
        self.gate = nn.Linear(cfg.n_embd, ffn, bias=False)
        self.up = nn.Linear(cfg.n_embd, ffn, bias=False)
        self.down = nn.Linear(ffn, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), freqs)
        x = x + self.mlp(self.ln2(x))
        return x


class KeyboardLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying (output.weight)

        self.register_buffer(
            "freqs",
            precompute_rope(cfg.rope_theta, cfg.n_embd // cfg.n_head, cfg.max_seq_len),
            persistent=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = token_ids.shape
        x = self.tok_emb(token_ids)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, self.freqs[:T])
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
