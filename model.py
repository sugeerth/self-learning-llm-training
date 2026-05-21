"""LLaMA-style decoder-only transformer with swappable components.

Components (all toggleable via BlockSpec / ModelConfig):
- Norm:        RMSNorm (LLaMA) or LayerNorm (GPT-2)
- Attention:   MHA | GQA | MQA (GQA groups Q heads into fewer KV heads)
- FFN:         SwiGLU (LLaMA) or GELU-MLP (GPT-2)
- Residual:    pre-norm (stable, modern) or post-norm (original Transformer)
- Positional:  RoPE (rotary) or learned embeddings (GPT-2 style)
- Tie embeds:  tok_emb weight shared with lm_head (saves vocab*d_model params)

Each layer can be configured independently via `layers: list[BlockSpec]`.
If `layers` is None we build n_layers identical blocks using uniform defaults.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BlockSpec:
    norm: str = "rms"        # rms | layer
    attn: str = "gqa"        # mha | gqa | mqa
    ffn: str = "swiglu"      # swiglu | gelu
    residual: str = "pre"    # pre | post


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2      # used when a block.attn == "gqa"
    d_ff_mult: float = 8 / 3
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    pos_type: str = "rope"   # rope | learned
    layers: Optional[list] = None  # list[dict|BlockSpec]; None => uniform


def _as_block_specs(cfg: ModelConfig) -> list[BlockSpec]:
    if cfg.layers is None:
        return [BlockSpec() for _ in range(cfg.n_layers)]
    specs = []
    for item in cfg.layers:
        if isinstance(item, BlockSpec):
            specs.append(item)
        else:
            specs.append(BlockSpec(**item))
    return specs


# ---------- norms ----------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "rms":
        return RMSNorm(dim)
    if kind == "layer":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm: {kind}")


# ---------- RoPE ----------
def precompute_rope(dim: int, max_seq_len: int, theta: float, device) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: dim // 2].float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[: x.size(1)].view(1, x.size(1), 1, -1)
    out = torch.view_as_real(xc * freqs_cis).flatten(-2)
    return out.type_as(x)


# ---------- attention ----------
class Attention(nn.Module):
    """Supports MHA (n_kv=n_heads), GQA (n_kv<n_heads), MQA (n_kv=1)."""

    def __init__(self, cfg: ModelConfig, attn_kind: str, use_rope: bool):
        super().__init__()
        if attn_kind == "mha":
            n_kv = cfg.n_heads
        elif attn_kind == "mqa":
            n_kv = 1
        else:  # gqa
            n_kv = cfg.n_kv_heads
        assert cfg.n_heads % n_kv == 0, f"n_heads={cfg.n_heads} must be divisible by n_kv={n_kv}"
        self.n_heads = cfg.n_heads
        self.n_kv_heads = n_kv
        self.head_dim = cfg.d_model // cfg.n_heads
        self.repeat = cfg.n_heads // n_kv
        self.use_rope = use_rope
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, n_kv * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, n_kv * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        if self.use_rope and freqs_cis is not None:
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)
        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=2)
            v = v.repeat_interleave(self.repeat, dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------- FFNs ----------
class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.d_model * cfg.d_ff_mult)
        hidden = 64 * ((hidden + 63) // 64)
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GELU_FFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.d_model * 4)  # GPT-2 convention
        hidden = 64 * ((hidden + 63) // 64)
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


def make_ffn(kind: str, cfg: ModelConfig) -> nn.Module:
    if kind == "swiglu":
        return SwiGLU(cfg)
    if kind == "gelu":
        return GELU_FFN(cfg)
    raise ValueError(f"unknown ffn: {kind}")


# ---------- block ----------
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, spec: BlockSpec):
        super().__init__()
        self.spec = spec
        self.attn_norm = make_norm(spec.norm, cfg.d_model)
        self.attn = Attention(cfg, spec.attn, use_rope=(cfg.pos_type == "rope"))
        self.ffn_norm = make_norm(spec.norm, cfg.d_model)
        self.ffn = make_ffn(spec.ffn, cfg)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor]) -> torch.Tensor:
        if self.spec.residual == "pre":
            x = x + self.attn(self.attn_norm(x), freqs_cis)
            x = x + self.ffn(self.ffn_norm(x))
        else:  # post-norm
            x = self.attn_norm(x + self.attn(x, freqs_cis))
            x = self.ffn_norm(x + self.ffn(x))
        return x


# ---------- LLM ----------
class LLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_type == "learned":
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        else:
            self.pos_emb = None
        specs = _as_block_specs(cfg)
        self.blocks = nn.ModuleList([Block(cfg, s) for s in specs])
        self.norm = make_norm(specs[0].norm if specs else "rms", cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        if cfg.pos_type == "rope":
            self.register_buffer(
                "freqs_cis",
                precompute_rope(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta, "cpu"),
                persistent=False,
            )
        else:
            self.freqs_cis = None

        self.apply(self._init)
        scale = 1.0 / math.sqrt(2 * max(len(specs), 1))
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        if self.cfg.tie_embeddings:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        freqs = self.freqs_cis.to(x.device) if self.freqs_cis is not None else None
        for blk in self.blocks:
            x = blk(x, freqs)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        eos_token: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_token is not None and (nxt == eos_token).all():
                break
        return idx
