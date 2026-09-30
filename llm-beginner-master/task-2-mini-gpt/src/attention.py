"""Causal MHA + KV Cache。"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope, build_rope_cache


KVCache = Tuple[torch.Tensor, torch.Tensor]  # (K, V)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # buffer，随 .to(device) 一起走
        cos, sin = build_rope_cache(self.d_k, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        return_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, KVCache]:
        B, T, C = x.shape
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)

        # RoPE 只加 Q/K；增量时从 past 长度接着编位置
        pos_offset = 0 if kv_cache is None else kv_cache[0].size(2)
        assert pos_offset + T <= self.max_seq_len, (
            f"序列长度 {pos_offset + T} 超过 max_seq_len={self.max_seq_len}"
        )
        q, k = apply_rope(q, k, self.rope_cos, self.rope_sin, position_offset=pos_offset)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache: KVCache = (k, v)

        T_k = k.size(2)
        scale = 1.0 / math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 全量需要 causal mask；增量 T=1 时 past 都可看
        if T == T_k:
            causal = torch.triu(
                torch.ones(T, T_k, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal, float("-inf"))
        elif T != 1:
            q_pos = torch.arange(pos_offset, pos_offset + T, device=x.device)[:, None]
            k_pos = torch.arange(T_k, device=x.device)[None, :]
            scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # view 前需 contiguous
        out = self.resid_drop(self.W_o(y))

        if return_cache:
            return out, new_cache
        return out


class TransformerBlock(nn.Module):
    """Pre-LN decoder block。"""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        return_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, KVCache]:
        if return_cache:
            a, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache, return_cache=True)
            x = x + a
            x = x + self.ffn(self.ln2(x))
            return x, new_cache
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache, return_cache=False)
        x = x + self.ffn(self.ln2(x))
        return x
