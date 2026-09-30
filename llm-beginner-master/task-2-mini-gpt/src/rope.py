"""RoPE：只加在 Q/K，前后折半配对（同 LLaMA）。"""
from __future__ import annotations

from typing import Tuple

import torch


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0, "RoPE 要求 head_dim 为偶数"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)  # 对齐 rotate_half
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] → [-x2, x1]。"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """position_offset 给 KV cache 增量。"""
    T = q.size(2)
    cos_t = cos[position_offset : position_offset + T].unsqueeze(0).unsqueeze(0)
    sin_t = sin[position_offset : position_offset + T].unsqueeze(0).unsqueeze(0)
    q_rot = q * cos_t + rotate_half(q) * sin_t
    k_rot = k * cos_t + rotate_half(k) * sin_t
    return q_rot, k_rot
