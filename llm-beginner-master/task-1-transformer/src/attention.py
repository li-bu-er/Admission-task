"""Scaled Dot-Product Attention + Multi-Head（手写，不用 nn.MultiheadAttention）。

mask: True = 屏蔽。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    return_weights: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # 填 -inf，别乘 0，不然 softmax 还会漏概率
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    attn_for_v = F.dropout(attn_weights, p=dropout_p, training=training)
    out = torch.matmul(attn_for_v, V)

    if return_weights:
        return out, attn_weights
    return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout_p = dropout

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.last_attn_weights: Optional[torch.Tensor] = None  # visualize 用

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, d_k = x.shape
        # transpose 后不连续，view 前要 contiguous
        return x.transpose(1, 2).contiguous().view(B, T, H * d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        Q = self._split_heads(self.W_q(x))
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        attn_out, attn_weights = scaled_dot_product_attention(
            Q,
            K,
            V,
            mask=mask,
            dropout_p=self.dropout_p,
            training=self.training,
            return_weights=True,
        )
        self.last_attn_weights = attn_weights.detach()

        out = self.W_o(self._merge_heads(attn_out))
        if return_weights:
            return out, attn_weights
        return out
