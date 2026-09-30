"""采样：greedy / temperature / top-k / top-p。"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """temperature<=0 走 greedy。"""
    squeeze = logits.dim() == 1
    if squeeze:
        logits = logits.unsqueeze(0)

    if temperature is None or temperature <= 0:
        next_id = torch.argmax(logits, dim=-1)
        return next_id.squeeze(0) if squeeze else next_id

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        # 保留刚超过 top_p 的那个，后面砍掉
        mask = cumprobs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return next_id.squeeze(0) if squeeze else next_id
