"""玩具 LM：复用 scaled_dot_product_attention + causal mask，验证不看未来。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.attention import scaled_dot_product_attention  # noqa: E402


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """上三角 True = 屏蔽未来。"""
    return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()


class ToyLM(nn.Module):
    """单层因果自注意力 LM。"""

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.emb = nn.Embedding(vocab_size, d_model)
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        x = self.emb(ids)
        Q, K, V = self.W_qkv(x).chunk(3, dim=-1)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn_out = scaled_dot_product_attention(
            split(Q), split(K), split(V), mask=causal_mask(T, ids.device)
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.lm_head(self.W_o(attn_out))


def demo_no_future_leak() -> None:
    """改未来 V，过去输出应不变。"""
    torch.manual_seed(0)
    B, H, T, D = 1, 1, 5, 8
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)
    mask = causal_mask(T, Q.device)

    out = scaled_dot_product_attention(Q, K, V, mask=mask)
    V2 = V.clone()
    V2[:, :, -1] = 999.0
    out2 = scaled_dot_product_attention(Q, K, V2, mask=mask)

    leaked = (out[:, :, :-1] - out2[:, :, :-1]).abs().max().item()
    print(f"causal leak diff = {leaked:.2e}  ({'OK' if leaked < 1e-6 else 'FAIL'})")


def demo_train_step() -> None:
    """随机 token 跑一步 CE。"""
    vocab_size, B, T = 32, 2, 16
    model = ToyLM(vocab_size)
    ids = torch.randint(0, vocab_size, (B, T))
    logits = model(ids)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), ids[:, 1:].reshape(-1))
    print(f"toy LM logits shape = {tuple(logits.shape)}, loss = {loss.item():.4f}")


if __name__ == "__main__":
    demo_no_future_leak()
    demo_train_step()
