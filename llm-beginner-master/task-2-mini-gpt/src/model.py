"""MiniGPT：Decoder-only + RoPE + KV Cache。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .attention import KVCache, TransformerBlock
from .sampling import sample_next_token
from .tokenizer import BPETokenizer


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        block_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.max_seq_len = block_size
        # eval 会一次喂 block_size+1 算 NLL
        self._max_forward_len = block_size + 1

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, self._max_forward_len, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        ids: torch.Tensor,
        kv_cache: Optional[List[KVCache]] = None,
        return_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[KVCache]]:
        B, T = ids.shape
        assert T <= self._max_forward_len, (
            f"T={T} > max_forward_len={self._max_forward_len} "
            f"(block_size={self.block_size})"
        )

        x = self.drop(self.tok_emb(ids))

        new_caches: List[KVCache] = []
        for i, layer in enumerate(self.layers):
            layer_cache = None if kv_cache is None else kv_cache[i]
            if return_cache:
                x, layer_new = layer(x, kv_cache=layer_cache, return_cache=True)
                new_caches.append(layer_new)
            else:
                x = layer(x, kv_cache=layer_cache, return_cache=False)

        logits = self.lm_head(self.ln_f(x))
        if return_cache:
            return logits, new_caches
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """自回归；内部走 KV cache。"""
        single = prompt_ids.dim() == 1
        if single:
            prompt_ids = prompt_ids.unsqueeze(0)

        self.eval()
        ids = prompt_ids[:, -self.block_size :]
        logits, cache = self.forward(ids, kv_cache=None, return_cache=True)

        for _ in range(max_new_tokens):
            # 超长就截断并重建 cache，简单可靠
            if ids.size(1) >= self.block_size:
                ids = ids[:, -self.block_size + 1 :]
                logits, cache = self.forward(ids, kv_cache=None, return_cache=True)

            next_logits = logits[:, -1, :]
            next_id = sample_next_token(
                next_logits, top_k=top_k, top_p=top_p, temperature=temperature
            )
            if next_id.dim() == 0:
                next_id = next_id.unsqueeze(0)
            next_id = next_id.view(-1, 1)

            ids = torch.cat([ids, next_id], dim=1)
            logits, cache = self.forward(next_id, kv_cache=cache, return_cache=True)

        return ids.squeeze(0) if single else ids


def load_for_eval(ckpt_path: str) -> Tuple[MiniGPT, BPETokenizer]:
    """加载 model+tokenizer；tokenizer 优先读 ckpt 旁 json。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = MiniGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        block_size=config["block_size"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tok_path = ckpt.get("tokenizer_path")
    if tok_path is None or not Path(tok_path).exists():
        tok_path = str(Path(ckpt_path).resolve().parent / "tokenizer.json")
    tokenizer = BPETokenizer.from_pretrained(tok_path)
    return model, tokenizer
