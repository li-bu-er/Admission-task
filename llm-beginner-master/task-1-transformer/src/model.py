"""Transformer 文本分类：Embedding + PE + Encoder + Pooling + Classifier。"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn

from .block import TransformerBlock


class AbsolutePositionalEncoding(nn.Module):
    """可学习绝对位置编码。"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        return x + self.pe(positions).unsqueeze(0)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int | None = None,
        num_classes: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = AbsolutePositionalEncoding(d_model, max_len)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff=d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)  # Pre-LN 栈顶再归一化一次
        self.classifier = nn.Linear(d_model, num_classes)

    def _padding_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """True = PAD，(B,1,1,T) 广播到 attention。"""
        return (ids == self.pad_id).unsqueeze(1).unsqueeze(2)

    def encode(
        self,
        ids: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        assert ids.size(1) <= self.max_len, f"序列长度超过 max_len={self.max_len}"

        mask = self._padding_mask(ids)
        x = self.embed_dropout(self.pos_enc(self.token_emb(ids)))

        attn_list: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x, mask=mask)
            if return_attn and layer.attn.last_attn_weights is not None:
                attn_list.append(layer.attn.last_attn_weights)

        x = self.final_ln(x)
        if return_attn:
            return x, attn_list
        return x

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(ids)

        # mean pooling，跳过 PAD
        mask_flat = ids != self.pad_id
        lengths = mask_flat.sum(dim=1).clamp(min=1).unsqueeze(1).float()
        pooled = (hidden * mask_flat.unsqueeze(-1).float()).sum(dim=1) / lengths
        return self.classifier(pooled)


PAD_TOKEN, UNK_TOKEN, CLS_TOKEN = "<pad>", "<unk>", "<cls>"


def build_char_vocab(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    from collections import Counter

    counter: Counter = Counter()
    for text in texts:
        counter.update(list(text))

    vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1, CLS_TOKEN: 2}
    for ch, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        if freq >= min_freq and ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def tokenize_text(text: str, vocab: Dict[str, int], max_len: int = 512) -> torch.Tensor:
    unk_id = vocab[UNK_TOKEN]
    cls_id = vocab[CLS_TOKEN]
    ids = [cls_id] + [vocab.get(ch, unk_id) for ch in text]
    return torch.tensor(ids[:max_len], dtype=torch.long)


def load_for_eval(ckpt_path: str) -> Tuple[TransformerClassifier, Callable[[str], torch.Tensor]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    vocab: Dict[str, int] = ckpt["vocab"]

    model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config.get("d_ff"),
        num_classes=config.get("num_classes", 2),
        max_len=config.get("max_len", 512),
        dropout=0.0,
        pad_id=vocab[PAD_TOKEN],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    max_len = config.get("max_len", 512)

    def tokenize_fn(text: str) -> torch.Tensor:
        return tokenize_text(text, vocab, max_len=max_len)

    return model, tokenize_fn
