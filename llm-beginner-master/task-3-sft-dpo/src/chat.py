"""Qwen chat template + 只对 assistant 正文计 loss。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

_DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B"


def format_messages(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """<|im_start|>role\\ncontent<|im_end|>\\n；推理时加 generation prompt。"""
    pieces: List[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        pieces.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    if add_generation_prompt:
        pieces.append(f"{IM_START}assistant\n")
    return "".join(pieces)


def _get_tokenizer(tokenizer=None):
    if tokenizer is not None:
        return tokenizer
    from transformers import AutoTokenizer

    if not _DEFAULT_MODEL.exists():
        raise FileNotFoundError(
            f"找不到 {_DEFAULT_MODEL}，请先运行 data/download.py 或传入 tokenizer="
        )
    return AutoTokenizer.from_pretrained(str(_DEFAULT_MODEL), trust_remote_code=True)


def build_labels(
    input_ids: torch.Tensor,
    messages: List[Dict[str, str]],
    tokenizer=None,
) -> torch.Tensor:
    """与 input_ids 同形；仅 assistant content 保留 id，其余 -100。"""
    tok = _get_tokenizer(tokenizer)
    squeeze = input_ids.dim() == 1
    if squeeze:
        ids = input_ids
    else:
        if input_ids.size(0) != 1:
            raise ValueError("build_labels 仅支持单条样本或 (L,)；batch 请逐条调用")
        ids = input_ids[0]

    text = format_messages(messages)
    id_list = ids.tolist()

    # 对齐 tokenization（是否 add_special_tokens）
    offset_mapping = None
    for add_special in (True, False):
        enc = tok(
            text,
            return_offsets_mapping=True,
            add_special_tokens=add_special,
            return_tensors=None,
        )
        if enc["input_ids"] == id_list:
            offset_mapping = enc["offset_mapping"]
            break

    if offset_mapping is None:
        enc = tok(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            return_tensors=None,
        )
        offset_mapping = enc["offset_mapping"]

    labels = torch.full_like(ids, -100)
    L = min(len(ids), len(offset_mapping))

    search_from = 0
    content_spans = []
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        header = f"{IM_START}assistant\n"
        pos = text.find(header, search_from)
        if pos < 0:
            continue
        c_start = pos + len(header)
        c_end = text.find(IM_END, c_start)
        if c_end < 0:
            c_end = len(text)
        content_spans.append((c_start, c_end))
        search_from = c_end

    for i in range(L):
        s, e = offset_mapping[i]
        if e <= s:
            continue
        for c_start, c_end in content_spans:
            # token 完全落在 assistant content 内才计 loss
            if s >= c_start and e <= c_end:
                labels[i] = ids[i]
                break

    if not squeeze:
        labels = labels.unsqueeze(0)
    return labels


def messages_to_ids(
    messages: List[Dict[str, str]],
    tokenizer,
    max_length: int = 1024,
    add_generation_prompt: bool = False,
) -> Dict[str, torch.Tensor]:
    """messages → input_ids / attention_mask / labels（未 pad）。"""
    text = format_messages(messages, add_generation_prompt=add_generation_prompt)
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"][0]
    # 先对完整序列算 labels，再按截断长度切
    full_text = format_messages(messages)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    full_labels = build_labels(full_ids, messages, tokenizer=tokenizer)
    labels = full_labels[: input_ids.size(0)].clone()
    if labels.size(0) < input_ids.size(0):
        pad = torch.full((input_ids.size(0) - labels.size(0),), -100, dtype=torch.long)
        labels = torch.cat([labels, pad])

    return {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"][0],
        "labels": labels,
    }
