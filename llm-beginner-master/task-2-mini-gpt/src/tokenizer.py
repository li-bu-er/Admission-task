"""Byte-level BPE（不用 tiktoken / sentencepiece）。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _get_pair_counts(ids: List[int]) -> Counter:
    counts: Counter = Counter()
    for i in range(len(ids) - 1):
        counts[(ids[i], ids[i + 1])] += 1
    return counts


def _merge_once(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """左到右非重叠 merge。"""
    a, b = pair
    out: List[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE：0..255 起步，按频次 merge。"""

    def __init__(
        self,
        merges: Optional[List[Tuple[int, int]]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        self.merges: List[Tuple[int, int]] = list(merges or [])
        self.special_tokens: List[str] = list(
            special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"]
        )
        self._rebuild_tables()

    def _rebuild_tables(self) -> None:
        self.id_to_bytes: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            new_id = 256 + i
            self.id_to_bytes[new_id] = self.id_to_bytes[a] + self.id_to_bytes[b]

        self.special_to_id: Dict[str, int] = {}
        base = 256 + len(self.merges)
        for j, name in enumerate(self.special_tokens):
            sid = base + j
            self.special_to_id[name] = sid
            self.id_to_bytes[sid] = b""

        self.pair_to_id: Dict[Tuple[int, int], int] = {
            pair: 256 + i for i, pair in enumerate(self.merges)
        }
        self.merge_rank: Dict[Tuple[int, int], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<eos>"]

    def train(self, text: str, vocab_size: int = 1000) -> None:
        n_special = len(self.special_tokens)
        assert vocab_size > 256 + n_special, "vocab_size 过小"
        num_merges = vocab_size - 256 - n_special

        ids = list(text.encode("utf-8"))
        self.merges = []

        for _ in range(num_merges):
            counts = _get_pair_counts(ids)
            if not counts:
                break
            pair, _ = counts.most_common(1)[0]
            new_id = 256 + len(self.merges)
            ids = _merge_once(ids, pair, new_id)
            self.merges.append(pair)

        self._rebuild_tables()

    def encode(self, text: str) -> List[int]:
        ids = list(text.encode("utf-8"))
        if not self.merges:
            return ids

        # 每次合训练时最早出现的那对，直到合不动
        while len(ids) >= 2:
            best_pair = None
            best_rank = len(self.merges) + 1
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            ids = _merge_once(ids, best_pair, self.pair_to_id[best_pair])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        chunks: List[bytes] = []
        for i in ids:
            i = int(i)
            if i in self.special_to_id.values():
                continue  # special 是空字节，别污染文本
            chunks.append(self.id_to_bytes[i])
        return b"".join(chunks).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        payload = {
            "merges": [list(p) for p in self.merges],
            "special_tokens": self.special_tokens,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(p) for p in payload["merges"]]
        specials = payload.get("special_tokens")
        return cls(merges=merges, special_tokens=specials)
