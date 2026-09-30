"""BGE Cross-Encoder 精排。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RERANK_PATH = ROOT / "models" / "bge-reranker-base"


class Reranker:
    def __init__(self, model_path: Optional[str] = None):
        from sentence_transformers import CrossEncoder

        path = str(model_path or DEFAULT_RERANK_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(path, device=device)

    def rerank(
        self,
        query: str,
        docs: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        if not docs:
            return []
        pairs = [(query, d.get("text", "")) for d in docs]
        scores = self.model.predict(pairs)
        ranked = []
        for d, s in zip(docs, scores):
            item = dict(d)
            item["score"] = float(s)
            ranked.append(item)
        ranked.sort(key=lambda x: x["score"], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked
