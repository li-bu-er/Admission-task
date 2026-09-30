"""端到端 RAG：retrieve → generate → {answer, sources}。"""
from __future__ import annotations

from typing import Dict, List, Optional

from .generator import generate_answer
from .retriever import Retriever

# 懒加载，避免每次 answer 都重建索引/模型
_retriever: Optional[Retriever] = None


def get_retriever(force_reload: bool = False) -> Retriever:
    global _retriever
    if _retriever is None or force_reload:
        _retriever = Retriever(use_reranker=True)
    return _retriever


def answer(query: str, k: int = 5) -> Dict:
    # 自检入口：{answer, sources}
    retriever = get_retriever()
    # 检索稍多再截断，兼顾召回与上下文长度
    sources: List[Dict] = retriever.retrieve(query, k=max(k, 10))[:k]
    if not sources:
        return {
            "answer": "根据提供的资料无法确定。",
            "sources": [{"text": "", "score": 0.0, "source": "none"}],
        }
    ans = generate_answer(query, sources)
    return {"answer": ans, "sources": sources}
