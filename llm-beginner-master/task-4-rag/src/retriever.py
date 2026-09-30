"""向量召回（+ 可选 Rerank）→ List[{text, score, source}]。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .indexer import (
    DEFAULT_INDEX_DIR,
    Indexer,
    embed_queries,
)

ROOT = Path(__file__).resolve().parents[1]


class Retriever:
    def __init__(
        self,
        index_dir: Optional[str] = None,
        use_reranker: bool = True,
        recall_multiplier: int = 4,
        auto_build: bool = True,
    ):
        self.index_dir = Path(index_dir or DEFAULT_INDEX_DIR)
        self.use_reranker = use_reranker
        self.recall_multiplier = max(1, recall_multiplier)

        if not (self.index_dir / "faiss.index").exists():
            if auto_build:
                self._auto_build_index()
            else:
                raise FileNotFoundError(
                    f"缺少索引 {self.index_dir}，请先运行 python build_index.py"
                )

        self.indexer = Indexer()
        self.indexer.load(self.index_dir)
        self.embedder = self.indexer.model

        self.reranker = None
        if use_reranker:
            rerank_path = ROOT / "models" / "bge-reranker-base"
            if rerank_path.exists():
                from .reranker import Reranker

                self.reranker = Reranker(str(rerank_path))
            else:
                print("[提示] reranker 模型不存在，仅用向量召回")

    def _auto_build_index(self) -> None:
        # 索引缺失时从 kb.pdf 建默认索引（recursive, 512/128）
        from .chunker import chunk_pdf
        from .indexer import Indexer as _Indexer

        pdf = ROOT / "data" / "kb.pdf"
        if not pdf.exists():
            raise FileNotFoundError(f"缺少 {pdf}，请先 python data/download.py")
        print("索引不存在，正在从 kb.pdf 自动构建 ...")
        chunks = chunk_pdf(pdf, chunk_size=512, overlap=128, strategy="recursive")
        ix = _Indexer()
        ix.build(chunks)
        ix.save(self.index_dir)

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        if k <= 0:
            return []
        index = self.indexer.index
        chunks = self.indexer.chunks
        if index is None or not chunks:
            return []

        # 初检多召回一些，留给 rerank
        candidate_n = min(len(chunks), max(k * self.recall_multiplier, k))
        q_vec = embed_queries(self.embedder, [query])
        scores, indices = index.search(q_vec, candidate_n)

        hits: List[Dict] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0:
                continue
            c = chunks[idx]
            hits.append(
                {
                    "text": c.get("text", ""),
                    "score": float(score),
                    "source": c.get("source", f"chunk:{idx}"),
                    "chunk_id": c.get("chunk_id", idx),
                }
            )

        if self.reranker is not None and hits:
            hits = self.reranker.rerank(query, hits, top_k=k)
        else:
            hits = hits[:k]

        return [
            {"text": h["text"], "score": float(h["score"]), "source": h["source"]}
            for h in hits
        ]
