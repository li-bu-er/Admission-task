"""BGE embedding + FAISS（IndexFlatIP + L2 normalize ≈ cosine）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import torch

# BGE 中文检索：query 侧必须加此前缀，文档侧不加
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBED_PATH = ROOT / "models" / "bge-small-zh-v1.5"
DEFAULT_INDEX_DIR = ROOT / "data" / "index"


def load_embedder(model_path: Optional[str] = None):
    from sentence_transformers import SentenceTransformer

    path = str(model_path or DEFAULT_EMBED_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(path, device=device)


def embed_documents(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    # 文档侧不加 query 前缀
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(emb, dtype=np.float32)


def embed_queries(model, queries: List[str], batch_size: int = 32) -> np.ndarray:
    # query 侧加 BGE_QUERY_PREFIX
    prefixed = [BGE_QUERY_PREFIX + q for q in queries]
    emb = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(emb, dtype=np.float32)


class Indexer:
    def __init__(self, embed_model_path: Optional[str] = None):
        self.model = load_embedder(embed_model_path)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []

    def build(self, chunks: List[Dict]) -> None:
        if not chunks:
            raise ValueError("chunks 为空，无法建索引")
        self.chunks = list(chunks)
        texts = [c["text"] for c in self.chunks]
        vectors = embed_documents(self.model, texts)
        dim = vectors.shape[1]
        # 内积 + 已归一化 ≡ cosine
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        print(f"FAISS 索引完成：N={len(chunks)}, dim={dim}")

    def save(self, index_dir: str | Path = DEFAULT_INDEX_DIR) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            raise RuntimeError("请先 build")
        # FAISS C++ FileIO 在 Windows 上打不开含非 ASCII 的路径，改走 Python 字节 IO
        index_path = index_dir / "faiss.index"
        buf = faiss.serialize_index(self.index)
        index_path.write_bytes(np.asarray(buf, dtype=np.uint8).tobytes())
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        meta = {
            "n_chunks": len(self.chunks),
            "dim": int(self.index.d),
            "embed_model": str(DEFAULT_EMBED_PATH),
            "metric": "ip_normalized",
        }
        (index_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已保存索引到 {index_dir}")

    def load(self, index_dir: str | Path = DEFAULT_INDEX_DIR) -> None:
        index_dir = Path(index_dir)
        index_path = index_dir / "faiss.index"
        raw = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
        self.index = faiss.deserialize_index(raw)
        self.chunks = []
        with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))
        print(f"已加载索引：{len(self.chunks)} chunks, dim={self.index.d}")
