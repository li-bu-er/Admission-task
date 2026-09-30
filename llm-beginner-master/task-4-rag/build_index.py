"""从 data/kb.pdf 构建 FAISS 索引（写入 data/index/）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chunker import chunk_pdf  # noqa: E402
from src.indexer import Indexer  # noqa: E402


def main(args: argparse.Namespace) -> None:
    pdf = ROOT / "data" / "kb.pdf"
    if not pdf.exists():
        raise SystemExit(f"缺少 {pdf}，请先运行 python data/download.py")

    print(
        f"切分策略={args.strategy}, chunk_size={args.chunk_size}, overlap={args.overlap}"
    )
    chunks = chunk_pdf(
        pdf,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        strategy=args.strategy,
    )
    print(f"得到 {len(chunks)} 个 chunk")
    if chunks:
        avg = sum(len(c["text"]) for c in chunks) / len(chunks)
        print(f"平均 chunk 长度 = {avg:.1f} 字符")

    indexer = Indexer(embed_model_path=args.embed_model)
    indexer.build(chunks)
    out = ROOT / "data" / "index"
    indexer.save(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build RAG index from kb.pdf")
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--strategy", choices=["fixed", "recursive"], default="recursive")
    p.add_argument(
        "--embed_model",
        type=str,
        default=None,
        help="默认 models/bge-small-zh-v1.5",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
