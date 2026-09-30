"""在 gold_qa.jsonl 上评测 Recall@k / MRR（与自检口径一致）。"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.retriever import Retriever  # noqa: E402


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def main(args: argparse.Namespace) -> None:
    gold_path = ROOT / "data" / "gold_qa.jsonl"
    gold = [
        json.loads(line)
        for line in gold_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retriever = Retriever(use_reranker=not args.no_rerank)
    ks = [1, 3, 5, 10]
    hits = {k: 0 for k in ks}
    rr = []

    for item in gold:
        anchors = [normalize_text(a) for a in item.get("gold_anchors", [])]
        results = retriever.retrieve(item["question"], k=10)
        rank = None
        for i, r in enumerate(results, 1):
            text = normalize_text(r.get("text", ""))
            if any(a and a in text for a in anchors):
                rank = i
                break
        if rank:
            for k in ks:
                if rank <= k:
                    hits[k] += 1
            rr.append(1.0 / rank)
        else:
            rr.append(0.0)
        mark = "HIT" if rank else "MISS"
        print(f"{item.get('id')}: {mark} rank={rank}")

    n = len(gold)
    metrics = {f"recall@{k}": round(hits[k] / n, 3) for k in ks}
    metrics["mrr"] = round(sum(rr) / n, 3)
    print("\n=== metrics ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    out = ROOT / "eval" / "retrieval_metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--no-rerank", action="store_true", help="关闭 reranker 做消融")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
