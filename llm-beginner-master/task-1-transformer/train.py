"""ChnSentiCorp 训练：AdamW + Cosine LR，按 dev acc 存 best。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import (  # noqa: E402
    PAD_TOKEN,
    TransformerClassifier,
    build_char_vocab,
    tokenize_text,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        ids = tokenize_text(self.texts[idx], self.vocab, max_len=self.max_len)
        return ids, int(self.labels[idx])


def collate_batch(batch, pad_id: int = 0):
    ids_list, labels = zip(*batch)
    max_len = max(x.size(0) for x in ids_list)
    B = len(ids_list)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(ids_list):
        ids[i, : seq.size(0)] = seq
    return ids, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        pred = model(ids).argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device = {device}")

    data_dir = ROOT / "data"
    train_df = pd.read_parquet(data_dir / "train.parquet")
    dev_df = pd.read_parquet(data_dir / "validation.parquet")

    # 二分类只允许 {0,1}；Kerwin11 那份是三分类别混进来
    label_vals = sorted({int(x) for x in train_df["label"].tolist()})
    if label_vals != [0, 1]:
        raise SystemExit(
            f"训练集 label 应为 {{0,1}} 二分类，实际为 {label_vals}。"
            "请重新运行: python data/download.py"
            "（勿使用 Kerwin11 三分类数据）"
        )
    num_classes = 2

    vocab = build_char_vocab(train_df["text"].tolist())
    pad_id = vocab[PAD_TOKEN]
    print(f"vocab_size = {len(vocab)}, num_classes = {num_classes}")

    train_ds = SentimentDataset(train_df["text"], train_df["label"], vocab, args.max_len)
    dev_ds = SentimentDataset(dev_df["text"], dev_df["label"], vocab, args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id),
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id),
        num_workers=0,
    )

    model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        num_classes=num_classes,
        max_len=args.max_len,
        dropout=args.dropout,
        pad_id=pad_id,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = ROOT / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ids), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)
        dev_acc = evaluate(model, dev_loader, device)
        history.append({"epoch": epoch, "loss": avg_loss, "dev_acc": dev_acc})
        print(f"epoch {epoch:02d}  loss={avg_loss:.4f}  dev_acc={dev_acc:.4f}")

        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "d_ff": args.d_ff or (4 * args.d_model),
                        "num_classes": 2,
                        "max_len": args.max_len,
                        "dropout": args.dropout,
                    },
                    "vocab": vocab,
                    "best_dev_acc": best_acc,
                },
                ckpt_dir / "best.pt",
            )
            print(f"  -> saved ckpt/best.pt (dev_acc={best_acc:.4f})")

    with open(ckpt_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\n训练结束。best_dev_acc = {best_acc:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TransformerClassifier on ChnSentiCorp")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
