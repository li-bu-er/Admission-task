"""预训练：BPE → MiniGPT，按 dev PPL 存 ckpt。"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import MiniGPT  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_batches(
    ids: list,
    block_size: int,
    batch_size: int,
    device: torch.device,
    stride: int | None = None,
):
    """切 (x,y)，y 右移一位；小语料用滑动窗凑 batch。"""
    if stride is None:
        stride = max(1, block_size // 4)
    chunks = []
    last = len(ids) - block_size - 1
    for i in range(0, max(0, last) + 1, stride):
        chunks.append(ids[i : i + block_size + 1])
    if not chunks:
        raise RuntimeError(
            f"语料过短：tokens={len(ids)}，需要 > block_size+1={block_size + 1}"
        )
    random.shuffle(chunks)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        arr = torch.tensor(batch, dtype=torch.long, device=device)
        yield arr[:, :-1], arr[:, 1:]


@torch.no_grad()
def eval_perplexity(model: MiniGPT, ids: list, block_size: int, device: torch.device) -> float:
    """非重叠分块累 NLL→PPL，窗长与 eval 一致。"""
    model.eval()
    nll, n_tok = 0.0, 0
    for i in range(0, max(1, len(ids) - 1), block_size):
        window = ids[i : i + block_size + 1]
        if len(window) < 2:
            break
        chunk = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(chunk)
        nll += F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
        n_tok += chunk.size(1) - 1
    if n_tok == 0:
        return float("inf")
    return math.exp(nll / n_tok)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device = {device}")

    data_dir = ROOT / "data"
    train_path = data_dir / "train.txt"
    dev_path = data_dir / "dev.txt"
    if not train_path.exists():
        raise SystemExit("缺少 data/train.txt，请先运行 python data/download.py")

    train_text = load_text(train_path)
    dev_text = load_text(dev_path) if dev_path.exists() else train_text[- max(100, len(train_text) // 10) :]

    ckpt_dir = ROOT / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tok_path = ckpt_dir / "tokenizer.json"

    # train+dev 一起训词表，避免 dev 罕见字把 PPL 撑高（LM 仍只用 train）
    if args.retrain_tokenizer or not tok_path.exists():
        print(f"训练 BPE tokenizer，vocab_size={args.vocab_size} ...")
        tok = BPETokenizer()
        tok.train(train_text + "\n" + dev_text, vocab_size=args.vocab_size)
        tok.save(str(tok_path))
        print(f"  saved {tok_path}  vocab_size={tok.vocab_size}")
    else:
        tok = BPETokenizer.from_pretrained(str(tok_path))
        print(f"加载已有 tokenizer，vocab_size={tok.vocab_size}")

    train_ids = tok.encode(train_text)
    dev_ids = tok.encode(dev_text)
    print(f"train tokens={len(train_ids)}, dev tokens={len(dev_ids)}")

    model = MiniGPT(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        block_size=args.block_size,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters = {n_params / 1e6:.2f}M")

    stride = args.stride if args.stride > 0 else max(1, args.block_size // 4)
    n_windows = max(0, len(train_ids) - args.block_size - 1) // stride + 1
    print(f"stride={stride}, windows/epoch≈{n_windows}, batches/epoch≈{math.ceil(n_windows / args.batch_size)}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_ppl = float("inf")
    history = []
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        batches = list(
            build_batches(
                train_ids,
                args.block_size,
                args.batch_size,
                device,
                stride=stride,
            )
        )
        for x, y in tqdm(batches, desc=f"epoch {epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        dev_ppl = eval_perplexity(model, dev_ids, args.block_size, device)
        history.append({"epoch": epoch, "loss": avg_loss, "dev_ppl": dev_ppl})
        print(f"epoch {epoch:02d}  loss={avg_loss:.4f}  dev_ppl={dev_ppl:.2f}")

        if dev_ppl < best_ppl - 1e-6:
            best_ppl = dev_ppl
            bad_epochs = 0
            payload = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": tok.vocab_size,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "block_size": args.block_size,
                    "dropout": args.dropout,
                },
                "tokenizer_path": str(tok_path),
                "best_dev_ppl": best_ppl,
            }
            torch.save(payload, ckpt_dir / "best.pt")
            print(f"  -> saved ckpt/best.pt (dev_ppl={best_ppl:.2f})")
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"早停：连续 {args.patience} 个 epoch dev_ppl 未提升")
                break

    with open(ckpt_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\n训练结束。best_dev_ppl = {best_ppl:.2f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain MiniGPT")
    p.add_argument("--vocab_size", type=int, default=800, help="BPE 词表大小")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument(
        "--stride",
        type=int,
        default=8,
        help="滑动窗口步长；越小更新越多",
    )
    p.add_argument("--patience", type=int, default=6, help="早停耐心；0 关闭")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--retrain_tokenizer", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
