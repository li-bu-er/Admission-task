"""SFT 训练：Qwen2.5-0.5B + 手写 LoRA + MOSS loss masking。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chat import messages_to_ids  # noqa: E402
from src.data_utils import load_sft_messages  # noqa: E402
from src.lora import inject_lora, save_lora  # noqa: E402


class SFTDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return messages_to_ids(self.samples[idx], self.tokenizer, self.max_length)


def collate_pad(batch, pad_id: int):
    max_len = max(x["input_ids"].size(0) for x in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        L = x["input_ids"].size(0)
        input_ids[i, :L] = x["input_ids"]
        attention_mask[i, :L] = x["attention_mask"]
        labels[i, :L] = x["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_path = ROOT / "models" / "Qwen2.5-0.5B"
    if not model_path.exists():
        raise SystemExit(f"缺少基座 {model_path}，请先 python data/download.py")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载 SFT 数据: {args.data}")
    samples = load_sft_messages(args.data, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("未解析到任何对话，请检查 MOSS jsonl/zip 路径与格式")
    random.shuffle(samples)
    print(f"样本数 = {len(samples)}")

    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float32)
    inject_lora(
        model,
        target_modules=args.target_modules,
        r=args.r,
        alpha=args.alpha,
    )
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable = {trainable} / {total} ({trainable / total:.4%})")

    ds = SFTDataset(samples, tokenizer, args.max_length)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, tokenizer.pad_token_id),
        num_workers=0,
    )

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model.train()
    global_step = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        total_loss, n = 0.0, 0
        pbar = tqdm(loader, desc=f"sft epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), args.grad_clip
            )
            optimizer.step()
            total_loss += loss.item()
            n += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if args.max_steps and global_step >= args.max_steps:
                break
        avg = total_loss / max(n, 1)
        history.append({"epoch": epoch, "loss": avg})
        print(f"epoch {epoch}  avg_loss={avg:.4f}")
        if args.max_steps and global_step >= args.max_steps:
            break

    out_dir = ROOT / "ckpt" / "sft"
    save_lora(
        model,
        str(out_dir),
        config={
            "target_modules": args.target_modules,
            "r": args.r,
            "alpha": args.alpha,
            "base_model": str(model_path),
            "stage": "sft",
        },
    )
    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已保存 LoRA 到 {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT with hand-written LoRA")
    p.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data" / "moss-sft"),
        help="MOSS jsonl / zip / 目录",
    )
    p.add_argument("--max_samples", type=int, default=5000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0, help=">0 时按 step 早停，便于冒烟")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16)
    p.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
