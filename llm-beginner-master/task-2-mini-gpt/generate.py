"""同一 prompt 下对比 greedy / top-k / top-p / temperature。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import load_for_eval  # noqa: E402


STRATEGIES = [
    {"name": "greedy", "temperature": 0.0, "top_k": None, "top_p": None},
    {"name": "temperature_0.8", "temperature": 0.8, "top_k": None, "top_p": None},
    {"name": "top_k_20", "temperature": 0.9, "top_k": 20, "top_p": None},
    {"name": "top_p_0.9", "temperature": 0.9, "top_k": None, "top_p": 0.9},
]


def main(args: argparse.Namespace) -> None:
    ckpt = ROOT / "ckpt" / "best.pt"
    if not ckpt.exists():
        raise SystemExit("缺少 ckpt/best.pt，请先运行 python train.py")

    model, tok = load_for_eval(str(ckpt))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    prompt = args.prompt
    prompt_ids = torch.tensor(tok.encode(prompt), dtype=torch.long, device=device)
    print(f"prompt = {prompt!r}\n")

    for cfg in STRATEGIES:
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            top_k=cfg["top_k"],
            top_p=cfg["top_p"],
            temperature=cfg["temperature"],
        )
        text = tok.decode(out_ids.tolist())
        print(f"===== {cfg['name']} =====")
        print(text)
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="床前明月光")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
