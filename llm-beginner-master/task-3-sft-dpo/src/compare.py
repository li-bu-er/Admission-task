"""对比 base / SFT / DPO 在同一指令上的生成。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chat import format_messages  # noqa: E402
from src.lora import load_lora_into_model  # noqa: E402

DEFAULT_PROMPTS = [
    "用三句话介绍什么是机器学习。",
    "请写一首关于春天的五言绝句。",
    "把下面句子翻译成英文：深度学习改变了人工智能。",
]


@torch.no_grad()
def generate_reply(model, tokenizer, user_text: str, max_new_tokens: int, device) -> str:
    messages = [{"role": "user", "content": user_text}]
    prompt = format_messages(messages, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].size(1) :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def load_stage(stage: str, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = ROOT / "models" / "Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float32)
    if stage == "sft":
        load_lora_into_model(model, str(ROOT / "ckpt" / "sft"))
    elif stage == "dpo":
        # DPO 在 SFT 上续训，直接加载 dpo adapter
        load_lora_into_model(model, str(ROOT / "ckpt" / "dpo"))
    model.to(device)
    model.eval()
    return model, tok


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    prompts = args.prompts or DEFAULT_PROMPTS

    stages = ["base"]
    if (ROOT / "ckpt" / "sft").exists():
        stages.append("sft")
    if (ROOT / "ckpt" / "dpo").exists():
        stages.append("dpo")

    models = {}
    tokenizer = None
    for s in stages:
        print(f"加载 {s} ...")
        models[s], tokenizer = load_stage(s, device)

    lines = []
    for i, prompt in enumerate(prompts, 1):
        print("\n" + "=" * 60)
        print(f"[{i}] USER: {prompt}")
        lines.append(f"## Prompt {i}\n{prompt}\n")
        for s in stages:
            reply = generate_reply(models[s], tokenizer, prompt, args.max_new_tokens, device)
            print(f"\n--- {s.upper()} ---\n{reply}")
            lines.append(f"### {s.upper()}\n{reply}\n")

    out = ROOT / "ckpt" / "compare.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n对比结果已写入 {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", nargs="*", default=None)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
