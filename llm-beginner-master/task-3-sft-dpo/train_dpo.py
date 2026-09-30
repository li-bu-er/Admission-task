"""DPO 训练：在 SFT LoRA 上继续训，reference 冻结只做 forward。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chat import format_messages  # noqa: E402
from src.data_utils import load_dpo_pairs  # noqa: E402
from src.lora import load_lora_into_model, save_lora  # noqa: E402


def build_dpo_example(tokenizer, pair: dict, max_length: int):
    """chosen/rejected 的 input_ids + labels（仅回答段计 logp）。"""
    ctx = list(pair["messages_context"])

    def encode_response(response: str):
        msgs = ctx + [{"role": "assistant", "content": response}]
        text = format_messages(msgs)
        prompt_text = format_messages(ctx, add_generation_prompt=True)
        full = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
        prompt = tokenizer(
            prompt_text, truncation=True, max_length=max_length, add_special_tokens=True
        )
        input_ids = torch.tensor(full["input_ids"], dtype=torch.long)
        labels = input_ids.clone()
        prompt_len = min(len(prompt["input_ids"]), input_ids.size(0))
        # prompt（含 assistant 头）全 -100，只对回答计 logp
        labels[:prompt_len] = -100
        attn = torch.ones_like(input_ids)
        return input_ids, attn, labels

    c_ids, c_attn, c_lab = encode_response(pair["chosen"])
    r_ids, r_attn, r_lab = encode_response(pair["rejected"])
    return {
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_attn,
        "chosen_labels": c_lab,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_attn,
        "rejected_labels": r_lab,
    }


class DPODataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return build_dpo_example(self.tokenizer, self.pairs[idx], self.max_length)


def collate_dpo(batch, pad_id: int):
    def pad_key(key, fill):
        max_len = max(b[key].size(0) for b in batch)
        out = torch.full((len(batch), max_len), fill, dtype=torch.long)
        for i, b in enumerate(batch):
            L = b[key].size(0)
            out[i, :L] = b[key]
        return out

    return {
        "chosen_input_ids": pad_key("chosen_input_ids", pad_id),
        "chosen_attention_mask": pad_key("chosen_attention_mask", 0),
        "chosen_labels": pad_key("chosen_labels", -100),
        "rejected_input_ids": pad_key("rejected_input_ids", pad_id),
        "rejected_attention_mask": pad_key("rejected_attention_mask", 0),
        "rejected_labels": pad_key("rejected_labels", -100),
    }


def sequence_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """labels != -100 的 token logp 求和 → (B,)"""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.clamp(min=0)
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = shift_labels != -100
    return (token_logps * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple:
    pi_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    margin = (pi_logratios - ref_logratios).detach().mean()
    return loss, margin


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def train(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_path = ROOT / "models" / "Qwen2.5-0.5B"
    sft_dir = ROOT / "ckpt" / "sft"
    if not model_path.exists():
        raise SystemExit(f"缺少基座 {model_path}")
    if not sft_dir.exists() or not any(sft_dir.iterdir()):
        raise SystemExit("缺少 ckpt/sft/，请先运行 python train_sft.py")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pairs = load_dpo_pairs(args.data, max_samples=args.max_samples)
    if not pairs:
        raise SystemExit("未解析到 DPO 样本，请检查 --data 路径与字段（prompt/chosen/rejected）")
    random.shuffle(pairs)
    print(f"DPO 样本数 = {len(pairs)}")

    policy = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float32)
    load_lora_into_model(policy, str(sft_dir))
    policy.to(device)

    ref = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float32)
    load_lora_into_model(ref, str(sft_dir))
    freeze_model(ref)
    ref.to(device)

    ds = DPODataset(pairs, tokenizer, args.max_length)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_dpo(b, tokenizer.pad_token_id),
        num_workers=0,
    )

    optimizer = AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.0,
    )

    policy.train()
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        total_loss, total_margin, n = 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"dpo epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            out_pc = policy(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )
            out_pr = policy(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            )
            with torch.no_grad():
                out_rc = ref(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"],
                )
                out_rr = ref(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"],
                )

            pc = sequence_logps(out_pc.logits, batch["chosen_labels"])
            pr = sequence_logps(out_pr.logits, batch["rejected_labels"])
            rc = sequence_logps(out_rc.logits, batch["chosen_labels"])
            rr = sequence_logps(out_rr.logits, batch["rejected_labels"])

            loss, margin = dpo_loss(pc, pr, rc, rr, beta=args.beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in policy.parameters() if p.requires_grad), args.grad_clip
            )
            optimizer.step()

            total_loss += loss.item()
            total_margin += margin.item()
            n += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", margin=f"{margin.item():.3f}")
            if args.max_steps and global_step >= args.max_steps:
                break

        avg_loss = total_loss / max(n, 1)
        avg_margin = total_margin / max(n, 1)
        history.append({"epoch": epoch, "loss": avg_loss, "reward_margin": avg_margin})
        print(f"epoch {epoch}  loss={avg_loss:.4f}  reward_margin={avg_margin:.4f}")
        if args.max_steps and global_step >= args.max_steps:
            break

    sft_meta = json.loads((sft_dir / "config.json").read_text(encoding="utf-8"))
    out_dir = ROOT / "ckpt" / "dpo"
    save_lora(
        policy,
        str(out_dir),
        config={
            **sft_meta,
            "stage": "dpo",
            "beta": args.beta,
            "base_sft": str(sft_dir),
        },
    )
    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已保存 DPO LoRA 到 {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO on top of SFT LoRA")
    p.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data" / "dpo"),
        help="含 prompt/chosen/rejected 的 jsonl/parquet/目录",
    )
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
