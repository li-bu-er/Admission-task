"""注意力热图：从 ckpt/best.pt 抽样本，保存到 figures/。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import CLS_TOKEN, PAD_TOKEN, load_for_eval  # noqa: E402

SAMPLES = [
    ("positive", "这家饭店的菜味道很不错，服务态度也很好，下次还来。"),
    ("negative", "质量太差了，用了两天就坏了，非常失望，不推荐购买。"),
    (
        "long",
        "酒店位置还可以，但是房间有点旧，空调声音很大影响睡眠，"
        "前台态度一般，早餐种类少味道一般，总体来说性价比不高，不会再住。",
    ),
]


def ids_to_tokens(ids: torch.Tensor, id2char: dict) -> list:
    tokens = []
    for i in ids.tolist():
        ch = id2char.get(i, "<unk>")
        if ch == CLS_TOKEN:
            tokens.append("[CLS]")
        elif ch == PAD_TOKEN:
            tokens.append("[PAD]")
        else:
            tokens.append(ch)
    return tokens


@torch.no_grad()
def plot_attention(
    tokens: list,
    attn: torch.Tensor,
    layer: int,
    head: int,
    title: str,
    out_path: Path,
) -> None:
    matrix = attn[0, head].cpu().numpy()
    T = len(tokens)

    fig, ax = plt.subplots(figsize=(max(6, T * 0.35), max(5, T * 0.35)))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_title(f"{title}\nlayer={layer} head={head}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    ckpt = ROOT / "ckpt" / "best.pt"
    if not ckpt.exists():
        raise SystemExit("缺少 ckpt/best.pt，请先运行 python train.py")

    model, tokenize_fn = load_for_eval(str(ckpt))
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    id2char = {i: ch for ch, i in payload["vocab"].items()}

    layer_idx = len(model.layers) - 1
    head_idx = 0

    for name, text in SAMPLES:
        ids = tokenize_fn(text)
        tokens = ids_to_tokens(ids, id2char)
        _, attn_list = model.encode(ids.unsqueeze(0), return_attn=True)
        attn = attn_list[layer_idx]
        out = ROOT / "figures" / f"attn_{name}_L{layer_idx}H{head_idx}.png"
        plot_attention(
            tokens,
            attn,
            layer=layer_idx,
            head=head_idx,
            title=f"{name}: {text[:24]}...",
            out_path=out,
        )


if __name__ == "__main__":
    # Windows 下中文字体兜底
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
