"""手写 LoRA：y = Wx + scaling·B(Ax)，scaling = alpha/r。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """冻结 Linear，旁路挂低秩 A/B。"""

    def __init__(self, linear: nn.Linear, r: int, alpha: float):
        super().__init__()
        assert r > 0
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_f = linear.in_features
        out_f = linear.out_features
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        # A: Kaiming；B: 全零，注入瞬间不改输出
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling

    def merged_weight(self) -> torch.Tensor:
        """W + scaling * (B @ A)"""
        delta = self.lora_B.weight @ self.lora_A.weight
        return self.linear.weight + self.scaling * delta


def _set_module(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    r: int,
    alpha: float,
) -> nn.Module:
    """把目标 Linear 换成 LoRALinear，只训 LoRA。"""
    targets = set(target_modules)

    for p in model.parameters():
        p.requires_grad = False

    to_replace: List[tuple] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            to_replace.append((name, module))

    for name, module in to_replace:
        _set_module(model, name, LoRALinear(module, r=r, alpha=alpha))

    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True

    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """合并 LoRA 进原权重，换回普通 Linear。"""
    to_merge: List[tuple] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            to_merge.append((name, module))

    for name, module in to_merge:
        linear = module.linear
        with torch.no_grad():
            linear.weight.copy_(module.merged_weight())
        linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False
        _set_module(model, name, linear)

    return model


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> nn.Module:
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "lora_" in k]
    if missing:
        raise KeyError(f"LoRA 权重缺失: {missing[:5]}...")
    if unexpected:
        raise KeyError(f"意外键: {unexpected[:5]}...")
    return model


def save_lora(model: nn.Module, out_dir: str, config: Optional[dict] = None) -> None:
    """adapter.pt + config.json"""
    import json

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), path / "adapter.pt")
    meta = config or {}
    (path / "config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_lora_into_model(
    model: nn.Module,
    adapter_dir: str,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """按 config 注入（如需）并加载 adapter。"""
    import json

    path = Path(adapter_dir)
    meta = json.loads((path / "config.json").read_text(encoding="utf-8"))
    targets = target_modules or meta.get("target_modules", ["q_proj", "v_proj"])
    r = int(meta.get("r", 8))
    alpha = float(meta.get("alpha", 16))

    has_lora = any(isinstance(m, LoRALinear) for m in model.modules())
    if not has_lora:
        inject_lora(model, target_modules=targets, r=r, alpha=alpha)

    state = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
    return load_lora_state_dict(model, state)
