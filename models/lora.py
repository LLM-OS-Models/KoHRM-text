"""LoRA helpers for KoHRM-Text.

This module wraps local LinearInit layers with a frozen base projection plus a
trainable low-rank delta. It is intentionally small and repo-local so the HRM
architecture can keep using its existing custom modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.layers import LinearInit


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: LinearInit,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        out_features, in_features = base.weight.shape
        self.lora_a = nn.Parameter(torch.empty(rank, in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, input: Tensor) -> Tensor:
        base_out = self.base(input)
        x = self.dropout(input)
        delta = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        return base_out + delta


def _get_parent_and_attr(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    suffixes = tuple(target_suffixes)
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, LinearInit):
            continue
        if suffixes and not any(name.endswith(suffix) for suffix in suffixes):
            continue
        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)
    if not replaced:
        raise ValueError(f"No LinearInit modules matched target suffixes: {suffixes}")
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)


def trainable_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            out[f"{name}.lora_a"] = module.lora_a.detach().cpu()
            out[f"{name}.lora_b"] = module.lora_b.detach().cpu()
    return out


def save_lora_adapter(
    model: nn.Module,
    output_dir: str | Path,
    config: dict,
    matched_modules: list[str],
    step: int,
    tag: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), out / f"lora_{tag}.pt")
    payload = dict(config)
    payload |= {
        "tag": tag,
        "step": int(step),
        "matched_modules": matched_modules,
        "num_lora_tensors": len(lora_state_dict(model)),
    }
    with (out / f"lora_{tag}_info.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with (out / "latest_lora.txt").open("w", encoding="utf-8") as f:
        f.write(f"{tag}\n")
