from __future__ import annotations

import re
from argparse import Namespace
from typing import Any

import torch


# Qwen3.5-9B has 27 visual blocks. Each block contributes qkv, proj,
# linear_fc1, and linear_fc2, so the frozen vision inventory is 27 * 4.
EXPECTED_INVENTORY = {"language": 248, "vision": 108, "merger": 2}


def discover_targets(model: torch.nn.Module) -> tuple[list[str], dict[str, int], dict[str, int], dict[str, str]]:
    language_suffixes = (
        "gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    )
    targets: list[str] = []
    ranks: dict[str, int] = {}
    alphas: dict[str, int] = {}
    kinds: dict[str, str] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        lowered = name.lower()
        is_vision = "visual" in lowered or "vision" in lowered
        suffix = name.rsplit(".", 1)[-1]
        if not is_vision and suffix in language_suffixes:
            kind = "language"
            rank, alpha = (4, 8) if suffix in {"in_proj_a", "in_proj_b"} else (16, 32)
        elif is_vision and re.search(r"\.merger\.linear_fc[12]$", lowered) and "deepstack" not in lowered:
            kind, rank, alpha = "merger", 32, 64
        elif re.search(r"(?:visual|vision).*\.blocks\.\d+\.", lowered) and suffix in {
            "qkv", "proj", "linear_fc1", "linear_fc2"
        }:
            kind, rank, alpha = "vision", 8, 16
        else:
            continue
        targets.append(name)
        ranks[name], alphas[name], kinds[name] = rank, alpha, kind
    inventory = {kind: sum(value == kind for value in kinds.values()) for kind in EXPECTED_INVENTORY}
    if inventory != EXPECTED_INVENTORY:
        raise RuntimeError(f"Unexpected Qwen3.5-9B linear inventory: {inventory}")
    selected = sorted(name for name in targets if kinds[name] in {"language", "merger"})
    return selected, {name: ranks[name] for name in selected}, {name: alphas[name] for name in selected}, {
        name: kinds[name] for name in selected
    }


def attach_lora(model: torch.nn.Module) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    targets, ranks, alphas, _ = discover_targets(model)
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=targets,
        rank_pattern=ranks,
        alpha_pattern=alphas,
        bias="none",
        task_type="CAUSAL_LM",
    )
    result = get_peft_model(model, config)
    trainable = [name for name, parameter in result.named_parameters() if parameter.requires_grad]
    if any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable")
    if not any(".merger." in name for name in trainable):
        raise RuntimeError("Final-merger LoRA was not attached")
    if any(("visual" in name or "vision" in name) and ".merger." not in name for name in trainable):
        raise RuntimeError("Vision blocks must remain frozen")
    return result


def optimizer_groups(model: torch.nn.Module, args: Namespace) -> list[dict[str, Any]]:
    language, merger = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (merger if ".merger." in name.lower() else language).append(parameter)
    if not language or not merger:
        raise RuntimeError("Both language and merger LoRA optimizer groups are required")
    return [
        {"params": language, "lr": args.language_lr, "weight_decay": args.weight_decay, "name": "language"},
        {"params": merger, "lr": args.merger_lr, "weight_decay": args.weight_decay, "name": "merger"},
    ]

