from __future__ import annotations

from typing import Any, Iterator

import torch


def _walk_wrappers(model: Any) -> Iterator[Any]:
    queue = [model]
    seen: set[int] = set()
    while queue:
        value = queue.pop(0)
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        yield value
        for name in ("base_model", "model", "module"):
            child = getattr(value, name, None)
            if child is not None and child is not value:
                queue.append(child)


def conditional_model(model: Any) -> Any:
    """Return the Qwen conditional-generation wrapper through PEFT/DDP wrappers."""

    for candidate in _walk_wrappers(model):
        inner = getattr(candidate, "model", None)
        if inner is not None and hasattr(inner, "get_rope_index") and hasattr(inner, "visual"):
            return candidate
    raise TypeError("Could not locate a Qwen3.5 conditional-generation model")


def multimodal_base(model: Any) -> Any:
    return conditional_model(model).model


def visual_tower(model: Any) -> Any:
    return multimodal_base(model).visual


def input_embedding_layer(model: Any) -> Any:
    conditional = conditional_model(model)
    layer = conditional.get_input_embeddings()
    if layer is None:
        raise RuntimeError("Qwen input embedding layer is unavailable")
    return layer


def freeze_vision_except_merger(model: Any) -> None:
    visual = visual_tower(model)
    for parameter in visual.parameters():
        parameter.requires_grad_(False)
    visual.patch_embed.eval()
    visual.pos_embed.eval()
    visual.rotary_pos_emb.eval()
    visual.blocks.eval()


def enable_training_gradient_checkpointing(model: Any) -> None:
    """Checkpoint Qwen decoder layers without the legacy reentrant restrictions."""

    conditional = conditional_model(model)
    if not getattr(conditional, "supports_gradient_checkpointing", False):
        raise RuntimeError("The loaded Qwen model does not support gradient checkpointing")
    conditional.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if not getattr(conditional, "is_gradient_checkpointing", False):
        raise RuntimeError("Transformers did not enable gradient checkpointing")


def assert_frozen_vision(model: Any) -> None:
    visual = visual_tower(model)
    leaked = [name for name, parameter in visual.named_parameters() if not name.startswith("merger.") and parameter.requires_grad]
    if leaked:
        raise RuntimeError(f"Frozen vision contract violated: {leaked[:5]}")


def trainable_parameter_summary(model: Any) -> dict[str, int]:
    return {
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }


def assert_gradient_contract(model: Any) -> dict[str, float]:
    language = 0.0
    control = 0.0
    merger = 0.0
    forbidden: list[str] = []
    nonfinite: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
            continue
        magnitude = float(parameter.grad.detach().float().abs().sum())
        if "lora_" in name and ".visual.merger." in name:
            merger += magnitude
        elif "lora_" in name and (".in_proj_a." in name or ".in_proj_b." in name):
            control += magnitude
        elif "lora_" in name:
            language += magnitude
        elif magnitude:
            forbidden.append(name)
    if language <= 0 or control <= 0 or merger <= 0 or forbidden or nonfinite:
        raise RuntimeError(
            "Gradient contract failed: "
            f"language={language:.4g}, control={control:.4g}, merger={merger:.4g}, "
            f"forbidden={forbidden[:5]}, nonfinite={nonfinite[:5]}"
        )
    return {
        "language_lora_grad_l1": language,
        "control_lora_grad_l1": control,
        "merger_lora_grad_l1": merger,
    }


def cuda_reserved_gib(device: torch.device | str = "cuda") -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_reserved(device) / 2**30
