from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .model_utils import conditional_model, input_embedding_layer, multimodal_base
from .sparse_vision import SparseVisionOutput


@dataclass(frozen=True)
class PreparedSparseInputs:
    model_inputs: dict[str, torch.Tensor]
    selected_visual_indices: torch.Tensor
    kept_sequence_indices: torch.Tensor
    original_sequence_length: int
    compressed_sequence_length: int
    rope_delta: torch.Tensor


def _get(batch: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    return batch.get(key, default) if isinstance(batch, Mapping) else getattr(batch, key, default)


def _derive_types(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    conditional = conditional_model(model)
    result = torch.zeros_like(input_ids, dtype=torch.int32)
    result[input_ids == int(conditional.config.image_token_id)] = 1
    result[input_ids == int(conditional.config.video_token_id)] = 2
    return result


def _full_positions(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    types: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    semantic, delta = multimodal_base(model).get_rope_index(
        input_ids,
        mm_token_type_ids=types,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )
    if semantic.ndim != 3 or semantic.shape[0] != 3:
        raise RuntimeError("Expected semantic M-RoPE positions shaped [3,batch,length]")
    physical = attention_mask.long().cumsum(-1) - 1
    physical.masked_fill_(attention_mask == 0, 0)
    return torch.cat((physical.unsqueeze(0), semantic), dim=0), delta


def prepare_sparse_inputs(
    model: Any,
    batch: Mapping[str, Any] | Any,
    vision: SparseVisionOutput,
    *,
    track: str,
    labels: torch.Tensor | None = None,
) -> PreparedSparseInputs:
    input_ids = _get(batch, "input_ids")
    attention = _get(batch, "attention_mask")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError("micro-batch size 1 is required by the training contract")
    if attention is None:
        attention = torch.ones_like(input_ids)
    conditional = conditional_model(model)
    is_segment = track.lower() == "segment"
    token_id = int(conditional.config.video_token_id if is_segment else conditional.config.image_token_id)
    visual_positions = torch.nonzero(input_ids[0] == token_id, as_tuple=False).flatten()
    if visual_positions.numel() != vision.original_group_count:
        raise RuntimeError(
            f"Processor placeholders ({visual_positions.numel()}) do not match native visual groups "
            f"({vision.original_group_count})"
        )
    selected = vision.selection.group_indices.to(input_ids.device)
    if selected.numel() != vision.pooler_output.shape[0]:
        raise RuntimeError("Selected indices and sparse merger features have different lengths")

    keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    visual_keep = torch.zeros(visual_positions.numel(), dtype=torch.bool, device=input_ids.device)
    visual_keep[selected] = True
    keep[visual_positions[~visual_keep]] = False
    kept = torch.nonzero(keep, as_tuple=False).flatten()

    types = _get(batch, "mm_token_type_ids")
    if types is None:
        types = _derive_types(model, input_ids)
    image_grid = _get(batch, "image_grid_thw")
    video_grid = _get(batch, "video_grid_thw")
    full_positions, _ = _full_positions(model, input_ids, attention, types, image_grid, video_grid)
    # Prune IDs before embedding. The old path materialized embeddings for all
    # discarded high-resolution video placeholders and then cloned that tensor.
    compressed_ids = input_ids.index_select(1, kept)
    compressed_embeddings = input_embedding_layer(model)(compressed_ids).clone()
    compressed_visual = torch.nonzero(compressed_ids[0] == token_id, as_tuple=False).flatten()
    compressed_embeddings[0, compressed_visual] = vision.pooler_output.to(compressed_embeddings)
    compressed_attention = attention.index_select(1, kept)
    semantic = full_positions[1:].index_select(2, kept)
    physical = compressed_attention.long().cumsum(-1) - 1
    physical.masked_fill_(compressed_attention == 0, 0)
    positions = torch.cat((physical.unsqueeze(0), semantic), dim=0)
    if positions.shape != (4, 1, compressed_embeddings.shape[1]):
        raise RuntimeError("Physical/temporal/height/width M-RoPE channels are misaligned")
    valid = compressed_attention[0].bool()
    rope_delta = (semantic[:, 0, valid].max() + 1 - int(valid.sum())).reshape(1, 1).long()
    multimodal_base(model).rope_deltas = rope_delta
    model_inputs: dict[str, torch.Tensor] = {
        "inputs_embeds": compressed_embeddings,
        "attention_mask": compressed_attention,
        "position_ids": positions,
    }
    if labels is not None:
        if labels.shape != input_ids.shape:
            raise ValueError("labels must match the uncompressed input_ids")
        model_inputs["labels"] = labels.index_select(1, kept)
    return PreparedSparseInputs(
        model_inputs,
        selected,
        kept,
        int(input_ids.shape[1]),
        int(compressed_embeddings.shape[1]),
        rope_delta,
    )


def answer_only_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_tokens: int) -> torch.Tensor:
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[:, :prompt_tokens] = -100
    return labels


def answer_only_loss_suffix(labels: torch.Tensor, ignore_index: int = -100) -> tuple[torch.Tensor, int]:
    """Return the minimal causal-LM suffix whose loss equals the full masked loss.

    If supervised answer labels occupy positions ``a..L-1``, causal logits at
    ``a-1..L-2`` predict them. Keeping labels/logits from ``a-1`` onward
    preserves exactly those terms while avoiding vocabulary logits for the
    masked multimodal prompt.
    """

    if labels.ndim != 2 or labels.shape[0] != 1:
        raise NotImplementedError("Answer-only loss requires micro-batch size 1")
    supervised = torch.nonzero(labels[0] != ignore_index, as_tuple=False).flatten()
    if supervised.numel() == 0:
        raise RuntimeError("Training row has no supervised answer tokens")
    first = int(supervised[0])
    last = int(supervised[-1])
    expected = torch.arange(first, last + 1, device=supervised.device)
    if first == 0 or not torch.equal(supervised, expected):
        raise RuntimeError("Supervised answer labels must be one contiguous suffix after the prompt")
    suffix = labels[:, first - 1 :]
    return suffix, int(suffix.shape[1])


def assert_identity_logits(native_logits: torch.Tensor, sparse_logits: torch.Tensor, atol: float = 1e-2) -> dict[str, float | bool]:
    if native_logits.shape != sparse_logits.shape:
        raise RuntimeError("Identity logits have different shapes")
    max_abs = float((native_logits.float() - sparse_logits.float()).abs().max())
    argmax = bool(torch.equal(native_logits.argmax(-1), sparse_logits.argmax(-1)))
    if max_abs >= atol or not argmax:
        raise RuntimeError(f"Identity parity failed: max_abs={max_abs:.6g}, argmax={argmax}")
    return {"max_abs": max_abs, "argmax_match": argmax}
