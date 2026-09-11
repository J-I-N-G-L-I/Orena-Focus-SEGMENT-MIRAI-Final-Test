from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from .config import ExperimentConfig
from .model_utils import visual_tower
from .selection import Layer8GroupSelector, SelectionResult
from .time_constraints import TimeConstraint
from .resolution import config_for_grid


@dataclass(frozen=True)
class VisionPhaseTimes:
    prefix_s: float
    selection_s: float
    suffix_s: float
    merger_s: float


@dataclass(frozen=True)
class SparseVisionOutput:
    pooler_output: torch.Tensor
    last_hidden_state: torch.Tensor
    selection: SelectionResult
    original_group_count: int
    times: VisionPhaseTimes


def _sync(tensor: torch.Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def _native_position_inputs(visual: Any, grid_thw: torch.Tensor, hidden_states: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    try:
        from transformers.vision_utils import (
            get_vision_bilinear_indices_and_weights,
            get_vision_cu_seqlens,
            get_vision_position_ids,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("This project requires transformers==5.9.0") from exc

    bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
        grid_thw,
        num_grid_per_side=visual.num_grid_per_side,
        spatial_merge_size=visual.config.spatial_merge_size,
    )
    pos_embeds = (visual.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
    position_ids = get_vision_position_ids(grid_thw, visual.spatial_merge_size)
    cu_seqlens = get_vision_cu_seqlens(grid_thw)
    rotary = visual.rotary_pos_emb(position_ids).reshape(hidden_states.shape[0], -1)
    emb = torch.cat((rotary, rotary), dim=-1)
    return pos_embeds, (emb.cos(), emb.sin()), cu_seqlens


def sparse_cu_seqlens(groups_per_unit: tuple[int, ...], device: torch.device) -> torch.Tensor:
    lengths = torch.tensor([count * 4 for count in groups_per_unit], dtype=torch.int32, device=device)
    return torch.nn.functional.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))


class SparseVisionEncoder:
    """Native Qwen3.5 vision forward split after zero-based blocks[8]."""

    def __init__(self, model: Any, config: ExperimentConfig):
        config.validate()
        self.model = model
        self.config = config
        self.selector = Layer8GroupSelector(config)
        visual = visual_tower(model)
        if len(visual.blocks) != config.vision_depth:
            raise RuntimeError(f"Expected {config.vision_depth} vision blocks, found {len(visual.blocks)}")
        for module in (visual.patch_embed, visual.pos_embed, visual.rotary_pos_emb, visual.blocks):
            module.eval()

    def encode_frame(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> SparseVisionOutput:
        """Frame examples bypass segmentation and token pruning."""

        visual = visual_tower(self.model)
        started = time.perf_counter()
        native = visual(pixel_values, grid_thw=image_grid_thw)
        _sync(pixel_values)
        groups = native.pooler_output.shape[0]
        group_indices = torch.arange(groups, device=pixel_values.device)
        token_indices = (group_indices[:, None] * 4 + torch.arange(4, device=pixel_values.device)).reshape(-1)
        selection = SelectionResult(group_indices, token_indices, (groups,), (), "identity", groups, groups)
        elapsed = time.perf_counter() - started
        return SparseVisionOutput(native.pooler_output, native.last_hidden_state, selection, groups, VisionPhaseTimes(elapsed, 0.0, 0.0, 0.0))

    def encode_segment(
        self,
        pixel_values: torch.Tensor,
        video_grid_thw: torch.Tensor,
        *,
        constraint: TimeConstraint | None = None,
    ) -> SparseVisionOutput:
        if video_grid_thw.ndim != 2 or video_grid_thw.shape[0] != 1:
            raise NotImplementedError("Sparse Segment encoding currently requires one video per micro-batch")
        visual = visual_tower(self.model)
        temporal, height, width = map(int, video_grid_thw[0].tolist())
        merge = int(visual.spatial_merge_size)
        if height % merge or width % merge:
            raise ValueError("Vision grid is not divisible by the native merger size")
        groups_per_unit = (height // merge) * (width // merge)
        row_config = config_for_grid(self.config, height, width)
        selector = Layer8GroupSelector(row_config)
        patches_per_unit = height * width
        if pixel_values.shape[0] != temporal * patches_per_unit:
            raise ValueError("Pixel tensor does not match the native video grid")

        _sync(pixel_values)
        prefix_started = time.perf_counter()
        with torch.no_grad():
            hidden = None
            # Vision attention is independent between temporal units. Chunking
            # reduces temporary activations without changing its attention scope.
            for start in range(0, temporal, self.config.vision_chunk_units):
                stop = min(temporal, start + self.config.vision_chunk_units)
                lo, hi = start * patches_per_unit, stop * patches_per_unit
                chunk = visual.patch_embed(pixel_values[lo:hi])
                grid = video_grid_thw.new_tensor([[stop - start, height, width]])
                pos, rotary, cu = _native_position_inputs(visual, grid, chunk)
                chunk = (chunk + pos.to(chunk.dtype)).reshape(chunk.shape[0], -1)
                for block in visual.blocks[: self.config.capture_layer_index + 1]:
                    chunk = block(chunk, cu_seqlens=cu, position_embeddings=rotary)
                if hidden is None:
                    hidden = chunk.new_empty((temporal * patches_per_unit, chunk.shape[-1]))
                hidden[lo:hi] = chunk
            if hidden is None:
                raise ValueError("Empty temporal input")
            del chunk, pos, rotary, cu
            captured = hidden.reshape(temporal, groups_per_unit, merge * merge, hidden.shape[-1]).mean(dim=2)
        _sync(pixel_values)
        prefix_s = time.perf_counter() - prefix_started

        selection_started = time.perf_counter()
        selection = selector.select(captured, constraint)
        del captured
        _sync(pixel_values)
        selection_s = time.perf_counter() - selection_started
        if selection.token_indices.numel() != selection.effective_budget * merge * merge:
            raise RuntimeError("Every selected merger group must expand to four original Layer-8 tokens")

        suffix_started = time.perf_counter()
        with torch.no_grad():
            _, unit_rotary, _ = _native_position_inputs(
                visual, video_grid_thw.new_tensor([[1, height, width]]), hidden[:patches_per_unit]
            )
            selected_hidden = hidden.new_empty((selection.token_indices.numel(), hidden.shape[-1]))
            token_offset = 0
            counts = selection.groups_per_unit
            for start in range(0, len(counts), self.config.vision_chunk_units):
                chunk_counts = counts[start:start + self.config.vision_chunk_units]
                token_stop = token_offset + sum(chunk_counts) * merge * merge
                if token_stop == token_offset:
                    continue
                indices = selection.token_indices[token_offset:token_stop]
                chunk = hidden.index_select(0, indices)
                spatial_indices = indices.remainder(patches_per_unit)
                rotary = tuple(value.index_select(0, spatial_indices) for value in unit_rotary)
                cu = sparse_cu_seqlens(tuple(c for c in chunk_counts if c), hidden.device)
                for block in visual.blocks[self.config.capture_layer_index + 1:]:
                    chunk = block(chunk, cu_seqlens=cu, position_embeddings=rotary)
                selected_hidden[token_offset:token_stop] = chunk
                token_offset = token_stop
            if token_offset != selection.token_indices.numel():
                raise RuntimeError("Sparse chunk cursor mismatch")
            del hidden, chunk, rotary, unit_rotary
            hidden = selected_hidden
        _sync(pixel_values)
        suffix_s = time.perf_counter() - suffix_started

        # Leave no_grad before the LoRA-instrumented native final merger.
        merger_started = time.perf_counter()
        merged = visual.merger(hidden)
        _sync(pixel_values)
        merger_s = time.perf_counter() - merger_started
        if merged.shape[0] != selection.effective_budget:
            raise RuntimeError("Native merger output does not match the exact post-merger budget")
        return SparseVisionOutput(
            merged,
            hidden,
            selection,
            temporal * groups_per_unit,
            VisionPhaseTimes(prefix_s, selection_s, suffix_s, merger_s),
        )

