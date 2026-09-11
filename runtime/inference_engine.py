from __future__ import annotations

import time
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from attention_runtime import (
    ATTN_IMPLEMENTATION,
    assert_lab_software_contract,
    audit_qwen_linear_attention_fast_path,
    configure_lab_attention,
)
from patch_selection.config import ExperimentConfig, MODEL_ID, MODEL_REVISION
from patch_selection.media_inputs import load_media, prepare_generation_batch
from patch_selection.model_utils import freeze_vision_except_merger, multimodal_base
from patch_selection.prompting import sanitize_prediction, system_prompt
from patch_selection.sparse_inputs import prepare_sparse_inputs
from patch_selection.sparse_vision import SparseVisionEncoder
from patch_selection.time_constraints import parse_time_constraint


@dataclass(frozen=True)
class InferenceProfile:
    decode_s: float
    prefix_s: float
    selection_s: float
    suffix_s: float
    merger_s: float
    llm_s: float
    total_s: float
    visual_tokens: int
    peak_reserved_gib: float
    native_size: tuple[int, int]
    padded_size: tuple[int, int]
    requested_visual_budget: int
    empty_generation: bool = False
    minimum_free_gib: float = 0.0


class InferenceEngine:
    @classmethod
    def from_components(
        cls, *, model: Any, processor: Any, encoder: SparseVisionEncoder,
        prompt: str, device: str | torch.device,
    ) -> "InferenceEngine":
        """Borrow the training model; the caller owns its mode and lifetime."""
        engine = cls.__new__(cls)
        engine.model, engine.processor, engine.encoder = model, processor, encoder
        engine.config, engine.system, engine.device = encoder.config, prompt, torch.device(device)
        engine.attention = {"policy": "reuse_training_model"}
        return engine

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        foreign_object_definitions: str,
        device: str = "cuda:0",
        selector_mode: str = "svd_crd",
        base_model: str | Path = MODEL_ID,
        require_fast_path: bool = True,
        local_files_only: bool = False,
    ) -> None:
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.attention = configure_lab_attention(require_fast_path=require_fast_path)
        checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(checkpoint / "processor", local_files_only=True)
        base = AutoModelForImageTextToText.from_pretrained(
            base_model,
            revision=None if Path(str(base_model)).is_dir() else MODEL_REVISION,
            cache_dir=os.environ.get("HF_HUB_CACHE"),
            dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        freeze_vision_except_merger(base)
        self.model = PeftModel.from_pretrained(base, checkpoint / "adapter", is_trainable=False)
        self.model.to(self.device).eval()
        if require_fast_path:
            audit_qwen_linear_attention_fast_path(self.model)
        manifest = json.loads((checkpoint / "run_manifest.json").read_text(encoding="utf-8"))
        self.config = ExperimentConfig(**manifest["config"]["experiment"])
        if self.config.selector_mode != selector_mode:
            raise ValueError("Requested selector differs from the trained checkpoint")
        self.config.validate(formal=selector_mode == "svd_crd")
        self.encoder = SparseVisionEncoder(self.model, self.config)
        self.system = system_prompt(foreign_object_definitions)

    @torch.inference_mode()
    def answer(self, row: dict[str, Any], *, max_new_tokens: int = 32) -> tuple[str, InferenceProfile]:
        torch.cuda.reset_peak_memory_stats(self.device)
        free_samples = [torch.cuda.mem_get_info(self.device)[0] / 2**30]
        total_started = time.perf_counter()
        decode_started = time.perf_counter()
        media, timestamps = load_media(row)
        batch = prepare_generation_batch(self.processor, row, media, timestamps, self.system, self.device)
        torch.cuda.synchronize(self.device)
        decode_s = time.perf_counter() - decode_started
        free_samples.append(torch.cuda.mem_get_info(self.device)[0] / 2**30)
        if row["track"] == "segment":
            duration = max(0.0, float(row["end_seconds"]) - float(row["start_seconds"]))
            constraint = parse_time_constraint(row["question"]).relative_to_clip(float(row["start_seconds"]), duration)
            vision = self.encoder.encode_segment(batch["pixel_values_videos"], batch["video_grid_thw"], constraint=constraint)
        else:
            vision = self.encoder.encode_frame(batch["pixel_values"], batch["image_grid_thw"])
        prepared = prepare_sparse_inputs(self.model, batch, vision, track=row["track"])
        free_samples.append(torch.cuda.mem_get_info(self.device)[0] / 2**30)
        multimodal_base(self.model).rope_deltas = prepared.rope_delta
        llm_started = time.perf_counter()
        generated = self.model.generate(
            **prepared.model_inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        torch.cuda.synchronize(self.device)
        free_samples.append(torch.cuda.mem_get_info(self.device)[0] / 2**30)
        llm_s = time.perf_counter() - llm_started
        # Decoder-only generation returns prompt + continuation.  Only the
        # continuation is a candidate answer for the official evaluator.
        # Generation with inputs_embeds and no input_ids returns only new IDs.
        # No fabricated visual prompt IDs should be sliced from this result.
        text = decode_continuation(self.processor, generated, 0)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
        cleaned = re.sub(r"^\s*(?:final\s+)?answer\s*:\s*", "", cleaned, flags=re.I)
        frame = media[0] if row["track"] == "segment" else media
        profile = InferenceProfile(
            decode_s=decode_s,
            prefix_s=vision.times.prefix_s,
            selection_s=vision.times.selection_s,
            suffix_s=vision.times.suffix_s,
            merger_s=vision.times.merger_s,
            llm_s=llm_s,
            total_s=time.perf_counter() - total_started,
            visual_tokens=int(vision.selection.effective_budget),
            peak_reserved_gib=torch.cuda.max_memory_reserved(self.device) / 2**30,
            native_size=tuple(frame.info.get("native_size", frame.size)),
            padded_size=tuple(frame.size),
            requested_visual_budget=int(vision.selection.requested_budget),
            empty_generation=not bool(cleaned.strip()),
            minimum_free_gib=min(free_samples),
        )
        return sanitize_prediction(text), profile


def profile_dict(value: InferenceProfile) -> dict[str, Any]:
    return asdict(value)


def decode_continuation(processor: Any, generated: torch.Tensor, input_length: int) -> str:
    if generated.ndim != 2 or not 0 <= input_length <= generated.shape[1]:
        raise ValueError((tuple(generated.shape), input_length))
    return processor.batch_decode(
        generated[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
