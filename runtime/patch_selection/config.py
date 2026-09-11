from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal


MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
JUDGE_MODEL_ID = "Qwen/Qwen3.5-4B"
JUDGE_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
OFFICIAL_ORENA_COMMIT = "7b7e5c537518e8a65f8d70b89dedc3b4518a856a"
BASE_INITIALIZATION = "fixed_base_with_fresh_lora"
PROJECT_VERSION = "qwen35-9b-layer8-crd-lab-raw-resolution-v2"
RUN_FAMILY = "qwen35_9b_crd_raw_resolution_seed2026_v2"
FINAL_SCHEDULE_VERSION = "capability_balanced_v2"

PROJECT_ROOT = Path(
    os.environ.get(
        "PROJECT_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
)
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data/orena"))
RUNTIME_ROOT = Path(
    os.environ.get("RUNTIME_ROOT", "./work")
)
CONDA_ENV = Path(os.environ.get("CONDA_ENV", "./.venv"))
HF_HOME = Path(os.environ.get("HF_HOME", "./.cache/huggingface"))
HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", str(HF_HOME / "hub")))
TMP_ROOT = Path(os.environ.get("TMPDIR", str(RUNTIME_ROOT / "tmp")))
OFFICIAL_SOURCE_ROOT = Path(
    os.environ.get(
        "OFFICIAL_SOURCE_ROOT",
        "./external/orena-focus",
    )
)
OFFICIAL_ROOT = Path(
    os.environ.get(
        "OFFICIAL_ROOT",
        f"./external/orena-focus-{OFFICIAL_ORENA_COMMIT[:12]}",
    )
)
FO_DEFINITIONS_PATH = OFFICIAL_ROOT / "src" / "focus" / "assets" / "FO_definitions.txt"
PREPARED_ROOT = RUNTIME_ROOT / "prepared" / RUN_FAMILY
RUNS_ROOT = RUNTIME_ROOT / "runs"
DEFAULT_RUN_NAME = RUN_FAMILY
JUDGE_MODEL_ROOT = Path(os.environ.get(
    "CRD_JUDGE_MODEL_ROOT", "./resources/judge_qwen35_4b"
))

SelectorMode = Literal["uniform", "svd_cr", "svd_crd", "identity"]


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    sampling_fps: float = 1.0
    # Reference grid/budget; encode_segment derives a per-video config from its
    # actual padded grid. These defaults do not control decoding dimensions.
    width: int = 512
    height: int = 288
    capture_layer_index: int = 8
    visual_token_budget: int = 5_184
    # Legacy defaults intentionally preserve manifests exported before v2.
    visual_budget_units: int = 36
    resolution_mode: str = "native_pad32"
    vision_chunk_units: int = int(os.environ.get("CRD_VISION_CHUNK_UNITS", "4"))
    activation_offload: bool = os.environ.get("CRD_ACTIVATION_OFFLOAD", "0") == "1"
    common_energy: float = 0.90
    min_common_energy: float = 0.70
    min_segment_units: int = 2
    max_segment_units: int = 16
    selector_mode: SelectorMode = "svd_crd"
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    vision_depth: int = 27
    segment_slots: int = 16_000
    frame_slots: int = 4_000
    schedule_version: str = "legacy_v1"
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    answer_only_logits: bool = True
    optimizer_steps: int = 5_000
    language_lora_rank: int = 16
    control_lora_rank: int = 4
    merger_lora_rank: int = 32
    language_lr: float = 3e-5
    merger_lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    max_reserved_gib: float = 46.0
    min_free_gib: float = 2.0
    save_steps: int = 25
    eval_steps: int = 500
    eval_size: int = 128
    official_eval_size: int = 0
    official_eval_steps: int = 500
    early_stopping_min_steps: int = 2_500
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.002

    @property
    def resolution(self) -> str:
        return self.resolution_mode

    @property
    def group_grid(self) -> tuple[int, int]:
        patch = 16
        return (
            self.height // patch // self.spatial_merge_size,
            self.width // patch // self.spatial_merge_size,
        )

    @property
    def groups_per_unit(self) -> int:
        gh, gw = self.group_grid
        return gh * gw

    @property
    def schedule_size(self) -> int:
        return self.segment_slots + self.frame_slots

    def validate(self, *, formal: bool = False) -> None:
        if self.sampling_fps != 1.0 or self.resolution_mode != "native_pad32":
            raise ValueError("The Segment contract requires 1 FPS and native resolution with edge padding")
        if min(self.width, self.height) <= 0 or self.width % 32 or self.height % 32:
            raise ValueError("Padded resolution must be positive and divisible by 32")
        if self.vision_chunk_units < 1:
            raise ValueError("vision_chunk_units must be positive")
        if self.capture_layer_index != 8 or self.vision_depth != 27:
            raise ValueError("Capture must be the output of zero-based visual.blocks[8]")
        if self.visual_budget_units not in (36, 38):
            raise ValueError("The validated visual budget candidates are 36 and 38 units")
        if self.visual_token_budget != self.visual_budget_units * self.groups_per_unit:
            raise ValueError("Visual budget must equal visual_budget_units * groups_per_unit")
        if self.selector_mode not in {"uniform", "svd_cr", "svd_crd", "identity"}:
            raise ValueError(f"Unknown selector_mode={self.selector_mode!r}")
        if formal and self.selector_mode != "svd_crd":
            raise ValueError("Formal laboratory training is frozen to selector_mode=svd_crd")
        if self.schedule_size != 20_000 or self.optimizer_steps != 5_000:
            raise ValueError("Formal training requires 20,000 slots and 5,000 target steps")
        expected_slots = {"legacy_v1": (16_000, 4_000), FINAL_SCHEDULE_VERSION: (17_000, 3_000)}
        if expected_slots.get(self.schedule_version) != (self.segment_slots, self.frame_slots):
            raise ValueError("Schedule version and Segment/Frame quotas disagree")
        expected_monitor = 250 if self.schedule_version == FINAL_SCHEDULE_VERSION else 0
        if self.official_eval_size != expected_monitor or self.official_eval_steps != 500:
            raise ValueError("Official monitor must match the frozen schedule version")
        if self.max_reserved_gib != 46.0 or self.min_free_gib != 2.0:
            raise ValueError("Memory gate requires <=46 GiB reserved and >=2 GiB device free")
        if self.gradient_accumulation_steps != 4:
            raise ValueError("Gradient accumulation is frozen at four")
        if not self.gradient_checkpointing or not self.answer_only_logits:
            raise ValueError("Gradient checkpointing and answer-only logits are mandatory")
        if self.eval_steps != 500 or self.eval_size != 128:
            raise ValueError("Formal evaluation is frozen to 128 rows every 500 steps")
        if (
            self.early_stopping_min_steps != 2_500
            or self.early_stopping_patience != 3
            or self.early_stopping_min_delta != 0.002
        ):
            raise ValueError("The early-stopping contract changed")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def final_training_config(*, visual_budget_units: int | None = None, **overrides: Any) -> ExperimentConfig:
    """Explicit v2 factory; inference continues to deserialize legacy defaults."""
    units = int(os.environ.get("CRD_VISUAL_BUDGET_UNITS", "38")) if visual_budget_units is None else visual_budget_units
    config = ExperimentConfig(
        visual_budget_units=units,
        visual_token_budget=units * 144,
        segment_slots=17_000,
        frame_slots=3_000,
        schedule_version=FINAL_SCHEDULE_VERSION,
        official_eval_size=250,
        activation_offload=True,
    )
    config = replace(config, **overrides)
    config.validate(formal=True)
    return config


def expected_hf_cache() -> dict[str, Path]:
    return {
        "HF_HOME": HF_HOME,
        "HF_HUB_CACHE": HF_HUB_CACHE,
        "HF_DATASETS_CACHE": HF_HOME / "datasets",
        "HF_XET_CACHE": HF_HOME / "xet",
        "HF_ASSETS_CACHE": HF_HOME / "assets",
    }


# Compatibility aliases retained for the laboratory benchmark/identity tools.
LAB_PROJECT_ROOT = PROJECT_ROOT
LAB_DATA_ROOT = DATA_ROOT
LAB_RUNTIME_ROOT = RUNTIME_ROOT
LAB_HF_CACHE = HF_HUB_CACHE
LAB_TMP_ROOT = TMP_ROOT
LAB_FO_DEFINITIONS = FO_DEFINITIONS_PATH
