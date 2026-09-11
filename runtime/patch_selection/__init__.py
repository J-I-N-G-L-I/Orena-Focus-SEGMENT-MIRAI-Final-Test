"""Layer-8 sparse visual selection for Qwen3.5-VL."""

from .config import ExperimentConfig, MODEL_REVISION
from .selection import Layer8GroupSelector, SelectionResult
from .time_constraints import TimeConstraint, parse_time_constraint

__all__ = [
    "ExperimentConfig",
    "MODEL_REVISION",
    "Layer8GroupSelector",
    "SelectionResult",
    "TimeConstraint",
    "parse_time_constraint",
]


