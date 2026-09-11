"""Preserve native pixels; pad only the right/bottom edges for the Qwen merger."""
from __future__ import annotations

from dataclasses import replace
import numpy as np


def padded_size(width: int, height: int) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError((width, height))
    return ((width + 31) // 32 * 32, (height + 31) // 32 * 32)


def budget_for_size(width: int, height: int, *, units: int = 36) -> int:
    if isinstance(units, bool) or not isinstance(units, int) or units < 1:
        raise ValueError("visual budget units must be a positive integer")
    width, height = padded_size(width, height)
    return units * (width // 32) * (height // 32)


def pad_native(array: np.ndarray) -> np.ndarray:
    if array.ndim not in (3, 4) or array.shape[-1] != 3:
        raise ValueError(f"Expected HWC or THWC RGB, got {array.shape}")
    height, width = array.shape[-3:-1]
    target_w, target_h = padded_size(width, height)
    padding = [(0, 0)] * array.ndim
    padding[-3] = (0, target_h - height)
    padding[-2] = (0, target_w - width)
    return np.pad(array, padding, mode="edge") if any(b for _, b in padding) else array


def config_for_grid(config, height_patches: int, width_patches: int):
    if height_patches % 2 or width_patches % 2:
        raise ValueError("Native processor grid must be merger-aligned")
    return replace(config, width=width_patches * 16, height=height_patches * 16,
                   visual_token_budget=budget_for_size(width_patches * 16, height_patches * 16,
                                                      units=config.visual_budget_units))
