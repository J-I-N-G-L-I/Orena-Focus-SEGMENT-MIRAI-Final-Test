from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class SampledVideo:
    frames: np.ndarray
    source_indices: tuple[int, ...]
    timestamps_s: tuple[float, ...]
    source_fps: float
    original_frame_count: int
    padded_last_frame: bool


def one_fps_indices(frame_count: int, source_fps: float, *, include_last: bool = True) -> list[int]:
    if frame_count <= 0:
        return []
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid source_fps={source_fps}")
    duration = (frame_count - 1) / source_fps
    indices = [min(frame_count - 1, int(round(second * source_fps))) for second in range(int(np.floor(duration)) + 1)]
    indices = list(dict.fromkeys(indices))
    if include_last and indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def pad_temporal_pairs(frames: np.ndarray, indices: list[int], source_fps: float) -> tuple[np.ndarray, list[int], bool]:
    if len(frames) != len(indices):
        raise ValueError("frames and indices differ in length")
    if len(frames) == 0:
        return frames, indices, False
    if len(frames) % 2 == 0:
        return frames, indices, False
    return np.concatenate([frames, frames[-1:]], axis=0), [*indices, indices[-1]], True


def decode_video_1fps(path: str | Path, *, width: int = -1, height: int = -1) -> SampledVideo:
    try:
        import decord
    except ImportError as exc:  # pragma: no cover - exercised in the lab environment
        raise RuntimeError("decord==0.6.0 is required for video decoding") from exc

    if (width, height) != (-1, -1):
        raise ValueError("Raw-resolution decoding must not resize")
    reader = decord.VideoReader(str(path), num_threads=2)
    source_fps = float(reader.get_avg_fps())
    indices = one_fps_indices(len(reader), source_fps)
    if not indices:
        raise ValueError(f"Empty video: {path}")
    frames = reader.get_batch(indices).asnumpy()
    frames, padded_indices, padded = pad_temporal_pairs(frames, indices, source_fps)
    timestamps = tuple(index / source_fps for index in padded_indices)
    return SampledVideo(frames, tuple(padded_indices), timestamps, source_fps, len(reader), padded)


def decode_window_1fps(
    path: str | Path,
    *,
    start_s: float,
    end_s: float,
    width: int = -1,
    height: int = -1,
) -> SampledVideo:
    try:
        import decord
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("decord==0.6.0 is required for video decoding") from exc
    if (width, height) != (-1, -1):
        raise ValueError("Raw-resolution decoding must not resize")
    reader = decord.VideoReader(str(path), num_threads=2)
    fps = float(reader.get_avg_fps())
    if end_s < start_s:
        raise ValueError("end_s precedes start_s")
    first = max(0, min(len(reader) - 1, int(round(start_s * fps))))
    last = max(first, min(len(reader) - 1, int(round(end_s * fps))))
    relative = one_fps_indices(last - first + 1, fps)
    indices = [first + value for value in relative]
    frames = reader.get_batch(indices).asnumpy()
    frames, padded_indices, padded = pad_temporal_pairs(frames, indices, fps)
    timestamps = tuple(index / fps for index in padded_indices)
    return SampledVideo(frames, tuple(padded_indices), timestamps, fps, len(reader), padded)


def decode_frame(path: str | Path, timestamp_s: float, *, long_edge: int | None = None):
    try:
        import decord
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("decord and Pillow are required") from exc
    reader = decord.VideoReader(str(path), num_threads=1)
    fps = float(reader.get_avg_fps())
    index = max(0, min(len(reader) - 1, int(round(timestamp_s * fps))))
    image = Image.fromarray(reader[index].asnumpy()).convert("RGB")
    if long_edge is not None:
        raise ValueError("Raw-resolution Frame decoding must not resize")
    return image, index / fps


def iter_temporal_units(sampled: SampledVideo) -> Iterator[np.ndarray]:
    if len(sampled.frames) % 2:
        raise ValueError("Temporal frames must have been padded to a multiple of two")
    for offset in range(0, len(sampled.frames), 2):
        yield sampled.frames[offset : offset + 2]
