from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .prompting import make_messages, normalize_answer, seconds
from .sparse_inputs import answer_only_labels
from .video_io import decode_frame, decode_video_1fps, decode_window_1fps
from .resolution import pad_native


def load_media(row: dict[str, Any]) -> tuple[Any, list[float]]:
    path = Path(row["video_path"])
    start, end = seconds(row["timestamp_start"]), seconds(row["timestamp_end"])
    if row["track"] == "frame":
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            with Image.open(path) as source:
                image = source.convert("RGB")
            timestamp = start
        else:
            image, timestamp = decode_frame(path, start)
        padded = Image.fromarray(pad_native(np.asarray(image)))
        padded.info["native_size"] = image.size
        return padded, [timestamp]
    if row.get("video_is_clip"):
        sampled = decode_video_1fps(path)
        timestamps = [start + value for value in sampled.timestamps_s]
    else:
        sampled = decode_window_1fps(path, start_s=start, end_s=end)
        timestamps = list(sampled.timestamps_s)
    native_size = (sampled.frames.shape[2], sampled.frames.shape[1])
    images = [Image.fromarray(pad_native(frame)) for frame in sampled.frames]
    for image in images:
        image.info["native_size"] = native_size
    return images, timestamps


def _vision_payload(messages: list[dict[str, Any]]) -> tuple[Any, Any, Any, dict[str, Any]]:
    # Bypass qwen-vl-utils resizing for BOTH image and video payloads.
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                frame = item.get("image")
                if not isinstance(frame, Image.Image):
                    raise TypeError("Frame must be a predecoded PIL image")
                return [frame], None, None, {}
            if item.get("type") != "video":
                continue
            frames = item.get("video")
            if not isinstance(frames, (list, tuple)):
                break
            if not frames:
                raise ValueError("A predecoded Segment video must contain at least one frame")
            expected_width, expected_height = frames[0].size
            if expected_width % 32 or expected_height % 32:
                raise ValueError("Input must be edge-padded to 32 before preprocessing")
            arrays = []
            for index, frame in enumerate(frames):
                if not isinstance(frame, Image.Image):
                    raise TypeError(f"Predecoded video frame {index} is not a PIL image")
                array = np.asarray(frame.convert("RGB"), dtype=np.uint8)
                if array.shape != (expected_height, expected_width, 3):
                    raise RuntimeError(
                        f"Segment frame {index} has shape {array.shape}; expected "
                        f"({expected_height}, {expected_width}, 3)"
                    )
                arrays.append(array)
            if len(arrays) % 2:
                raise RuntimeError("Segment frames must be padded to an even temporal-patch count")
            fps = float(item.get("fps", 1.0))
            if fps != 1.0:
                raise RuntimeError(f"Segment payload must remain at 1 FPS, received {fps}")
            video = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
            metadata = [{
                "fps": fps,
                "frames_indices": list(range(len(arrays))),
                "total_num_frames": float(len(arrays)),
            }]
            return None, [video], metadata, {"do_sample_frames": False}

    raise ValueError("Expected exactly one predecoded image or video payload")


def _assert_processor_grid(batch: Any, media: Any, track: str) -> None:
    key = "video_grid_thw" if track == "segment" else "image_grid_thw"
    grid = batch.get(key)
    if grid is None or tuple(grid.shape) != (1, 3):
        raise RuntimeError(f"Processor must return one {key} row")
    _temporal, height, width = map(int, grid[0].tolist())
    expected_temporal = len(media) // 2 if track == "segment" else 1
    if _temporal != expected_temporal:
        raise RuntimeError(f"Processor changed temporal sampling: {_temporal} != {expected_temporal}")
    frame = media[0] if track == "segment" else media
    expected_width, expected_height = (value // 16 for value in frame.size)
    if (height, width) != (expected_height, expected_width):
        raise RuntimeError(
            f"Processor changed the native padded input grid to {height}x{width}; "
            f"expected {expected_height}x{expected_width}"
        )


def prepare_training_batch(
    processor: Any,
    row: dict[str, Any],
    media: Any,
    timestamps: list[float],
    system: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prompt = make_messages(row, media, timestamps, system)
    full = [*prompt, {"role": "assistant", "content": normalize_answer(row["answer"])}]
    prompt_text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    images, videos, video_metadata, kwargs = _vision_payload(prompt)
    common = {
        "images": images,
        "videos": videos,
        "video_metadata": video_metadata,
        **kwargs,
        "return_tensors": "pt",
        "padding": True,
    }
    common["do_resize"] = False
    full_batch = processor(text=[full_text], **common)
    _assert_processor_grid(full_batch, media, row["track"])
    # Only process the large video once. Expanded image/video tokens lie entirely
    # in the shared prefix, so they cancel when counting the textual answer suffix.
    prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = processor.tokenizer(full_text, add_special_tokens=False).input_ids
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Chat template changed: prompt is not an exact token prefix")
    answer_tokens = len(full_ids) - len(prompt_ids)
    if answer_tokens <= 0:
        raise RuntimeError("Empty answer suffix")
    actual_suffix = full_batch.input_ids[0, -answer_tokens:].tolist()
    if actual_suffix != full_ids[-answer_tokens:]:
        raise RuntimeError("Processor altered the supervised textual suffix")
    prompt_tokens = int(full_batch.attention_mask[0].sum()) - answer_tokens
    full_batch["labels"] = answer_only_labels(full_batch.input_ids, full_batch.attention_mask, prompt_tokens)
    if not bool(full_batch["labels"].ne(-100).any()):
        raise RuntimeError(f"No answer tokens for {row['internal_id']}")
    return _move_batch(full_batch, device)


def prepare_generation_batch(
    processor: Any,
    row: dict[str, Any],
    media: Any,
    timestamps: list[float],
    system: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = make_messages(row, media, timestamps, system)
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    images, videos, video_metadata, kwargs = _vision_payload(messages)
    batch = processor(
        text=[rendered], images=images, videos=videos, video_metadata=video_metadata,
        **kwargs, do_resize=False, return_tensors="pt",
    )
    _assert_processor_grid(batch, media, row["track"])
    return _move_batch(batch, device)


def _move_batch(batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
    # Frozen vision receives BF16 just as native patch_embed would cast it;
    # avoid holding a full-resolution FP32 pixel tensor on the GPU.
    return {key: value.to(device=device, dtype=torch.bfloat16 if key.startswith("pixel_values") else value.dtype)
            for key, value in batch.items() if isinstance(value, torch.Tensor)}
