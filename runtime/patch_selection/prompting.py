from __future__ import annotations

import re
from typing import Any


OUTPUT_RULES = (
    "Output only the answer, with no explanation. Use yes or no for binary questions; "
    "one non-negative integer for counts; one numeric percentage for percentages; "
    "procedure-clock HH:MM:SS timestamps for time questions; official foreign-object class names, "
    "or None, for classification; and only selected choice text for multiple choice."
)


def seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        parts = text.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
        raise ValueError(f"Invalid timestamp {value!r}")


def hhmmss(value: float) -> str:
    total = max(0, int(round(value)))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def system_prompt(foreign_object_definitions: str = "") -> str:
    suffix = f"\n{foreign_object_definitions.strip()}" if foreign_object_definitions.strip() else ""
    return "You are a surgical assistant analyzing endoscopic evidence. Answer precisely and concisely." + suffix


def _timestamp_table(timestamps: list[float], start: float, track: str) -> str:
    if track == "frame":
        return f"frame 1: procedure={hhmmss(timestamps[0])}; clip-relative={hhmmss(timestamps[0] - start)}"
    rows = []
    for unit, offset in enumerate(range(0, len(timestamps), 2), start=1):
        left = timestamps[offset]
        right = timestamps[min(offset + 1, len(timestamps) - 1)]
        rows.append(
            f"u{unit}={hhmmss(left)}..{hhmmss(right)}/"
            f"+{max(0, int(round(left - start)))}..+{max(0, int(round(right - start)))}s"
        )
    return "; ".join(rows)


def user_text(row: dict[str, Any], timestamps: list[float]) -> str:
    track = str(row["track"]).lower()
    start, end = seconds(row["timestamp_start"]), seconds(row["timestamp_end"])
    return "\n".join(
        [
            f"Track: {track.upper()}",
            f"Procedure window: {hhmmss(start)} to {hhmmss(end)}.",
            "For a time answer, use procedure clock rather than clip-relative time.",
            "Visual-time table (procedure/clip-relative): " + _timestamp_table(timestamps, start, track),
            f"Output rules: {OUTPUT_RULES}",
            f"Question: {str(row['question']).strip()}",
        ]
    )


def make_messages(row: dict[str, Any], media: Any, timestamps: list[float], system: str) -> list[dict[str, Any]]:
    track = str(row["track"]).lower()
    visual = {"type": "video", "video": media, "fps": 1.0} if track == "segment" else {"type": "image", "image": media}
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [visual, {"type": "text", "text": user_text(row, timestamps)}]},
    ]


def normalize_answer(value: Any) -> str:
    answer = " ".join(str(value).strip().split())
    if not answer:
        raise ValueError("Empty training answer")
    return answer[:300]


def sanitize_prediction(value: str) -> str:
    value = re.sub(r"<think>.*?</think>", "", str(value), flags=re.I | re.S)
    value = re.sub(r"^\s*(?:final\s+)?answer\s*:\s*", "", value, flags=re.I)
    return (" ".join(value.split()) or "None")[:300]


