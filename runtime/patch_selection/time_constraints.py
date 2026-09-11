from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ConstraintKind = Literal["global", "range", "discrete"]

_CLOCK = r"(?<!\d)(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:\.(\d+))?(?!\d)"
_UNIT_TIME = r"(?<![\w.])(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)(?!\w)"
_RANGE_WORDS = re.compile(
    r"\b(?:between\b.+?\band\b|from\b.+?\b(?:to|until|through|thru)\b)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class TimeConstraint:
    kind: ConstraintKind
    anchors_s: tuple[float, ...] = ()
    start_s: float | None = None
    end_s: float | None = None

    @property
    def is_true_range(self) -> bool:
        return self.kind == "range"

    def unit_indices(self, unit_seconds: float, unit_count: int) -> tuple[int, ...]:
        if unit_count <= 0:
            return ()
        values = self.anchors_s
        if self.kind == "range" and self.start_s is not None and self.end_s is not None:
            values = (self.start_s, self.end_s)
        indices = {
            max(0, min(unit_count - 1, int(value // unit_seconds)))
            for value in values
            if value >= 0
        }
        return tuple(sorted(indices))

    def range_unit_bounds(self, unit_seconds: float, unit_count: int) -> tuple[int, int] | None:
        if not self.is_true_range or self.start_s is None or self.end_s is None or unit_count <= 0:
            return None
        lo = max(0, min(unit_count - 1, int(self.start_s // unit_seconds)))
        hi = max(0, min(unit_count - 1, int(self.end_s // unit_seconds)))
        return min(lo, hi), max(lo, hi)

    def relative_to_clip(self, procedure_start_s: float, duration_s: float) -> "TimeConstraint":
        """Map procedure-clock anchors to clip time without cropping out-of-context questions."""

        if self.kind == "global":
            return self
        def convert(value: float) -> float:
            if procedure_start_s <= value <= procedure_start_s + duration_s:
                return value - procedure_start_s
            return value
        if self.kind == "range" and self.start_s is not None and self.end_s is not None:
            start, end = convert(self.start_s), convert(self.end_s)
            if end < 0 or start > duration_s or start > end:
                return TimeConstraint(kind="global")
            start, end = max(0.0, start), min(duration_s, end)
            return TimeConstraint(kind="range", anchors_s=(start, end), start_s=start, end_s=end)
        anchors = tuple(value for value in (convert(item) for item in self.anchors_s) if 0 <= value <= duration_s)
        return TimeConstraint(kind="discrete", anchors_s=anchors)


def _extract_times(text: str) -> list[tuple[int, float]]:
    found: list[tuple[int, float]] = []
    occupied: list[tuple[int, int]] = []
    for match in re.finditer(_CLOCK, text):
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        fraction = float(f"0.{match.group(4)}") if match.group(4) else 0.0
        found.append((match.start(), hours * 3600.0 + minutes * 60.0 + seconds + fraction))
        occupied.append(match.span())
    for match in re.finditer(_UNIT_TIME, text, re.IGNORECASE):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("h"):
            value *= 3600.0
        elif unit.startswith("m"):
            value *= 60.0
        found.append((match.start(), value))
    return sorted(found)


def parse_time_constraint(question: str) -> TimeConstraint:
    """Parse explicit clock constraints without turning arbitrary timestamps into a range."""

    times = _extract_times(question)
    values = tuple(value for _, value in times)
    if not values:
        return TimeConstraint(kind="global")
    if len(values) >= 2 and _RANGE_WORDS.search(question):
        start, end = values[0], values[1]
        return TimeConstraint(kind="range", anchors_s=(start, end), start_s=min(start, end), end_s=max(start, end))
    return TimeConstraint(kind="discrete", anchors_s=tuple(dict.fromkeys(values)))

