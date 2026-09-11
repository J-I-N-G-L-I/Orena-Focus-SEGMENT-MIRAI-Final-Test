from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path


# Literal "None" is an official FO-class answer, not evidence of a failed
# generation. Empty continuations are identified separately by the engine.
INVALID_ANSWER_CONTENT = frozenset({"null", "n/a", "na", "unanswered", "error"})


def is_placeholder_answer(content: str) -> bool:
    normalized = " ".join(str(content).split()).casefold().strip(" .")
    return not normalized or normalized in INVALID_ANSWER_CONTENT


def fallback_answer(question: str, *, start_time: float = 0.0) -> str:
    """Return a schema-safe best-effort answer after a per-question failure.

    The evaluation interface requires a string response for every qID.  A
    malformed model answer must therefore reduce that question's score, not
    terminate the whole hidden batch.  These conservative values also parse
    for the common binary/count/percentage/time formats.
    """
    text = " ".join(str(question).casefold().split())
    if any(token in text for token in ("how many", "number of", "count of")):
        return "0"
    if "percent" in text or "percentage" in text:
        return "0%"
    if any(token in text for token in ("what time", "timestamp", "when did", "when does")):
        total = max(0, int(round(float(start_time))))
        return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
    binary_prefixes = (
        "is ", "are ", "was ", "were ", "does ", "do ", "did ",
        "has ", "have ", "had ", "can ", "could ", "will ", "would ",
    )
    if text.startswith(binary_prefixes):
        return "No."
    return "None"


@dataclass(frozen=True)
class Request:
    qID: str
    videoID: str
    start_time: float
    end_time: float
    procedure_type: str
    question: str


@dataclass(frozen=True)
class Response:
    qID: str
    content: str
    latency: float


def load_requests(path: str | Path) -> list[Request]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError("request.json must contain a list")
    required = {"qID", "videoID", "start_time", "end_time", "procedure_type", "question"}
    result = []
    for index, row in enumerate(value):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Invalid request schema at row {index}")
        qid = str(row["qID"]).strip()
        start, end = float(row["start_time"]), float(row["end_time"])
        if not qid or not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError(f"Invalid qID/time values at row {index}")
        result.append(Request(qid, str(row["videoID"]), start, end, str(row["procedure_type"]), str(row["question"])))
    if len({item.qID for item in result}) != len(result):
        raise ValueError("Duplicate request qID")
    return result


def save_responses(responses: list[Response], path: str | Path) -> None:
    if len({item.qID for item in responses}) != len(responses):
        raise ValueError("Duplicate response qID")
    rows = []
    for item in responses:
        content = " ".join(str(item.content).split()) or "None"
        latency = float(item.latency)
        if not math.isfinite(latency) or latency < 0:
            latency = 0.0
        rows.append({**asdict(item), "content": content[:300], "latency": latency})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
