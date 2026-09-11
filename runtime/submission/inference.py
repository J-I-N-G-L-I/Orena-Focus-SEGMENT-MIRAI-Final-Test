"""ORena Segment entry point using the exact exported native-resolution runtime."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import time

START = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from challenge_io import Response, fallback_answer, is_placeholder_answer, load_requests, save_responses


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    source = Path(os.environ.get("ORENA_INPUT", "/input"))
    output = Path(os.environ.get("ORENA_OUTPUT", "/output")) / "answer.json"
    checkpoint = Path(os.environ.get("RAW_CHECKPOINT", "/opt/app/raw_checkpoint"))
    base = Path(os.environ.get("RAW_BASE_MODEL", "/opt/app/resources/base_qwen35_9b"))
    requests = load_requests(source / "request.json")
    responses = [Response(q.qID, fallback_answer(q.question, start_time=q.start_time), 0.) for q in requests]
    save_responses(responses, output)
    if not requests:
        return 0
    allowed = 120. + 15. * len(requests)
    deadline = START + allowed - max(15., min(60., .05 * allowed))
    success, fallback, skipped = 0, 0, 0
    try:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Submission requires a BF16 CUDA GPU")
        if os.environ.get("CRD_SUBMISSION_PROFILE", "legacy-fallback") == "h100-accelerated":
            from submission.acceleration import validate_installed
            logging.info("[accelerated-runtime] %s", json.dumps(validate_installed(require_h100=True), sort_keys=True))
        from inference_engine import InferenceEngine, profile_dict
        value = json.loads((source / "FO_definitions.json").read_text(encoding="utf-8"))
        definitions = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        engine = InferenceEngine(checkpoint, foreign_object_definitions=definitions, base_model=base,
                                 require_fast_path=False, local_files_only=True)
        logging.info("[runtime] gpu=%s resolution=%s config=%s", torch.cuda.get_device_name(0),
                     engine.config.resolution_mode, engine.config.to_dict())
        logging.info("[runtime] attention=%s", json.dumps(engine.attention, sort_keys=True))
    except Exception:
        logging.exception("[setup] model initialization failed; responses are fallbacks")
        logging.info("[batch-summary] model_success=0 fallback=%d deadline_skip=0 total=%d", len(requests), len(requests))
        return 0
    for index, q in enumerate(requests):
        if time.monotonic() >= deadline:
            skipped = len(requests) - index
            break
        start = time.perf_counter()
        row = {"internal_id": q.qID, "track": "segment", "video_is_clip": True,
               "video_path": str(source / "plain" / f"{q.qID}.mp4"),
               "timestamp_start": q.start_time, "timestamp_end": q.end_time,
               "start_seconds": q.start_time, "end_seconds": q.end_time,
               "procedure_type": q.procedure_type, "question": q.question}
        try:
            answer, profile = engine.answer(row, max_new_tokens=32)
            if profile.empty_generation or is_placeholder_answer(answer):
                raise RuntimeError("Model produced placeholder content")
            success += 1
            logging.info("qID=%s profile=%s", q.qID, json.dumps(profile_dict(profile), sort_keys=True))
        except Exception:
            logging.exception("qID=%s inference failed", q.qID)
            answer = fallback_answer(q.question, start_time=q.start_time)
            fallback += 1
            torch.cuda.empty_cache()
        responses[index] = Response(q.qID, answer, time.perf_counter() - start)
        save_responses(responses, output)
    logging.info("[batch-summary] model_success=%d fallback=%d deadline_skip=%d total=%d", success, fallback, skipped, len(requests))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
