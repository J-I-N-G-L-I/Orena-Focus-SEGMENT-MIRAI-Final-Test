#!/usr/bin/env python3
"""Run the frozen SEGMENT entry point on an official-format input directory."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Contains request.json, FO_definitions.json and plain/<qID>.mp4")
    parser.add_argument("--output", type=Path, required=True, help="Receives answer.json")
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--base-model", type=Path, default=ROOT / "resources" / "base_qwen35_9b")
    args = parser.parse_args()
    for path in (args.input / "request.json", args.input / "FO_definitions.json",
                 args.model / "adapter" / "adapter_model.safetensors",
                 args.model / "processor" / "processor_config.json", args.model / "run_manifest.json",
                 args.base_model / "config.json"):
        if not path.is_file():
            parser.error(f"Missing required file: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({
        "ORENA_INPUT": str(args.input.resolve()), "ORENA_OUTPUT": str(args.output.resolve()),
        "RAW_CHECKPOINT": str(args.model.resolve()), "RAW_BASE_MODEL": str(args.base_model.resolve()),
        "CRD_ATTENTION": "sdpa", "CRD_SUBMISSION_PROFILE": "legacy-fallback",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    return subprocess.run([sys.executable, str(ROOT / "runtime" / "submission" / "inference.py")],
                          env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
