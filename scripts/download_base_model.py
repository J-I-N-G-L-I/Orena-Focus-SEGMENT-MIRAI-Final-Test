#!/usr/bin/env python3
"""Download the exact upstream Qwen3.5-9B revision used by this release."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def default_output() -> Path:
    scratch = os.environ.get("SCRATCH_ROOT")
    if not scratch and Path.home().parent == Path("/users"):
        scratch = str(Path("/mnt/scratch") / Path.home().name)
    if scratch:
        root = Path(scratch)
        os.environ.setdefault("HF_HOME", str(root / "hf_home"))
        hf_home = Path(os.environ["HF_HOME"])
        for variable, directory in (("HF_HUB_CACHE", "hub"), ("HF_DATASETS_CACHE", "datasets"),
                                    ("HF_XET_CACHE", "xet"), ("HF_ASSETS_CACHE", "assets")):
            os.environ.setdefault(variable, str(hf_home / directory))
        return root / "mirai-segment" / "resources" / "base_qwen35_9b"
    return ROOT / "resources" / "base_qwen35_9b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output())
    args = parser.parse_args()
    # Import after configuring scratch/cache paths: the Hub reads them at import.
    from huggingface_hub import snapshot_download

    target = args.output.expanduser().resolve()
    location = snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=target)
    print(f"Downloaded {MODEL_ID}@{REVISION}\nLocation: {location}")


if __name__ == "__main__":
    main()
