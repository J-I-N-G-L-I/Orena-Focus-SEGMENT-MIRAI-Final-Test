#!/usr/bin/env python3
"""Restore the public model release using only the Python standard library.

The checked-in RELEASE_MANIFEST.json authenticates the downloaded archive and
each restored file. Only the exact listed regular files under model/ are allowed.
The slim run_manifest contains the original experiment configuration; removing
unrelated training metadata does not change inference configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str, *, directory: bool = False) -> str:
    value = value.rstrip("/") if directory else value
    path = PurePosixPath(value)
    if (not value or "\\" in value or ":" in value or path.is_absolute()
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or path.parts[0] != "model" or (len(path.parts) < 2 and not directory)):
        raise ValueError(f"Unsafe model archive path: {value!r}")
    return path.as_posix()


def check_file(path: Path, expected: dict) -> None:
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Size mismatch: {path.name}")
    if sha256(path) != expected["sha256"].lower():
        raise ValueError(f"SHA256 mismatch: {path.name}")


def restore(archive: Path, manifest_path: Path, destination: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: dict[str, dict] = {}
    for item in manifest["model_files"]:
        name = safe_name(item["path"])
        if name in files:
            raise ValueError(f"Duplicate file in manifest: {name}")
        if not isinstance(item["bytes"], int) or item["bytes"] < 0:
            raise ValueError(f"Invalid file size: {name}")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", item["sha256"]):
            raise ValueError(f"Invalid SHA256: {name}")
        files[name] = item
    if not files:
        raise ValueError("The release manifest contains no model files")
    check_file(archive, manifest["model_archive"])
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    directories = {parent.as_posix() for name in files
                   for parent in PurePosixPath(name).parents if str(parent) != "."}
    # Extract by copying file streams; never call extract/extractall on a tar.
    with tempfile.TemporaryDirectory(prefix=".restore-model-", dir=destination) as temporary:
        staging = Path(temporary)
        seen: set[str] = set()
        with tarfile.open(archive, mode="r:gz") as package:
            for member in package:
                if member.isdir():
                    if safe_name(member.name, directory=True) not in directories:
                        raise ValueError(f"Unexpected archive directory: {member.name}")
                    continue
                if not member.isfile():
                    raise ValueError(f"Only regular files are allowed: {member.name}")
                name = safe_name(member.name)
                if name not in files or name in seen:
                    raise ValueError(f"Unexpected or duplicate archive file: {name}")
                expected = files[name]
                if member.size != expected["bytes"]:
                    raise ValueError(f"Archive member size mismatch: {name}")
                path = staging.joinpath(*PurePosixPath(name).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                source = package.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot read archive member: {name}")
                digest = hashlib.sha256()
                with source, path.open("xb") as target:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                        target.write(chunk)
                if path.stat().st_size != expected["bytes"] or digest.hexdigest() != expected["sha256"].lower():
                    raise ValueError(f"Extracted file hash/size mismatch: {name}")
                seen.add(name)
        if seen != set(files):
            raise ValueError(f"Archive is missing files: {sorted(set(files) - seen)}")
        # Reject redirects before replacing any file in an existing checkout.
        for name in files:
            target = destination.joinpath(*PurePosixPath(name).parts)
            for candidate in (target, *target.parents):
                if candidate == destination:
                    break
                if candidate.is_symlink():
                    raise ValueError(f"Destination contains a symlink: {candidate}")
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Destination escapes release folder: {name}")
            if target.exists() and not target.is_file():
                raise ValueError(f"Destination is not a file: {name}")
        for name in sorted(files):
            relative = PurePosixPath(name).parts
            target = destination.joinpath(*relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging.joinpath(*relative), target)
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Downloaded model .tar.gz release asset")
    parser.add_argument("--manifest", type=Path, default=ROOT / "RELEASE_MANIFEST.json")
    parser.add_argument("--destination", type=Path, default=ROOT, help="Repository root")
    args = parser.parse_args()
    count = restore(args.archive, args.manifest, args.destination)
    print(f"Verified and restored {count} model files to {args.destination.resolve() / 'model'}")


if __name__ == "__main__":
    main()
