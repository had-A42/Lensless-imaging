"""Shared loading and structural validation for dataset split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SPLIT_NAMES = ("train", "validation", "test")


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset split manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid dataset split manifest JSON: {manifest_path}"
        ) from error
    validate_split_manifest(payload)
    return payload


def validate_split_manifest(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or set(payload) != {"seed", "splits"}:
        raise ValueError("Dataset split manifest must contain only seed and splits")
    if not isinstance(payload["seed"], int) or payload["seed"] < 0:
        raise ValueError("Manifest seed must be a non-negative integer")
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES):
        raise ValueError("Manifest splits must contain train, validation and test")
    seen: set[str] = set()
    for split in SPLIT_NAMES:
        values = splits[split]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Manifest split {split!r} must be a non-empty list")
        for value in values:
            if not isinstance(value, str) or not value:
                raise ValueError("Manifest IDs must be non-empty strings")
            if value in seen:
                raise ValueError(
                    "Manifest IDs must be globally unique; "
                    f"duplicate occurs in multiple splits: {value}"
                )
            seen.add(value)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split_manifest(
    payload: dict[str, Any],
    output: str | Path,
    *,
    rewrite: bool,
) -> dict[str, Any]:
    validate_split_manifest(payload)
    target = Path(output)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    created = not target.exists()
    rewritten = False
    target.parent.mkdir(parents=True, exist_ok=True)
    should_write = created
    if not created and target.read_text(encoding="utf-8") != text:
        if not rewrite:
            raise ValueError(
                f"Existing manifest differs from generated split: {target}"
            )
        rewritten = True
        should_write = True
    if should_write:
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
    return {
        "created": created,
        "rewritten": rewritten,
        "sha256": file_sha256(target),
    }


def split_manifest_summary(
    *,
    dataset: str,
    root: str | Path,
    output: str | Path,
    payload: dict[str, Any],
    write_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "pass",
        "dataset": dataset,
        "root": str(Path(root).expanduser().resolve()),
        "manifest": str(Path(output).expanduser().resolve()),
        "seed": payload["seed"],
        "counts": {split: len(payload["splits"][split]) for split in SPLIT_NAMES},
        **write_result,
    }


__all__ = [
    "SPLIT_NAMES",
    "file_sha256",
    "load_split_manifest",
    "split_manifest_summary",
    "validate_split_manifest",
    "write_split_manifest",
]
