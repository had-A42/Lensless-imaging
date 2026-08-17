from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

MIRFLICKR_IMAGE_COUNT = 25_000
DIGICAM_MULTIMASK_PERIOD = 100
DIGICAM_REAL_TEST_MASK_COUNT = 15

_ORIGINAL_FILENAME = re.compile(r"^im(?P<one_based>[1-9][0-9]*)$", re.IGNORECASE)
_RENAMED_FILENAME = re.compile(r"^(?P<zero_based>0|[1-9][0-9]*)$")


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mirflickr_source_index(path: str | Path) -> int:
    path = Path(path)
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError(f"MIRFLICKR image must be JPEG, got {path.name!r}")

    original = _ORIGINAL_FILENAME.fullmatch(path.stem)
    if original is not None:
        return int(original.group("one_based")) - 1

    renamed = _RENAMED_FILENAME.fullmatch(path.stem)
    if renamed is not None:
        return int(renamed.group("zero_based"))

    raise ValueError(f"Not a canonical MIRFLICKR filename: {path.name!r}")


def discover_mirflickr_images(root_dir: str | Path) -> dict[int, Path]:
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MIRFLICKR root directory not found: {root}")

    images: dict[int, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        try:
            source_index = mirflickr_source_index(path)
        except ValueError:
            continue
        previous = images.get(source_index)
        if previous is not None:
            raise ValueError(
                f"Duplicate MIRFLICKR source index {source_index}: "
                f"{previous} and {path}"
            )
        images[source_index] = path.resolve()

    if not images:
        raise ValueError(f"No canonical MIRFLICKR JPEGs found below {root}")
    return images


def verify_mirflickr_images(images: dict[int, Path]) -> None:
    for source_index, path in images.items():
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            raise ValueError(
                f"Invalid MIRFLICKR image at source index {source_index}: {path}"
            ) from error


def is_external_real_test_scene(source_index: int) -> bool:
    source_index = int(source_index)
    if source_index < 0:
        raise ValueError("source_index must be non-negative")
    return source_index % DIGICAM_MULTIMASK_PERIOD < DIGICAM_REAL_TEST_MASK_COUNT


def _normalize_split_counts(split_counts: dict[str, int]) -> dict[str, int]:
    required_order = ("train", "validation", "test")
    if set(split_counts) != set(required_order):
        raise ValueError("split_counts must contain exactly train, validation and test")
    normalized = {name: int(split_counts[name]) for name in required_order}
    if any(value <= 0 for value in normalized.values()):
        raise ValueError("Every scene split must contain at least one scene")
    return normalized


def build_mirflickr_splits(
    *,
    root_dir: str | Path,
    output_path: str | Path,
    split_counts: dict[str, int],
    seed: int,
    expected_image_count: int | None = MIRFLICKR_IMAGE_COUNT,
) -> dict[str, Any]:
    root = Path(root_dir).expanduser().resolve()
    images = discover_mirflickr_images(root)
    if expected_image_count is not None and len(images) != int(expected_image_count):
        raise ValueError(
            f"Expected {int(expected_image_count)} MIRFLICKR images, found {len(images)}"
        )

    counts = _normalize_split_counts(split_counts)
    eligible = sorted(
        source_index
        for source_index in images
        if not is_external_real_test_scene(source_index)
    )
    requested = sum(counts.values())
    if requested > len(eligible):
        raise ValueError(
            f"Requested {requested} scenes but only {len(eligible)} are eligible"
        )
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")

    rng = np.random.default_rng(int(seed))
    selected = [int(value) for value in rng.permutation(eligible)[:requested]]
    splits: dict[str, list[str]] = {}
    offset = 0
    for split, count in counts.items():
        splits[split] = [
            images[source_index].relative_to(root).as_posix()
            for source_index in selected[offset : offset + count]
        ]
        offset += count

    payload = {"seed": int(seed), "splits": splits}

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def load_mirflickr_splits(path: str | Path) -> dict[str, Any]:
    splits_path = Path(path).expanduser()
    if not splits_path.is_file():
        raise FileNotFoundError(f"MIRFLICKR splits not found: {splits_path}")
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    validate_mirflickr_splits(payload)
    return payload


def validate_mirflickr_splits(
    payload: dict[str, Any],
    *,
    root_dir: str | Path | None = None,
) -> None:
    if not isinstance(payload, dict) or set(payload) != {"seed", "splits"}:
        raise ValueError("Split file must contain only seed and splits")
    if not isinstance(payload["seed"], int) or payload["seed"] < 0:
        raise ValueError("Split seed must be a non-negative integer")
    splits = payload["splits"]
    if not isinstance(splits, dict) or set(splits) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("Splits must contain train, validation and test")

    source_indices: set[int] = set()
    relative_paths: set[str] = set()
    root = Path(root_dir).expanduser().resolve() if root_dir is not None else None

    for split, paths in splits.items():
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"Split {split} must be a non-empty list")
        for relative_path in paths:
            if not isinstance(relative_path, str):
                raise ValueError("Split paths must be safe relative strings")
            path = Path(relative_path)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Split paths must be safe relative strings")
            source_index = mirflickr_source_index(path)
            if source_index in source_indices or relative_path in relative_paths:
                raise ValueError("Split files must be globally unique")
            if is_external_real_test_scene(source_index):
                raise ValueError(
                    f"Scene {source_index} violates the external real-test exclusion"
                )
            if root is not None and not (root / path).is_file():
                raise FileNotFoundError(root / path)
            source_indices.add(source_index)
            relative_paths.add(relative_path)


class MirFlickrSceneDataset(Dataset):
    def __init__(
        self,
        *,
        root_dir: str | Path,
        splits_path: str | Path,
        split: str,
        image_size: list[int] | None = None,
        verify_files: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.splits_path = Path(splits_path).expanduser().resolve()
        self.split = str(split)
        if image_size is not None:
            image_size = tuple(int(value) for value in image_size)
            if len(image_size) != 2 or min(image_size) <= 0:
                raise ValueError("image_size must contain positive [height, width]")
        self.image_size = image_size

        payload = load_mirflickr_splits(self.splits_path)
        validate_mirflickr_splits(
            payload,
            root_dir=self.root_dir if verify_files else None,
        )
        available_splits = set(payload["splits"])
        if self.split not in available_splits:
            raise ValueError(
                f"Unknown split {self.split!r}; expected one of {sorted(available_splits)}"
            )
        self.records = tuple(
            {
                "relative_path": relative_path,
                "source_index": mirflickr_source_index(relative_path),
            }
            for relative_path in payload["splits"][self.split]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self.root_dir / record["relative_path"]
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        target = torch.from_numpy(np.moveaxis(rgb, -1, 0).copy()).contiguous()
        if self.image_size is not None and target.shape[-2:] != self.image_size:
            target = F.interpolate(
                target.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        target = target.clamp(0.0, 1.0)
        scene_id = f'mirflickr_{record["source_index"]:05d}'
        return {
            "target": target,
            "sample_id": scene_id,
            "scene_id": scene_id,
            "source_index": record["source_index"],
            "split": self.split,
        }
