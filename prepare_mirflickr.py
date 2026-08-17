from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.mirflickr import (
    build_mirflickr_splits,
    discover_mirflickr_images,
    file_digest,
    validate_mirflickr_splits,
    verify_mirflickr_images,
)


def _safe_extract(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(archive) as file:
        for member in file.infolist():
            destination = (root / member.filename).resolve()
            if root != destination and root not in destination.parents:
                raise ValueError(f"Unsafe zip member path: {member.filename!r}")
        file.extractall(root)


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="prepare_mirflickr",
)
def main(config) -> None:
    archive = Path(to_absolute_path(config.source.archive_path))
    extract_dir = Path(to_absolute_path(config.source.extract_dir))
    splits_path = Path(to_absolute_path(config.splits.output_path))
    if not archive.is_file():
        raise FileNotFoundError(
            f"MIRFLICKR archive not found: {archive}. Download source.selected_url "
            "before running preparation."
        )

    archive_md5 = file_digest(archive, "md5")
    archive_sha256 = file_digest(archive, "sha256")
    archive_bytes = archive.stat().st_size
    expected_md5 = config.source.expected_archive_md5
    if expected_md5 is not None and archive_md5.lower() != str(expected_md5).lower():
        raise ValueError(
            f"Archive MD5 mismatch: expected {expected_md5}, got {archive_md5}"
        )
    expected_sha256 = config.source.expected_archive_sha256
    if archive_sha256.lower() != str(expected_sha256).lower():
        raise ValueError(
            "Archive SHA-256 mismatch: "
            f"expected {expected_sha256}, got {archive_sha256}"
        )
    expected_bytes = int(config.source.expected_archive_bytes)
    if archive_bytes != expected_bytes:
        raise ValueError(
            f"Archive size mismatch: expected {expected_bytes}, got {archive_bytes}"
        )

    if bool(config.extract):
        _safe_extract(archive, extract_dir)

    discovered_images = discover_mirflickr_images(extract_dir)
    decode_verified_image_count = None
    if bool(config.verify_images):
        verify_mirflickr_images(discovered_images)
        decode_verified_image_count = len(discovered_images)

    split_counts = OmegaConf.to_container(config.splits.split_counts, resolve=True)
    payload = build_mirflickr_splits(
        root_dir=extract_dir,
        output_path=splits_path,
        split_counts=split_counts,
        seed=int(config.splits.seed),
        expected_image_count=int(config.source.expected_image_count),
    )
    validate_mirflickr_splits(payload, root_dir=extract_dir)
    summary = {
        "archive": str(archive),
        "archive_bytes": archive_bytes,
        "archive_md5": archive_md5,
        "archive_sha256": archive_sha256,
        "extract_dir": str(extract_dir),
        "splits_path": str(splits_path),
        "split_seed": payload["seed"],
        "split_counts": {
            split: len(paths) for split, paths in payload["splits"].items()
        },
        "discovered_image_count": len(discovered_images),
        "decode_verified_image_count": decode_verified_image_count,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
