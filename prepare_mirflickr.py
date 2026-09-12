import json
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path

from src.datasets.mirflickr import (
    discover_mirflickr_images,
    make_mirflickr_splits,
    validate_mirflickr_splits,
)
from src.datasets.preparation import download_http, extract_zip, verify_images
from src.datasets.split_manifest import split_manifest_summary, write_split_manifest


def prepare(config) -> dict:
    root = Path(to_absolute_path(config.root_dir))
    archive = Path(to_absolute_path(config.source.archive_path))
    extract_dir = Path(to_absolute_path(config.source.extract_dir))
    root.mkdir(parents=True, exist_ok=True)

    downloaded = download_http(
        str(config.source.download_url),
        archive,
        enabled=bool(config.download),
        expected_bytes=int(config.source.expected_archive_bytes),
        expected_digests={
            "sha256": config.source.expected_archive_sha256,
        },
    )

    expected_count = int(config.source.expected_image_count)
    images = {}
    try:
        images = discover_mirflickr_images(extract_dir)
    except (FileNotFoundError, ValueError):
        pass
    extraction_complete = len(images) == expected_count
    extracted_files = 0
    if not extraction_complete:
        extracted_files = extract_zip(archive, extract_dir)
        images = discover_mirflickr_images(extract_dir)
    if len(images) != expected_count:
        raise ValueError(
            f"Expected {expected_count} MIRFLICKR images below {extract_dir}, "
            f"found {len(images)}"
        )

    verified_files = verify_images(
        images.values(),
        description="Checking MIRFLICKR images",
    )

    manifest_output = Path(to_absolute_path(config.manifest.output_path))
    manifest_payload = make_mirflickr_splits(
        root_dir=extract_dir,
        split_counts={
            "train": int(config.manifest.counts.train),
            "validation": int(config.manifest.counts.validation),
            "test": int(config.manifest.counts.test),
        },
        seed=int(config.manifest.seed),
        expected_image_count=expected_count,
    )
    validate_mirflickr_splits(manifest_payload, root_dir=extract_dir)
    manifest_write = write_split_manifest(
        manifest_payload,
        manifest_output,
        rewrite=bool(config.manifest.rewrite),
    )
    manifest_summary = split_manifest_summary(
        dataset="MIRFLICKR-25000",
        root=extract_dir,
        output=manifest_output,
        payload=manifest_payload,
        write_result=manifest_write,
    )

    return {
        "status": "pass",
        "dataset": "MIRFLICKR-25000",
        "root": str(root.resolve()),
        "downloaded_files": [str(archive)] if downloaded else [],
        "extracted_files": extracted_files,
        "verified_files": verified_files,
        "archive": str(archive),
        "discovered_images": len(images),
        "manifest": manifest_summary,
    }


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="prepare_mirflickr",
)
def main(config) -> None:
    print(json.dumps(prepare(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
