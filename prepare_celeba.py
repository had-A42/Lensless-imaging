import json
import zipfile
from pathlib import Path

import gdown
import hydra
from hydra.utils import to_absolute_path
from torchvision.datasets import CelebA

from src.datasets.celeba import celeba_filenames
from src.datasets.preparation import extract_zip, verify_file, verify_images
from src.datasets.split_manifest import split_manifest_summary, write_split_manifest

SOURCE_RECORDS = {
    filename: {"file_id": file_id, "md5": md5}
    for file_id, md5, filename in CelebA.file_list
}


def _valid_partition(path: Path) -> None:
    with path.open(encoding="utf-8") as stream:
        first_row = stream.readline().split()
    if len(first_row) != 2 or not first_row[0].endswith(".jpg") or first_row[1] != "0":
        raise ValueError(f"Invalid CelebA partition file: {path}")


def _get_source_file(
    *,
    root: Path,
    filename: str,
    download: bool,
) -> tuple[Path, bool]:
    if filename not in SOURCE_RECORDS:
        raise ValueError(f"Unknown CelebA source file: {filename}")
    target = root / filename
    downloaded = False
    if not target.is_file():
        if not download:
            raise FileNotFoundError(
                f"Dataset file is missing: {target}. Re-run with download=true."
            )
        partial = target.with_suffix(target.suffix + ".part")
        result = gdown.download(
            id=SOURCE_RECORDS[filename]["file_id"],
            output=str(partial),
            quiet=False,
            resume=True,
        )
        if result is None or not partial.is_file():
            raise RuntimeError(f"Could not download CelebA file {filename}")
        if filename.endswith(".zip") and not zipfile.is_zipfile(partial):
            raise ValueError("Google Drive did not return the CelebA ZIP archive")
        if filename.endswith(".txt"):
            _valid_partition(partial)
        verify_file(
            partial,
            expected_digests={"md5": SOURCE_RECORDS[filename]["md5"]},
        )
        partial.replace(target)
        downloaded = True
    else:
        verify_file(
            target,
            expected_digests={"md5": SOURCE_RECORDS[filename]["md5"]},
        )
    return target, downloaded


def prepare(config) -> dict:
    root = Path(to_absolute_path(config.root_dir))
    root.mkdir(parents=True, exist_ok=True)
    partition_name = str(config.source.partition_file)
    archive_name = str(config.source.archive_file)
    partition, partition_downloaded = _get_source_file(
        root=root,
        filename=partition_name,
        download=bool(config.download),
    )
    _valid_partition(partition)

    seed = int(config.manifest.seed)
    train_count = int(config.selection.train_count)
    validation_count = int(config.selection.validation_count)
    selected = {
        "train": celeba_filenames(root, "train", train_count, seed),
        "validation": celeba_filenames(root, "validation", validation_count, seed),
    }
    image_root = root / "img_align_celeba"
    required_names = [name for names in selected.values() for name in names]
    missing = [name for name in required_names if not (image_root / name).is_file()]

    archive = root / archive_name
    archive_downloaded = False
    extracted_files = 0
    if missing:
        archive, archive_downloaded = _get_source_file(
            root=root,
            filename=archive_name,
            download=bool(config.download),
        )
        extracted_files = extract_zip(
            archive,
            root,
            members=[f"img_align_celeba/{name}" for name in missing],
        )
    elif archive.is_file():
        _get_source_file(
            root=root,
            filename=archive_name,
            download=False,
        )

    verified_files = verify_images(
        [image_root / name for name in required_names],
        description="Checking CelebA images",
    )

    manifest_output = Path(to_absolute_path(config.manifest.output_path))
    manifest_payload = {
        "seed": seed,
        "splits": {
            "train": selected["train"],
            "validation": selected["validation"],
            "test": celeba_filenames(root, "test", None, seed),
        },
    }
    manifest_write = write_split_manifest(
        manifest_payload,
        manifest_output,
        rewrite=bool(config.manifest.rewrite),
    )
    manifest_summary = split_manifest_summary(
        dataset="CelebA",
        root=root,
        output=manifest_output,
        payload=manifest_payload,
        write_result=manifest_write,
    )

    downloaded_files = []
    if partition_downloaded:
        downloaded_files.append(str(partition))
    if archive_downloaded:
        downloaded_files.append(str(archive))
    return {
        "status": "pass",
        "dataset": "CelebA",
        "root": str(root.resolve()),
        "downloaded_files": downloaded_files,
        "extracted_files": extracted_files,
        "verified_files": verified_files,
        "selected_counts": {split: len(names) for split, names in selected.items()},
        "manifest": manifest_summary,
    }


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="prepare_celeba",
)
def main(config) -> None:
    print(json.dumps(prepare(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
