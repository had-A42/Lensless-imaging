"""Create a portable source-only archive for an isolated server checkout."""

import argparse
import tarfile
from pathlib import Path

from .common import ROOT, save_json, sha256, source_hashes


def bundle(destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    paths = [
        ROOT / "src",
        ROOT / "scripts/scene_scaling",
        ROOT / "train.py",
        ROOT / "inference.py",
        ROOT / "requirements.txt",
        ROOT / "pytest.ini",
        ROOT / "tests",
        ROOT / "manifests/mirflickr25k_splits.json",
    ]

    def exclude(info):
        if "__pycache__" in Path(info.name).parts or info.name.endswith(
            (".pyc", ".DS_Store")
        ):
            return None
        return info

    with tarfile.open(destination, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)), filter=exclude)
    save_json(
        destination.with_suffix(destination.suffix + ".json"),
        {
            "archive_sha256": sha256(destination),
            "source_hashes": source_hashes(),
            "contents": "source, tests, archived reference configs, original manifest; no image data or weights",
            "deployment": "unpack in a new project directory, then run preparation there",
        },
    )
    print(destination)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", default="outputs/scene_scaling_20260907/ss01-source.tar.gz"
    )
    bundle(p.parse_args().output)
