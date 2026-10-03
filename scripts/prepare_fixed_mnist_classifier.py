"""Freeze the fixed classifier protocol used for MNIST reconstruction accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PROGRAM = Path("scripts/train_fixed_mnist_classifier.py")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-root", default=str(REPO))
    parser.add_argument(
        "--output", default="outputs/coursework_mnist_classifier_local_v2_20260912"
    )
    args = parser.parse_args()
    execution_root = Path(args.execution_root).resolve()
    output_rel = Path(args.output)
    if output_rel.is_absolute():
        raise ValueError("--output must be relative to the repository root")
    output = REPO / output_rel
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source_split = REPO / "manifests/mnist_splits.json"
    frozen_split = output / "mnist_splits.json"
    frozen_split.write_bytes(source_split.read_bytes())
    manifest = {
        "schema_version": 1,
        "status": "frozen_pre_execution",
        "purpose": "fixed downstream digit classifier for evaluating reconstructed MNIST images",
        "root": str(execution_root),
        "program": str(PROGRAM),
        "program_sha256": sha256(REPO / PROGRAM),
        "split_manifest": str(output_rel / "mnist_splits.json"),
        "split_manifest_sha256": sha256(frozen_split),
        "mnist_root": "data/raw/mnist",
        "seed": 20260912,
        "epochs": 5,
        "batch_size": 256,
        "num_workers": 0,
        "learning_rate": 0.001,
        "architecture": "two convolutional blocks and a fixed 128-unit classification head",
        "endpoint_selection": "final epoch only; no hyperparameter search",
        "data_partitions": ["train", "development"],
        "official_test_accessed": False,
        "output": str(output_rel / "training"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": "prepared", "epochs": 5}, indent=2))


if __name__ == "__main__":
    main()
