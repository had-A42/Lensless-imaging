"""Freeze fixed-classifier evaluation of the six current MNIST endpoints."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/home/hadhad/project/Lensless-imaging-research-508a878")
LONG_REL = Path("outputs/coursework_long_scaling_v4_20260911")
CLASSIFIER = REPO / "outputs/coursework_mnist_classifier_local_v2_20260912/training/classifier_final.pth"
OUTPUT_REL = Path("outputs/coursework_mnist_reconstruction_accuracy_v3_20260912")
OUTPUT = REPO / OUTPUT_REL
EVALUATOR = Path("scripts/evaluate_mnist_reconstruction_accuracy.py")
LAUNCHER = Path("scripts/launch_mnist_reconstruction_accuracy.py")
SOURCE_REVISION = "508a878ec55768861270624e311418624f6eefd7"
SOURCE_FILES = (
    "src/datasets/on_the_fly.py",
    "src/datasets/mnist.py",
    "src/model/psf_free_drunet.py",
    "src/model/psf_aware_drunet.py",
    "src/metrics/reconstruction.py",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revision_sha256(path: str) -> str:
    value = subprocess.check_output(
        ["git", "show", f"{SOURCE_REVISION}:{path}"], cwd=REPO
    )
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    classifier_copy = OUTPUT / "classifier_final.pth"
    classifier_copy.write_bytes(CLASSIFIER.read_bytes())
    entries = []
    long = REPO / LONG_REL
    for seed in (42, 52, 62):
        for regime in ("psf_free", "psf_aware"):
            token = regime.replace("_", "-")
            name = f"cw-mnist-{token}-finite100-50k-seed{seed}-v2"
            run = long / "training" / name
            complete = json.loads((run / "job_complete.json").read_text())
            config = run / "config.yaml"
            entries.append(
                {
                    "id": f"mnist-{token}-seed{seed}",
                    "seed": seed,
                    "information_regime": regime,
                    "config": str(REMOTE_ROOT / LONG_REL / "training" / name / "config.yaml"),
                    "config_sha256": sha256(config),
                    "checkpoint": complete["checkpoint"],
                    "checkpoint_sha256": complete["checkpoint_sha256"],
                    "output": str(OUTPUT_REL / f"mnist-{token}-seed{seed}"),
                }
            )
    split_path = REPO / "manifests/mnist_splits.json"
    split = json.loads(split_path.read_text())
    validation_ids = split["splits"]["validation"]
    scene_ids = [
        scene_id
        for digit in range(10)
        for scene_id in validation_ids[digit * 500 : digit * 500 + 10]
    ]
    manifest = {
        "schema_version": 1,
        "status": "frozen_pre_execution",
        "root": str(REMOTE_ROOT),
        "source_revision": SOURCE_REVISION,
        "source_hashes": {path: revision_sha256(path) for path in SOURCE_FILES},
        "evaluator": str(EVALUATOR),
        "evaluator_sha256": sha256(REPO / EVALUATOR),
        "launcher": str(LAUNCHER),
        "launcher_sha256": sha256(REPO / LAUNCHER),
        "classifier_checkpoint": str(OUTPUT_REL / "classifier_final.pth"),
        "classifier_checkpoint_sha256": sha256(classifier_copy),
        "classifier_development_accuracy": 0.9902,
        "classifier_endpoint": "fixed epoch 5",
        "scene_manifest_sha256": sha256(split_path),
        "scene_selection": "first ten manifest validation IDs from each class block",
        "scene_ids": scene_ids,
        "entries": entries,
        "grid": {
            "masks": 32,
            "scenes_per_mask": 100,
            "scenes_per_digit": 10,
            "samples": 3200
        },
        "data_partition": "development",
        "official_test_accessed": False,
        "final_test_accessed": False,
        "automatic_retry": False,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": "prepared", "entries": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
