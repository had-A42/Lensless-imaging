"""Freeze the three-seed MNIST PSF-conditioning diagnostic."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/home/hadhad/project/Lensless-imaging-research-508a878")
LONG_REL = Path("outputs/coursework_long_scaling_v4_20260911")
OUTPUT_REL = Path("outputs/coursework_mnist_psf_conditioning_v2_20260912")
OUTPUT = REPO / OUTPUT_REL
SOURCE_REVISION = "508a878ec55768861270624e311418624f6eefd7"
EVALUATOR = Path("scripts/evaluate_mnist_psf_conditioning.py")
LAUNCHER = Path("scripts/launch_mnist_psf_conditioning.py")
SOURCE_FILES = (
    "src/datasets/on_the_fly.py",
    "src/datasets/mnist.py",
    "src/model/psf_aware_drunet.py",
    "src/model/psf_free_drunet.py",
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
    long_root = REPO / LONG_REL
    entries = []
    for seed in (42, 52, 62):
        name = f"cw-mnist-psf-aware-finite100-50k-seed{seed}-v2"
        run = long_root / "training" / name
        complete = json.loads((run / "job_complete.json").read_text())
        config = run / "config.yaml"
        reference = run / "validation_per_mask_epoch0010.csv"
        if complete["status"] != "complete" or complete["global_step"] != 50000:
            raise ValueError(f"Incomplete source endpoint: {name}")
        entries.append(
            {
                "id": f"mnist-psf-aware-seed{seed}",
                "seed": seed,
                "config": str(REMOTE_ROOT / LONG_REL / "training" / name / "config.yaml"),
                "config_sha256": sha256(config),
                "checkpoint": complete["checkpoint"],
                "checkpoint_sha256": complete["checkpoint_sha256"],
                "reference_per_mask": str(
                    REMOTE_ROOT
                    / LONG_REL
                    / "training"
                    / name
                    / "validation_per_mask_epoch0010.csv"
                ),
                "reference_per_mask_sha256": sha256(reference),
                "output": str(OUTPUT_REL / f"mnist-psf-aware-seed{seed}"),
                "selection_basis": "all retained PSF-aware seeds from the frozen 50k matched matrix",
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "frozen_pre_execution",
        "question": "Does each MNIST PSF-aware endpoint use the corresponding PSF, and is seed62 degradation reproducible?",
        "root": str(REMOTE_ROOT),
        "source_revision": SOURCE_REVISION,
        "source_hashes": {path: revision_sha256(path) for path in SOURCE_FILES},
        "evaluator": str(EVALUATOR),
        "evaluator_sha256": sha256(REPO / EVALUATOR),
        "launcher": str(LAUNCHER),
        "launcher_sha256": sha256(REPO / LAUNCHER),
        "parent_manifest": str(REMOTE_ROOT / LONG_REL / "manifest.json"),
        "parent_manifest_sha256": sha256(long_root / "manifest.json"),
        "data_partition": "development",
        "final_test_accessed": False,
        "conditions": ["Correct PSF", "cyclic next-mask PSF"],
        "grid": {"masks": 32, "scenes_per_mask": 32, "samples": 1024},
        "entries": entries,
        "gpus": [0, 1, 2],
        "automatic_retry": False,
        "checkpoint_substitution": False,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": "prepared", "entries": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
