"""Freeze the mandatory supervisor PSF-aware DRUNet queue."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/home/hadhad/project/Lensless-imaging-research-508a878")
BUNDLE_REL = Path("outputs/coursework_supervisor_psf_aware_v1_20260914")
BUNDLE = REPO / BUNDLE_REL
SOURCE_REVISION = "508a878ec55768861270624e311418624f6eefd7"

SOURCE_PATHS = (
    "requirements.txt",
    "train.py",
    "src/datasets/__init__.py",
    "src/datasets/base_dataset.py",
    "src/datasets/data_utils.py",
    "src/datasets/mirflickr.py",
    "src/datasets/on_the_fly.py",
    "src/digicam_synth/mask_protocol.py",
    "src/digicam_synth/pipeline.py",
    "src/digicam_synth/psf_cache.py",
    "src/logger/__init__.py",
    "src/logger/logger.py",
    "src/logger/wandb.py",
    "src/loss/__init__.py",
    "src/loss/reconstruction.py",
    "src/metrics/__init__.py",
    "src/metrics/base_metric.py",
    "src/metrics/reconstruction.py",
    "src/metrics/tracker.py",
    "src/model/__init__.py",
    "src/model/psf_aware_drunet.py",
    "src/model/psf_free_drunet.py",
    "src/trainer/__init__.py",
    "src/trainer/base_trainer.py",
    "src/trainer/trainer.py",
    "src/utils/__init__.py",
    "src/utils/init_utils.py",
    "src/utils/io_utils.py",
)

PROGRAM_PATHS = (
    "scripts/prepare_supervisor_psf_aware.py",
    "scripts/preflight_supervisor_psf_aware.py",
    "scripts/launch_supervisor_psf_aware.py",
    "scripts/evaluate_mirflickr_psf_conditioning.py",
    "scripts/summarize_supervisor_psf_aware.py",
)

CONTROL_RUNS = {
    42: {
        "run_relative": Path(
            "outputs/coursework_long_scaling_v4_20260911/training/"
            "cw-dr-scene16384-finite100-100k-seed42-v1"
        ),
        "config_relative": Path(
            "outputs/coursework_long_scaling_v4_20260911/configs/"
            "cw-dr-scene16384-finite100-100k-seed42-v1.yaml"
        ),
        "checkpoint_sha256": "2edf54e9e6acba5d821490de5474efb2d378afa5f54983c733215b2b127c3a02",
        "checkpoint_size": 98083613,
        "config_sha256": "21b45e73d4137db436696e4909f19f05578fb9765207712d4f5016cd1b63c733",
    },
    52: {
        "run_relative": Path(
            "outputs/coursework_consistency_scaling_v2_20260912/training/"
            "cw-consistency-dr-scene16384-finite100-100k-seed52-v1"
        ),
        "config_relative": Path(
            "outputs/coursework_consistency_scaling_v2_20260912/configs/"
            "cw-consistency-dr-scene16384-finite100-100k-seed52-v1.yaml"
        ),
        "checkpoint_sha256": "36046957f9d92c85bdcd5a5287aacc5a4e4287607895337a385ddfeb4bbd07c9",
        "checkpoint_size": 98083933,
        "config_sha256": "29122f379e59fe65d24df001fa78c615d2b81b91040733722363ceae25517d29",
    },
    62: {
        "run_relative": Path(
            "outputs/coursework_consistency_scaling_v2_20260912/training/"
            "cw-consistency-dr-scene16384-finite100-100k-seed62-v1"
        ),
        "config_relative": Path(
            "outputs/coursework_consistency_scaling_v2_20260912/configs/"
            "cw-consistency-dr-scene16384-finite100-100k-seed62-v1.yaml"
        ),
        "checkpoint_sha256": "73222fbaa23716f9937ae5bebfab8d1e0f9c7099d474d584a5803698e9b77c77",
        "checkpoint_size": 98083997,
        "config_sha256": "a22194991a0c3448b940b13c1a0f3458a7009424c805f0608154606b781a1864",
    },
}


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


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build_config(seed: int, source: Path) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    name = f"supervisor-mir-psf-aware-16k-finite100-100k-seed{seed}-v1"
    if config["trainer"]["seed"] != seed:
        raise ValueError(f"Control seed mismatch: {source}")

    config["model"]["_target_"] = "src.model.psf_aware_drunet.PSFAwareDRUNet"
    config["dataloader_builder"]["return_psf"] = True
    config["trainer"]["device_tensors"] = ["measurement", "psf", "target"]
    config["trainer"]["save_dir"] = str(REMOTE_ROOT / BUNDLE_REL / "training")

    writer = config["writer"]
    writer.pop("run_id", None)
    writer["run_name"] = name
    writer["group"] = "coursework-supervisor-psf-aware-v1"
    writer["job_type"] = "matched-psf-aware"
    writer["tags"] = [
        "synthetic",
        "psf-aware",
        "mirflickr",
        "16k-scenes",
        "finite100",
        "100k",
        "supervisor-required-v1",
    ]

    protocol = config["protocol"]
    protocol["experiment"] = "SUPERVISOR-PSF-AWARE-01"
    protocol["queue"] = "coursework-supervisor-psf-aware-v1"
    protocol["information_regime"] = "measurement + corresponding PSF"
    protocol["comparison"] = (
        "matched against the registered 16k/finite100/100k PSF-free endpoint "
        "within the same seed"
    )
    protocol["data_partition"] = "train/development"
    protocol["final_test_accessed"] = False
    return config


def main() -> None:
    if BUNDLE.exists():
        raise FileExistsError(BUNDLE)
    for relative in PROGRAM_PATHS:
        if not (REPO / relative).is_file():
            raise FileNotFoundError(REPO / relative)

    configs_dir = BUNDLE / "configs"
    configs_dir.mkdir(parents=True)
    jobs = []
    controls = {}
    for gpu, seed in enumerate((42, 52, 62)):
        record = CONTROL_RUNS[seed]
        local_run = REPO / record["run_relative"]
        remote_run = REMOTE_ROOT / record["run_relative"]
        control_config = REPO / record["config_relative"]
        remote_config = REMOTE_ROOT / record["config_relative"]
        control_metrics = local_run / "validation_per_mask_epoch0010.csv"
        control_complete = local_run / "job_complete.json"
        for path in (control_config, control_metrics, control_complete):
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256(control_config) != record["config_sha256"]:
            raise ValueError(f"Control config hash drift for seed {seed}")

        name = f"supervisor-mir-psf-aware-16k-finite100-100k-seed{seed}-v1"
        config_path = configs_dir / f"{name}.yaml"
        OmegaConf.save(
            OmegaConf.create(build_config(seed, control_config)),
            config_path,
            resolve=True,
        )
        controls[str(seed)] = {
            "name": Path(record["run_relative"]).name,
            "config": str(remote_config),
            "config_sha256": record["config_sha256"],
            "checkpoint": str(remote_run / "checkpoint-epoch10.pth"),
            "checkpoint_sha256": record["checkpoint_sha256"],
            "checkpoint_size": record["checkpoint_size"],
            "metrics_csv": str(remote_run / "validation_per_mask_epoch0010.csv"),
            "metrics_sha256": sha256(control_metrics),
            "job_complete": str(remote_run / "job_complete.json"),
            "job_complete_sha256": sha256(control_complete),
            "global_step": 100000,
            "sampler_step": 100000,
            "source_revision": SOURCE_REVISION,
        }
        jobs.append(
            {
                "name": name,
                "seed": seed,
                "gpu": gpu,
                "steps": 100000,
                "epochs": 10,
                "config": str(BUNDLE_REL / "configs" / config_path.name),
                "config_sha256": sha256(config_path),
                "control_seed": seed,
                "output": str(BUNDLE_REL / "training" / name),
                "evaluation_output": str(BUNDLE_REL / "evaluation" / name),
            }
        )

    scene_manifest = (
        REPO
        / "outputs/coursework_long_scaling_v4_20260911/manifests/scenes16384.json"
    )
    if not scene_manifest.is_file():
        raise FileNotFoundError(scene_manifest)
    scene_hash = sha256(scene_manifest)
    expected_scene_hash = (
        "c45e9ec1ddd06c0dbb4a246e7fa1896b5cd6a61b2a4182fd68362d03f1e42442"
    )
    if scene_hash != expected_scene_hash:
        raise ValueError("16k scene manifest hash drift")

    manifest = {
        "schema_version": 1,
        "status": "frozen_pre_execution",
        "question": (
            "Does access to the corresponding PSF improve matched DRUNet "
            "reconstruction on unseen MIRFLICKR masks?"
        ),
        "root": str(REMOTE_ROOT),
        "bundle": str(BUNDLE_REL),
        "source_revision": SOURCE_REVISION,
        "source_hashes": {path: revision_sha256(path) for path in SOURCE_PATHS},
        "program_hashes": {path: sha256(REPO / path) for path in PROGRAM_PATHS},
        "scene_manifest": {
            "path": str(
                REMOTE_ROOT
                / "outputs/coursework_long_scaling_v4_20260911/manifests/"
                "scenes16384.json"
            ),
            "sha256": scene_hash,
            "training_scene_count": 16384,
        },
        "data_partitions": ["train", "development"],
        "final_test_accessed_by_this_queue": False,
        "uses_final_test_for_selection": False,
        "protocol": {
            "architecture": "PSFAwareDRUNet concat conditioning",
            "training_scenes": 16384,
            "finite_mask_count": 100,
            "train_mask_seed": 42,
            "evaluation_mask_seed": 42,
            "steps": 100000,
            "batch_size": 4,
            "precision": "FP32",
            "optimizer": "Adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "scheduler": "CosineAnnealingLR",
            "eta_min": 0.000001,
            "loss": "independently peak-normalized MSE + LPIPS-VGG",
            "development_grid": {
                "masks": 32,
                "scenes_per_mask": 32,
                "pairs": 1024,
            },
            "primary_endpoint": "final step only",
        },
        "controls": controls,
        "jobs": jobs,
        "evaluation": {
            "conditions": ["Measurement only", "Correct PSF", "Shuffled PSF"],
            "shuffle": "cyclic next development mask; measurement and target fixed",
            "metrics": ["PSNR", "SSIM", "LPIPS"],
            "qualitative_sample_indices": [0, 341, 682, 1023],
            "correct_replay_tolerance": {
                "PSNR": 0.00005,
                "SSIM": 0.000005,
                "LPIPS": 0.000005,
            },
        },
        "minimum_free_disk_bytes": 3221225472,
        "automatic_retry": False,
        "checkpoint_substitution": False,
        "stopping_rule": (
            "all three fixed endpoints and all correct/shuffled replays must finish; "
            "no checkpoint selection or silent recipe change"
        ),
    }
    save_json(BUNDLE / "manifest.json", manifest)
    save_json(
        BUNDLE / "preparation.json",
        {
            "status": "prepared",
            "manifest": str(BUNDLE / "manifest.json"),
            "manifest_sha256": sha256(BUNDLE / "manifest.json"),
            "job_count": len(jobs),
            "optimizer_steps": sum(job["steps"] for job in jobs),
            "data_partitions": ["train", "development"],
            "final_test_accessed": False,
        },
    )
    print((BUNDLE / "preparation.json").read_text(), end="")


if __name__ == "__main__":
    main()
