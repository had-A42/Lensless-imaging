"""Prepare the frozen DRUNet scene-scaling and MNIST PSF-aware queue."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "508a878ec55768861270624e311418624f6eefd7"
REMOTE_ROOT = Path("/home/hadhad/project/Lensless-imaging-research-508a878")
BUNDLE_REL = Path("outputs/coursework_long_scaling_v4_20260911")
BUNDLE = REPO / BUNDLE_REL
REMOTE_BUNDLE = REMOTE_ROOT / BUNDLE_REL

SOURCE_PATHS = (
    "train.py",
    "src/datasets/data_utils.py",
    "src/datasets/on_the_fly.py",
    "src/datasets/mirflickr.py",
    "src/datasets/mnist.py",
    "src/digicam_synth/mask_protocol.py",
    "src/digicam_synth/psf_cache.py",
    "src/model/psf_free_drunet.py",
    "src/model/psf_aware_drunet.py",
    "src/loss/reconstruction.py",
    "src/metrics/reconstruction.py",
    "src/trainer/base_trainer.py",
    "src/trainer/trainer.py",
    "src/utils/init_utils.py",
)
PROGRAM_PATHS = (
    "scripts/preflight_long_scaling_queue.py",
    "scripts/launch_long_scaling_queue.py",
    "scripts/summarize_long_scaling_results.py",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def revision_file(path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{SOURCE_REVISION}:{path}"], cwd=REPO
    )


def save_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_resolved(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def common_storage(config: dict, name: str, group: str) -> None:
    config["writer"].update(
        {
            "run_name": name,
            "mode": "offline",
            "group": group,
            "job_type": name,
            "log_checkpoints": False,
            "log_best_checkpoint_at_end": False,
            "log_images": False,
            "save_code": False,
            "log_git_state": False,
        }
    )
    config["writer"].pop("run_id", None)
    config["trainer"].update(
        {
            "device": "cuda",
            "save_dir": str(REMOTE_BUNDLE / "training"),
            "monitor": "off",
            "early_stop": -1,
            "save_period": 1,
            "keep_last_checkpoints": 2,
            "save_initial_checkpoint": True,
            "resume_from": None,
            "from_pretrained": None,
            "override": False,
            "skip_oom": False,
        }
    )


def set_remote_data_paths(config: dict) -> None:
    for dataset in config["datasets"].values():
        if "root_dir" in dataset:
            relative = Path(dataset["root_dir"])
            dataset["root_dir"] = str(
                relative if relative.is_absolute() else REMOTE_ROOT / relative
            )
        if "splits_path" in dataset:
            relative = Path(dataset["splits_path"])
            dataset["splits_path"] = str(
                relative if relative.is_absolute() else REMOTE_ROOT / relative
            )
    cache = config["dataloader_builder"].get("psf_cache")
    if cache is not None:
        cache["root_dir"] = str(REMOTE_ROOT / "data/psf_cache")
        cache["mode"] = "read_write"


def drunet_config(condition: str, scenes: int) -> tuple[dict, Path]:
    if condition == "finite100":
        source = REPO / "saved/cv1-drunet-small-finite-100-100000step-seed42-508a878/config.yaml"
        mode, count, label = "finite", 100, "finite-100"
    elif condition in {"finite1000", "finite10000"}:
        source = REPO / "saved/cv1-drunet-small-finite-1000-100000step-seed42-508a878/config.yaml"
        mode = "finite"
        count = 1000 if condition == "finite1000" else 10000
        label = f"finite-{count}"
    elif condition == "streaming":
        source = REPO / "saved/cv1-drunet-small-infinite-100000step-seed42-508a878/config.yaml"
        mode, count, label = "infinite", None, "infinite"
    else:
        raise ValueError(condition)
    config = load_resolved(source)
    name = f"cw-dr-scene{scenes}-{condition}-100k-seed42-v1"
    common_storage(config, name, "coursework-dr16k-mask-scaling")
    set_remote_data_paths(config)
    config["condition_name"] = label
    config["mask_mode"] = mode
    config["mask_count"] = count
    config["scale_condition"] = {
        "name": label,
        "mask_mode": mode,
        "mask_count": count,
    }
    loader = config["dataloader_builder"]
    loader["train_mode"] = mode
    loader["finite_mask_count"] = count
    loader["train_steps"] = 100000
    loader["run_seed"] = 42
    loader["train_mask_seed"] = 42
    loader["evaluation_mask_seed"] = 42
    loader["validation_seed"] = 52
    loader["psf_cache"]["request_modes"] = ["finite"]
    loader["psf_cache"]["warmup"] = mode == "finite"
    config["trainer"].update(
        {
            "seed": 42,
            "n_epochs": 10,
            "epoch_len": 10000,
            "total_steps": 100000,
            "evaluation_period": 1,
        }
    )
    config["lr_scheduler"]["T_max"] = 100000
    config["datasets"]["train"]["splits_path"] = str(
        REMOTE_BUNDLE / "manifests" / (
            "scenes16384.json" if scenes == 16384 else "scenes4096.json"
        )
    )
    config["protocol"]["experiment"] = "DR16K-MASK-SCALE-01"
    config["protocol"]["training_scenes"] = scenes
    config["protocol"]["data_partition"] = "train/development"
    config["protocol"]["final_test_accessed"] = False
    return config, source


def mnist_config(seed: int, psf_aware: bool) -> tuple[dict, Path]:
    source = REPO / "saved/a800-mnist-finite100-dice-50k-seed42/config.yaml"
    config = load_resolved(source)
    regime = "psf-aware" if psf_aware else "psf-free"
    name = f"cw-mnist-{regime}-finite100-50k-seed{seed}-v2"
    common_storage(config, name, "coursework-mnist-long-psf-comparison-v2")
    set_remote_data_paths(config)
    config["writer"]["tags"] = [
        "synthetic",
        regime,
        "mnist",
        "50k",
        "matched-v2",
    ]
    config["trainer"].update(
        {
            "seed": seed,
            "device_tensors": (
                ["measurement", "psf", "target"]
                if psf_aware
                else ["measurement", "target"]
            ),
            "n_epochs": 10,
            "epoch_len": 5000,
            "total_steps": 50000,
            "evaluation_period": 1,
        }
    )
    config["lr_scheduler"]["T_max"] = 50000
    loader = config["dataloader_builder"]
    loader["train_steps"] = 50000
    loader["run_seed"] = seed
    loader["train_mask_seed"] = 42
    loader["evaluation_mask_seed"] = 52
    loader["validation_seed"] = 52
    loader["return_psf"] = psf_aware
    loader["psf_cache"]["request_modes"] = ["finite"]
    loader["psf_cache"]["warmup"] = True
    if psf_aware:
        config["model"]["_target_"] = "src.model.psf_aware_drunet.PSFAwareDRUNet"
    else:
        config["model"]["_target_"] = "src.model.psf_free_drunet.PSFFreeDRUNet"
    config["protocol"]["experiment"] = "MNIST-LONG-PSF-COMPARE-02"
    config["protocol"]["information_regime"] = (
        "measurement + corresponding PSF" if psf_aware else "measurement only"
    )
    config["protocol"]["comparison"] = (
        "paired 10k-to-50k learning curves and matched PSF-free/PSF-aware seeds42/52/62"
    )
    config["protocol"]["data_partition"] = "train/development"
    config["protocol"]["final_test_accessed"] = False
    return config, source


def write_config(name: str, config: dict) -> tuple[str, str]:
    relative = BUNDLE_REL / "configs" / f"{name}.yaml"
    path = REPO / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), path, resolve=True)
    return str(relative), sha256(path)


def control_record(condition: str) -> dict | None:
    names = {
        "finite100": "cv1-drunet-small-finite-100-100000step-seed42-508a878",
    }
    if condition not in names:
        return None
    name = names[condition]
    local = REPO / "saved" / name
    checkpoint = local / "checkpoint-epoch10.pth"
    metrics = local / "validation_per_mask_epoch0010.csv"
    config = local / "config.yaml"
    for path in (checkpoint, metrics, config):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "training_scenes": 4096,
        "checkpoint": str(
            Path("/home/hadhad/project/Lensless-imaging-research-508a878/saved")
            / name
            / checkpoint.name
        ),
        "checkpoint_sha256": sha256(checkpoint),
        "metrics_csv": str(
            Path("/home/hadhad/project/Lensless-imaging-research-508a878/saved")
            / name
            / metrics.name
        ),
        "metrics_sha256": sha256(metrics),
        "config": str(
            Path("/home/hadhad/project/Lensless-imaging-research-508a878/saved")
            / name
            / config.name
        ),
        "config_sha256": sha256(config),
        "source_revision": SOURCE_REVISION,
        "endpoint_steps": 100000,
    }


def main() -> None:
    if BUNDLE.exists():
        raise FileExistsError(BUNDLE)
    (BUNDLE / "manifests").mkdir(parents=True)
    original_manifest = REPO / "manifests/mirflickr25k_splits.json"
    expanded_manifest = REPO / "outputs/scene_scaling_20260907/manifests/scenes16384.json"
    (BUNDLE / "manifests/scenes4096.json").write_bytes(original_manifest.read_bytes())
    (BUNDLE / "manifests/scenes16384.json").write_bytes(expanded_manifest.read_bytes())

    jobs = []
    references = {}
    for condition in ("finite100", "finite1000", "finite10000", "streaming"):
        config, source = drunet_config(condition, 16384)
        name = config["writer"]["run_name"]
        config_file, config_sha = write_config(name, config)
        references[condition] = control_record(condition)
        jobs.append(
            {
                "name": name,
                "experiment": "drunet_scene_scaling",
                "condition": condition,
                "training_scenes": 16384,
                "seed": 42,
                "steps": 100000,
                "epochs": 10,
                "config": config_file,
                "config_sha256": config_sha,
                "source_config": str(source.relative_to(REPO)),
                "source_config_sha256": sha256(source),
                "output": str(BUNDLE_REL / "training" / name),
            }
        )
    # Only finite100 remains on A800. Train fresh 4096-scene controls for
    # every other regime before comparing them with the 16384-scene arms.
    for condition, record in references.items():
        if record is not None:
            continue
        control_config, source = drunet_config(condition, 4096)
        control_name = control_config["writer"]["run_name"]
        config_file, config_sha = write_config(control_name, control_config)
        jobs.append(
            {
                "name": control_name,
                "experiment": "drunet_scene_scaling",
                "condition": condition,
                "training_scenes": 4096,
                "seed": 42,
                "steps": 100000,
                "epochs": 10,
                "config": config_file,
                "config_sha256": config_sha,
                "source_config": str(source.relative_to(REPO)),
                "source_config_sha256": sha256(source),
                "output": str(BUNDLE_REL / "training" / control_name),
            }
        )
    for seed in (42, 52, 62):
        for aware in (False, True):
            config, source = mnist_config(seed, aware)
            name = config["writer"]["run_name"]
            config_file, config_sha = write_config(name, config)
            jobs.append(
                {
                    "name": name,
                    "experiment": "mnist_long_psf_comparison",
                    "information_regime": "psf_aware" if aware else "psf_free",
                    "condition": "finite100",
                    "seed": seed,
                    "steps": 50000,
                    "epochs": 10,
                    "config": config_file,
                    "config_sha256": config_sha,
                    "source_config": str(source.relative_to(REPO)),
                    "source_config_sha256": sha256(source),
                    "output": str(BUNDLE_REL / "training" / name),
                }
            )

    by_name = {job["name"]: job for job in jobs}
    lanes = [
        {
            "lane_id": 0,
            "physical_gpu": 0,
            "jobs": [
                "cw-dr-scene16384-finite100-100k-seed42-v1",
                "cw-mnist-psf-free-finite100-50k-seed42-v2",
                "cw-mnist-psf-aware-finite100-50k-seed42-v2",
            ],
        },
        {
            "lane_id": 1,
            "physical_gpu": 1,
            "jobs": [
                "cw-dr-scene4096-finite1000-100k-seed42-v1",
                "cw-dr-scene16384-finite1000-100k-seed42-v1",
                "cw-mnist-psf-free-finite100-50k-seed52-v2",
                "cw-mnist-psf-aware-finite100-50k-seed52-v2",
            ],
        },
        {
            "lane_id": 2,
            "physical_gpu": 2,
            "jobs": [
                "cw-dr-scene4096-streaming-100k-seed42-v1",
                "cw-dr-scene16384-streaming-100k-seed42-v1",
                "cw-mnist-psf-free-finite100-50k-seed62-v2",
                "cw-mnist-psf-aware-finite100-50k-seed62-v2",
            ],
        },
        {
            "lane_id": 3,
            "physical_gpu": 3,
            "jobs": [
                "cw-dr-scene4096-finite10000-100k-seed42-v1",
                "cw-dr-scene16384-finite10000-100k-seed42-v1",
            ],
        },
    ]
    assigned = [name for lane in lanes for name in lane["jobs"]]
    if set(assigned) != set(by_name) or len(assigned) != len(set(assigned)):
        raise ValueError("Lane assignment is not an exact job partition")
    manifest = {
        "schema_version": 1,
        "status": "frozen_pre_execution",
        "prepared_after_final_test": True,
        "uses_final_test_for_selection": False,
        "data_partitions": ["train", "development"],
        "final_test_accessed_by_this_queue": False,
        "root": str(REMOTE_ROOT),
        "bundle": str(BUNDLE_REL),
        "source_revision": SOURCE_REVISION,
        "source_hashes": {
            path: bytes_sha256(revision_file(path)) for path in SOURCE_PATHS
        },
        "program_hashes": {path: sha256(REPO / path) for path in PROGRAM_PATHS},
        "scene_manifests": {
            "4096": {
                "path": str(BUNDLE_REL / "manifests/scenes4096.json"),
                "sha256": sha256(BUNDLE / "manifests/scenes4096.json"),
            },
            "16384": {
                "path": str(BUNDLE_REL / "manifests/scenes16384.json"),
                "sha256": sha256(BUNDLE / "manifests/scenes16384.json"),
            },
        },
        "drunet_protocol": {
            "primary_effects": "scenes16384 minus scenes4096 within each mask regime",
            "mask_regimes": ["finite100", "finite1000", "finite10000", "streaming"],
            "seed": 42,
            "steps": 100000,
            "historical_controls": references,
            "missing_control_policy": "train fresh 4096-scene controls before the corresponding 16384-scene arms",
            "fresh_control_conditions": [
                condition for condition, record in references.items() if record is None
            ],
            "remote_inventory_audit": {
                "finite100": "present",
                "finite1000": "missing",
                "finite10000": "missing",
                "streaming": "missing"
            },
        },
        "mnist_protocol": {
            "primary_effect": "PSF-aware minus PSF-free at 50000 steps, paired within seed",
            "secondary_effect": "50000 minus 10000 steps within the same run",
            "seeds": [42, 52, 62],
            "mask_regime": "finite100",
            "steps": 50000,
            "comparison_endpoints": [10000, 50000],
            "shared_initialization_check_required": True,
        },
        "jobs": jobs,
        "lanes": lanes,
        "stopping_rule": "each job runs once to its fixed endpoint; no validation selection, automatic retry, or checkpoint substitution",
        "estimated_wall_time_hours": 7,
    }
    save_json(BUNDLE / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "prepared",
                "jobs": len(jobs),
                "lanes": len(lanes),
                "optimizer_steps": sum(job["steps"] for job in jobs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
