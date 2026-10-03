"""Prepare the multi-seed DRUNet factorial and MNIST replication extension."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/home/hadhad/project/Lensless-imaging-research-508a878")
SOURCE_REVISION = "508a878ec55768861270624e311418624f6eefd7"
PARENT_REL = Path("outputs/coursework_long_scaling_v4_20260911")
OUTPUT_REL = Path("outputs/coursework_consistency_scaling_v2_20260912")
OUTPUT = REPO / OUTPUT_REL
REMOTE_OUTPUT = REMOTE_ROOT / OUTPUT_REL
SOURCE_FILES = (
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
)
PROGRAM_FILES = (
    "scripts/preflight_consistency_scaling_queue.py",
    "scripts/launch_consistency_scaling_queue.py",
    "scripts/summarize_consistency_scaling_results.py",
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


def save_config(config: dict, name: str) -> tuple[str, str]:
    relative = OUTPUT_REL / "configs" / f"{name}.yaml"
    path = REPO / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), path, resolve=True)
    return str(relative), sha256(path)


def storage(config: dict, name: str, group: str) -> None:
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
            "save_dir": str(REMOTE_OUTPUT / "training"),
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


def drunet_config(condition: str, scenes: int, seed: int) -> dict:
    parent = REPO / PARENT_REL / "configs"
    template = parent / f"cw-dr-scene16384-{condition}-100k-seed42-v1.yaml"
    config = OmegaConf.to_container(OmegaConf.load(template), resolve=True)
    name = f"cw-consistency-dr-scene{scenes}-{condition}-100k-seed{seed}-v1"
    storage(config, name, "coursework-dr-scene-mask-factorial-v1")
    config["trainer"]["seed"] = seed
    loader = config["dataloader_builder"]
    loader["run_seed"] = seed
    loader["train_mask_seed"] = 42
    loader["psf_cache"].update(
        {
            "mode": "read_only",
            "root_dir": str(REMOTE_ROOT / "data/psf_cache"),
            "request_modes": ["finite"],
            "warmup": condition != "streaming",
        }
    )
    config["datasets"]["train"]["splits_path"] = str(
        REMOTE_OUTPUT
        / "manifests"
        / ("scenes16384.json" if scenes == 16384 else "scenes4096.json")
    )
    config["protocol"].update(
        {
            "experiment": "DR-SCENE-MASK-FACTORIAL-02",
            "training_scenes": scenes,
            "fixed_train_mask_bank_seed": 42,
            "repetition_seed": seed,
            "data_partition": "train/development",
            "final_test_accessed": False,
        }
    )
    return config


def mnist_config(regime: str, seed: int) -> dict:
    parent = REPO / PARENT_REL / "configs"
    template = parent / f"cw-mnist-{regime}-finite100-50k-seed42-v2.yaml"
    config = OmegaConf.to_container(OmegaConf.load(template), resolve=True)
    name = f"cw-consistency-mnist-{regime}-finite100-50k-seed{seed}-v1"
    storage(config, name, "coursework-mnist-long-psf-comparison-v3")
    config["trainer"]["seed"] = seed
    loader = config["dataloader_builder"]
    loader["run_seed"] = seed
    loader["train_mask_seed"] = 42
    loader["psf_cache"].update(
        {
            "mode": "read_only",
            "root_dir": str(REMOTE_ROOT / "data/psf_cache"),
            "request_modes": ["finite"],
            "warmup": True,
        }
    )
    config["protocol"].update(
        {
            "experiment": "MNIST-LONG-PSF-COMPARE-03",
            "fixed_train_mask_bank_seed": 42,
            "repetition_seed": seed,
            "data_partition": "train/development",
            "final_test_accessed": False,
        }
    )
    return config


def job(name: str, experiment: str, seed: int, steps: int, config: dict, **fields) -> dict:
    config_path, config_hash = save_config(config, name)
    return {
        "name": name,
        "experiment": experiment,
        "seed": seed,
        "steps": steps,
        "epochs": 10,
        "config": config_path,
        "config_sha256": config_hash,
        "output": str(OUTPUT_REL / "training" / name),
        **fields,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    (OUTPUT / "manifests").mkdir(parents=True)
    for source, target in (
        (REPO / PARENT_REL / "manifests/scenes4096.json", OUTPUT / "manifests/scenes4096.json"),
        (REPO / PARENT_REL / "manifests/scenes16384.json", OUTPUT / "manifests/scenes16384.json"),
    ):
        target.write_bytes(source.read_bytes())
    jobs = []
    for seed in (52, 62):
        for scenes in (4096, 16384):
            for condition in ("finite100", "finite1000", "finite10000", "streaming"):
                config = drunet_config(condition, scenes, seed)
                name = config["writer"]["run_name"]
                jobs.append(
                    job(
                        name,
                        "drunet_scene_mask_factorial",
                        seed,
                        100000,
                        config,
                        training_scenes=scenes,
                        condition=condition,
                    )
                )
    for seed in (72, 82):
        for regime in ("psf-free", "psf-aware"):
            config = mnist_config(regime, seed)
            name = config["writer"]["run_name"]
            jobs.append(
                job(
                    name,
                    "mnist_replication_extension",
                    seed,
                    50000,
                    config,
                    information_regime=regime.replace("-", "_"),
                    condition="finite100",
                )
            )
    lanes = [
        {
            "lane_id": 0,
            "preferred_gpu": 0,
            "jobs": [
                "cw-consistency-mnist-psf-free-finite100-50k-seed72-v1",
                "cw-consistency-dr-scene4096-finite100-100k-seed52-v1",
                "cw-consistency-dr-scene16384-finite100-100k-seed52-v1",
                "cw-consistency-dr-scene4096-finite1000-100k-seed52-v1",
                "cw-consistency-dr-scene16384-finite1000-100k-seed52-v1",
            ],
        },
        {
            "lane_id": 1,
            "preferred_gpu": 1,
            "jobs": [
                "cw-consistency-mnist-psf-aware-finite100-50k-seed72-v1",
                "cw-consistency-dr-scene4096-finite10000-100k-seed52-v1",
                "cw-consistency-dr-scene16384-finite10000-100k-seed52-v1",
                "cw-consistency-dr-scene4096-streaming-100k-seed52-v1",
                "cw-consistency-dr-scene16384-streaming-100k-seed52-v1",
            ],
        },
        {
            "lane_id": 2,
            "preferred_gpu": 2,
            "jobs": [
                "cw-consistency-mnist-psf-free-finite100-50k-seed82-v1",
                "cw-consistency-dr-scene4096-finite100-100k-seed62-v1",
                "cw-consistency-dr-scene16384-finite100-100k-seed62-v1",
                "cw-consistency-dr-scene4096-finite1000-100k-seed62-v1",
                "cw-consistency-dr-scene16384-finite1000-100k-seed62-v1",
            ],
        },
        {
            "lane_id": 3,
            "preferred_gpu": 3,
            "jobs": [
                "cw-consistency-mnist-psf-aware-finite100-50k-seed82-v1",
                "cw-consistency-dr-scene4096-finite10000-100k-seed62-v1",
                "cw-consistency-dr-scene16384-finite10000-100k-seed62-v1",
                "cw-consistency-dr-scene4096-streaming-100k-seed62-v1",
                "cw-consistency-dr-scene16384-streaming-100k-seed62-v1",
            ],
        },
    ]
    names = [item["name"] for item in jobs]
    assigned = [name for lane in lanes for name in lane["jobs"]]
    if sorted(names) != sorted(assigned) or len(assigned) != len(set(assigned)):
        raise ValueError("Lane mapping is not an exact job partition")
    manifest = {
        "schema_version": 1,
        "status": "frozen_pre_execution",
        "root": str(REMOTE_ROOT),
        "bundle": str(OUTPUT_REL),
        "source_revision": SOURCE_REVISION,
        "source_hashes": {path: revision_sha256(path) for path in SOURCE_FILES},
        "program_hashes": {path: sha256(REPO / path) for path in PROGRAM_FILES},
        "parent_manifest": str(REMOTE_ROOT / PARENT_REL / "manifest.json"),
        "parent_manifest_sha256": sha256(REPO / PARENT_REL / "manifest.json"),
        "scene_manifests": {
            "4096": {
                "path": str(OUTPUT_REL / "manifests/scenes4096.json"),
                "sha256": sha256(OUTPUT / "manifests/scenes4096.json"),
            },
            "16384": {
                "path": str(OUTPUT_REL / "manifests/scenes16384.json"),
                "sha256": sha256(OUTPUT / "manifests/scenes16384.json"),
            },
        },
        "drunet_protocol": {
            "seeds": [42, 52, 62],
            "new_seeds": [52, 62],
            "training_scenes": [4096, 16384],
            "mask_regimes": ["finite100", "finite1000", "finite10000", "streaming"],
            "steps": 100000,
            "fixed_train_mask_bank_seed": 42,
            "primary_effect": "16384 minus 4096 scenes paired within mask regime and repetition seed",
        },
        "mnist_protocol": {
            "existing_seeds": [42, 52, 62],
            "new_seeds": [72, 82],
            "information_regimes": ["psf_free", "psf_aware"],
            "steps": 50000,
            "fixed_train_mask_bank_seed": 42,
            "primary_effect": "PSF-aware minus PSF-free paired within repetition seed",
        },
        "jobs": jobs,
        "lanes": lanes,
        "allowed_gpu_count": 4,
        "gpu_conflict_policy": "never start a lane on an occupied GPU; wait or remap only before execution",
        "data_partitions": ["train", "development"],
        "final_test_accessed": False,
        "automatic_retry": False,
        "checkpoint_substitution": False,
        "endpoint_selection": "fixed final step only",
        "total_new_optimizer_steps": sum(item["steps"] for item in jobs),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "jobs": len(jobs),
                "optimizer_steps": manifest["total_new_optimizer_steps"],
                "lanes": len(lanes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
