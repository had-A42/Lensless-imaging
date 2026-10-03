"""Fail-closed metadata and one-batch GPU preflight for the long queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scientific_signature(config: dict) -> dict:
    value = json.loads(json.dumps(config))
    for dataset in value["datasets"].values():
        dataset.pop("root_dir", None)
        dataset.pop("splits_path", None)
    cache = value["dataloader_builder"].get("psf_cache", {})
    cache.pop("root_dir", None)
    cache.pop("warmup", None)
    for key in (
        "run_name",
        "run_id",
        "group",
        "job_type",
        "tags",
        "mode",
    ):
        value["writer"].pop(key, None)
    for key in (
        "save_dir",
        "monitor",
        "early_stop",
        "save_period",
        "keep_last_checkpoints",
        "save_initial_checkpoint",
        "resume_from",
        "from_pretrained",
        "override",
        "evaluation_period",
    ):
        value["trainer"].pop(key, None)
    value["protocol"].pop("training_scenes", None)
    value["protocol"].pop("information_regime", None)
    value["protocol"].pop("comparison", None)
    value["protocol"].pop("data_partition", None)
    value["protocol"].pop("final_test_accessed", None)
    value["protocol"].pop("experiment", None)
    return value


def validate_metadata(manifest_path: Path) -> tuple[dict, Path]:
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    if manifest["data_partitions"] != ["train", "development"]:
        raise ValueError("Only train/development partitions are allowed")
    if manifest["final_test_accessed_by_this_queue"] is not False:
        raise ValueError("Final-test access must be false")
    if manifest["uses_final_test_for_selection"] is not False:
        raise ValueError("Final results cannot select these jobs")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if revision != manifest["source_revision"]:
        raise ValueError(f"Source revision drift: {revision}")
    for relative, expected in manifest["source_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Source hash drift: {relative}")
    for relative, expected in manifest["program_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Program hash drift: {relative}")
    for record in manifest["scene_manifests"].values():
        if sha256(root / record["path"]) != record["sha256"]:
            raise ValueError(f"Scene manifest drift: {record['path']}")
    for condition, control in manifest["drunet_protocol"][
        "historical_controls"
    ].items():
        if control is None:
            if condition not in manifest["drunet_protocol"]["fresh_control_conditions"]:
                raise ValueError(f"Unexpected missing control: {condition}")
            continue
        for field, hash_field in (
            ("checkpoint", "checkpoint_sha256"),
            ("metrics_csv", "metrics_sha256"),
            ("config", "config_sha256"),
        ):
            if sha256(control[field]) != control[hash_field]:
                raise ValueError(f"Historical control drift: {condition}/{field}")

    jobs = manifest["jobs"]
    assigned = [name for lane in manifest["lanes"] for name in lane["jobs"]]
    names = [job["name"] for job in jobs]
    if sorted(assigned) != sorted(names) or len(assigned) != len(set(assigned)):
        raise ValueError("GPU lanes are not an exact partition")
    configs = {}
    for job in jobs:
        config_path = root / job["config"]
        if sha256(config_path) != job["config_sha256"]:
            raise ValueError(f"Config drift: {job['name']}")
        output = root / job["output"]
        if output.exists():
            raise FileExistsError(f"Training output already exists: {output}")
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        configs[job["name"]] = config
        trainer = config["trainer"]
        loader = config["dataloader_builder"]
        if trainer["n_epochs"] * trainer["epoch_len"] != trainer["total_steps"]:
            raise ValueError(f"Iteration budget mismatch: {job['name']}")
        if trainer["total_steps"] != loader["train_steps"]:
            raise ValueError(f"Sampler budget mismatch: {job['name']}")
        if trainer["total_steps"] != config["lr_scheduler"]["T_max"]:
            raise ValueError(f"Scheduler budget mismatch: {job['name']}")
        if trainer["seed"] != job["seed"] or trainer["total_steps"] != job["steps"]:
            raise ValueError(f"Job/config identity mismatch: {job['name']}")
        if trainer["monitor"] != "off" or trainer["resume_from"] is not None:
            raise ValueError(f"Endpoint or resume policy drift: {job['name']}")
        if config["protocol"]["final_test_accessed"] is not False:
            raise ValueError(f"Final-test guard missing: {job['name']}")
        if {dataset["split"] for dataset in config["datasets"].values()} != {
            "train",
            "validation",
        }:
            raise ValueError(f"Unexpected data split: {job['name']}")

    # The only scientific DR16k change relative to its 4096-scene counterpart
    # is the training manifest.  The finite10000 pair is fully generated here.
    for condition, control in manifest["drunet_protocol"][
        "historical_controls"
    ].items():
        if control is None:
            continue
        new_name = f"cw-dr-scene16384-{condition}-100k-seed42-v1"
        source = OmegaConf.to_container(
            OmegaConf.load(control["config"]),
            resolve=True,
        )
        if scientific_signature(configs[new_name]) != scientific_signature(source):
            raise ValueError(f"DRUNet scientific signature drift: {condition}")
    for condition in manifest["drunet_protocol"]["fresh_control_conditions"]:
        control = configs[f"cw-dr-scene4096-{condition}-100k-seed42-v1"]
        expanded = configs[f"cw-dr-scene16384-{condition}-100k-seed42-v1"]
        if scientific_signature(control) != scientific_signature(expanded):
            raise ValueError(f"Fresh scene-count pair is not matched: {condition}")

    for seed in manifest["mnist_protocol"]["seeds"]:
        free = configs[f"cw-mnist-psf-free-finite100-50k-seed{seed}-v2"]
        aware = configs[f"cw-mnist-psf-aware-finite100-50k-seed{seed}-v2"]
        normalized = json.loads(json.dumps(aware))
        normalized["model"]["_target_"] = free["model"]["_target_"]
        normalized["dataloader_builder"]["return_psf"] = False
        normalized["trainer"]["device_tensors"] = ["measurement", "target"]
        normalized["protocol"]["information_regime"] = free["protocol"][
            "information_regime"
        ]
        normalized["writer"]["tags"] = free["writer"]["tags"]
        if scientific_signature(normalized) != scientific_signature(free):
            raise ValueError(f"MNIST matched-pair drift: seed{seed}")
    return manifest, root


def assert_shared_initialization(configs: dict, seeds: list[int]) -> dict:
    from src.utils.init_utils import set_random_seed

    results = {}
    for seed in seeds:
        free_config = configs[f"cw-mnist-psf-free-finite100-50k-seed{seed}-v2"]
        aware_config = configs[f"cw-mnist-psf-aware-finite100-50k-seed{seed}-v2"]
        set_random_seed(seed)
        free = instantiate(free_config.model).state_dict()
        set_random_seed(seed)
        aware = instantiate(aware_config.model).state_dict()
        for name, value in free.items():
            if name == "network.m_head.weight":
                if not torch.equal(value[:, :3], aware[name][:, :3]):
                    raise ValueError(f"Shared measurement head differs for seed{seed}")
                if not torch.equal(value[:, -1:], aware[name][:, -1:]):
                    raise ValueError(f"Shared noise head differs for seed{seed}")
                if torch.count_nonzero(aware[name][:, 3:6]).item() != 0:
                    raise ValueError(f"PSF channels are not zero initialized for seed{seed}")
            elif not torch.equal(value, aware[name]):
                raise ValueError(f"Shared parameter differs for seed{seed}: {name}")
        results[str(seed)] = {
            "shared_state_keys": len(free),
            "shared_parameters_exact": True,
            "extra_psf_head_channels_zero": True,
        }
    return results


def smoke_jobs(manifest: dict, root: Path) -> list[dict]:
    from src.datasets.data_utils import get_dataloaders
    from src.utils.init_utils import set_random_seed

    results = []
    for job in manifest["jobs"]:
        config = OmegaConf.load(root / job["config"])
        config.dataloader_builder.num_workers = 0
        config.dataloader_builder.persistent_workers = False
        config.dataloader_builder.psf_cache.mode = "off"
        config.dataloader_builder.psf_cache.warmup = False
        set_random_seed(job["seed"])
        loaders, transforms = get_dataloaders(config, "cuda")
        if transforms:
            raise ValueError(f"Unexpected transforms: {job['name']}")
        batch = next(iter(loaders["train"]))
        required = list(config.trainer.device_tensors)
        if any(name not in batch for name in required):
            raise KeyError(f"Smoke batch misses required tensors: {job['name']}")
        set_random_seed(job["seed"])
        model = instantiate(config.model).cuda().train()
        criterion = instantiate(config.loss_function).cuda()
        moved = {name: batch[name].cuda() for name in required}
        model.zero_grad(set_to_none=True)
        output = model(**moved)
        losses = criterion(prediction=output["prediction"], target=moved["target"])
        loss = losses["loss"]
        loss.backward()
        grad_norm = math.sqrt(
            sum(
                float(parameter.grad.detach().float().square().sum())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )
        if not torch.isfinite(loss) or not math.isfinite(grad_norm) or grad_norm == 0:
            raise ValueError(f"Invalid smoke loss/gradient: {job['name']}")
        mask_ids = batch.get("mask_id", [])
        scene_ids = batch.get("scene_id", [])
        results.append(
            {
                "name": job["name"],
                "loss": float(loss.detach()),
                "grad_norm": grad_norm,
                "batch_size": int(moved["measurement"].shape[0]),
                "measurement_shape": list(moved["measurement"].shape),
                "target_shape": list(moved["target"].shape),
                "psf_present": "psf" in moved,
                "mask_ids": [str(value) for value in mask_ids],
                "scene_ids": [str(value) for value in scene_ids],
                "optimizer_steps": 0,
            }
        )
        del model, criterion, output, loss, losses, moved, batch, loaders
        torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--smoke-gpu", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest, root = validate_metadata(manifest_path)
    sys.path.insert(0, str(root))
    configs = {
        job["name"]: OmegaConf.load(root / job["config"])
        for job in manifest["jobs"]
    }
    shared = assert_shared_initialization(
        configs, manifest["mnist_protocol"]["seeds"]
    )
    result = {
        "status": "metadata_pass",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "source_revision": manifest["source_revision"],
        "job_count": len(manifest["jobs"]),
        "lane_count": len(manifest["lanes"]),
        "shared_mnist_initialization": shared,
        "gpu_smoke": [],
        "optimizer_steps": 0,
        "final_test_accessed": False,
    }
    if args.smoke_gpu:
        result["gpu_smoke"] = smoke_jobs(manifest, root)
        result["status"] = "pass"
        save_json(root / manifest["bundle"] / "preflight.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
