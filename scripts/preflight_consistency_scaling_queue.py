"""Fail-closed metadata and one-batch checks for the consistency queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def signature(config: dict, ignore_repetition: bool = False) -> dict:
    value = json.loads(json.dumps(config))
    for dataset in value["datasets"].values():
        dataset.pop("root_dir", None)
        dataset.pop("splits_path", None)
    cache = value["dataloader_builder"].get("psf_cache", {})
    cache.pop("root_dir", None)
    cache.pop("warmup", None)
    for key in ("run_name", "run_id", "group", "job_type", "tags", "mode"):
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
    ):
        value["trainer"].pop(key, None)
    for key in (
        "experiment",
        "training_scenes",
        "fixed_train_mask_bank_seed",
        "repetition_seed",
        "data_partition",
        "final_test_accessed",
    ):
        value["protocol"].pop(key, None)
    if ignore_repetition:
        value["trainer"].pop("seed", None)
        value["dataloader_builder"].pop("run_seed", None)
    return value


def metadata(manifest_path: Path) -> tuple[dict, Path, dict[str, dict]]:
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if revision != manifest["source_revision"]:
        raise ValueError("Source revision drift")
    for relative, expected in manifest["source_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Source hash drift: {relative}")
    for relative, expected in manifest["program_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Program hash drift: {relative}")
    if sha256(manifest["parent_manifest"]) != manifest["parent_manifest_sha256"]:
        raise ValueError("Parent manifest drift")
    for record in manifest["scene_manifests"].values():
        if sha256(root / record["path"]) != record["sha256"]:
            raise ValueError("Scene manifest drift")
    if manifest["final_test_accessed"] or manifest["data_partitions"] != [
        "train",
        "development",
    ]:
        raise ValueError("Only train/development execution is allowed")
    names = [job["name"] for job in manifest["jobs"]]
    assigned = [name for lane in manifest["lanes"] for name in lane["jobs"]]
    if sorted(names) != sorted(assigned) or len(assigned) != len(set(assigned)):
        raise ValueError("Lane mapping drift")
    configs = {}
    for job in manifest["jobs"]:
        path = root / job["config"]
        if sha256(path) != job["config_sha256"]:
            raise ValueError(f"Config drift: {job['name']}")
        if (root / job["output"]).exists():
            raise FileExistsError(root / job["output"])
        config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        configs[job["name"]] = config
        trainer = config["trainer"]
        loader = config["dataloader_builder"]
        if not (
            trainer["n_epochs"] * trainer["epoch_len"]
            == trainer["total_steps"]
            == loader["train_steps"]
            == config["lr_scheduler"]["T_max"]
            == job["steps"]
        ):
            raise ValueError(f"Step budget mismatch: {job['name']}")
        if trainer["seed"] != job["seed"] or loader["run_seed"] != job["seed"]:
            raise ValueError(f"Repetition seed mismatch: {job['name']}")
        if loader["train_mask_seed"] != 42 or trainer["monitor"] != "off":
            raise ValueError(f"Mask-bank or endpoint policy drift: {job['name']}")
        if config["protocol"]["final_test_accessed"]:
            raise ValueError(f"Final-test guard missing: {job['name']}")

    for seed in manifest["drunet_protocol"]["new_seeds"]:
        for condition in manifest["drunet_protocol"]["mask_regimes"]:
            small = configs[
                f"cw-consistency-dr-scene4096-{condition}-100k-seed{seed}-v1"
            ]
            large = configs[
                f"cw-consistency-dr-scene16384-{condition}-100k-seed{seed}-v1"
            ]
            if signature(small) != signature(large):
                raise ValueError(f"Scene-count pair drift: seed{seed}/{condition}")
    for scenes in manifest["drunet_protocol"]["training_scenes"]:
        for condition in manifest["drunet_protocol"]["mask_regimes"]:
            first = configs[
                f"cw-consistency-dr-scene{scenes}-{condition}-100k-seed52-v1"
            ]
            second = configs[
                f"cw-consistency-dr-scene{scenes}-{condition}-100k-seed62-v1"
            ]
            if signature(first, True) != signature(second, True):
                raise ValueError(f"Cross-seed recipe drift: {scenes}/{condition}")
    return manifest, root, configs


def shared_mnist_initialization(configs: dict[str, dict], seeds: list[int]) -> dict:
    from src.utils.init_utils import set_random_seed

    results = {}
    for seed in seeds:
        free_name = f"cw-consistency-mnist-psf-free-finite100-50k-seed{seed}-v1"
        aware_name = f"cw-consistency-mnist-psf-aware-finite100-50k-seed{seed}-v1"
        free_cfg, aware_cfg = configs[free_name], configs[aware_name]
        normalized = json.loads(json.dumps(aware_cfg))
        normalized["model"]["_target_"] = free_cfg["model"]["_target_"]
        normalized["dataloader_builder"]["return_psf"] = False
        normalized["trainer"]["device_tensors"] = ["measurement", "target"]
        normalized["protocol"]["information_regime"] = "measurement only"
        normalized["writer"]["tags"] = free_cfg["writer"]["tags"]
        if signature(normalized) != signature(free_cfg):
            raise ValueError(f"MNIST pair recipe drift: seed{seed}")
        set_random_seed(seed)
        free = instantiate(free_cfg["model"]).state_dict()
        set_random_seed(seed)
        aware = instantiate(aware_cfg["model"]).state_dict()
        for name, value in free.items():
            if name == "network.m_head.weight":
                if not torch.equal(value[:, :3], aware[name][:, :3]):
                    raise ValueError("MNIST measurement head initialization drift")
                if not torch.equal(value[:, -1:], aware[name][:, -1:]):
                    raise ValueError("MNIST noise head initialization drift")
                if torch.count_nonzero(aware[name][:, 3:6]).item():
                    raise ValueError("MNIST PSF channels must start at zero")
            elif not torch.equal(value, aware[name]):
                raise ValueError(f"MNIST shared initialization drift: {name}")
        results[str(seed)] = {
            "shared_state_keys": len(free),
            "shared_parameters_exact": True,
            "extra_psf_channels_zero": True,
        }
    return results


def smoke(manifest: dict, root: Path, configs: dict[str, dict]) -> list[dict]:
    from src.datasets.data_utils import get_dataloaders
    from src.utils.init_utils import set_random_seed

    results = []
    for job in manifest["jobs"]:
        config = OmegaConf.create(configs[job["name"]])
        config.dataloader_builder.num_workers = 0
        config.dataloader_builder.persistent_workers = False
        config.dataloader_builder.psf_cache.warmup = False
        set_random_seed(job["seed"])
        loaders, transforms = get_dataloaders(config, "cuda")
        if transforms:
            raise ValueError("Unexpected transforms")
        batch = next(iter(loaders["train"]))
        required = list(config.trainer.device_tensors)
        moved = {name: batch[name].cuda() for name in required}
        set_random_seed(job["seed"])
        model = instantiate(config.model).cuda().train()
        criterion = instantiate(config.loss_function).cuda()
        model.zero_grad(set_to_none=True)
        prediction = model(**moved)["prediction"]
        loss = criterion(prediction=prediction, target=moved["target"])["loss"]
        loss.backward()
        grad = math.sqrt(
            sum(
                float(parameter.grad.detach().float().square().sum())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )
        if not torch.isfinite(loss) or not math.isfinite(grad) or grad == 0:
            raise ValueError(f"Invalid smoke result: {job['name']}")
        results.append(
            {
                "name": job["name"],
                "loss": float(loss.detach()),
                "grad_norm": grad,
                "mask_ids": [str(value) for value in batch["mask_id"]],
                "scene_ids": [str(value) for value in batch["scene_id"]],
                "psf_present": "psf" in moved,
                "optimizer_steps": 0,
            }
        )
        del model, criterion, prediction, loss, batch, moved, loaders
        torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--smoke-gpu", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest, root, configs = metadata(manifest_path)
    sys.path.insert(0, str(root))
    shared = shared_mnist_initialization(
        configs, manifest["mnist_protocol"]["new_seeds"]
    )
    result = {
        "status": "metadata_pass",
        "manifest_sha256": sha256(manifest_path),
        "job_count": len(manifest["jobs"]),
        "lane_count": len(manifest["lanes"]),
        "drunet_pair_contracts_pass": True,
        "mnist_shared_initialization": shared,
        "gpu_smoke": [],
        "optimizer_steps": 0,
        "final_test_accessed": False,
    }
    if args.smoke_gpu:
        result["gpu_smoke"] = smoke(manifest, root, configs)
        result["status"] = "pass"
        save_json(root / manifest["bundle"] / "preflight.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
