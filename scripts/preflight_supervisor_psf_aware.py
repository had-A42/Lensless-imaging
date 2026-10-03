"""Fail-closed preflight for the mandatory supervisor PSF-aware queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def plain(config) -> dict:
    return OmegaConf.to_container(config, resolve=True)


def copy_or_remove(target: dict, source: dict, key: str) -> None:
    if key in source:
        target[key] = json.loads(json.dumps(source[key]))
    else:
        target.pop(key, None)


def normalize_allowed_differences(candidate: dict, control: dict) -> dict:
    value = json.loads(json.dumps(candidate))
    value["model"]["_target_"] = control["model"]["_target_"]
    copy_or_remove(
        value["dataloader_builder"], control["dataloader_builder"], "return_psf"
    )
    value["trainer"]["device_tensors"] = control["trainer"]["device_tensors"]
    copy_or_remove(value["trainer"], control["trainer"], "save_dir")
    for key in ("run_name", "run_id", "group", "job_type", "tags"):
        copy_or_remove(value["writer"], control["writer"], key)
    for key in ("experiment", "queue", "information_regime", "comparison"):
        copy_or_remove(value["protocol"], control["protocol"], key)
    return value


def diff_paths(left, right, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(diff_paths(left_item, right_item, f"{prefix}[{index}]"))
        return paths
    return [] if left == right else [prefix]


def validate_mask_metrics(path: Path) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 32 or len({row["mask_id"] for row in rows}) != 32:
        raise ValueError(f"Expected 32 unique masks: {path}")
    if {int(row["sample_count"]) for row in rows} != {32}:
        raise ValueError(f"Expected 32 scenes per mask: {path}")
    for row in rows:
        for metric in ("PSNR", "SSIM", "LPIPS"):
            if not math.isfinite(float(row[metric])):
                raise ValueError(f"Non-finite {metric}: {path}")
    return {"mask_count": 32, "scenes_per_mask": 32, "all_metrics_finite": True}


def no_test_access(config: dict) -> None:
    splits = {dataset["split"] for dataset in config["datasets"].values()}
    if splits != {"train", "validation"}:
        raise ValueError(f"Unexpected dataset splits: {sorted(splits)}")

    def walk(value, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                if key == "allow_test" and child is True:
                    raise ValueError(f"Test authorization is forbidden: {child_path}")
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(config)
    if config["protocol"]["final_test_accessed"] is not False:
        raise ValueError("Config does not explicitly forbid final-test access")


def validate_checkpoint(path: Path, control: dict) -> dict:
    if path.stat().st_size != int(control["checkpoint_size"]):
        raise ValueError(f"Checkpoint size drift: {path}")
    state = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    if state.get("global_step") != 100000 or state.get("sampler_step") != 100000:
        raise ValueError(f"Control is not the exact 100k endpoint: {path}")
    scheduler = state.get("lr_scheduler")
    if not isinstance(scheduler, dict) or scheduler.get("T_max") != 100000:
        raise ValueError(f"Control scheduler horizon drift: {path}")
    result = {
        "global_step": int(state["global_step"]),
        "sampler_step": int(state["sampler_step"]),
        "scheduler_T_max": int(scheduler["T_max"]),
        "checkpoint_size": path.stat().st_size,
    }
    del state
    return result


def validate_config(job: dict, aware: dict, control: dict) -> None:
    no_test_access(aware)
    trainer = aware["trainer"]
    loader = aware["dataloader_builder"]
    if trainer["seed"] != job["seed"] or loader["run_seed"] != job["seed"]:
        raise ValueError(f"Seed mismatch: {job['name']}")
    if trainer["n_epochs"] * trainer["epoch_len"] != 100000:
        raise ValueError(f"Epoch budget mismatch: {job['name']}")
    if not (
        trainer["total_steps"]
        == loader["train_steps"]
        == aware["lr_scheduler"]["T_max"]
        == job["steps"]
        == 100000
    ):
        raise ValueError(f"Step budget mismatch: {job['name']}")
    if trainer["device_tensors"] != ["measurement", "psf", "target"]:
        raise ValueError(f"Device tensor contract drift: {job['name']}")
    if loader.get("return_psf") is not True:
        raise ValueError(f"PSF return is disabled: {job['name']}")
    if aware["model"]["_target_"] != "src.model.psf_aware_drunet.PSFAwareDRUNet":
        raise ValueError(f"Model target drift: {job['name']}")
    if loader["finite_mask_count"] != 100 or loader["train_mask_seed"] != 42:
        raise ValueError(f"Training-mask protocol drift: {job['name']}")
    if loader["validation_mask_count"] != 32:
        raise ValueError(f"Development-mask count drift: {job['name']}")
    if loader["validation_scenes_per_mask"] != 32:
        raise ValueError(f"Development-scene count drift: {job['name']}")
    if trainer["monitor"] != "off" or trainer["resume_from"] is not None:
        raise ValueError(f"Endpoint selection or resume drift: {job['name']}")

    normalized = normalize_allowed_differences(aware, control)
    if normalized != control:
        differences = diff_paths(normalized, control)
        raise ValueError(
            f"Scientific signature drift for {job['name']}: {differences[:20]}"
        )


def compare_initialization(seed: int, aware_config, control_config) -> tuple[object, dict]:
    from src.utils.init_utils import set_random_seed

    set_random_seed(seed)
    free_model = instantiate(control_config.model).cpu()
    set_random_seed(seed)
    aware_model = instantiate(aware_config.model).cpu()
    free = free_model.state_dict()
    aware = aware_model.state_dict()
    for name, value in free.items():
        if name == "network.m_head.weight":
            if not torch.equal(value[:, :3], aware[name][:, :3]):
                raise ValueError(f"Measurement initialization mismatch for seed {seed}")
            if not torch.equal(value[:, -1:], aware[name][:, -1:]):
                raise ValueError(f"Noise initialization mismatch for seed {seed}")
            if torch.count_nonzero(aware[name][:, 3:6]).item() != 0:
                raise ValueError(f"PSF channels are not zero for seed {seed}")
        elif not torch.equal(value, aware[name]):
            raise ValueError(f"Shared initialization mismatch for seed {seed}: {name}")
    parameter_delta = aware_model.num_parameters - free_model.num_parameters
    expected_delta = 3 * int(aware_config.model.nc[0]) * 3 * 3
    if parameter_delta != expected_delta:
        raise ValueError(f"Unexpected capacity delta for seed {seed}: {parameter_delta}")
    del free_model
    return aware_model, {
        "shared_parameters_exact": True,
        "extra_psf_head_channels_zero": True,
        "parameter_delta": parameter_delta,
    }


def loader_config(config):
    value = OmegaConf.create(plain(config))
    value.dataloader_builder.num_workers = 0
    value.dataloader_builder.pin_memory = False
    value.dataloader_builder.persistent_workers = False
    value.dataloader_builder.psf_cache.mode = "read_only"
    value.dataloader_builder.psf_cache.warmup = False
    return value


def gpu_smoke(job: dict, aware_config, control_config, aware_model) -> dict:
    from src.datasets.data_utils import get_dataloaders
    from src.utils.init_utils import set_random_seed

    free_loader_config = loader_config(control_config)
    aware_loader_config = loader_config(aware_config)
    set_random_seed(job["seed"])
    free_loaders, free_transforms = get_dataloaders(free_loader_config, "cuda")
    set_random_seed(job["seed"])
    aware_loaders, aware_transforms = get_dataloaders(aware_loader_config, "cuda")
    if free_transforms or aware_transforms:
        raise ValueError(f"Unexpected batch transforms: {job['name']}")
    free_batch = next(iter(free_loaders["train"]))
    aware_batch = next(iter(aware_loaders["train"]))
    if not torch.equal(free_batch["measurement"], aware_batch["measurement"]):
        raise ValueError(f"Matched measurement batch differs: {job['name']}")
    if not torch.equal(free_batch["target"], aware_batch["target"]):
        raise ValueError(f"Matched target batch differs: {job['name']}")
    if list(free_batch["mask_id"]) != list(aware_batch["mask_id"]):
        raise ValueError(f"Matched mask IDs differ: {job['name']}")
    if list(free_batch["scene_id"]) != list(aware_batch["scene_id"]):
        raise ValueError(f"Matched scene IDs differ: {job['name']}")

    expected_shapes = {
        "measurement": [4, 3, 380, 507],
        "psf": [4, 3, 380, 507],
        "target": [4, 3, 200, 266],
    }
    actual_shapes = {
        name: list(aware_batch[name].shape) for name in expected_shapes
    }
    if actual_shapes != expected_shapes:
        raise ValueError(f"Tensor shape drift: {job['name']}: {actual_shapes}")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    aware_model = aware_model.to(device).train()
    criterion = instantiate(aware_config.loss_function).to(device)
    moved = {
        name: aware_batch[name].to(device)
        for name in ("measurement", "psf", "target")
    }
    aware_model.zero_grad(set_to_none=True)
    output = aware_model(**moved)
    losses = criterion(prediction=output["prediction"], target=moved["target"])
    loss = losses["loss"]
    loss.backward()
    gradients = [
        parameter.grad.detach().float()
        for parameter in aware_model.parameters()
        if parameter.grad is not None
    ]
    grad_norm = math.sqrt(sum(float(value.square().sum()) for value in gradients))
    psf_gradient = aware_model.network.m_head.weight.grad[:, 3:6]
    psf_grad_norm = float(psf_gradient.detach().float().square().sum().sqrt())
    if not torch.isfinite(loss) or not math.isfinite(grad_norm) or grad_norm <= 0:
        raise ValueError(f"Invalid smoke loss/gradient: {job['name']}")
    if not torch.isfinite(psf_gradient).all() or psf_grad_norm <= 0:
        raise ValueError(f"PSF input has no finite learning signal: {job['name']}")
    result = {
        "loss": float(loss.detach()),
        "grad_norm": grad_norm,
        "psf_channel_grad_norm": psf_grad_norm,
        "shapes": actual_shapes,
        "measurement_batch_exact": True,
        "target_batch_exact": True,
        "mask_ids": [str(value) for value in aware_batch["mask_id"]],
        "scene_ids": [str(value) for value in aware_batch["scene_id"]],
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "optimizer_steps": 0,
    }
    del aware_model, criterion, output, losses, loss, moved
    del free_batch, aware_batch, free_loaders, aware_loaders
    torch.cuda.empty_cache()
    return result


def validate_metadata(manifest_path: Path) -> tuple[dict, Path, list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    if manifest["data_partitions"] != ["train", "development"]:
        raise ValueError("Only train/development partitions are allowed")
    if manifest["final_test_accessed_by_this_queue"] is not False:
        raise ValueError("Final-test access must be false")
    if manifest["uses_final_test_for_selection"] is not False:
        raise ValueError("Final test cannot select these jobs")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if revision != manifest["source_revision"]:
        raise ValueError(f"Source revision drift: {revision}")
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
    ).strip()
    if tracked_status:
        raise ValueError(f"Tracked worktree is dirty: {tracked_status}")
    for relative, expected in manifest["source_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Source hash drift: {relative}")
    for relative, expected in manifest["program_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Program hash drift: {relative}")
    scene = manifest["scene_manifest"]
    if sha256(scene["path"]) != scene["sha256"]:
        raise ValueError("Training scene manifest drift")
    usage = shutil.disk_usage(root)
    if usage.free < manifest["minimum_free_disk_bytes"]:
        raise OSError(f"Insufficient free disk: {usage.free} bytes")

    audits = []
    for job in manifest["jobs"]:
        config_path = root / job["config"]
        if sha256(config_path) != job["config_sha256"]:
            raise ValueError(f"Config hash drift: {job['name']}")
        if (root / job["output"]).exists():
            raise FileExistsError(f"Training output exists: {root / job['output']}")
        if (root / job["evaluation_output"]).exists():
            raise FileExistsError(
                f"Evaluation output exists: {root / job['evaluation_output']}"
            )
        control = manifest["controls"][str(job["control_seed"])]
        for field, hash_field in (
            ("config", "config_sha256"),
            ("checkpoint", "checkpoint_sha256"),
            ("metrics_csv", "metrics_sha256"),
            ("job_complete", "job_complete_sha256"),
        ):
            if sha256(control[field]) != control[hash_field]:
                raise ValueError(f"Control {field} hash drift for seed {job['seed']}")
        checkpoint_audit = validate_checkpoint(Path(control["checkpoint"]), control)
        metric_audit = validate_mask_metrics(Path(control["metrics_csv"]))
        aware = plain(OmegaConf.load(config_path))
        free = plain(OmegaConf.load(control["config"]))
        validate_config(job, aware, free)
        audits.append(
            {
                "seed": job["seed"],
                "control_hashes_exact": True,
                "scientific_signature_exact_after_allowed_normalization": True,
                **checkpoint_audit,
                **metric_audit,
            }
        )
    return manifest, root, audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest, root, control_audits = validate_metadata(manifest_path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    initialization = {}
    smokes = []
    for job in manifest["jobs"]:
        aware_config = OmegaConf.load(root / job["config"])
        control = manifest["controls"][str(job["control_seed"])]
        control_config = OmegaConf.load(control["config"])
        aware_model, init_audit = compare_initialization(
            job["seed"], aware_config, control_config
        )
        initialization[str(job["seed"])] = init_audit
        if args.metadata_only:
            del aware_model
        else:
            smokes.append(
                {
                    "name": job["name"],
                    **gpu_smoke(job, aware_config, control_config, aware_model),
                }
            )

    result = {
        "status": "metadata_pass" if args.metadata_only else "pass",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "source_revision": manifest["source_revision"],
        "tracked_worktree_clean": True,
        "control_audits": control_audits,
        "shared_initialization": initialization,
        "gpu_smoke": smokes,
        "optimizer_steps": 0,
        "data_partitions": ["train", "development"],
        "final_test_accessed": False,
    }
    if not args.metadata_only:
        save_json(root / manifest["bundle"] / "preflight.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
