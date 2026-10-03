"""Replay a PSF-aware MNIST endpoint with correct and shuffled PSFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn import functional as F


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def dice_loss_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = F.avg_pool2d(prediction, 8, 8)
    target = F.avg_pool2d(target, 8, 8)
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    total = prediction.square().sum(dim=(1, 2, 3)) + target.square().sum(
        dim=(1, 2, 3)
    )
    return 1 - (2 * intersection + 1e-8) / (total + 1e-8)


def metrics() -> list:
    from src.metrics.reconstruction import (
        PooledPSNRMetric,
        PooledSSIMMetric,
        PSNRMetric,
        SSIMMetric,
    )

    return [
        PSNRMetric(name="PSNR", normalize_by_max=False),
        SSIMMetric(name="SSIM", normalize_by_max=False),
        PooledPSNRMetric(name="PSNR_32", pooling_factor=8, normalize_by_max=False),
        PooledSSIMMetric(name="SSIM_32", pooling_factor=8, normalize_by_max=False),
    ]


def metric_rows(metric_objects, prediction, target) -> dict[str, np.ndarray]:
    values = {
        metric.name: metric.per_image(prediction=prediction, target=target)
        .detach()
        .float()
        .cpu()
        .numpy()
        for metric in metric_objects
    }
    values["Dice_loss_32"] = (
        dice_loss_per_image(prediction, target).detach().float().cpu().numpy()
    )
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("Non-finite metric values")
    return values


def evaluate(manifest_path: Path, identifier: str) -> None:
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    sys.path.insert(0, str(root))
    entry = next(item for item in manifest["entries"] if item["id"] == identifier)
    for field, hash_field in (
        ("config", "config_sha256"),
        ("checkpoint", "checkpoint_sha256"),
        ("reference_per_mask", "reference_per_mask_sha256"),
    ):
        if sha256(entry[field]) != entry[hash_field]:
            raise ValueError(f"{field} hash drift")
    output = root / entry["output"]
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "entry.json", entry)
    started = time.monotonic()

    config = OmegaConf.load(entry["config"])
    config.model.checkpoint_path = None
    config.dataloader_builder.evaluation_only = True
    config.dataloader_builder.batch_size = 4
    config.dataloader_builder.num_workers = 4
    config.dataloader_builder.persistent_workers = False
    config.dataloader_builder.return_psf = True
    config.dataloader_builder.psf_cache.mode = "read_only"
    from src.datasets.data_utils import get_dataloaders
    from src.utils.init_utils import set_random_seed

    set_random_seed(entry["seed"])
    loaders, transforms = get_dataloaders(config, "cuda")
    if transforms:
        raise ValueError("Unexpected batch transforms")
    loader = loaders["validation"]
    state = torch.load(
        entry["checkpoint"], map_location="cpu", weights_only=False, mmap=True
    )
    if state["global_step"] != 50000 or state["sampler_step"] != 50000:
        raise ValueError("Checkpoint endpoint mismatch")
    model = instantiate(config.model).cuda().eval()
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    metric_objects = metrics()
    metric_names = [metric.name for metric in metric_objects] + ["Dice_loss_32"]
    rows = []
    consistency_rows = []
    first_psf = None
    group = []
    current_mask = None
    mask_order = []
    sample_index = 0
    torch.cuda.reset_peak_memory_stats()

    def predict(measurement: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        return model(measurement=measurement.cuda(), psf=psf.cuda())[
            "prediction"
        ].float()

    def finish_group(other_mask: str, other_psf: torch.Tensor) -> None:
        nonlocal group
        if not group:
            return
        if len(group) != 8 or sum(len(item["scene_ids"]) for item in group) != 32:
            raise ValueError("Expected eight batches and 32 scenes per mask")
        with torch.no_grad():
            for item in group:
                measurement = item["measurement"].cuda()
                target = item["target"].cuda()
                correct = predict(measurement, item["psf"])
                shuffled_psf = other_psf.expand(measurement.shape[0], -1, -1, -1)
                shuffled = predict(measurement, shuffled_psf)
                correct_values = metric_rows(metric_objects, correct, target)
                shuffled_values = metric_rows(metric_objects, shuffled, target)
                difference = correct - shuffled
                for index, scene_id in enumerate(item["scene_ids"]):
                    common = {
                        "sample_index": item["sample_indices"][index],
                        "mask_id": item["mask_id"],
                        "other_psf_mask_id": other_mask,
                        "scene_id": scene_id,
                    }
                    rows.append(
                        {
                            "condition": "Correct PSF",
                            **common,
                            **{name: float(correct_values[name][index]) for name in metric_names},
                        }
                    )
                    rows.append(
                        {
                            "condition": "Shuffled PSF",
                            **common,
                            **{name: float(shuffled_values[name][index]) for name in metric_names},
                        }
                    )
                    consistency_rows.append(
                        {
                            **common,
                            "prediction_MAE": float(difference[index].abs().mean()),
                            "prediction_RMSE": float(
                                difference[index].square().mean().sqrt()
                            ),
                        }
                    )
        group = []

    for batch in loader:
        mask_id = str(batch["mask_id"][0])
        if current_mask is None:
            current_mask = mask_id
            first_psf = batch["psf"][0:1].float().cpu()
            mask_order.append(mask_id)
        elif mask_id != current_mask:
            finish_group(mask_id, batch["psf"][0:1].float().cpu())
            current_mask = mask_id
            mask_order.append(mask_id)
        batch_size = int(batch["measurement"].shape[0])
        group.append(
            {
                "mask_id": mask_id,
                "sample_indices": list(range(sample_index, sample_index + batch_size)),
                "scene_ids": [str(value) for value in batch["scene_id"]],
                "measurement": batch["measurement"].float().cpu(),
                "target": batch["target"].float().cpu(),
                "psf": batch["psf"].float().cpu(),
            }
        )
        sample_index += batch_size
    finish_group(mask_order[0], first_psf)

    frame = pd.DataFrame(rows).sort_values(["condition", "sample_index"])
    if len(frame) != 2048 or sample_index != 1024 or len(mask_order) != 32:
        raise ValueError("Evaluation grid count mismatch")
    frame.to_csv(output / "per_image.csv", index=False)
    per_mask = (
        frame.groupby(["condition", "mask_id"], sort=False)[metric_names]
        .mean()
        .reset_index()
    )
    per_mask["sample_count"] = 32
    per_mask.to_csv(output / "per_mask.csv", index=False)
    consistency = pd.DataFrame(consistency_rows)
    consistency.to_csv(output / "prediction_consistency.csv", index=False)
    summary = per_mask.groupby("condition")[metric_names].mean().to_dict("index")

    reference = pd.read_csv(entry["reference_per_mask"]).sort_values("mask_id")
    correct = per_mask[per_mask["condition"] == "Correct PSF"].sort_values("mask_id")
    if list(reference["mask_id"].astype(str)) != list(correct["mask_id"].astype(str)):
        raise ValueError("Reference mask identity mismatch")
    parity = {
        name: float(
            np.max(
                np.abs(
                    reference[name].to_numpy(dtype=float)
                    - correct[name].to_numpy(dtype=float)
                )
            )
        )
        for name in ("PSNR", "SSIM", "PSNR_32", "SSIM_32")
    }
    for name, delta in parity.items():
        if delta > (1e-5 if "PSNR" in name else 1e-6):
            raise ValueError(f"Reference parity failed for {name}: {delta}")
    paired = {}
    for name in metric_names:
        correct_values = frame[frame["condition"] == "Correct PSF"].sort_values(
            "sample_index"
        )[name].to_numpy()
        shuffled_values = frame[frame["condition"] == "Shuffled PSF"].sort_values(
            "sample_index"
        )[name].to_numpy()
        paired[name] = {
            "mean": float(np.mean(shuffled_values - correct_values)),
            "mean_absolute": float(np.mean(np.abs(shuffled_values - correct_values))),
        }
    summary_payload = {
        "status": "complete",
        "id": identifier,
        "seed": entry["seed"],
        "conditions": summary,
        "shuffled_minus_correct": paired,
        "prediction_consistency": {
            name: float(consistency[name].mean())
            for name in ("prediction_MAE", "prediction_RMSE")
        },
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "final_test_accessed": False,
    }
    save_json(output / "summary.json", summary_payload)
    validation = {
        "status": "pass",
        "sample_count": 1024,
        "mask_count": 32,
        "scenes_per_mask": 32,
        "conditions": ["Correct PSF", "Shuffled PSF"],
        "reference_parity_max_abs": parity,
        "all_metrics_finite": bool(np.isfinite(frame[metric_names].to_numpy()).all()),
        "checkpoint_endpoint": 50000,
        "data_partition": "development",
        "final_test_accessed": False,
    }
    save_json(output / "validation.json", validation)
    print(json.dumps(summary_payload, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("identifier")
    args = parser.parse_args()
    evaluate(Path(args.manifest).resolve(), args.identifier)


if __name__ == "__main__":
    main()
