"""Replay a MIRFLICKR PSF-aware endpoint with correct and shuffled PSFs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


METRICS = ("PSNR", "SSIM", "LPIPS")


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


def metric_rows(metric_objects, prediction, target) -> dict[str, np.ndarray]:
    values = {
        metric.name: metric.per_image(prediction=prediction, target=target)
        .detach()
        .float()
        .cpu()
        .numpy()
        for metric in metric_objects
    }
    if set(values) != set(METRICS):
        raise ValueError(f"Unexpected metric set: {sorted(values)}")
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("Non-finite metric values")
    return values


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def make_qualitative(path: Path, samples: dict[int, dict], expected: list[int]) -> None:
    if sorted(samples) != sorted(expected):
        raise ValueError(
            f"Qualitative sample set mismatch: {sorted(samples)} != {sorted(expected)}"
        )
    fig, axes = plt.subplots(len(expected), 3, figsize=(10, 2.8 * len(expected)))
    for row_index, sample_index in enumerate(expected):
        sample = samples[sample_index]
        for column, key in enumerate(("target", "correct", "shuffled")):
            image = sample[key].permute(1, 2, 0).numpy().clip(0, 1)
            axes[row_index, column].imshow(image)
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_title(
            f"Target\n{sample['mask_id']} / {sample['scene_id']}", fontsize=8
        )
        axes[row_index, 1].set_title("Correct PSF", fontsize=9)
        axes[row_index, 2].set_title(
            f"Shuffled PSF\n{sample['other_psf_mask_id']}", fontsize=8
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate(manifest_path: Path, identifier: str) -> None:
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if manifest["final_test_accessed_by_this_queue"] is not False:
        raise ValueError("Final test is forbidden")
    evaluator = "scripts/evaluate_mirflickr_psf_conditioning.py"
    if sha256(root / evaluator) != manifest["program_hashes"][evaluator]:
        raise ValueError("Evaluator hash drift")
    for relative, expected in manifest["source_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Source hash drift: {relative}")

    job = next(item for item in manifest["jobs"] if item["name"] == identifier)
    control = manifest["controls"][str(job["control_seed"])]
    for field, hash_field in (
        ("config", "config_sha256"),
        ("checkpoint", "checkpoint_sha256"),
        ("metrics_csv", "metrics_sha256"),
    ):
        if sha256(control[field]) != control[hash_field]:
            raise ValueError(f"Control {field} hash drift")
    config_path = root / job["config"]
    if sha256(config_path) != job["config_sha256"]:
        raise ValueError("Training config hash drift")

    training_output = root / job["output"]
    complete_path = training_output / "job_complete.json"
    complete = json.loads(complete_path.read_text())
    if complete["status"] != "complete" or complete["global_step"] != 100000:
        raise ValueError("PSF-aware final endpoint is incomplete")
    for field, hash_field in (
        ("checkpoint", "checkpoint_sha256"),
        ("metrics_csv", "metrics_sha256"),
        ("config", "config_sha256"),
    ):
        if sha256(complete[field]) != complete[hash_field]:
            raise ValueError(f"PSF-aware endpoint {field} hash drift")

    output = root / job["evaluation_output"]
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "run_state.json",
        {
            "status": "running",
            "name": identifier,
            "seed": job["seed"],
            "data_partition": "development",
            "final_test_accessed": False,
        },
    )
    started = time.monotonic()

    config = OmegaConf.load(config_path)
    config.model.checkpoint_path = None
    config.dataloader_builder.evaluation_only = True
    config.dataloader_builder.batch_size = 4
    config.dataloader_builder.num_workers = 4
    config.dataloader_builder.persistent_workers = False
    config.dataloader_builder.return_psf = True
    config.dataloader_builder.psf_cache.mode = "read_only"
    config.dataloader_builder.psf_cache.warmup = False

    from src.datasets.data_utils import get_dataloaders
    from src.utils.init_utils import set_random_seed

    set_random_seed(job["seed"])
    loaders, transforms = get_dataloaders(config, "cuda")
    if transforms:
        raise ValueError("Unexpected batch transforms")
    loader = loaders["validation"]
    state = torch.load(
        str(complete["checkpoint"]),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if state.get("global_step") != 100000 or state.get("sampler_step") != 100000:
        raise ValueError("Checkpoint endpoint mismatch")
    if state.get("lr_scheduler", {}).get("T_max") != 100000:
        raise ValueError("Checkpoint scheduler mismatch")
    model = instantiate(config.model).cuda().eval()
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    metric_objects = [instantiate(item) for item in config.metrics.inference]
    if [metric.name for metric in metric_objects] != list(METRICS):
        raise ValueError("Metric order or identity drift")

    rows = []
    consistency_rows = []
    qualitative = {}
    qualitative_indices = [
        int(value) for value in manifest["evaluation"]["qualitative_sample_indices"]
    ]
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
        with torch.inference_mode():
            for item in group:
                measurement = item["measurement"].cuda()
                target = item["target"].cuda()
                correct = predict(measurement, item["psf"])
                shuffled_psf = other_psf.expand(measurement.shape[0], -1, -1, -1)
                shuffled = predict(measurement, shuffled_psf)
                correct_values = metric_rows(metric_objects, correct, target)
                shuffled_values = metric_rows(metric_objects, shuffled, target)
                difference = shuffled - correct
                for index, scene_id in enumerate(item["scene_ids"]):
                    global_index = item["sample_indices"][index]
                    common = {
                        "seed": job["seed"],
                        "sample_index": global_index,
                        "mask_id": item["mask_id"],
                        "other_psf_mask_id": other_mask,
                        "scene_id": scene_id,
                    }
                    rows.append(
                        {
                            "condition": "Correct PSF",
                            **common,
                            **{
                                name: float(correct_values[name][index])
                                for name in METRICS
                            },
                        }
                    )
                    rows.append(
                        {
                            "condition": "Shuffled PSF",
                            **common,
                            **{
                                name: float(shuffled_values[name][index])
                                for name in METRICS
                            },
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
                    if global_index in qualitative_indices:
                        qualitative[global_index] = {
                            **common,
                            "target": target[index].detach().cpu().clamp(0, 1),
                            "correct": correct[index].detach().cpu().clamp(0, 1),
                            "shuffled": shuffled[index].detach().cpu().clamp(0, 1),
                        }
        group = []

    for batch in loader:
        mask_ids = [str(value) for value in batch["mask_id"]]
        if len(set(mask_ids)) != 1:
            raise ValueError("A validation batch contains multiple masks")
        mask_id = mask_ids[0]
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
    if frame[list(METRICS)].isna().any().any():
        raise ValueError("Evaluation contains missing metrics")
    frame.to_csv(output / "per_image.csv", index=False)
    per_mask = (
        frame.groupby(["condition", "mask_id"], sort=False)[list(METRICS)]
        .mean()
        .reset_index()
    )
    per_mask["sample_count"] = 32
    per_mask.to_csv(output / "per_mask.csv", index=False)
    consistency = pd.DataFrame(consistency_rows).sort_values("sample_index")
    consistency.to_csv(output / "prediction_consistency.csv", index=False)
    summary = per_mask.groupby("condition")[list(METRICS)].mean().to_dict("index")

    reference = pd.DataFrame(load_rows(Path(complete["metrics_csv"]))).sort_values(
        "mask_id"
    )
    correct = per_mask[per_mask["condition"] == "Correct PSF"].sort_values(
        "mask_id"
    )
    if list(reference["mask_id"].astype(str)) != list(correct["mask_id"].astype(str)):
        raise ValueError("Training/replay mask identity mismatch")
    parity = {
        name: float(
            np.max(
                np.abs(
                    reference[name].to_numpy(dtype=float)
                    - correct[name].to_numpy(dtype=float)
                )
            )
        )
        for name in METRICS
    }
    for name, delta in parity.items():
        tolerance = float(manifest["evaluation"]["correct_replay_tolerance"][name])
        if delta > tolerance:
            raise ValueError(f"Correct replay parity failed for {name}: {delta}")

    paired = {}
    for name in METRICS:
        correct_values = frame[frame["condition"] == "Correct PSF"].sort_values(
            "sample_index"
        )[name].to_numpy()
        shuffled_values = frame[frame["condition"] == "Shuffled PSF"].sort_values(
            "sample_index"
        )[name].to_numpy()
        paired[name] = {
            "mean": float(np.mean(shuffled_values - correct_values)),
            "mean_absolute": float(
                np.mean(np.abs(shuffled_values - correct_values))
            ),
        }
    make_qualitative(output / "qualitative.png", qualitative, qualitative_indices)

    summary_payload = {
        "status": "complete",
        "name": identifier,
        "seed": job["seed"],
        "conditions": summary,
        "shuffled_minus_correct": paired,
        "prediction_consistency": {
            name: float(consistency[name].mean())
            for name in ("prediction_MAE", "prediction_RMSE")
        },
        "correct_replay_max_abs": parity,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "data_partition": "development",
        "final_test_accessed": False,
    }
    save_json(output / "summary.json", summary_payload)
    validation = {
        "status": "pass",
        "sample_count": 1024,
        "mask_count": 32,
        "scenes_per_mask": 32,
        "conditions": ["Correct PSF", "Shuffled PSF"],
        "correct_replay_max_abs": parity,
        "correct_replay_within_tolerance": True,
        "all_metrics_finite": bool(
            np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all()
        ),
        "checkpoint_endpoint": 100000,
        "qualitative_sample_indices": qualitative_indices,
        "qualitative_complete": True,
        "data_partition": "development",
        "final_test_accessed": False,
    }
    save_json(output / "validation.json", validation)
    save_json(
        output / "provenance.json",
        {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "checkpoint": complete["checkpoint"],
            "checkpoint_sha256": complete["checkpoint_sha256"],
            "training_metrics": complete["metrics_csv"],
            "training_metrics_sha256": complete["metrics_sha256"],
            "evaluator": str(root / evaluator),
            "evaluator_sha256": sha256(root / evaluator),
        },
    )
    save_json(
        output / "run_state.json",
        {
            "status": "complete",
            "name": identifier,
            "seed": job["seed"],
            "data_partition": "development",
            "final_test_accessed": False,
        },
    )
    print(json.dumps(summary_payload, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("identifier")
    args = parser.parse_args()
    evaluate(Path(args.manifest).resolve(), args.identifier)


if __name__ == "__main__":
    main()
