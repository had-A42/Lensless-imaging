"""Development-only zero-input and cross-mask consistency evaluation.

The same-scene/other-mask target score is included for completeness.  On a
complete balanced grid it is a permutation of the Correct scores, so the
scientifically useful quantity is the paired change and the distance between
the two reconstructions, not its aggregate mean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True
    )
    return hashlib.sha256(
        header.encode() + b"\0" + value.numpy().tobytes()
    ).hexdigest()


def save_json(path: str | Path, value: object) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def validate_job(job: dict) -> dict:
    required = {
        "name",
        "root",
        "config",
        "checkpoint",
        "baseline_csv",
        "output",
        "dataset",
        "steps",
        "seed",
        "expected_samples",
        "expected_masks",
        "expected_scenes_per_mask",
        "expected_checkpoint_sha256",
        "expected_config_sha256",
        "expected_baseline_sha256",
        "data_partition",
        "selection_basis",
    }
    missing = sorted(required - set(job))
    if missing:
        raise ValueError(f"Missing job fields: {missing}")
    if job["data_partition"] != "development":
        raise ValueError("Only development data are allowed")
    if job.get("final_test_accessed") is not False:
        raise ValueError("The job must explicitly state final_test_accessed=false")
    if job.get("selected_using_final_test") is not False:
        raise ValueError("Post-final checkpoint selection is forbidden")
    if job["dataset"] not in {"mirflickr", "celeba"}:
        raise ValueError(f"Unsupported dataset: {job['dataset']}")
    if job["expected_samples"] != (
        job["expected_masks"] * job["expected_scenes_per_mask"]
    ):
        raise ValueError("Expected grid dimensions are inconsistent")
    forbidden_baseline_parts = {"final_run", "final_run_revised"}
    if forbidden_baseline_parts.intersection(Path(job["baseline_csv"]).parts):
        raise ValueError("Final-run artifacts cannot be used as a baseline")
    checks = {
        "status": "pass",
        "name": job["name"],
        "data_partition": "development",
        "final_test_accessed": False,
        "selected_using_final_test": False,
        "files": {},
    }
    for field, expected_field in (
        ("config", "expected_config_sha256"),
        ("checkpoint", "expected_checkpoint_sha256"),
        ("baseline_csv", "expected_baseline_sha256"),
    ):
        path = Path(job[field])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != job[expected_field]:
            raise ValueError(f"{field} SHA256 drift: {actual}")
        checks["files"][field] = {"path": str(path), "sha256": actual}
    return checks


def get_config(job: dict):
    config = OmegaConf.load(job["config"])
    config.model.checkpoint_path = None
    if config.get("initialization") is not None:
        config.initialization.checkpoint_path = None
    config.metrics.device = "cuda"
    if config.get("trainer") is not None:
        config.trainer.from_pretrained = str(job["checkpoint"])
    return config


def get_loader(config):
    from src.datasets.data_utils import get_dataloaders

    config.dataloader_builder.evaluation_only = True
    config.dataloader_builder.batch_size = 1
    config.dataloader_builder.num_workers = 4
    config.dataloader_builder.persistent_workers = False
    loaders, transforms = get_dataloaders(config, "cuda")
    if transforms:
        raise ValueError("Unexpected evaluation batch transforms")
    return loaders["validation"]


def build_metrics(dataset: str):
    from src.metrics.reconstruction import LPIPSMetric, PSNRMetric, SSIMMetric

    normalize = dataset != "celeba"
    metrics = [
        PSNRMetric(name="PSNR", normalize_by_max=normalize),
        SSIMMetric(name="SSIM", normalize_by_max=normalize),
        LPIPSMetric(
            name="LPIPS", net_type="vgg", device="cuda", normalize_by_max=normalize
        ),
    ]
    if dataset == "celeba":
        from src.metrics.grid_artifacts import BlockResidualRMSE, PeriodicResidualRMSE
        from src.metrics.reconstruction import PooledPSNRMetric, PooledSSIMMetric

        metrics.extend(
            [
                PooledPSNRMetric(
                    name="PSNR_coarse", pooling_factor=8, normalize_by_max=False
                ),
                PooledSSIMMetric(
                    name="SSIM_coarse", pooling_factor=8, normalize_by_max=False
                ),
                BlockResidualRMSE(name="block_residual_rmse", block_size=8),
                PeriodicResidualRMSE(name="periodic16_residual_rmse", period=16),
            ]
        )
    return metrics


def metric_values(metrics, prediction: torch.Tensor, target: torch.Tensor) -> dict:
    values = {}
    for metric in metrics:
        value = metric.per_image(
            prediction=prediction.unsqueeze(0).cuda(),
            target=target.unsqueeze(0).cuda(),
        ).reshape(-1)[0]
        values[metric.name] = float(value)
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite metrics: {values}")
    return values


def baseline_rows(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "condition" in frame:
        frame = frame[frame["condition"].isin(["Correct", "correct"])].copy()
    required = {"sample_index", "mask_id", "scene_id", "PSNR", "SSIM", "LPIPS"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Baseline columns missing: {sorted(required - set(frame))}")
    return frame.sort_values("sample_index").reset_index(drop=True)


def evaluate(job: dict) -> None:
    from src.utils.init_utils import set_random_seed

    preflight = validate_job(job)
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "job.json", job)
    save_json(output / "preflight.json", preflight)

    set_random_seed(job["seed"])
    torch.set_num_threads(4)
    started = time.monotonic()
    config = get_config(job)
    loader = get_loader(config)
    checkpoint = torch.load(
        str(job["checkpoint"]),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if checkpoint["global_step"] != job["steps"]:
        raise ValueError(
            f"Checkpoint endpoint mismatch: {checkpoint['global_step']} != {job['steps']}"
        )
    model = instantiate(config.model).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    metrics = build_metrics(job["dataset"])
    metric_names = [metric.name for metric in metrics]
    precision = job.get("evaluation_precision", "fp32").lower()
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"Unsupported precision: {precision}")

    def predict(measurement: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=precision == "bf16"
        ):
            result = model(measurement=measurement.cuda())["prediction"]
        return result[0].float().cpu()

    OmegaConf.save(config, output / "resolved_config.yaml", resolve=True)
    provenance = {
        "checkpoint": job["checkpoint"],
        "checkpoint_sha256": job["expected_checkpoint_sha256"],
        "source_config": job["config"],
        "source_config_sha256": job["expected_config_sha256"],
        "baseline_csv": job["baseline_csv"],
        "baseline_sha256": job["expected_baseline_sha256"],
        "evaluation_precision": precision.upper(),
        "evaluator_sha256": sha256(Path(__file__)),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_partition": "development",
        "final_test_accessed": False,
        "selected_using_final_test": False,
        "selection_basis": job["selection_basis"],
        "cross_mask_mapping": "cyclic next unseen validation mask for the same scene",
        "aggregate_other_mask_note": "a balanced cyclic mask permutation preserves the aggregate target score",
    }
    save_json(output / "provenance.json", provenance)

    rows = []
    correct_by_mask: OrderedDict[str, OrderedDict[str, dict]] = OrderedDict()
    targets: dict[str, torch.Tensor] = {}
    target_hashes: dict[str, str] = {}
    zero_prediction = None
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for index, batch in enumerate(loader):
            mask_id = str(
                batch["mask_id"][0].item()
                if torch.is_tensor(batch["mask_id"])
                else batch["mask_id"][0]
            )
            scene_id = str(batch["scene_id"][0])
            target = batch["target"][0].float().cpu()
            target_sha = tensor_hash(target)
            if scene_id in target_hashes and target_hashes[scene_id] != target_sha:
                raise ValueError(f"Target drift across masks for {scene_id}")
            target_hashes.setdefault(scene_id, target_sha)
            targets.setdefault(scene_id, target)
            prediction = predict(batch["measurement"])
            if prediction.shape != target.shape:
                raise ValueError(
                    f"Prediction/target shape mismatch: {prediction.shape} != {target.shape}"
                )
            if zero_prediction is None:
                zero_prediction = predict(torch.zeros_like(batch["measurement"]))
            correct_values = metric_values(metrics, prediction, target)
            zero_values = metric_values(metrics, zero_prediction, target)
            common = {
                "sample_index": index,
                "mask_id": mask_id,
                "scene_id": scene_id,
                "target_sha256": target_sha,
            }
            rows.append({"condition": "Correct", **common, **correct_values})
            rows.append({"condition": "Zero", **common, **zero_values})
            correct_by_mask.setdefault(mask_id, OrderedDict())[scene_id] = {
                "sample_index": index,
                "prediction": prediction,
                "metrics": correct_values,
            }
            if (index + 1) % 128 == 0:
                print(job["name"], index + 1, "/", len(loader), flush=True)

    mask_ids = list(correct_by_mask)
    if len(mask_ids) != job["expected_masks"]:
        raise ValueError(f"Mask count mismatch: {len(mask_ids)}")
    scene_order = list(next(iter(correct_by_mask.values())))
    if len(scene_order) != job["expected_scenes_per_mask"]:
        raise ValueError(f"Scenes-per-mask mismatch: {len(scene_order)}")
    for mask_id, group in correct_by_mask.items():
        if list(group) != scene_order:
            raise ValueError(f"Scene order drift for mask {mask_id}")
    if sum(len(group) for group in correct_by_mask.values()) != job["expected_samples"]:
        raise ValueError("Sample count mismatch")

    consistency_rows = []
    paired_rows = []
    for mask_index, mask_id in enumerate(mask_ids):
        other_mask_id = mask_ids[(mask_index + 1) % len(mask_ids)]
        current_group = correct_by_mask[mask_id]
        other_group = correct_by_mask[other_mask_id]
        for scene_id in scene_order:
            current = current_group[scene_id]
            other = other_group[scene_id]
            same_other_values = other["metrics"]
            common = {
                "sample_index": current["sample_index"],
                "mask_id": mask_id,
                "scene_id": scene_id,
                "target_sha256": target_hashes[scene_id],
                "measurement_mask_id": other_mask_id,
            }
            rows.append(
                {
                    "condition": "Same scene, other mask",
                    **common,
                    **same_other_values,
                }
            )
            deltas = {
                name: same_other_values[name] - current["metrics"][name]
                for name in metric_names
            }
            paired_rows.append({**common, **deltas})
            pair_values = metric_values(
                metrics, current["prediction"], other["prediction"]
            )
            difference = current["prediction"] - other["prediction"]
            consistency_rows.append(
                {
                    **common,
                    **{f"prediction_pair_{key}": value for key, value in pair_values.items()},
                    "prediction_pair_MAE": float(difference.abs().mean()),
                    "prediction_pair_RMSE": float(difference.square().mean().sqrt()),
                }
            )
            if current["sample_index"] < 4:
                torch.save(
                    {
                        "target": targets[scene_id],
                        "correct": current["prediction"],
                        "same_scene_other_mask": other["prediction"],
                        "zero": zero_prediction,
                        "scene_id": scene_id,
                        "mask_id": mask_id,
                        "other_mask_id": other_mask_id,
                    },
                    output / f"example_{current['sample_index']:04d}.pth",
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_image.csv", index=False)
    per_mask = frame.groupby(["condition", "mask_id"])[metric_names].mean().reset_index()
    per_mask["sample_count"] = (
        frame.groupby(["condition", "mask_id"]).size().to_numpy()
    )
    per_mask.to_csv(output / "per_mask.csv", index=False)
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(output / "paired_other_mask_minus_correct.csv", index=False)
    consistency = pd.DataFrame(consistency_rows)
    consistency.to_csv(output / "cross_mask_consistency.csv", index=False)

    summary = per_mask.groupby("condition")[metric_names].mean().to_dict("index")
    baseline = baseline_rows(job["baseline_csv"])
    correct = frame[frame["condition"] == "Correct"].sort_values("sample_index")
    if len(baseline) != len(correct):
        raise ValueError("Baseline row count mismatch")
    if list(baseline["scene_id"].astype(str)) != list(correct["scene_id"].astype(str)):
        raise ValueError("Baseline scene identity mismatch")
    if list(baseline["mask_id"].astype(str)) != list(correct["mask_id"].astype(str)):
        raise ValueError("Baseline mask identity mismatch")
    baseline_summary = baseline.groupby("mask_id")[["PSNR", "SSIM", "LPIPS"]].mean().mean()
    baseline_deltas = {
        name: summary["Correct"][name] - float(baseline_summary[name])
        for name in ("PSNR", "SSIM", "LPIPS")
    }
    for name, delta in baseline_deltas.items():
        tolerance = 0.01 if name == "PSNR" else 0.001
        if abs(delta) > tolerance:
            raise ValueError(f"Baseline parity failed for {name}: {delta}")
    other_deltas = {
        name: summary["Same scene, other mask"][name] - summary["Correct"][name]
        for name in metric_names
    }
    if any(abs(value) > 1e-10 for value in other_deltas.values()):
        raise ValueError(f"Balanced cyclic permutation invariance failed: {other_deltas}")
    validation = {
        "status": "pass",
        "complete": True,
        "data_partition": "development",
        "final_test_accessed": False,
        "sample_count": len(correct),
        "mask_count": len(mask_ids),
        "scenes_per_mask": len(scene_order),
        "baseline_deltas": baseline_deltas,
        "balanced_other_mask_aggregate_deltas": other_deltas,
        "all_metrics_finite": bool(
            np.isfinite(frame[metric_names].to_numpy()).all()
            and np.isfinite(consistency.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "post_final_checkpoint_selection": False,
    }
    if not validation["all_metrics_finite"]:
        raise ValueError("Non-finite result values")
    save_json(output / "validation.json", validation)
    pair_columns = [column for column in consistency if column.startswith("prediction_pair_")]
    summary_payload = {
        "status": "complete",
        "conditions": summary,
        "paired_other_mask_minus_correct_mean": {
            name: float(paired_frame[name].mean()) for name in metric_names
        },
        "paired_other_mask_minus_correct_mean_absolute": {
            name: float(paired_frame[name].abs().mean()) for name in metric_names
        },
        "cross_mask_consistency_mean": {
            name: float(consistency[name].mean()) for name in pair_columns
        },
        "sample_count": len(correct),
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "data_partition": "development",
        "final_test_accessed": False,
    }
    save_json(output / "summary.json", summary_payload)
    print(json.dumps(summary_payload, indent=2, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("--execute-development", action="store_true")
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    os.chdir(job["root"])
    sys.path.insert(0, job["root"])
    if args.execute_development:
        evaluate(job)
    else:
        print(json.dumps(validate_job(job), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
