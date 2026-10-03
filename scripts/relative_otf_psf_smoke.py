"""Calibrate an unknown DigiCam mask relative to a known reference mask.

The same ordinary scenes are captured once with a calibrated reference mask and
once with an unknown mask.  In Fourier space the scene approximately cancels:

    P_unknown ~= P_reference * sum(Y_unknown * conj(Y_reference))
                               / (sum(|Y_reference|^2) + lambda).

No point source, clean target, or learned PSF estimator is used.  The script
uses only the upstream DigiCam train split and is a development smoke test.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.psf_estimator_cascade_smoke import (  # noqa: E402
    DATASET_REPO,
    DATASET_REVISION,
    MODEL_REPO,
    MODEL_REVISION,
    ROI,
    SENSOR_SIZE,
    TARGET_SIZE,
    load_psfs,
    normalize_l2,
    psnr_per_image,
    source_index,
    write_json,
)


def normalized_measurements(value: Tensor) -> Tensor:
    value = value.clamp_min(0)
    return value / value.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)


def load_mask_rows(source_dataset, mask_id: int, row_slots: list[int]) -> list[dict]:
    from src.datasets.digicam import DigiCamRealDataset

    dataset = DigiCamRealDataset(
        repo_id=DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        source_dataset=source_dataset,
        indices=[source_index(mask_id, slot) for slot in row_slots],
        force_rgb=True,
        rotate_measurement=True,
        measurement_downsample=1.0,
        target_size=list(TARGET_SIZE),
        target_resize_mode="bilinear",
    )
    output = []
    for slot, sample in zip(row_slots, dataset):
        if int(sample["mask_id"]) != mask_id:
            raise ValueError("mask/row mismatch")
        output.append(
            {
                "mask_id": mask_id,
                "row_slot": slot,
                "measurement": sample["measurement"],
                "target": sample["target"],
            }
        )
    return output


def estimate_relative_psf(
    unknown: Tensor,
    reference: Tensor,
    reference_psf: Tensor,
    relative_lambda: float,
    window: bool,
) -> Tensor:
    if unknown.shape != reference.shape or unknown.ndim != 4:
        raise ValueError("paired measurements must have equal NCHW shape")
    unknown = normalized_measurements(unknown)
    reference = normalized_measurements(reference)
    if window:
        vertical = torch.hann_window(unknown.shape[-2], periodic=False)
        horizontal = torch.hann_window(unknown.shape[-1], periodic=False)
        taper = vertical[:, None] * horizontal[None, :]
        unknown = unknown * taper
        reference = reference * taper
    unknown_spectrum = torch.fft.fft2(unknown, dim=(-2, -1))
    reference_spectrum = torch.fft.fft2(reference, dim=(-2, -1))
    cross = (unknown_spectrum * reference_spectrum.conj()).sum(dim=0)
    power = reference_spectrum.abs().square().sum(dim=0)
    floor = relative_lambda * power.amax(dim=(-2, -1), keepdim=True)
    relative_transfer = cross / (power + floor)

    reference_transfer = torch.fft.fft2(
        torch.fft.ifftshift(reference_psf, dim=(-2, -1)), dim=(-2, -1)
    )
    estimated_transfer = reference_transfer * relative_transfer
    estimated = torch.fft.fftshift(
        torch.fft.ifft2(estimated_transfer, dim=(-2, -1)).real,
        dim=(-2, -1),
    )
    return normalize_l2(estimated.clamp_min(0))


def blend_psf(mean_psf: Tensor, estimate: Tensor, alpha: float) -> Tensor:
    return normalize_l2(((1 - alpha) * mean_psf + alpha * estimate).clamp_min(0))


@torch.inference_mode()
def evaluate(rows: list[dict], psf_arms: dict[str, Tensor]) -> list[dict]:
    from src.model.psf_aware_lensless import PSFAwareLenslessModel

    model = PSFAwareLenslessModel(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        cache_dir="data/huggingface",
        output_crop=list(ROI),
    ).eval()
    output = []
    names = list(psf_arms)
    for row in rows:
        for start in range(0, len(names), 2):
            selected = names[start : start + 2]
            measurement = (
                row["measurement"].unsqueeze(0).expand(len(selected), -1, -1, -1)
            )
            target = row["target"].unsqueeze(0).expand(len(selected), -1, -1, -1)
            psfs = torch.stack([psf_arms[name] for name in selected])
            prediction = model(measurement=measurement, psf=psfs)["prediction"].float()
            scores = psnr_per_image(prediction, target)
            for arm, score in zip(selected, scores):
                output.append(
                    {
                        "mask_id": row["mask_id"],
                        "row_slot": row["row_slot"],
                        "arm": arm,
                        "PSNR": float(score),
                    }
                )
    return output


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-start", type=int, default=0)
    parser.add_argument("--mask-count", type=int, default=1)
    parser.add_argument("--reference-masks", type=parse_int_list, default=[45])
    parser.add_argument("--calibration-scenes", type=int, default=8)
    parser.add_argument("--calibration-row-start", type=int, default=20)
    parser.add_argument("--evaluation-scenes", type=int, default=1)
    parser.add_argument("--evaluation-row-start", type=int, default=40)
    parser.add_argument("--lambdas", type=parse_float_list, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument(
        "--alphas", type=parse_float_list, default=[0.1, 0.25, 0.5, 1.0]
    )
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--ensemble-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data/huggingface"))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache/matplotlib"))
    from datasets import load_dataset

    from src.digicam_protocol import build_digicam_mask_split

    torch.set_num_threads(4)
    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    train_masks, held_out_masks = build_digicam_mask_split()
    if not args.reference_masks or any(
        mask_id not in train_masks for mask_id in args.reference_masks
    ):
        raise ValueError("reference masks must belong to the calibrated train set")
    if len(set(args.reference_masks)) != len(args.reference_masks):
        raise ValueError("reference masks must be unique")
    selected_masks = held_out_masks[args.mask_start : args.mask_start + args.mask_count]
    if len(selected_masks) != args.mask_count:
        raise ValueError("requested mask range exceeds the held-out split")
    all_psfs = load_psfs(
        train_masks,
        held_out_masks,
        REPO_ROOT / "data/ref_psff_real/prepared_psfs_v1.npz",
        REPO_ROOT / "data/hf/DigiCam-Mirflickr-MultiMask-1K/masks",
        REPO_ROOT / "src/configs/simulator/digicam_article.yaml",
        REPO_ROOT / "outputs/psf_estimator_smoke_cache/outer17_psfs.npz",
    )
    mean_psf = normalize_l2(
        torch.stack([all_psfs[mask_id] for mask_id in train_masks]).mean(dim=0)
    )
    source_dataset = load_dataset(
        DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        cache_dir=str(REPO_ROOT / "data/huggingface"),
    )
    calibration_slots = list(
        range(
            args.calibration_row_start,
            args.calibration_row_start + args.calibration_scenes,
        )
    )
    evaluation_slots = list(
        range(
            args.evaluation_row_start,
            args.evaluation_row_start + args.evaluation_scenes,
        )
    )
    reference_measurements = {}
    for reference_mask in args.reference_masks:
        reference_rows = load_mask_rows(
            source_dataset, reference_mask, calibration_slots
        )
        reference_measurements[reference_mask] = torch.stack(
            [row["measurement"] for row in reference_rows]
        )
    all_rows = []
    psf_rows = []
    for offset, mask_id in enumerate(selected_masks, start=1):
        unknown_rows = load_mask_rows(source_dataset, mask_id, calibration_slots)
        unknown_measurement = torch.stack([row["measurement"] for row in unknown_rows])
        evaluation_rows = load_mask_rows(source_dataset, mask_id, evaluation_slots)
        true_psf = all_psfs[mask_id]
        wrong_psf = all_psfs[
            held_out_masks[(held_out_masks.index(mask_id) + 1) % len(held_out_masks)]
        ]
        arms = {
            "true_psf": true_psf,
            "mean_train_psf": mean_psf,
            "wrong_true_psf": wrong_psf,
        }
        for relative_lambda in args.lambdas:
            estimates = []
            for reference_mask in args.reference_masks:
                estimates.append(
                    estimate_relative_psf(
                        unknown_measurement,
                        reference_measurements[reference_mask],
                        all_psfs[reference_mask],
                        relative_lambda,
                        args.window,
                    )
                )
            estimates.append(normalize_l2(torch.stack(estimates).mean(dim=0)))
            estimate_names = [f"ref{mask_id}" for mask_id in args.reference_masks] + [
                f"ensemble{len(args.reference_masks)}"
            ]
            if args.ensemble_only:
                estimates = estimates[-1:]
                estimate_names = estimate_names[-1:]
            for estimate_name, estimate in zip(estimate_names, estimates):
                for alpha in args.alphas:
                    name = f"relative_{estimate_name}_l{relative_lambda:g}_a{alpha:g}"
                    candidate = blend_psf(mean_psf, estimate, alpha)
                    arms[name] = candidate
                    psf_rows.append(
                        {
                            "mask_id": mask_id,
                            "estimate": estimate_name,
                            "relative_lambda": relative_lambda,
                            "alpha": alpha,
                            "cosine": float(
                                F.cosine_similarity(
                                    candidate.flatten(), true_psf.flatten(), dim=0
                                )
                            ),
                            "l1": float((candidate - true_psf).abs().mean()),
                        }
                    )
        all_rows.extend(evaluate(evaluation_rows, arms))
        print(f"mask {offset}/{len(selected_masks)} complete: {mask_id}", flush=True)

    for name, rows in (("per_sample.csv", all_rows), ("psf_metrics.csv", psf_rows)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in all_rows:
        grouped[row["arm"]].append(row["PSNR"])
    summary = {
        "status": "complete",
        "purpose": "development smoke test",
        "official_test_accessed": False,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "selected_masks": selected_masks,
        "reconstruction": {
            arm: {
                "PSNR_mean": float(np.mean(values)),
                "sample_count": len(values),
            }
            for arm, values in grouped.items()
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
