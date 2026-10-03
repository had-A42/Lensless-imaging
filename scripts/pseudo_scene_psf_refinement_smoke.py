"""Targetless PSF refinement from draft reconstructions of ordinary scenes.

This diagnostic asks whether a measurement-only reconstructor can provide
enough approximate scene information to refine a shared PSF over several
measurements from the same mask.  The refined PSF is then passed to the
published PSF-aware reconstructor.

Only the upstream DigiCam train split is used.  The official test split is not
accessed.  This is a development smoke test, not a final benchmark.
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
from scipy.fftpack import next_fast_len
from torch import Tensor
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.psf_estimator_cascade_smoke import (  # noqa: E402
    DATASET_REPO,
    DATASET_REVISION,
    FEATURE_SIZE,
    MODEL_REPO,
    MODEL_REVISION,
    PSF_WORK_SIZE,
    ROI,
    SENSOR_SIZE,
    TARGET_SIZE,
    independent_max_normalize,
    load_psfs,
    normalize_l2,
    psnr_per_image,
    resize_psf,
    source_index,
    write_json,
)

PUBLISHED_PSF_FREE_CHECKPOINT = Path(
    "data/models/digicam-mirflickr-multi-25k-unet8M/recon_epochBEST"
)


def centered_pad(
    value: Tensor, size: tuple[int, int]
) -> tuple[Tensor, tuple[int, int]]:
    height, width = value.shape[-2:]
    pad_height = size[0] - height
    pad_width = size[1] - width
    if pad_height < 0 or pad_width < 0:
        raise ValueError("target padding size is smaller than the tensor")
    top = pad_height // 2
    left = pad_width // 2
    return F.pad(
        value,
        (left, pad_width - left, top, pad_height - top),
    ), (top, left)


def fft_forward_roi(scene: Tensor, psf: Tensor) -> Tensor:
    """Match LenslessPiCam's padded FFT convention with a differentiable PSF."""

    if scene.ndim != 4 or scene.shape[1:] != (3, TARGET_SIZE[0], TARGET_SIZE[1]):
        raise ValueError(f"unexpected scene shape: {tuple(scene.shape)}")
    if psf.shape != (3, *SENSOR_SIZE):
        raise ValueError(f"unexpected PSF shape: {tuple(psf.shape)}")
    top, left, height, width = ROI
    canvas = F.pad(
        scene,
        (left, SENSOR_SIZE[1] - left - width, top, SENSOR_SIZE[0] - top - height),
    )
    padded_size = (
        next_fast_len(2 * SENSOR_SIZE[0] - 1),
        next_fast_len(2 * SENSOR_SIZE[1] - 1),
    )
    canvas_pad, crop_start = centered_pad(canvas, padded_size)
    psf_pad, _ = centered_pad(psf.unsqueeze(0), padded_size)
    scene_spectrum = torch.fft.rfft2(canvas_pad, dim=(-2, -1))
    psf_spectrum = torch.fft.rfft2(psf_pad, norm="ortho", dim=(-2, -1))
    measurement = torch.fft.irfft2(
        scene_spectrum * psf_spectrum,
        s=padded_size,
        dim=(-2, -1),
    )
    measurement = torch.fft.ifftshift(measurement, dim=(-2, -1))
    crop_top, crop_left = crop_start
    return measurement[
        ...,
        crop_top : crop_top + SENSOR_SIZE[0],
        crop_left : crop_left + SENSOR_SIZE[1],
    ]


def validate_forward(psf: Tensor) -> float:
    from lensless.recon.rfft_convolve import RealFFTConvolve2D

    generator = torch.Generator().manual_seed(7)
    scene = torch.rand(2, 3, *TARGET_SIZE, generator=generator)
    expected_operator = RealFFTConvolve2D(
        psf=psf.movedim(0, -1).unsqueeze(0), dtype=torch.float32
    )
    top, left, height, width = ROI
    canvas = torch.zeros(2, 1, *SENSOR_SIZE, 3)
    canvas[:, 0, top : top + height, left : left + width] = scene.movedim(1, -1)
    expected = expected_operator.convolve(canvas)[:, 0].movedim(-1, 1)
    actual = fft_forward_roi(scene, psf)
    error = float((actual - expected).abs().max())
    if error > 1e-5:
        raise RuntimeError(f"differentiable forward mismatch: {error}")
    return error


def load_rows(source_dataset, mask_id: int, row_slots: list[int]) -> list[dict]:
    from src.datasets.digicam import DigiCamRealDataset

    dataset = DigiCamRealDataset(
        repo_id=DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        source_dataset=source_dataset,
        indices=[source_index(mask_id, row_slot) for row_slot in row_slots],
        force_rgb=True,
        rotate_measurement=True,
        measurement_downsample=1.0,
        target_size=list(TARGET_SIZE),
        target_resize_mode="bilinear",
    )
    rows = []
    for row_slot, sample in zip(row_slots, dataset):
        if int(sample["mask_id"]) != mask_id:
            raise ValueError("dataset row does not match requested mask")
        rows.append(
            {
                "mask_id": mask_id,
                "row_slot": row_slot,
                "measurement": sample["measurement"],
                "target": sample["target"],
            }
        )
    return rows


@torch.inference_mode()
def draft_scenes(rows: list[dict], model, batch_size: int) -> Tensor:
    predictions = []
    for start in range(0, len(rows), batch_size):
        measurement = torch.stack(
            [row["measurement"] for row in rows[start : start + batch_size]]
        )
        predictions.append(model(measurement=measurement)["prediction"].float())
    return torch.cat(predictions)


def refine_psf(
    draft: Tensor,
    measurement: Tensor,
    mean_psf: Tensor,
    *,
    steps: int,
    learning_rate: float,
    regularization: float,
) -> tuple[Tensor, list[dict]]:
    mean_work = resize_psf(mean_psf, PSF_WORK_SIZE)
    parameter = mean_work.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([parameter], lr=learning_rate)
    target_measurement = independent_max_normalize(measurement)
    curve = []
    for step in range(steps + 1):
        psf = resize_psf(parameter.clamp_min(0), SENSOR_SIZE)
        predicted_measurement = independent_max_normalize(fft_forward_roi(draft, psf))
        consistency = (predicted_measurement - target_measurement).square().mean()
        prior = (parameter - mean_work).square().mean() / mean_work.square().mean()
        loss = consistency + regularization * prior
        if step == 0 or step % 5 == 0 or step == steps:
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "consistency": float(consistency.detach()),
                    "relative_prior": float(prior.detach()),
                }
            )
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(min=0)
            parameter.copy_(normalize_l2(parameter))
    return resize_psf(parameter.detach(), SENSOR_SIZE), curve


@torch.inference_mode()
def consistency_score(draft: Tensor, measurement: Tensor, psf: Tensor) -> float:
    predicted = independent_max_normalize(fft_forward_roi(draft, psf))
    target = independent_max_normalize(measurement)
    return float((predicted - target).square().mean())


@torch.inference_mode()
def evaluate_rows(rows: list[dict], psf_arms: dict[str, Tensor]) -> list[dict]:
    from src.model.psf_aware_lensless import PSFAwareLenslessModel

    model = PSFAwareLenslessModel(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        cache_dir="data/huggingface",
        output_crop=list(ROI),
    ).eval()
    output = []
    for row in rows:
        names = list(psf_arms)
        for start in range(0, len(names), 2):
            selected = names[start : start + 2]
            measurement = (
                row["measurement"].unsqueeze(0).expand(len(selected), -1, -1, -1)
            )
            target = row["target"].unsqueeze(0).expand(len(selected), -1, -1, -1)
            psfs = torch.stack([psf_arms[name] for name in selected])
            prediction = model(measurement=measurement, psf=psfs)["prediction"].float()
            psnr = psnr_per_image(prediction, target)
            for name, value in zip(selected, psnr):
                output.append(
                    {
                        "mask_id": row["mask_id"],
                        "row_slot": row["row_slot"],
                        "arm": name,
                        "PSNR": float(value),
                    }
                )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-count", type=int, default=1)
    parser.add_argument("--calibration-scenes", type=int, default=4)
    parser.add_argument("--calibration-row-start", type=int, default=20)
    parser.add_argument("--evaluation-scenes", type=int, default=1)
    parser.add_argument("--evaluation-row-start", type=int, default=40)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data/huggingface"))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache/matplotlib"))
    from datasets import load_dataset

    from src.digicam_protocol import build_digicam_mask_split
    from src.model.psf_free_drunet import PSFFreeDRUNet

    torch.manual_seed(20260911)
    torch.set_num_threads(4)
    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    train_masks, held_out_masks = build_digicam_mask_split()
    selected_masks = held_out_masks[: args.mask_count]
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
    forward_error = validate_forward(all_psfs[selected_masks[0]])
    source_dataset = load_dataset(
        DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        cache_dir=str(REPO_ROOT / "data/huggingface"),
    )
    draft_model = PSFFreeDRUNet(
        checkpoint_path=REPO_ROOT / PUBLISHED_PSF_FREE_CHECKPOINT,
        output_crop=list(ROI),
    ).eval()

    all_curve = []
    all_rows = []
    psf_metrics = []
    for offset, mask_id in enumerate(selected_masks, start=1):
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
        calibration_rows = load_rows(source_dataset, mask_id, calibration_slots)
        evaluation_rows = load_rows(source_dataset, mask_id, evaluation_slots)
        draft = draft_scenes(calibration_rows, draft_model, args.batch_size)
        measurement = torch.stack([row["measurement"] for row in calibration_rows])
        refined, curve = refine_psf(
            draft,
            measurement,
            mean_psf,
            steps=args.steps,
            learning_rate=args.learning_rate,
            regularization=args.regularization,
        )
        true_psf = all_psfs[mask_id]
        wrong_psf = all_psfs[
            held_out_masks[(held_out_masks.index(mask_id) + 1) % len(held_out_masks)]
        ]
        psf_arms = {
            "true_psf": true_psf,
            "refined_psf": refined,
            "mean_train_psf": mean_psf,
            "wrong_true_psf": wrong_psf,
        }
        all_rows.extend(evaluate_rows(evaluation_rows, psf_arms))
        for row in curve:
            all_curve.append({"mask_id": mask_id, **row})
        psf_metrics.append(
            {
                "mask_id": mask_id,
                "refined_cosine": float(
                    F.cosine_similarity(refined.flatten(), true_psf.flatten(), dim=0)
                ),
                "mean_cosine": float(
                    F.cosine_similarity(mean_psf.flatten(), true_psf.flatten(), dim=0)
                ),
                "refined_l1": float((refined - true_psf).abs().mean()),
                "mean_l1": float((mean_psf - true_psf).abs().mean()),
                "true_consistency": consistency_score(draft, measurement, true_psf),
                "refined_consistency": consistency_score(draft, measurement, refined),
                "mean_consistency": consistency_score(draft, measurement, mean_psf),
                "wrong_consistency": consistency_score(draft, measurement, wrong_psf),
            }
        )
        print(f"mask {offset}/{len(selected_masks)} complete: {mask_id}", flush=True)

    for name, rows in (
        ("optimization_curve.csv", all_curve),
        ("per_sample.csv", all_rows),
    ):
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
        "forward_parity_max_error": forward_error,
        "psf_metrics": psf_metrics,
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
