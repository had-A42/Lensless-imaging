"""Test shift/flip self-ensembling for measurement-only lensless inference.

Strided encoder-decoder networks are not exactly translation equivariant.  The
same full-sensor measurement is shifted by a few pixels, reconstructed, shifted
back, and averaged.  The target and checkpoint are never changed.  This is a
training-free development smoke test on the upstream DigiCam train split.
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
    ROI,
    TARGET_SIZE,
    source_index,
    write_json,
)

CHECKPOINT = Path("data/models/digicam-mirflickr-multi-25k-unet8M/recon_epochBEST")


def translate_zero(value: Tensor, dy: int, dx: int) -> Tensor:
    result = torch.zeros_like(value)
    height, width = value.shape[-2:]
    source_top = max(-dy, 0)
    source_left = max(-dx, 0)
    target_top = max(dy, 0)
    target_left = max(dx, 0)
    copy_height = height - abs(dy)
    copy_width = width - abs(dx)
    if copy_height <= 0 or copy_width <= 0:
        raise ValueError("translation is larger than the image")
    result[
        ...,
        target_top : target_top + copy_height,
        target_left : target_left + copy_width,
    ] = value[
        ...,
        source_top : source_top + copy_height,
        source_left : source_left + copy_width,
    ]
    return result


def transform(value: Tensor, name: str) -> Tensor:
    if name == "identity":
        return value
    if name == "shift_x4":
        return translate_zero(value, 0, 4)
    if name == "shift_y4":
        return translate_zero(value, 4, 0)
    if name == "shift_xy4":
        return translate_zero(value, 4, 4)
    if name == "flip_x":
        return value.flip(-1)
    if name == "flip_y":
        return value.flip(-2)
    if name == "flip_xy":
        return value.flip((-2, -1))
    raise ValueError(f"unknown transformation: {name}")


def inverse_transform(value: Tensor, name: str) -> Tensor:
    if name == "identity":
        return value
    if name == "shift_x4":
        return translate_zero(value, 0, -4)
    if name == "shift_y4":
        return translate_zero(value, -4, 0)
    if name == "shift_xy4":
        return translate_zero(value, -4, -4)
    if name == "flip_x":
        return value.flip(-1)
    if name == "flip_y":
        return value.flip(-2)
    if name == "flip_xy":
        return value.flip((-2, -1))
    raise ValueError(f"unknown transformation: {name}")


TRANSFORMS = (
    "identity",
    "shift_x4",
    "shift_y4",
    "shift_xy4",
    "flip_x",
    "flip_y",
    "flip_xy",
)
ARMS = {
    "baseline": ("identity",),
    "shift4_mean": ("identity", "shift_x4", "shift_y4", "shift_xy4"),
    "flip4_mean": ("identity", "flip_x", "flip_y", "flip_xy"),
    "shift_flip7_mean": TRANSFORMS,
}


def load_rows(source_dataset, mask_ids: list[int], row_slots: list[int]) -> list[dict]:
    from src.datasets.digicam import DigiCamRealDataset

    identities = [
        (source_index(mask_id, slot), mask_id, slot)
        for mask_id in mask_ids
        for slot in row_slots
    ]
    dataset = DigiCamRealDataset(
        repo_id=DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        source_dataset=source_dataset,
        indices=[item[0] for item in identities],
        force_rgb=True,
        rotate_measurement=True,
        measurement_downsample=1.0,
        target_size=list(TARGET_SIZE),
        target_resize_mode="bilinear",
    )
    rows = []
    for (_, mask_id, slot), sample in zip(identities, dataset):
        if int(sample["mask_id"]) != mask_id:
            raise ValueError("row identity mismatch")
        rows.append(
            {
                "mask_id": mask_id,
                "row_slot": slot,
                "measurement": sample["measurement"],
                "target": sample["target"],
            }
        )
    return rows


def crop_roi(value: Tensor) -> Tensor:
    top, left, height, width = ROI
    return value[..., top : top + height, left : left + width]


@torch.inference_mode()
def reconstruct_orbit(model, measurement: Tensor, batch_size: int) -> dict[str, Tensor]:
    transformed = torch.stack([transform(measurement, name) for name in TRANSFORMS])
    aligned = {}
    for start in range(0, len(TRANSFORMS), batch_size):
        names = TRANSFORMS[start : start + batch_size]
        prediction = model(measurement=transformed[start : start + batch_size])[
            "prediction"
        ].float()
        for name, item in zip(names, prediction):
            aligned[name] = crop_roi(inverse_transform(item, name))
    predictions = {
        arm: torch.stack([aligned[name] for name in members]).mean(dim=0)
        for arm, members in ARMS.items()
    }
    baseline = predictions["baseline"]
    for source in ("shift4_mean", "flip4_mean", "shift_flip7_mean"):
        for alpha in (0.02, 0.05, 0.1, 0.2):
            predictions[f"{source}_blend{alpha:g}"] = (
                1 - alpha
            ) * baseline + alpha * predictions[source]
    difference = predictions["shift4_mean"] - baseline
    for kernel_size in (5, 17, 33):
        smooth_difference = F.avg_pool2d(
            difference.unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )[0]
        for alpha in (0.25, 0.5, 1.0):
            predictions[f"shift4_lowpass_k{kernel_size}_a{alpha:g}"] = (
                baseline + alpha * smooth_difference
            )
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-start", type=int, default=0)
    parser.add_argument("--mask-count", type=int, default=1)
    parser.add_argument("--scene-count", type=int, default=1)
    parser.add_argument("--row-start", type=int, default=80)
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
    from src.metrics.reconstruction import LPIPSMetric, PSNRMetric, SSIMMetric
    from src.model.psf_free_drunet import PSFFreeDRUNet

    torch.set_num_threads(4)
    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _, held_out_masks = build_digicam_mask_split()
    selected_masks = held_out_masks[args.mask_start : args.mask_start + args.mask_count]
    if len(selected_masks) != args.mask_count:
        raise ValueError("requested mask range exceeds held-out development masks")
    row_slots = list(range(args.row_start, args.row_start + args.scene_count))
    source_dataset = load_dataset(
        DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        cache_dir=str(REPO_ROOT / "data/huggingface"),
    )
    rows = load_rows(source_dataset, selected_masks, row_slots)
    model = PSFFreeDRUNet(
        checkpoint_path=REPO_ROOT / CHECKPOINT,
        output_crop=None,
    ).eval()
    metrics = {
        "PSNR": PSNRMetric(normalize_by_max=True),
        "SSIM": SSIMMetric(normalize_by_max=True),
        "LPIPS": LPIPSMetric(net_type="vgg", device="cpu", normalize_by_max=True),
    }
    output_rows = []
    for index, row in enumerate(rows, start=1):
        predictions = reconstruct_orbit(model, row["measurement"], args.batch_size)
        target = row["target"].unsqueeze(0)
        for arm, prediction in predictions.items():
            values = {
                name: float(
                    metric.per_image(prediction.unsqueeze(0), target).reshape(-1)[0]
                )
                for name, metric in metrics.items()
            }
            output_rows.append(
                {
                    "mask_id": row["mask_id"],
                    "row_slot": row["row_slot"],
                    "arm": arm,
                    **values,
                }
            )
        print(f"sample {index}/{len(rows)} complete", flush=True)

    with (output / "per_sample.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {name: [] for name in metrics}
    )
    for row in output_rows:
        for name in metrics:
            grouped[row["arm"]][name].append(row[name])
    summary = {
        "status": "complete",
        "purpose": "training-free development smoke test",
        "official_test_accessed": False,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "selected_masks": selected_masks,
        "metrics": {
            arm: {name: float(np.mean(values[name])) for name in metrics}
            for arm, values in grouped.items()
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
