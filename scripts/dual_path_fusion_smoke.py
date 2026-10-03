"""Fuse measurement-only and mean-PSF reconstruction paths.

The measurement-only path is calibration-free but learns the inverse only
implicitly.  The second path uses the population mean of training PSFs, which
is intentionally not the true test-mask PSF.  This development smoke test
checks whether their errors are complementary under simple convex and
frequency-split fusion.  Only the upstream DigiCam train split is read.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.psf_estimator_cascade_smoke import (  # noqa: E402
    DATASET_REPO,
    DATASET_REVISION,
    MODEL_REPO,
    MODEL_REVISION,
    ROI,
    TARGET_SIZE,
    load_psfs,
    normalize_l2,
    source_index,
    write_json,
)
from src.model.operator_uncertainty_fusion import (  # noqa: E402
    normalize_nonnegative_max as independent_max_normalize,
    reflected_box_blur,
)

CHECKPOINT = Path("data/models/digicam-mirflickr-multi-25k-unet8M/recon_epochBEST")


def load_rows(source_dataset, mask_ids: list[int], slots: list[int]) -> list[dict]:
    from src.datasets.digicam import DigiCamRealDataset

    identities = [
        (source_index(mask_id, slot), mask_id, slot)
        for mask_id in mask_ids
        for slot in slots
    ]
    dataset = DigiCamRealDataset(
        repo_id=DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        source_dataset=source_dataset,
        indices=[row[0] for row in identities],
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


def fusion_arms(
    free: Tensor,
    physics: Tensor,
    confirmation: bool = False,
    fusion_alpha: float = 0.05,
    fusion_kind: str = "convex",
    fusion_kernel: int = 33,
) -> dict[str, Tensor]:
    free = independent_max_normalize(free)
    physics = independent_max_normalize(physics)
    arms = {"psf_free": free, "physics_path": physics}
    if confirmation:
        if fusion_kind == "convex":
            arms[f"convex_a{fusion_alpha:g}"] = (
                1 - fusion_alpha
            ) * free + fusion_alpha * physics
        elif fusion_kind == "lowfreq":
            low = reflected_box_blur(physics - free, fusion_kernel)
            arms[f"lowfreq_k{fusion_kernel}_a{fusion_alpha:g}"] = (
                free + fusion_alpha * low
            )
        elif fusion_kind == "luma_lowfreq":
            difference = physics - free
            luma = (
                0.299 * difference[:, 0:1]
                + 0.587 * difference[:, 1:2]
                + 0.114 * difference[:, 2:3]
            )
            low = reflected_box_blur(luma, fusion_kernel)
            arms[f"luma_lowfreq_k{fusion_kernel}_a{fusion_alpha:g}"] = (
                free + fusion_alpha * low
            )
        else:
            raise ValueError(f"unknown fusion kind: {fusion_kind}")
        return arms
    for alpha in (0.02, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 0.75, 0.9):
        arms[f"convex_a{alpha:g}"] = (1 - alpha) * free + alpha * physics
    difference = physics - free
    luma_difference = (
        0.299 * difference[:, 0:1]
        + 0.587 * difference[:, 1:2]
        + 0.114 * difference[:, 2:3]
    )
    for kernel in (5, 17, 33, 65):
        low = reflected_box_blur(difference, kernel)
        luma_low = reflected_box_blur(luma_difference, kernel)
        for alpha in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
            arms[f"lowfreq_k{kernel}_a{alpha:g}"] = free + alpha * low
            arms[f"luma_lowfreq_k{kernel}_a{alpha:g}"] = free + alpha * luma_low
    return arms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-start", type=int, default=0)
    parser.add_argument("--mask-count", type=int, default=1)
    parser.add_argument("--row-start", type=int, default=100)
    parser.add_argument("--scene-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--fusion-alpha", type=float, default=0.05)
    parser.add_argument(
        "--fusion-kind",
        choices=("convex", "lowfreq", "luma_lowfreq"),
        default="convex",
    )
    parser.add_argument("--fusion-kernel", type=int, default=33)
    parser.add_argument(
        "--physics-mode", choices=("mean_psf", "marginal"), default="mean_psf"
    )
    parser.add_argument("--marginal-psfs", type=int, default=8)
    parser.add_argument(
        "--marginal-aggregate",
        choices=(
            "mean",
            "median",
            "trimmed",
            "consensus2",
            "hf_consensus2",
            "uncertainty4",
            "all",
        ),
        default="mean",
    )
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
    from src.model.psf_aware_lensless import PSFAwareLenslessModel
    from src.model.psf_free_drunet import PSFFreeDRUNet

    torch.set_num_threads(4)
    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    train_masks, held_out_masks = build_digicam_mask_split()
    selected_masks = held_out_masks[args.mask_start : args.mask_start + args.mask_count]
    slots = list(range(args.row_start, args.row_start + args.scene_count))
    source_dataset = load_dataset(
        DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        cache_dir=str(REPO_ROOT / "data/huggingface"),
    )
    rows = load_rows(source_dataset, selected_masks, slots)
    all_psfs = load_psfs(
        train_masks,
        held_out_masks,
        REPO_ROOT / "data/ref_psff_real/prepared_psfs_v1.npz",
        REPO_ROOT / "data/hf/DigiCam-Mirflickr-MultiMask-1K/masks",
        REPO_ROOT / "src/configs/simulator/digicam_article.yaml",
        REPO_ROOT / "outputs/psf_estimator_smoke_cache/outer17_psfs.npz",
    )
    population_psf = normalize_l2(
        torch.stack([all_psfs[mask_id] for mask_id in train_masks]).mean(dim=0)
    )
    free_model = PSFFreeDRUNet(
        checkpoint_path=REPO_ROOT / CHECKPOINT,
        output_crop=list(ROI),
    ).eval()
    aware_model = PSFAwareLenslessModel(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        cache_dir="data/huggingface",
        output_crop=list(ROI),
    ).eval()
    metrics = {
        "PSNR": PSNRMetric(normalize_by_max=True),
        "SSIM": SSIMMetric(normalize_by_max=True),
        "LPIPS": LPIPSMetric(net_type="vgg", device="cpu", normalize_by_max=True),
    }
    output_rows = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        measurement = torch.stack([row["measurement"] for row in batch_rows])
        target = torch.stack([row["target"] for row in batch_rows])
        with torch.inference_mode():
            free_prediction = free_model(measurement=measurement)["prediction"].float()
            if args.physics_mode == "mean_psf":
                psfs = population_psf.unsqueeze(0).expand(len(batch_rows), -1, -1, -1)
                physics_prediction = aware_model(measurement=measurement, psf=psfs)[
                    "prediction"
                ].float()
            else:
                if not 1 <= args.marginal_psfs <= len(train_masks):
                    raise ValueError("marginal-psfs exceeds the train PSF bank")
                samples = []
                for mask_id in train_masks[: args.marginal_psfs]:
                    psfs = (
                        all_psfs[mask_id]
                        .unsqueeze(0)
                        .expand(len(batch_rows), -1, -1, -1)
                    )
                    sample = aware_model(measurement=measurement, psf=psfs)[
                        "prediction"
                    ].float()
                    samples.append(independent_max_normalize(sample))
                stack = torch.stack(samples)
                physics_candidates = {
                    "mean": stack.mean(dim=0),
                    "median": stack.median(dim=0).values,
                }
                if args.marginal_psfs >= 3:
                    physics_candidates["trimmed"] = (
                        stack.sort(dim=0).values[1:-1].mean(dim=0)
                    )
                normalized_free = independent_max_normalize(free_prediction)

                def closest_two(candidate_stack, proposal):
                    distances = (candidate_stack - proposal.unsqueeze(0)).square()
                    distances = distances.mean(dim=(2, 3, 4))
                    indices = distances.topk(2, dim=0, largest=False).indices
                    index = indices[..., None, None, None].expand(
                        2,
                        stack.shape[1],
                        stack.shape[2],
                        stack.shape[3],
                        stack.shape[4],
                    )
                    return torch.gather(stack, 0, index).mean(dim=0)

                physics_candidates["consensus2"] = closest_two(stack, normalized_free)
                stack_high = stack - reflected_box_blur(
                    stack.flatten(0, 1), 33
                ).reshape_as(stack)
                free_high = normalized_free - reflected_box_blur(normalized_free, 33)
                physics_candidates["hf_consensus2"] = closest_two(stack_high, free_high)
                if args.marginal_psfs >= 2:
                    variance = stack.var(dim=0, correction=1).mean(dim=1, keepdim=True)
                    variance_scale = variance.mean(dim=(2, 3), keepdim=True).clamp_min(
                        1e-8
                    )
                    confidence = torch.exp(-4 * variance / variance_scale)
                    physics_candidates["uncertainty4"] = (
                        normalized_free
                        + confidence * (physics_candidates["mean"] - normalized_free)
                    )
                elif args.marginal_aggregate in {"uncertainty4", "all"}:
                    raise ValueError("uncertainty aggregation needs at least two PSFs")
                if args.marginal_aggregate == "all":
                    arms = {}
                    for aggregate, physics_prediction in physics_candidates.items():
                        partial = fusion_arms(
                            free_prediction,
                            physics_prediction,
                            confirmation=args.confirmation,
                            fusion_alpha=args.fusion_alpha,
                            fusion_kind=args.fusion_kind,
                            fusion_kernel=args.fusion_kernel,
                        )
                        for name, prediction in partial.items():
                            if name == "psf_free":
                                arms.setdefault(name, prediction)
                            else:
                                arms[f"{aggregate}_{name}"] = prediction
                else:
                    physics_prediction = physics_candidates[args.marginal_aggregate]
            if args.physics_mode == "mean_psf" or args.marginal_aggregate != "all":
                arms = fusion_arms(
                    free_prediction,
                    physics_prediction,
                    confirmation=args.confirmation,
                    fusion_alpha=args.fusion_alpha,
                    fusion_kind=args.fusion_kind,
                    fusion_kernel=args.fusion_kernel,
                )
            for arm, prediction in arms.items():
                values = {
                    name: metric.per_image(prediction, target).reshape(-1).tolist()
                    for name, metric in metrics.items()
                }
                for index, row in enumerate(batch_rows):
                    output_rows.append(
                        {
                            "mask_id": row["mask_id"],
                            "row_slot": row["row_slot"],
                            "arm": arm,
                            **{name: values[name][index] for name in metrics},
                        }
                    )
        print(f"rows {start + len(batch_rows)}/{len(rows)} complete", flush=True)

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
        "purpose": "development smoke test",
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
