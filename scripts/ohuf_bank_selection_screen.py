"""Development screen for fixed, selected, and generated OHUF PSF banks.

The script only reads the upstream DigiCam training split.  The 68 calibration
PSFs form the candidate pool, while the 16 gate masks and their true PSFs are
unavailable to every selector.  Targets are consumed only after reconstruction
to compute metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.extract_psf_diversity_features import fourier_features  # noqa: E402
from scripts.psf_estimator_cascade_smoke import (  # noqa: E402
    DATASET_REPO,
    DATASET_REVISION,
    MODEL_REPO,
    MODEL_REVISION,
    ROI,
    TARGET_SIZE,
    source_index,
)
from src.model.operator_uncertainty_fusion import (  # noqa: E402
    normalize_nonnegative_max,
    reflected_box_blur,
)

METRICS = ("PSNR", "SSIM", "LPIPS")
CHECKPOINT = Path("data/models/digicam-mirflickr-multi-25k-unet8M/recon_epochBEST")
PSF_BUNDLE = Path("data/ref_psff_real/prepared_psfs_v1.npz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalize_psf(psf: Tensor) -> Tensor:
    psf = psf.float().clamp_min(0)
    norm = psf.square().sum().sqrt()
    if norm.item() <= 0:
        raise ValueError("PSF has no positive energy")
    return psf / norm


def zero_shift_psf(psf: Tensor, dy: int, dx: int) -> Tensor:
    """Translate a CHW PSF without circular wraparound."""

    if psf.ndim != 3:
        raise ValueError("PSF must be CHW")
    height, width = psf.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError("shift must be smaller than the PSF support")
    output = torch.zeros_like(psf)
    src_top, dst_top = max(0, -dy), max(0, dy)
    src_left, dst_left = max(0, -dx), max(0, dx)
    extent_y = height - abs(dy)
    extent_x = width - abs(dx)
    output[:, dst_top : dst_top + extent_y, dst_left : dst_left + extent_x] = psf[
        :, src_top : src_top + extent_y, src_left : src_left + extent_x
    ]
    return normalize_psf(output)


def greedy_kcenter(features: np.ndarray, count: int) -> list[int]:
    """Deterministic k-center indices, initialized by the global medoid proxy."""

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("features must be a finite matrix")
    if not 1 <= count <= len(features):
        raise ValueError("invalid k-center count")
    center = features.mean(axis=0)
    selected = [int(np.argmin(np.sum((features - center) ** 2, axis=1)))]
    minimum_distance = np.sum((features - features[selected[0]]) ** 2, axis=1)
    while len(selected) < count:
        minimum_distance[selected] = -1
        index = int(np.argmax(minimum_distance))
        selected.append(index)
        distance = np.sum((features - features[index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    return selected


def select_hypotheses(stack: Tensor, indices: Tensor) -> Tensor:
    """Select K hypotheses per image from a [H,B,C,Y,X] stack."""

    if stack.ndim != 5 or indices.ndim != 2:
        raise ValueError("invalid hypothesis selection shapes")
    hypotheses, batch = stack.shape[:2]
    if indices.shape[0] != batch:
        raise ValueError("selection batch does not match hypotheses")
    if indices.numel() and (indices.min() < 0 or indices.max() >= hypotheses):
        raise ValueError("hypothesis index is out of range")
    by_image = stack.permute(1, 0, 2, 3, 4)
    gather_index = indices[..., None, None, None].expand(
        batch,
        indices.shape[1],
        stack.shape[2],
        stack.shape[3],
        stack.shape[4],
    )
    return torch.gather(by_image, 1, gather_index).permute(1, 0, 2, 3, 4)


def ohuf_fuse(
    proposal: Tensor,
    hypotheses: Tensor,
    *,
    beta: float,
    kernel: int,
    alpha: float,
) -> Tensor:
    if hypotheses.ndim != 5 or hypotheses.shape[1:] != proposal.shape:
        raise ValueError("hypotheses must have [K,B,C,H,W] shape")
    if hypotheses.shape[0] < 2:
        raise ValueError("OHUF needs at least two hypotheses")
    marginal = hypotheses.mean(dim=0)
    variance = hypotheses.var(dim=0, correction=1).mean(dim=1, keepdim=True)
    scale = variance.mean(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    confidence = torch.exp(-float(beta) * variance / scale)
    guide = normalize_nonnegative_max(proposal + confidence * (marginal - proposal))
    correction = reflected_box_blur(guide - proposal, int(kernel))
    return proposal + float(alpha) * correction


def low_frequency_scores(stack: Tensor, reference: Tensor, kernel: int) -> Tensor:
    if stack.shape[1:] != reference.shape:
        raise ValueError("reference shape mismatch")
    residual = stack - reference.unsqueeze(0)
    low = reflected_box_blur(residual.flatten(0, 1), kernel).reshape_as(residual)
    return low.square().mean(dim=(2, 3, 4))


def load_psf_pool(mask_ids: list[int], path: Path) -> tuple[Tensor, np.ndarray]:
    psfs = []
    features = []
    with np.load(path, allow_pickle=False) as bundle:
        for mask_id in mask_ids:
            key = f"mask_{mask_id}"
            if key not in bundle:
                raise KeyError(f"{key} is missing from {path}")
            array = np.asarray(bundle[key], dtype=np.float32)
            psf = torch.from_numpy(array).squeeze(0).movedim(-1, 0).contiguous()
            psfs.append(normalize_psf(psf))
            feature, _, _ = fourier_features(
                array,
                feature_side=8,
                radial_bands=12,
            )
            features.append(feature)
    return torch.stack(psfs), np.stack(features)


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
        indices=[identity[0] for identity in identities],
        force_rgb=True,
        rotate_measurement=True,
        measurement_downsample=1.0,
        target_size=list(TARGET_SIZE),
        target_resize_mode="bilinear",
        return_psf=False,
    )
    rows = []
    for identity, sample in zip(identities, dataset):
        _, mask_id, row_slot = identity
        if int(sample["mask_id"]) != mask_id:
            raise ValueError("upstream row identity mismatch")
        rows.append(
            {
                "sample_id": f"mask{mask_id}-row{row_slot}",
                "mask_id": mask_id,
                "row_slot": row_slot,
                "measurement": sample["measurement"],
                "target": sample["target"],
            }
        )
    return rows


def t_interval(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) == 1:
        return {"mean": mean, "low": None, "high": None}
    margin = float(
        t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array))
    )
    return {"mean": mean, "low": mean - margin, "high": mean + margin}


def gain(metric: str, baseline: float, candidate: float) -> float:
    return baseline - candidate if metric == "LPIPS" else candidate - baseline


def summarize(rows: list[dict], banks: dict, selection_counts: dict) -> dict:
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    if "baseline" not in by_arm:
        raise ValueError("baseline rows are missing")
    baseline = {(row["sample_id"], row["mask_id"]): row for row in by_arm["baseline"]}
    arms = {}
    for arm, arm_rows in sorted(by_arm.items()):
        means = {
            metric: float(np.mean([row[metric] for row in arm_rows]))
            for metric in METRICS
        }
        arm_summary = {"metrics": means, "sample_count": len(arm_rows)}
        if arm != "baseline":
            grouped = defaultdict(lambda: defaultdict(list))
            image_wins = {metric: 0 for metric in METRICS}
            for row in arm_rows:
                base = baseline[(row["sample_id"], row["mask_id"])]
                for metric in METRICS:
                    value = gain(metric, base[metric], row[metric])
                    grouped[row["mask_id"]][metric].append(value)
                    image_wins[metric] += value > 0
            mask_means = {
                metric: [
                    float(np.mean(grouped[mask_id][metric]))
                    for mask_id in sorted(grouped)
                ]
                for metric in METRICS
            }
            arm_summary.update(
                {
                    "gain": {
                        metric: t_interval(mask_means[metric]) for metric in METRICS
                    },
                    "mask_wins": {
                        metric: int(sum(value > 0 for value in mask_means[metric]))
                        for metric in METRICS
                    },
                    "image_wins": image_wins,
                }
            )
        if arm in banks:
            arm_summary["bank"] = banks[arm]
        if arm in selection_counts:
            arm_summary["selected_mask_histogram"] = dict(
                sorted(selection_counts[arm].items())
            )
        arms[arm] = arm_summary

    random_distribution = {}
    for count in (4, 8):
        names = [
            name
            for name in arms
            if name.startswith("random_r") and name.endswith(f"_k{count}")
        ]
        if not names:
            continue
        random_distribution[f"k{count}"] = {
            metric: {
                "mean": float(
                    np.mean([arms[name]["gain"][metric]["mean"] for name in names])
                ),
                "standard_deviation": (
                    float(
                        np.std(
                            [arms[name]["gain"][metric]["mean"] for name in names],
                            ddof=1,
                        )
                    )
                    if len(names) > 1
                    else None
                ),
                "minimum": float(
                    min(arms[name]["gain"][metric]["mean"] for name in names)
                ),
                "maximum": float(
                    max(arms[name]["gain"][metric]["mean"] for name in names)
                ),
            }
            for metric in METRICS
        }
    return {"arms": arms, "random_bank_distribution": random_distribution}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-start", type=int, default=1)
    parser.add_argument("--mask-count", type=int, default=16)
    parser.add_argument("--row-start", type=int, default=120)
    parser.add_argument("--scenes-per-mask", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pool-limit", type=int, default=68)
    parser.add_argument("--random-banks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--fusion-kernel", type=int, default=65)
    parser.add_argument("--fusion-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This screen requires one visible CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU through CUDA_VISIBLE_DEVICES")
    if args.pool_limit < 8:
        raise ValueError("pool-limit must be at least 8")
    if args.random_banks < 1:
        raise ValueError("random-banks must be positive")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data/huggingface"))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache/matplotlib"))
    from datasets import load_dataset

    from src.digicam_protocol import build_digicam_mask_split
    from src.metrics.reconstruction import LPIPSMetric, PSNRMetric, SSIMMetric
    from src.model.psf_aware_lensless import PSFAwareLenslessModel
    from src.model.psf_free_drunet import PSFFreeDRUNet

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_state.json", {"status": "running"})
    started = time.monotonic()

    train_masks, held_out_masks = build_digicam_mask_split()
    pool_masks = train_masks[: args.pool_limit]
    evaluation_masks = held_out_masks[
        args.mask_start : args.mask_start + args.mask_count
    ]
    if len(evaluation_masks) != args.mask_count:
        raise ValueError("requested mask range exceeds held-out development split")
    if set(pool_masks) & set(evaluation_masks):
        raise ValueError("candidate and evaluation masks overlap")
    psf_path = REPO_ROOT / PSF_BUNDLE
    pool_cpu, features = load_psf_pool(pool_masks, psf_path)
    kcenter_order = greedy_kcenter(features, 8)

    banks = {
        "prefix_k4": pool_masks[:4],
        "prefix_k8": pool_masks[:8],
        "otf_kcenter_k4": [pool_masks[index] for index in kcenter_order[:4]],
        "otf_kcenter_k8": [pool_masks[index] for index in kcenter_order],
    }
    rng = np.random.default_rng(args.seed)
    random_indices = []
    for repeat in range(args.random_banks):
        order = rng.permutation(len(pool_masks)).tolist()
        random_indices.append(order)
        for count in (4, 8):
            banks[f"random_r{repeat:02d}_k{count}"] = [
                pool_masks[index] for index in order[:count]
            ]

    shift_specs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    banks["generated_axis_shift_k8"] = [
        *pool_masks[:4],
        *[
            f"mask_{mask_id}_shift_dy{dy:+d}_dx{dx:+d}"
            for mask_id, (dy, dx) in zip(pool_masks[:4], shift_specs)
        ],
    ]

    source_dataset = load_dataset(
        DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        cache_dir=os.environ["HF_HOME"],
    )
    rows = load_rows(
        source_dataset,
        evaluation_masks,
        list(range(args.row_start, args.row_start + args.scenes_per_mask)),
    )

    proposal_model = (
        PSFFreeDRUNet(
            checkpoint_path=REPO_ROOT / CHECKPOINT,
            output_crop=list(ROI),
        )
        .to(device)
        .eval()
    )
    aware_model = (
        PSFAwareLenslessModel(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            cache_dir=os.environ["HF_HOME"],
            output_crop=list(ROI),
        )
        .to(device)
        .eval()
    )
    pool = pool_cpu.to(device)
    shifted = torch.stack(
        [
            zero_shift_psf(pool_cpu[index], dy, dx)
            for index, (dy, dx) in enumerate(shift_specs)
        ]
    ).to(device)
    metrics = {
        "PSNR": PSNRMetric(normalize_by_max=True),
        "SSIM": SSIMMetric(normalize_by_max=True),
        "LPIPS": LPIPSMetric(net_type="vgg", device="cuda", normalize_by_max=True),
    }

    output_rows = []
    selection_counts = defaultdict(Counter)
    index_by_mask = {mask_id: index for index, mask_id in enumerate(pool_masks)}
    fixed_indices = {
        arm: [index_by_mask[mask_id] for mask_id in mask_ids]
        for arm, mask_ids in banks.items()
        if arm != "generated_axis_shift_k8"
    }
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        measurement = torch.stack([row["measurement"] for row in batch_rows]).to(device)
        target = torch.stack([row["target"] for row in batch_rows]).to(device)
        with torch.inference_mode():
            proposal = normalize_nonnegative_max(
                proposal_model(measurement=measurement)["prediction"].float()
            )
            candidates = []
            for psf in pool:
                psfs = psf.unsqueeze(0).expand(len(batch_rows), -1, -1, -1)
                prediction = aware_model(measurement=measurement, psf=psfs)[
                    "prediction"
                ].float()
                candidates.append(normalize_nonnegative_max(prediction))
            stack = torch.stack(candidates)

            arms = {"baseline": proposal}
            for arm, indices in fixed_indices.items():
                subset = stack[indices]
                arms[arm] = ohuf_fuse(
                    proposal,
                    subset,
                    beta=args.beta,
                    kernel=args.fusion_kernel,
                    alpha=args.fusion_alpha,
                )

            proposal_scores = low_frequency_scores(stack, proposal, args.fusion_kernel)
            consensus = stack.mean(dim=0)
            consensus_scores = low_frequency_scores(
                stack, consensus, args.fusion_kernel
            )
            for selector_name, scores in (
                ("conditional_proposal", proposal_scores),
                ("conditional_consensus", consensus_scores),
            ):
                for count in (4, 8):
                    arm = f"{selector_name}_k{count}"
                    indices = scores.topk(count, dim=0, largest=False).indices.T
                    subset = select_hypotheses(stack, indices)
                    arms[arm] = ohuf_fuse(
                        proposal,
                        subset,
                        beta=args.beta,
                        kernel=args.fusion_kernel,
                        alpha=args.fusion_alpha,
                    )
                    for image_indices in indices.detach().cpu().tolist():
                        selection_counts[arm].update(
                            pool_masks[index] for index in image_indices
                        )

            shifted_candidates = []
            for psf in shifted:
                psfs = psf.unsqueeze(0).expand(len(batch_rows), -1, -1, -1)
                prediction = aware_model(measurement=measurement, psf=psfs)[
                    "prediction"
                ].float()
                shifted_candidates.append(normalize_nonnegative_max(prediction))
            generated_stack = torch.cat(
                (stack[:4], torch.stack(shifted_candidates)), dim=0
            )
            arms["generated_axis_shift_k8"] = ohuf_fuse(
                proposal,
                generated_stack,
                beta=args.beta,
                kernel=args.fusion_kernel,
                alpha=args.fusion_alpha,
            )

            arm_names = list(arms)
            values_by_arm = {name: {} for name in arm_names}
            for metric_name, metric in metrics.items():
                for arm_start in range(0, len(arm_names), 4):
                    names = arm_names[arm_start : arm_start + 4]
                    prediction_batch = torch.cat([arms[name] for name in names])
                    target_batch = target.repeat(len(names), 1, 1, 1)
                    values = metric.per_image(prediction_batch, target_batch)
                    values = values.reshape(len(names), len(batch_rows)).cpu()
                    for index, name in enumerate(names):
                        values_by_arm[name][metric_name] = values[index].tolist()

        for arm in arm_names:
            for index, row in enumerate(batch_rows):
                output_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "mask_id": row["mask_id"],
                        "row_slot": row["row_slot"],
                        "arm": arm,
                        **{
                            metric: values_by_arm[arm][metric][index]
                            for metric in METRICS
                        },
                    }
                )
        print(f"samples {start + len(batch_rows)}/{len(rows)} complete", flush=True)

    with (output / "per_sample.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    result = summarize(output_rows, banks, selection_counts)
    summary = {
        "status": "complete",
        "purpose": "post-hoc development screen, not a final benchmark",
        "official_test_accessed": False,
        "target_used_for_selection": False,
        "true_psf_used_for_selection_or_reconstruction": False,
        "published_reconstructor_unseen_masks": False,
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "split": "train",
            "calibration_psf_masks": pool_masks,
            "evaluation_masks": evaluation_masks,
            "row_slots": list(
                range(args.row_start, args.row_start + args.scenes_per_mask)
            ),
            "sample_count": len(rows),
        },
        "configuration": vars(args)
        | {
            "output": str(output),
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hostname": platform.node(),
            "otf_feature": "8x8 pooled log magnitude, DC removed, L2 normalized",
            "generated_shift_specs": shift_specs,
        },
        "inputs": {
            "checkpoint": str((REPO_ROOT / CHECKPOINT).resolve()),
            "checkpoint_sha256": sha256(REPO_ROOT / CHECKPOINT),
            "psf_bundle": str(psf_path.resolve()),
            "psf_bundle_sha256": sha256(psf_path),
        },
        **result,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "run_state.json",
        {
            "status": "complete",
            "summary": str(output / "summary.json"),
            "elapsed_seconds": summary["elapsed_seconds"],
        },
    )
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
