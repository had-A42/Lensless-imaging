"""Calibrate and evaluate the frozen matched-backbone OHUF large study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ohuf_bank_selection_screen import (  # noqa: E402
    load_psf_pool,
    load_rows,
    ohuf_fuse,
    zero_shift_psf,
)
from scripts.psf_estimator_cascade_smoke import (  # noqa: E402
    DATASET_REPO,
    DATASET_REVISION,
    ROI,
)
from src.model.operator_uncertainty_fusion import (  # noqa: E402
    normalize_nonnegative_max,
)

BASE_BANK = (45, 93, 17, 63)
PREFIX_K8 = (45, 93, 17, 63, 49, 81, 25, 76)
OTF_KCENTER_K4 = (33, 95, 29, 79)
SHIFTS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALPHAS = (0.25, 0.5, 0.75, 1.0)
METRICS = ("PSNR", "SSIM", "LPIPS")
PSF_BUNDLE = Path("data/ref_psff_real/prepared_psfs_v1.npz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def phase_protocol(phase: str, outer_masks: list[int]) -> tuple[list[int], list[int]]:
    if phase == "calibration":
        return [outer_masks[0]], list(range(225, 233))
    if phase == "confirmation":
        return outer_masks[1:], list(range(233, 250))
    raise ValueError(f"unknown phase: {phase}")


def build_psf_specs(
    pool_masks: list[int], pool: torch.Tensor
) -> dict[str, torch.Tensor]:
    by_mask = {mask_id: pool[index] for index, mask_id in enumerate(pool_masks)}
    required = set(BASE_BANK) | set(PREFIX_K8) | set(OTF_KCENTER_K4)
    if not required <= set(by_mask):
        raise ValueError("PSF bundle does not contain every frozen bank member")
    specs = {f"mask_{mask_id}": by_mask[mask_id] for mask_id in sorted(required)}
    for mask_id in BASE_BANK:
        shifted = []
        for dy, dx in SHIFTS:
            value = zero_shift_psf(by_mask[mask_id], dy, dx)
            specs[f"mask_{mask_id}_dy{dy:+d}_dx{dx:+d}"] = value
            shifted.append(value)
        jittered = torch.stack([by_mask[mask_id], *shifted]).mean(dim=0)
        jittered = jittered / jittered.square().sum().sqrt().clamp_min(1e-12)
        specs[f"mask_{mask_id}_jitter_mean"] = jittered
    return specs


def assemble_methods(predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    def stack_masks(mask_ids):
        return torch.stack([predictions[f"mask_{mask_id}"] for mask_id in mask_ids])

    methods = {
        "prefix_k4": stack_masks(BASE_BANK),
        "prefix_k8": stack_masks(PREFIX_K8),
        "otf_kcenter_k4": stack_masks(OTF_KCENTER_K4),
        "jitter_psf_k4": torch.stack(
            [predictions[f"mask_{mask_id}_jitter_mean"] for mask_id in BASE_BANK]
        ),
    }
    grouped = []
    axis_shift = []
    for mask_id, (axis_dy, axis_dx) in zip(BASE_BANK, SHIFTS):
        center = predictions[f"mask_{mask_id}"]
        group = [center]
        for dy, dx in SHIFTS:
            group.append(predictions[f"mask_{mask_id}_dy{dy:+d}_dx{dx:+d}"])
        grouped.append(torch.stack(group).mean(dim=0))
        axis_shift.extend(
            [center, predictions[f"mask_{mask_id}_dy{axis_dy:+d}_dx{axis_dx:+d}"]]
        )
    methods["grouped_jitter_k4"] = torch.stack(grouped)
    methods["axis_shift_k8"] = torch.stack(axis_shift)
    return methods


def choose_alphas(rows: list[dict], methods: list[str]) -> dict[str, float]:
    selected = {}
    for method in methods:
        candidates = []
        for alpha in ALPHAS:
            values = [
                row["PSNR"]
                for row in rows
                if row["method"] == method and row["alpha"] == alpha
            ]
            if not values:
                raise ValueError(
                    f"missing calibration rows for {method}, alpha={alpha}"
                )
            candidates.append((float(np.mean(values)), -alpha, alpha))
        selected[method] = float(max(candidates)[2])
    return selected


def t_interval(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) == 1:
        return {"mean": mean, "low": None, "high": None}
    margin = float(
        t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array))
    )
    return {"mean": mean, "low": mean - margin, "high": mean + margin}


def oriented_gain(metric: str, baseline: float, candidate: float) -> float:
    return baseline - candidate if metric == "LPIPS" else candidate - baseline


def paired_contrast(
    by_method: dict[str, list[dict]], candidate: str, reference: str
) -> dict:
    reference_rows = {
        (row["sample_id"], row["mask_id"]): row for row in by_method[reference]
    }
    grouped = defaultdict(lambda: defaultdict(list))
    image_wins = {metric: 0 for metric in METRICS}
    for row in by_method[candidate]:
        base = reference_rows[(row["sample_id"], row["mask_id"])]
        for metric in METRICS:
            value = oriented_gain(metric, base[metric], row[metric])
            grouped[row["mask_id"]][metric].append(value)
            image_wins[metric] += value > 0
    mask_values = {
        metric: [
            float(np.mean(grouped[mask_id][metric])) for mask_id in sorted(grouped)
        ]
        for metric in METRICS
    }
    return {
        "candidate": candidate,
        "reference": reference,
        "gain": {metric: t_interval(mask_values[metric]) for metric in METRICS},
        "mask_wins": {
            metric: int(sum(value > 0 for value in mask_values[metric]))
            for metric in METRICS
        },
        "image_wins": image_wins,
    }


def summarize(rows: list[dict]) -> dict:
    alphas_by_method = defaultdict(set)
    for row in rows:
        alphas_by_method[row["method"]].add(float(row["alpha"]))
    by_method = defaultdict(list)
    for row in rows:
        method = row["method"]
        if method != "baseline" and len(alphas_by_method[method]) > 1:
            method = f"{method}_a{float(row['alpha']):g}"
        by_method[method].append(row)
    metrics = {
        method: {
            metric: float(np.mean([row[metric] for row in method_rows]))
            for metric in METRICS
        }
        for method, method_rows in sorted(by_method.items())
    }
    contrasts = {}
    if "baseline" in by_method:
        for method in sorted(set(by_method) - {"baseline"}):
            contrasts[f"{method}_vs_baseline"] = paired_contrast(
                by_method, method, "baseline"
            )
    if "prefix_k4" in by_method:
        for method in ("grouped_jitter_k4", "jitter_psf_k4", "otf_kcenter_k4"):
            if method in by_method:
                contrasts[f"{method}_vs_prefix_k4"] = paired_contrast(
                    by_method, method, "prefix_k4"
                )
    if "prefix_k8" in by_method and "axis_shift_k8" in by_method:
        contrasts["axis_shift_k8_vs_prefix_k8"] = paired_contrast(
            by_method, "axis_shift_k8", "prefix_k8"
        )
    return {"metrics": metrics, "contrasts": contrasts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("calibration", "confirmation"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--proposal-checkpoint", type=Path, required=True)
    parser.add_argument("--aware-checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--fusion-kernel", type=int, default=65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU")
    if args.phase == "confirmation" and args.selection is None:
        raise ValueError("confirmation requires --selection")
    if args.phase == "calibration" and args.selection is not None:
        raise ValueError("calibration does not accept --selection")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache/matplotlib"))
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data/huggingface"))
    from datasets import load_dataset

    from src.digicam_protocol import build_digicam_mask_split
    from src.metrics.reconstruction import LPIPSMetric, PSNRMetric, SSIMMetric
    from src.model.psf_aware_drunet import PSFAwareDRUNet
    from src.model.psf_free_drunet import PSFFreeDRUNet

    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_state.json", {"status": "running"})
    started = time.monotonic()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)

    train_masks, outer_masks = build_digicam_mask_split()
    evaluation_masks, row_slots = phase_protocol(args.phase, outer_masks)
    psf_path = REPO_ROOT / PSF_BUNDLE
    pool, _ = load_psf_pool(train_masks, psf_path)
    specs = {
        name: value.to(device)
        for name, value in build_psf_specs(train_masks, pool).items()
    }

    source_dataset = load_dataset(
        DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        cache_dir=os.environ["HF_HOME"],
    )
    rows = load_rows(source_dataset, evaluation_masks, row_slots)
    proposal_checkpoint = (REPO_ROOT / args.proposal_checkpoint).resolve()
    aware_checkpoint = (REPO_ROOT / args.aware_checkpoint).resolve()
    checkpoint_hashes = {
        "proposal": sha256(proposal_checkpoint),
        "aware": sha256(aware_checkpoint),
    }
    proposal_model = (
        PSFFreeDRUNet(
            checkpoint_path=proposal_checkpoint,
            output_crop=list(ROI),
        )
        .to(device)
        .eval()
    )
    aware_model = (
        PSFAwareDRUNet(
            checkpoint_path=aware_checkpoint,
            output_crop=list(ROI),
        )
        .to(device)
        .eval()
    )

    if args.phase == "calibration":
        selected_alphas = None
        alphas_by_method = None
    else:
        selection_path = (REPO_ROOT / args.selection).resolve()
        selection = json.loads(selection_path.read_text())
        if selection.get("status") != "frozen" or int(selection["seed"]) != args.seed:
            raise ValueError("selection artifact does not match confirmation seed")
        if selection.get("checkpoint_hashes") != checkpoint_hashes:
            raise ValueError("selection checkpoint hashes do not match confirmation")
        selected_alphas = {
            str(method): float(alpha)
            for method, alpha in selection["selected_alphas"].items()
        }
        alphas_by_method = selected_alphas

    metrics = {
        "PSNR": PSNRMetric(normalize_by_max=True),
        "SSIM": SSIMMetric(normalize_by_max=True),
        "LPIPS": LPIPSMetric(net_type="vgg", device="cuda", normalize_by_max=True),
    }
    output_rows = []
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        measurement = torch.stack([row["measurement"] for row in batch_rows]).to(device)
        target = torch.stack([row["target"] for row in batch_rows]).to(device)
        with torch.inference_mode():
            proposal = normalize_nonnegative_max(
                proposal_model(measurement=measurement)["prediction"].float()
            )
            predictions = {}
            for name, psf in specs.items():
                psfs = psf.unsqueeze(0).expand(len(batch_rows), -1, -1, -1)
                value = aware_model(measurement=measurement, psf=psfs)[
                    "prediction"
                ].float()
                predictions[name] = normalize_nonnegative_max(value)
            method_stacks = assemble_methods(predictions)
            arms = {("baseline", 0.0): proposal}
            for method, hypotheses in method_stacks.items():
                alphas = (
                    ALPHAS if alphas_by_method is None else (alphas_by_method[method],)
                )
                for alpha in alphas:
                    arms[(method, float(alpha))] = ohuf_fuse(
                        proposal,
                        hypotheses,
                        beta=args.beta,
                        kernel=args.fusion_kernel,
                        alpha=alpha,
                    )

            arm_keys = list(arms)
            metric_values = {key: {} for key in arm_keys}
            for metric_name, metric in metrics.items():
                for arm_start in range(0, len(arm_keys), 4):
                    keys = arm_keys[arm_start : arm_start + 4]
                    prediction_batch = torch.cat([arms[key] for key in keys])
                    target_batch = target.repeat(len(keys), 1, 1, 1)
                    values = metric.per_image(prediction_batch, target_batch)
                    values = values.reshape(len(keys), len(batch_rows)).cpu()
                    for index, key in enumerate(keys):
                        metric_values[key][metric_name] = values[index].tolist()

        for method, alpha in arm_keys:
            for index, row in enumerate(batch_rows):
                output_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "mask_id": row["mask_id"],
                        "row_slot": row["row_slot"],
                        "method": method,
                        "alpha": alpha,
                        **{
                            metric: metric_values[(method, alpha)][metric][index]
                            for metric in METRICS
                        },
                    }
                )
        print(f"{args.phase}: {start + len(batch_rows)}/{len(rows)}", flush=True)

    with (output / "per_sample.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    result = summarize(output_rows)
    summary = {
        "status": "complete",
        "phase": args.phase,
        "seed": args.seed,
        "official_test_accessed": False,
        "target_used_for_fusion": False,
        "true_psf_used_for_fusion": False,
        "target_used_for_alpha_selection": args.phase == "calibration",
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "split": "train",
            "mask_ids": evaluation_masks,
            "row_slots": row_slots,
            "sample_count": len(rows),
        },
        "configuration": {
            "base_bank": list(BASE_BANK),
            "prefix_k8": list(PREFIX_K8),
            "otf_kcenter_k4": list(OTF_KCENTER_K4),
            "shifts": [list(value) for value in SHIFTS],
            "alpha_grid": list(ALPHAS) if args.phase == "calibration" else None,
            "selected_alphas": selected_alphas,
            "beta": args.beta,
            "fusion_kernel": args.fusion_kernel,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hostname": platform.node(),
        },
        "inputs": {
            "proposal_checkpoint": str(proposal_checkpoint),
            "aware_checkpoint": str(aware_checkpoint),
            "checkpoint_hashes": checkpoint_hashes,
            "psf_bundle": str(psf_path.resolve()),
            "psf_bundle_sha256": sha256(psf_path),
        },
        **result,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(output / "summary.json", summary)

    if args.phase == "calibration":
        methods = sorted({row["method"] for row in output_rows} - {"baseline"})
        chosen = choose_alphas(output_rows, methods)
        selection = {
            "status": "frozen",
            "seed": args.seed,
            "criterion": "maximum mean PSNR on fusion-calibration mask 41; ties choose smaller alpha",
            "mask_ids": evaluation_masks,
            "row_slots": row_slots,
            "selected_alphas": chosen,
            "checkpoint_hashes": checkpoint_hashes,
            "calibration_summary": str((output / "summary.json").resolve()),
            "calibration_summary_sha256": sha256(output / "summary.json"),
        }
        write_json(output / "selection.json", selection)

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
