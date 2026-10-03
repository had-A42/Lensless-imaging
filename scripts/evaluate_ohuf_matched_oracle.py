"""Post-hoc oracle diagnostic for the matched PSF-aware OHUF backbone.

This is not a calibration-free method: the ``true_psf`` arm deliberately uses
the held-out operator to measure whether the conditioned backbone has useful
PSF sensitivity.  The official DigiCam test split remains closed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ohuf_bank_selection_screen import (  # noqa: E402
    load_psf_pool,
    load_rows,
    normalize_psf,
)
from scripts.ohuf_matched_large_evaluate import (  # noqa: E402
    METRICS,
    PSF_BUNDLE,
    phase_protocol,
    sha256,
    summarize,
    write_json,
)
from scripts.psf_estimator_cascade_smoke import (  # noqa: E402
    DATASET_REPO,
    DATASET_REVISION,
    ROI,
)
from src.model.operator_uncertainty_fusion import (  # noqa: E402
    normalize_nonnegative_max,
)

SEEDS = (42, 52, 62)


def load_outer_psfs(path: Path, mask_ids: list[int]) -> dict[int, torch.Tensor]:
    result = {}
    with np.load(path, allow_pickle=False) as bundle:
        for mask_id in mask_ids:
            key = f"mask_{mask_id}"
            if key not in bundle:
                raise KeyError(f"{key} missing from outer PSF bundle")
            value = torch.from_numpy(np.asarray(bundle[key])).squeeze(0).movedim(-1, 0)
            result[mask_id] = normalize_psf(value.contiguous())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-psfs", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU")
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
    torch.set_num_threads(4)
    device = torch.device("cuda")

    train_masks, outer_masks = build_digicam_mask_split()
    evaluation_masks, row_slots = phase_protocol("confirmation", outer_masks)
    outer_path = (REPO_ROOT / args.outer_psfs).resolve()
    outer_psfs = load_outer_psfs(outer_path, outer_masks)
    train_pool, _ = load_psf_pool(train_masks, REPO_ROOT / PSF_BUNDLE)
    mean_train_psf = normalize_psf(train_pool.mean(dim=0)).to(device)
    source_dataset = load_dataset(
        DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        cache_dir=os.environ["HF_HOME"],
    )
    rows = load_rows(source_dataset, evaluation_masks, row_slots)
    metrics = {
        "PSNR": PSNRMetric(normalize_by_max=True),
        "SSIM": SSIMMetric(normalize_by_max=True),
        "LPIPS": LPIPSMetric(net_type="vgg", device="cuda", normalize_by_max=True),
    }

    seed_summaries = {}
    all_rows = []
    checkpoint_hashes = {}
    torch.cuda.reset_peak_memory_stats()
    for seed in SEEDS:
        proposal_path = REPO_ROOT / (
            f"saved/ohuf-matched-large-v1-psf-free-seed{seed}/model-state-epoch10.pth"
        )
        aware_path = REPO_ROOT / (
            f"saved/ohuf-matched-large-v1-psf-aware-seed{seed}/model-state-epoch10.pth"
        )
        checkpoint_hashes[str(seed)] = {
            "proposal": sha256(proposal_path),
            "aware": sha256(aware_path),
        }
        proposal_model = (
            PSFFreeDRUNet(checkpoint_path=proposal_path, output_crop=list(ROI))
            .to(device)
            .eval()
        )
        aware_model = (
            PSFAwareDRUNet(checkpoint_path=aware_path, output_crop=list(ROI))
            .to(device)
            .eval()
        )
        seed_rows = []
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            measurement = torch.stack([row["measurement"] for row in batch_rows]).to(
                device
            )
            target = torch.stack([row["target"] for row in batch_rows]).to(device)
            true_psf = torch.stack(
                [outer_psfs[row["mask_id"]] for row in batch_rows]
            ).to(device)
            wrong_psf = torch.stack(
                [
                    outer_psfs[
                        outer_masks[
                            (outer_masks.index(row["mask_id"]) + 1) % len(outer_masks)
                        ]
                    ]
                    for row in batch_rows
                ]
            ).to(device)
            mean_psf = mean_train_psf.unsqueeze(0).expand(len(batch_rows), -1, -1, -1)
            with torch.inference_mode():
                arms = {
                    "baseline": normalize_nonnegative_max(
                        proposal_model(measurement=measurement)["prediction"].float()
                    ),
                    "true_psf": normalize_nonnegative_max(
                        aware_model(measurement=measurement, psf=true_psf)[
                            "prediction"
                        ].float()
                    ),
                    "wrong_outer_psf": normalize_nonnegative_max(
                        aware_model(measurement=measurement, psf=wrong_psf)[
                            "prediction"
                        ].float()
                    ),
                    "mean_train_psf": normalize_nonnegative_max(
                        aware_model(measurement=measurement, psf=mean_psf)[
                            "prediction"
                        ].float()
                    ),
                }
                values = {arm: {} for arm in arms}
                prediction_batch = torch.cat(list(arms.values()))
                target_batch = target.repeat(len(arms), 1, 1, 1)
                for metric_name, metric in metrics.items():
                    metric_values = metric.per_image(prediction_batch, target_batch)
                    metric_values = metric_values.reshape(
                        len(arms), len(batch_rows)
                    ).cpu()
                    for index, arm in enumerate(arms):
                        values[arm][metric_name] = metric_values[index].tolist()
            for arm in arms:
                for index, row in enumerate(batch_rows):
                    seed_rows.append(
                        {
                            "sample_id": row["sample_id"],
                            "mask_id": row["mask_id"],
                            "row_slot": row["row_slot"],
                            "method": arm,
                            "alpha": 0.0,
                            **{
                                metric: values[arm][metric][index] for metric in METRICS
                            },
                        }
                    )
            print(f"seed {seed}: {start + len(batch_rows)}/{len(rows)}", flush=True)

        seed_dir = output / f"seed{seed}"
        seed_dir.mkdir()
        with (seed_dir / "per_sample.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
            writer.writeheader()
            writer.writerows(seed_rows)
        seed_summary = summarize(seed_rows)
        write_json(seed_dir / "summary.json", seed_summary)
        seed_summaries[str(seed)] = seed_summary
        all_rows.extend({"seed": seed, **row} for row in seed_rows)
        del proposal_model, aware_model
        torch.cuda.empty_cache()

    aggregate = {}
    for arm in ("true_psf", "wrong_outer_psf", "mean_train_psf"):
        aggregate[arm] = {}
        for metric in METRICS:
            values = [
                seed_summaries[str(seed)]["contrasts"][f"{arm}_vs_baseline"]["gain"][
                    metric
                ]["mean"]
                for seed in SEEDS
            ]
            aggregate[arm][metric] = {
                "per_seed": dict(zip(map(str, SEEDS), values)),
                "mean": float(np.mean(values)),
            }
    result = {
        "status": "complete",
        "purpose": "post-hoc oracle diagnostic, not a calibration-free result",
        "official_test_accessed": False,
        "true_psf_arm_is_oracle": True,
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "split": "train",
            "mask_ids": evaluation_masks,
            "row_slots": row_slots,
            "samples_per_seed": len(rows),
        },
        "outer_psf_bundle": str(outer_path),
        "outer_psf_bundle_sha256": sha256(outer_path),
        "checkpoint_hashes": checkpoint_hashes,
        "seed_summaries": seed_summaries,
        "aggregate_gain_vs_baseline": aggregate,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": platform.node(),
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(output / "summary.json", result)
    write_json(
        output / "run_state.json",
        {"status": "complete", "summary": str(output / "summary.json")},
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
