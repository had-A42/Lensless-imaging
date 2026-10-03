"""Benchmark X-Restormer batch size on the frozen synthetic development grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS = ("PSNR", "SSIM", "LPIPS")
ALLOWED_BATCH_SIZES = (1, 2, 4, 8)
EXPECTED_CHECKPOINT_SHA256 = (
    "fb6ad518d1f75b6391985c43d1847611f1249149a997bdf0dc60e16826ae4000"
)
PARITY_LIMITS = {
    "per_image_max_abs": {"PSNR": 0.01, "SSIM": 1e-4, "LPIPS": 1e-4},
    "mask_balanced_abs": {"PSNR": 0.001, "SSIM": 1e-5, "LPIPS": 1e-5},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def scalar_list(value, count: int) -> list:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * count


def load_reference(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["condition"] == "Correct"]
    rows.sort(key=lambda row: int(row["sample_index"]))
    if len(rows) != 1024:
        raise ValueError("Batch-1 development reference must contain 1024 Correct rows")
    return rows


def build_loader(batch_size: int):
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from src.datasets.mirflickr import MirFlickrSceneDataset
    from src.datasets.on_the_fly import DigiCamOnTheFlyDataset, DigiCamValidationBatchSampler
    from src.digicam_synth.mask_protocol import get_mask_records

    scenes = MirFlickrSceneDataset(
        root_dir=REPO_ROOT / "data/raw/mirflickr25k/extracted",
        splits_path=REPO_ROOT / "manifests/mirflickr25k_splits.json",
        split="validation",
        image_size=None,
        verify_files=True,
    )
    if len(scenes) != 128:
        raise ValueError("Frozen development scene split must contain 128 scenes")
    masks = get_mask_records(42, "validation", 32)
    simulator = OmegaConf.load(REPO_ROOT / "src/configs/simulator/digicam_article.yaml")
    dataset = DigiCamOnTheFlyDataset(
        scenes,
        simulator,
        measurement_size=None,
        target_size=(200, 266),
        simulation_mode="roi_convolution",
        roi=(80, 100, 200, 266),
        finite_cache_size=2,
        psf_cache={
            "mode": "read_only",
            "root_dir": str(REPO_ROOT / "data/psf_cache"),
            "request_modes": ["finite"],
        },
    )
    sampler = DigiCamValidationBatchSampler(
        scene_count=len(scenes),
        batch_size=batch_size,
        mask_records=masks,
        run_seed=52,
        scenes_per_mask=32,
        scene_offset=0,
        scene_selector_salt=59,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0), masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch_size not in ALLOWED_BATCH_SIZES:
        raise ValueError(f"batch-size must be one of {ALLOWED_BATCH_SIZES}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"0", "1", "2", "3", "4", "5"}:
        raise RuntimeError("Use exactly one physical GPU from 0 through 5")

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from src.metrics.reconstruction import LPIPSMetric, PSNRMetric, SSIMMetric
    from src.utils.init_utils import set_random_seed
    from scripts.evaluate_frozen_final_synthetic import static_checkpoint_metadata

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_state.json", {"status": "preflight"})
    checkpoint = Path(args.checkpoint).resolve()
    reference_path = Path(args.reference_csv).resolve()
    if not checkpoint.is_file() or not reference_path.is_file():
        raise FileNotFoundError("Checkpoint or reference CSV is missing")
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Unexpected development benchmark checkpoint")
    reference = load_reference(reference_path)
    set_random_seed(42)
    torch.set_num_threads(4)
    loader, masks = build_loader(args.batch_size)

    model_config = OmegaConf.load(REPO_ROOT / "src/configs/model/psf_free_xrestormer.yaml")
    model_config.checkpoint_path = None
    model_config.output_crop = [80, 100, 200, 266]
    model = instantiate(model_config).cuda().eval()
    endpoint = static_checkpoint_metadata(checkpoint)
    if endpoint != {
        "epoch": 10,
        "global_step": 100_000,
        "sampler_step": 100_000,
        "T_max": 100_000,
    }:
        raise ValueError(f"Unexpected checkpoint endpoint: {endpoint}")
    checkpoint_value = torch.load(
        str(checkpoint), map_location="cpu", weights_only=False, mmap=True
    )
    model.load_state_dict(checkpoint_value["state_dict"], strict=True)
    del checkpoint_value
    metrics = (
        PSNRMetric(name="PSNR", normalize_by_max=True),
        SSIMMetric(name="SSIM", normalize_by_max=True),
        LPIPSMetric(name="LPIPS", net_type="vgg", device="cuda", normalize_by_max=True),
    )
    write_json(
        output / "preflight.json",
        {
            "status": "pass",
            "dataset_split": "validation",
            "mask_partition": "validation",
            "scene_count_available": 128,
            "scene_count_used_per_mask": 32,
            "mask_count": len(masks),
            "expected_samples": 1024,
            "batch_size": args.batch_size,
            "development_psf_source": "read-only cache used by the historical reference",
            "psf_cache_config_hash": "3a2a4d87833fb3e1a8e9618f3c365255616331ada32f68305272b0bd0f8e82d1",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_endpoint": endpoint,
            "reference_csv": str(reference_path),
            "reference_csv_sha256": sha256(reference_path),
            "parity_limits": PARITY_LIMITS,
            "test_scene_files_opened": False,
            "test_masks_generated": False,
            "final_test_model_forward_executed": False,
        },
    )

    rows = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    write_json(output / "run_state.json", {"status": "running"})
    with torch.inference_mode():
        for batch in loader:
            measurement = batch["measurement"].cuda()
            target = batch["target"].cuda()
            prediction = model(measurement=measurement)["prediction"].float()
            if prediction.shape != target.shape:
                raise ValueError("Prediction/target shape mismatch")
            values = {
                metric.name: metric.per_image(prediction=prediction, target=target)
                .detach()
                .cpu()
                .reshape(-1)
                for metric in metrics
            }
            count = prediction.shape[0]
            mask_ids = scalar_list(batch["mask_id"], count)
            scene_ids = scalar_list(batch["scene_id"], count)
            for offset in range(count):
                rows.append(
                    {
                        "sample_index": len(rows),
                        "mask_id": str(mask_ids[offset]),
                        "scene_id": str(scene_ids[offset]),
                        **{
                            name: float(metric_values[offset])
                            for name, metric_values in values.items()
                        },
                    }
                )
            if len(rows) % 128 == 0:
                print(f"batch{args.batch_size}: {len(rows)}/1024", flush=True)
    elapsed = time.monotonic() - started
    if len(rows) != 1024:
        raise ValueError("Development benchmark did not process exactly 1024 samples")
    identity_match = all(
        int(reference_row["sample_index"]) == row["sample_index"]
        and reference_row["mask_id"] == row["mask_id"]
        and reference_row["scene_id"] == row["scene_id"]
        for reference_row, row in zip(reference, rows)
    )
    if not identity_match:
        raise ValueError("Development row identities drifted from batch-1 reference")
    per_image_deltas = {
        metric: np.asarray(
            [row[metric] - float(reference_row[metric]) for row, reference_row in zip(rows, reference)]
        )
        for metric in METRICS
    }
    grouped = defaultdict(lambda: {metric: [] for metric in METRICS})
    reference_grouped = defaultdict(lambda: {metric: [] for metric in METRICS})
    for row, reference_row in zip(rows, reference):
        for metric in METRICS:
            grouped[row["mask_id"]][metric].append(row[metric])
            reference_grouped[row["mask_id"]][metric].append(float(reference_row[metric]))
    mask_balanced = {
        metric: float(
            np.mean([np.mean(values[metric]) for values in grouped.values()])
        )
        for metric in METRICS
    }
    reference_mask_balanced = {
        metric: float(
            np.mean([np.mean(values[metric]) for values in reference_grouped.values()])
        )
        for metric in METRICS
    }
    parity = {
        "per_image_max_abs": {
            metric: float(np.max(np.abs(values)))
            for metric, values in per_image_deltas.items()
        },
        "per_image_mean_abs": {
            metric: float(np.mean(np.abs(values)))
            for metric, values in per_image_deltas.items()
        },
        "mask_balanced_abs": {
            metric: abs(mask_balanced[metric] - reference_mask_balanced[metric])
            for metric in METRICS
        },
    }
    parity_pass = all(
        parity[group][metric] <= PARITY_LIMITS[group][metric]
        for group in PARITY_LIMITS
        for metric in METRICS
    )
    metrics_finite = all(np.isfinite(row[metric]) for row in rows for metric in METRICS)
    with (output / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "pass" if parity_pass and metrics_finite else "fail",
        "batch_size": args.batch_size,
        "sample_count": len(rows),
        "mask_count": len(grouped),
        "scenes_per_mask": sorted({len(values["PSNR"]) for values in grouped.values()}),
        "mask_balanced": mask_balanced,
        "reference_mask_balanced": reference_mask_balanced,
        "parity": parity,
        "parity_limits": PARITY_LIMITS,
        "elapsed_seconds": elapsed,
        "samples_per_second": len(rows) / elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "physical_gpu": visible,
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "final_test_model_forward_executed": False,
    }
    write_json(output / "summary.json", summary)
    provenance = {
        "status": summary["status"],
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in (
                REPO_ROOT / "src/model/psf_free_xrestormer.py",
                REPO_ROOT / "src/datasets/on_the_fly.py",
                REPO_ROOT / "src/datasets/mirflickr.py",
                REPO_ROOT / "src/digicam_synth/mask_protocol.py",
                REPO_ROOT / "src/digicam_synth/pipeline.py",
                REPO_ROOT / "src/metrics/reconstruction.py",
            )
        },
        "dataset_split": "validation",
        "mask_partition": "validation",
        "final_test_accessed": False,
    }
    write_json(output / "provenance.json", provenance)
    write_json(
        output / "run_state.json",
        {"status": "complete" if summary["status"] == "pass" else "parity_failed"},
    )
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
