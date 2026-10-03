"""Evaluate the published PSF-aware reference on frozen real development rows.

Only the upstream ``train`` split is allowed. The script verifies row identities
against an existing PSF-free evaluation before constructing the dataset and
never accesses the official real test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


DATASET_REPO = "bezzam/DigiCam-Mirflickr-MultiMask-25K"
DATASET_REVISION = "21d82b67662ed1e590a40c98688c32cb3c74f079"
MODEL_REPO = "bezzam/digicam-mirflickr-multi-25k-unet4M-unrolled-admm5-unet4M-wave-psfNN"
MODEL_REVISION = "9c965e99a6b9048eaa0eea3b8cdb2c5b9039416e"
METRICS = ("PSNR", "SSIM", "LPIPS")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_contract(manifest_path: Path, view: str) -> dict:
    manifest = json.loads(manifest_path.read_text())
    dataset = manifest["dataset"]
    if dataset["repo_id"] != DATASET_REPO or dataset["revision"] != DATASET_REVISION:
        raise ValueError("Unexpected dataset identity")
    if dataset["split"] != "train":
        raise ValueError("Only the upstream train split is allowed")

    if view == "inner68":
        masks = [int(value) for value in manifest["mask_split"]["development_mask_ids"]]
        slots = [int(value) for value in manifest["row_split"]["inner_validation_slots"]]
        expected_rows_per_mask = 25
        claim = "contextual privileged reference on new rows of operators seen by the published checkpoint"
    elif view == "outer17":
        masks = [int(value) for value in manifest["evaluation_rows"]["mask_ids"]]
        slots = [int(value) for value in manifest["evaluation_rows"]["row_slots"]]
        expected_rows_per_mask = 250
        claim = "contextual privileged reference; these operators were held out from our PSF-free training but not from published-model training"
        policy = manifest.get("access_policy", {})
        if policy.get("official_test_allowed") is not False:
            raise ValueError("Outer17 manifest must explicitly forbid official test access")
    else:
        raise ValueError("view must be inner68 or outer17")

    if not masks or min(masks) < 15 or max(masks) > 99:
        raise ValueError("Development masks must be within upstream train IDs 15..99")
    if len(set(masks)) != len(masks) or len(set(slots)) != len(slots):
        raise ValueError("Mask IDs and row slots must be unique")
    if len(slots) != expected_rows_per_mask:
        raise ValueError("Unexpected rows per mask")

    indices = sorted(slot * 85 + mask - 15 for slot in slots for mask in masks)
    identities = sorted(
        (slot * 85 + mask - 15, mask, slot) for slot in slots for mask in masks
    )
    return {
        "masks": masks,
        "slots": slots,
        "indices": indices,
        "identities": identities,
        "expected_rows": len(masks) * len(slots),
        "expected_masks": len(masks),
        "expected_rows_per_mask": expected_rows_per_mask,
        "claim": claim,
    }


def read_reference_identities(path: Path) -> list[tuple[int, int, int]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"source_index", "mask_id", "row_slot"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("Reference CSV does not contain row identities")
    identities = sorted(
        (int(row["source_index"]), int(row["mask_id"]), int(row["row_slot"]))
        for row in rows
    )
    if len(set(identities)) != len(identities):
        raise ValueError("Reference CSV contains duplicate identities")
    return identities


def scalar_list(value, count: int) -> list:
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * count


def find_model_checkpoint(snapshot: Path) -> Path:
    candidates = sorted(
        path
        for path in snapshot.rglob("*")
        if path.is_file() and path.name in {"recon_epochBEST", "model_best.pth"}
    )
    if len(candidates) != 1:
        raise ValueError(f"Expected one published model checkpoint, found {candidates}")
    return candidates[0]


def evaluate(args: argparse.Namespace) -> None:
    from src.datasets.digicam import DigiCamRealDataset
    from src.metrics.reconstruction import LPIPSMetric, PSNRMetric, SSIMMetric
    from src.model.psf_aware_lensless import PSFAwareLenslessModel
    from src.utils.init_utils import set_random_seed

    set_random_seed(42)
    torch.set_num_threads(4)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_state.json", {"status": "preflight", "view": args.view})

    manifest_path = Path(args.manifest).resolve()
    reference_path = Path(args.reference_csv).resolve()
    simulator_path = Path(args.simulator_config).resolve()
    for path in (manifest_path, reference_path, simulator_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    contract = load_contract(manifest_path, args.view)
    reference_identities = read_reference_identities(reference_path)
    if contract["identities"] != reference_identities:
        raise ValueError("Prepared rows do not exactly match the PSF-free reference CSV")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir).resolve()
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            cache_dir=str(cache_dir),
            local_files_only=not args.allow_model_download,
        )
    )
    model_checkpoint = find_model_checkpoint(snapshot)
    model_checkpoint_hash = sha256(model_checkpoint)
    simulator = OmegaConf.load(simulator_path)

    preflight = {
        "status": "passed",
        "view": args.view,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": "train",
        "official_real_test_accessed": False,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "reference_csv": str(reference_path),
        "reference_csv_sha256": sha256(reference_path),
        "simulator_config": str(simulator_path),
        "simulator_config_sha256": sha256(simulator_path),
        "expected_rows": contract["expected_rows"],
        "expected_masks": contract["expected_masks"],
        "expected_rows_per_mask": contract["expected_rows_per_mask"],
        "row_identity_match": True,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_snapshot": str(snapshot),
        "model_checkpoint": str(model_checkpoint),
        "model_checkpoint_sha256": model_checkpoint_hash,
        "model_download_allowed": bool(args.allow_model_download),
        "interpretation": contract["claim"],
    }
    write_json(output / "preflight.json", preflight)
    if args.preflight_only:
        write_json(output / "run_state.json", {"status": "preflight_complete", "view": args.view})
        print(json.dumps(preflight, indent=2), flush=True)
        return

    dataset = DigiCamRealDataset(
        repo_id=DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        indices=contract["indices"],
        cache_dir=str(cache_dir),
        force_rgb=True,
        rotate_measurement=True,
        measurement_downsample=1.0,
        target_size=[200, 266],
        target_resize_mode="bilinear",
        return_psf=True,
        simulator_config=simulator,
        expected_mask_count=contract["expected_masks"],
        expected_scenes_per_mask=contract["expected_rows_per_mask"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model = PSFAwareLenslessModel(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        cache_dir=str(cache_dir),
        output_crop=[80, 100, 200, 266],
    ).eval()
    metrics = (
        PSNRMetric(name="PSNR", normalize_by_max=True),
        SSIMMetric(name="SSIM", normalize_by_max=True),
        LPIPSMetric(
            name="LPIPS",
            net_type="vgg",
            device=str(device),
            normalize_by_max=True,
        ),
    )

    rows: list[dict] = []
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    processed = 0
    with torch.inference_mode():
        for batch in loader:
            measurement = batch["measurement"].to(device)
            target = batch["target"].to(device)
            psf = batch["psf"].to(device)
            prediction = model(measurement=measurement, psf=psf)["prediction"].float()
            if prediction.shape != target.shape:
                raise ValueError(f"Prediction/target mismatch: {prediction.shape} != {target.shape}")
            metric_values = {
                metric.name: metric.per_image(prediction=prediction, target=target).cpu()
                for metric in metrics
            }
            count = prediction.shape[0]
            mask_ids = scalar_list(batch["mask_id"], count)
            scene_ids = scalar_list(batch["scene_id"], count)
            for offset in range(count):
                source_index = contract["indices"][processed + offset]
                mask_id = int(mask_ids[offset])
                row_slot = source_index // 85
                expected_mask = source_index % 85 + 15
                if mask_id != expected_mask:
                    raise ValueError("Dataset mask ID does not match frozen row identity")
                rows.append(
                    {
                        "source_index": source_index,
                        "mask_id": mask_id,
                        "row_slot": row_slot,
                        "scene_id": str(scene_ids[offset]),
                        **{
                            name: float(values[offset])
                            for name, values in metric_values.items()
                        },
                    }
                )
            processed += count
            if processed % 128 == 0 or processed == contract["expected_rows"]:
                print(f"{args.view}: {processed}/{contract['expected_rows']}", flush=True)

    if processed != contract["expected_rows"] or len(rows) != contract["expected_rows"]:
        raise ValueError("Evaluation did not process the exact frozen row count")
    if sorted((row["source_index"], row["mask_id"], row["row_slot"]) for row in rows) != reference_identities:
        raise ValueError("Evaluated rows drifted from the frozen reference identities")
    if not all(np.isfinite(row[metric]) for row in rows for metric in METRICS):
        raise ValueError("Non-finite metric detected")

    per_image_path = output / "per_image.csv"
    with per_image_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in METRICS}
    )
    for row in rows:
        for metric in METRICS:
            grouped[row["mask_id"]][metric].append(row[metric])
    if set(map(len, (values["PSNR"] for values in grouped.values()))) != {
        contract["expected_rows_per_mask"]
    }:
        raise ValueError("Rows are not balanced across masks")
    per_mask = [
        {
            "mask_id": mask_id,
            "sample_count": len(values["PSNR"]),
            **{metric: float(np.mean(values[metric])) for metric in METRICS},
        }
        for mask_id, values in sorted(grouped.items())
    ]
    with (output / "per_mask.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_mask[0]))
        writer.writeheader()
        writer.writerows(per_mask)

    scores = {
        metric: float(np.mean([row[metric] for row in per_mask])) for metric in METRICS
    }
    elapsed = time.monotonic() - started
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    provenance = {
        **preflight,
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "source_hashes": {
            str(path): sha256(path)
            for path in (
                REPO_ROOT / "src/model/psf_aware_lensless.py",
                REPO_ROOT / "src/datasets/digicam.py",
                REPO_ROOT / "src/metrics/reconstruction.py",
                REPO_ROOT / "src/digicam_synth/pipeline.py",
            )
        },
        "device": str(device),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": args.batch_size,
        "normalization": "independent per-image maximum for prediction and target",
        "aggregation": "image -> equal mean within mask -> equal mean across masks",
        "crop": [80, 100, 200, 266],
        "measurement_rotation_degrees": 180,
        "checkpoint_selection": "published fixed checkpoint; no selection on these rows",
        "comparison_status": "contextual privileged reference, not a matched causal comparison",
    }
    write_json(output / "provenance.json", provenance)
    summary = {
        "status": "complete",
        "view": args.view,
        "scores": scores,
        "sample_count": len(rows),
        "mask_count": len(per_mask),
        "rows_per_mask": contract["expected_rows_per_mask"],
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": peak_vram,
        "interpretation": contract["claim"],
        "final_synthetic_test_accessed": False,
        "official_real_test_accessed": False,
    }
    write_json(output / "summary.json", summary)
    validation = {
        "complete": True,
        "row_identity_match": True,
        "sample_count_match": len(rows) == contract["expected_rows"],
        "mask_count_match": len(per_mask) == contract["expected_masks"],
        "rows_per_mask_match": all(
            row["sample_count"] == contract["expected_rows_per_mask"] for row in per_mask
        ),
        "metrics_finite": True,
        "manifest_sha256": preflight["manifest_sha256"],
        "reference_csv_sha256": preflight["reference_csv_sha256"],
        "model_checkpoint_sha256": model_checkpoint_hash,
        "evaluator_sha256": provenance["evaluator_sha256"],
        "official_real_test_accessed": False,
        "final_synthetic_test_accessed": False,
    }
    write_json(output / "validation.json", validation)
    write_json(output / "run_state.json", {"status": "complete", "view": args.view})
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=("inner68", "outer17"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-csv", required=True)
    parser.add_argument("--simulator-config", default="src/configs/simulator/digicam_article.yaml")
    parser.add_argument("--cache-dir", default="data/huggingface")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow downloading only the exact pinned published model revision.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
