"""Freeze BF16 development references from each selected checkpoint's own run."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_frozen_final_synthetic import sha256, write_json  # noqa: E402


def mask_balanced_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 32 or len({row["mask_id"] for row in rows}) != 32:
        raise ValueError(f"Expected 32 mask rows: {path}")
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in ("PSNR", "SSIM", "LPIPS")
    }


def main() -> None:
    shortlist_path = (
        REPO_ROOT
        / "outputs/coursework_pre_final_20260910/frozen_shortlist_v2/frozen_shortlist.json"
    )
    inventory_path = (
        REPO_ROOT
        / "outputs/coursework_status_20260909/core_xrest50k_checkpoint_inventory.csv"
    )
    output = REPO_ROOT / "outputs/coursework_final_runner_v4_20260910/development_reference_v2"
    output.mkdir(parents=True, exist_ok=False)
    shortlist = json.loads(shortlist_path.read_text())
    with inventory_path.open(newline="") as stream:
        inventory = list(csv.DictReader(stream))
    inventory_by_key = {
        (int(row["masks"]), row["initialization"], int(row["seed"])): row
        for row in inventory
    }
    rows = []
    source_hashes = {}
    for entry in shortlist["entries"]:
        checkpoint = Path(entry["checkpoint_local_path"])
        epoch = int(entry["epoch"])
        metric_path = checkpoint.parent / f"validation_per_mask_epoch{epoch:04d}.csv"
        if not metric_path.is_file():
            raise FileNotFoundError(metric_path)
        metrics = mask_balanced_metrics(metric_path)
        if entry["role"] == "core_50k_matrix":
            inventory_row = inventory_by_key[
                (
                    int(entry["training_masks"]),
                    entry["initialization"],
                    int(entry["seed"]),
                )
            ]
            inventory_metrics = {
                metric: float(inventory_row[f"endpoint_{metric}"])
                for metric in ("PSNR", "SSIM", "LPIPS")
            }
        else:
            inventory_metrics = {metric: metrics[metric] for metric in metrics}
        rows.append(
            {
                "shortlist_id": entry["shortlist_id"],
                "reference_type": "selected checkpoint own final-endpoint validation",
                "reference_precision": "BF16 autocast",
                **metrics,
                "PSNR_abs_tolerance": 0.01,
                "SSIM_abs_tolerance": 0.001,
                "LPIPS_abs_tolerance": 0.001,
                **{
                    f"inventory_delta_{metric}": metrics[metric]
                    - inventory_metrics[metric]
                    for metric in metrics
                },
                "source": str(metric_path),
                "source_sha256": sha256(metric_path),
            }
        )
        source_hashes[str(metric_path)] = sha256(metric_path)
    expected_ids = [entry["shortlist_id"] for entry in shortlist["entries"]]
    if [row["shortlist_id"] for row in rows] != expected_ids:
        raise ValueError("V2 development reference order drift")
    csv_path = output / "reference.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    validation = {
        "status": "pass",
        "checkpoint_count": len(rows),
        "exact_shortlist_order": True,
        "reference_precision": "BF16 autocast",
        "tolerances_frozen_before_bf16_runner_validation": True,
        "inventory_mismatches_preserved": [
            {
                "shortlist_id": row["shortlist_id"],
                **{
                    f"inventory_delta_{metric}": row[f"inventory_delta_{metric}"]
                    for metric in ("PSNR", "SSIM", "LPIPS")
                },
            }
            for row in rows
            if any(
                abs(float(row[f"inventory_delta_{metric}"])) > 1e-9
                for metric in ("PSNR", "SSIM", "LPIPS")
            )
        ],
        "source_hashes": source_hashes,
        "csv": str(csv_path),
        "csv_sha256": sha256(csv_path),
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "final_test_model_forward_executed": False,
    }
    write_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
