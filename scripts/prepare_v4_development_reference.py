"""Freeze expected development metrics for all V4 shortlist checkpoints."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_frozen_final_synthetic import sha256, write_json  # noqa: E402


def main() -> None:
    core_path = (
        REPO_ROOT
        / "outputs/coursework_status_20260909/core_xrest50k_checkpoint_inventory.csv"
    )
    shortlist_path = (
        REPO_ROOT
        / "outputs/coursework_pre_final_20260910/frozen_shortlist_v2/frozen_shortlist.json"
    )
    finalist_path = (
        REPO_ROOT
        / "outputs/coursework_completion_20260907/evaluations/E2-mirflickr-xrest100k-seed42/summary.json"
    )
    output = REPO_ROOT / "outputs/coursework_final_runner_v4_20260910/development_reference"
    output.mkdir(parents=True, exist_ok=False)
    with core_path.open(newline="") as stream:
        core = list(csv.DictReader(stream))
    shortlist = json.loads(shortlist_path.read_text())
    finalist = json.loads(finalist_path.read_text())["conditions"]["Correct"]
    by_key = {
        (
            int(entry["training_masks"]),
            entry["initialization"],
            int(entry["seed"]),
        ): entry["shortlist_id"]
        for entry in shortlist["entries"]
        if entry["role"] == "core_50k_matrix"
    }
    rows = []
    for row in core:
        key = (int(row["masks"]), row["initialization"], int(row["seed"]))
        rows.append(
            {
                "shortlist_id": by_key[key],
                "reference_type": "historical final-50k endpoint",
                "reference_precision": "training validation autocast",
                "PSNR": row["endpoint_PSNR"],
                "SSIM": row["endpoint_SSIM"],
                "LPIPS": row["endpoint_LPIPS"],
                "PSNR_abs_tolerance": 0.02,
                "SSIM_abs_tolerance": 0.002,
                "LPIPS_abs_tolerance": 0.002,
                "source": row["metric_source"],
            }
        )
    rows.append(
        {
            "shortlist_id": "xrest100k-m100-gopro-seed42",
            "reference_type": "standalone FP32 development evaluation",
            "reference_precision": "FP32",
            "PSNR": finalist["PSNR"],
            "SSIM": finalist["SSIM"],
            "LPIPS": finalist["LPIPS"],
            "PSNR_abs_tolerance": 0.01,
            "SSIM_abs_tolerance": 0.001,
            "LPIPS_abs_tolerance": 0.001,
            "source": str(finalist_path),
        }
    )
    expected_ids = {entry["shortlist_id"] for entry in shortlist["entries"]}
    if len(rows) != 13 or {row["shortlist_id"] for row in rows} != expected_ids:
        raise ValueError("Development reference does not cover the exact shortlist")
    csv_path = output / "reference.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    validation = {
        "status": "pass",
        "checkpoint_count": len(rows),
        "exact_shortlist_coverage": True,
        "tolerances_frozen_before_v4_development_run": True,
        "sources": {
            str(core_path): sha256(core_path),
            str(shortlist_path): sha256(shortlist_path),
            str(finalist_path): sha256(finalist_path),
        },
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
