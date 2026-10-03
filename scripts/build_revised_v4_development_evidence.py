"""Validate the 11 retained model outputs from the completed V4 development run."""

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
    shortlist_path = (
        REPO_ROOT
        / "outputs/coursework_final_runner_v4_20260910/revised_shortlist_v1/shortlist.json"
    )
    run_root = REPO_ROOT / "outputs/coursework_final_runner_v4_20260910/development_run"
    output = (
        REPO_ROOT
        / "outputs/coursework_final_runner_v4_20260910/revised_development_evidence_v2"
    )
    output.mkdir(parents=True, exist_ok=False)
    shortlist = json.loads(shortlist_path.read_text())
    wanted = [entry["shortlist_id"] for entry in shortlist["entries"]]
    located = {}
    for shard_id in (0, 1, 2):
        shard_root = run_root / f"shard{shard_id}"
        with (shard_root / "summary.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["shortlist_id"] in wanted:
                    located[row["shortlist_id"]] = (shard_id, row)
    if set(located) != set(wanted):
        raise ValueError("Completed development run does not cover revised shortlist")
    rows = []
    grid_hashes = set()
    sources = []
    for identifier in wanted:
        shard_id, source = located[identifier]
        model_root = run_root / f"shard{shard_id}" / identifier
        validation_path = model_root / "validation.json"
        summary_path = model_root / "summary.json"
        validation = json.loads(validation_path.read_text())
        summary = json.loads(summary_path.read_text())
        if (
            validation.get("status") != "pass"
            or validation.get("test_accessed") is not False
            or summary.get("mode") != "development"
            or summary.get("sample_count") != 1024
            or summary.get("mask_count") != 32
            or summary.get("test_accessed") is not False
        ):
            raise ValueError(f"Invalid retained development output: {identifier}")
        grid_hashes.add(summary["grid_identity_sha256"])
        rows.append(
            {
                "shortlist_id": identifier,
                "analysis_role": next(
                    entry["analysis_role"]
                    for entry in shortlist["entries"]
                    if entry["shortlist_id"] == identifier
                ),
                "source_shard": shard_id,
                "PSNR": summary["mask_balanced"]["PSNR"],
                "SSIM": summary["mask_balanced"]["SSIM"],
                "LPIPS": summary["mask_balanced"]["LPIPS"],
                "PSNR_abs_tolerance": 0.01,
                "SSIM_abs_tolerance": 0.001,
                "LPIPS_abs_tolerance": 0.001,
                "sample_count": summary["sample_count"],
                "mask_count": summary["mask_count"],
                "grid_identity_sha256": summary["grid_identity_sha256"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "validation": str(validation_path),
                "validation_sha256": sha256(validation_path),
            }
        )
        sources.extend(
            [
                {"path": str(validation_path), "sha256": sha256(validation_path)},
                {"path": str(summary_path), "sha256": sha256(summary_path)},
            ]
        )
    if len(grid_hashes) != 1:
        raise ValueError("Retained models do not share one development grid")
    csv_path = output / "summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    validation = {
        "status": "pass",
        "checkpoint_count": len(rows),
        "primary_matched_count": sum(
            row["analysis_role"] == "primary_matched_matrix" for row in rows
        ),
        "supplementary_count": sum(
            row["analysis_role"] == "supplementary_corrected_seed42" for row in rows
        ),
        "finalist_count": sum(
            row["analysis_role"] == "predeclared_100k_finalist" for row in rows
        ),
        "single_grid_identity_sha256": next(iter(grid_hashes)),
        "all_model_validations_pass": True,
        "source_worker_count": 3,
        "source_artifacts": sources,
        "summary": str(csv_path),
        "summary_sha256": sha256(csv_path),
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "final_test_model_forward_executed": False,
    }
    write_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
