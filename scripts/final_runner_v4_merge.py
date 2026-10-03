"""Merge three frozen V4 worker outputs without metric-based selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_frozen_final_synthetic import sha256, write_json  # noqa: E402
from scripts.final_runner_v4_worker import load_v4_metadata  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=("development", "final"), required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest, _, entries = load_v4_metadata(manifest_path)
    input_root = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    expected_order = [entry["shortlist_id"] for entry in entries]
    expected_by_shard = {
        int(shard["shard_id"]): shard["checkpoint_ids"] for shard in manifest["shards"]
    }
    rows_by_id = {}
    source_artifacts = []
    grid_hashes = set()
    for shard_id in (0, 1, 2):
        shard_root = input_root / f"shard{shard_id}"
        validation_path = shard_root / "validation.json"
        summary_path = shard_root / "summary.csv"
        provenance_path = shard_root / "provenance.json"
        for path in (validation_path, summary_path, provenance_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            source_artifacts.append({"path": str(path), "sha256": sha256(path)})
        validation = json.loads(validation_path.read_text())
        if (
            validation.get("status") != "pass"
            or validation.get("mode") != args.mode
            or int(validation.get("shard_id", -1)) != shard_id
            or validation.get("checkpoint_ids") != expected_by_shard[shard_id]
            or validation.get("completed_checkpoint_ids") != expected_by_shard[shard_id]
            or validation.get("exact_checkpoint_partition") is not True
            or validation.get("all_model_validations_pass") is not True
            or validation.get("test_accessed") != (args.mode == "final")
        ):
            raise ValueError(f"Shard {shard_id} validation failed")
        with summary_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [row["shortlist_id"] for row in rows] != expected_by_shard[shard_id]:
            raise ValueError(f"Shard {shard_id} summary order drift")
        for row in rows:
            identifier = row["shortlist_id"]
            if identifier in rows_by_id:
                raise ValueError(f"Duplicate checkpoint output: {identifier}")
            for metric in ("PSNR", "SSIM", "LPIPS"):
                if not math.isfinite(float(row[metric])):
                    raise ValueError(f"Non-finite {metric}: {identifier}")
            expected_samples = 1024 if args.mode == "development" else 25_600
            expected_masks = 32 if args.mode == "development" else 100
            if int(row["sample_count"]) != expected_samples or int(row["mask_count"]) != expected_masks:
                raise ValueError(f"Grid count mismatch: {identifier}")
            grid_hashes.add(row["grid_identity_sha256"])
            rows_by_id[identifier] = row
    if set(rows_by_id) != set(expected_order) or len(rows_by_id) != len(expected_order):
        raise ValueError("Merged checkpoint set is not the frozen revised shortlist")
    if len(grid_hashes) != 1:
        raise ValueError("Workers did not evaluate an identical ordered grid")

    development_parity = []
    if args.mode == "development":
        reference_path = (
            REPO_ROOT / manifest["development_validation_contract"]["reference"]
        ).resolve()
        with reference_path.open(newline="") as stream:
            reference_rows = list(csv.DictReader(stream))
        reference_by_id = {row["shortlist_id"]: row for row in reference_rows}
        if set(reference_by_id) != set(expected_order):
            raise ValueError("Development reference does not cover the exact shortlist")
        for identifier in expected_order:
            observed = rows_by_id[identifier]
            reference = reference_by_id[identifier]
            deltas = {
                metric: float(observed[metric]) - float(reference[metric])
                for metric in ("PSNR", "SSIM", "LPIPS")
            }
            tolerances = {
                metric: float(reference[f"{metric}_abs_tolerance"])
                for metric in ("PSNR", "SSIM", "LPIPS")
            }
            passed = all(abs(deltas[metric]) <= tolerances[metric] for metric in deltas)
            development_parity.append(
                {
                    "shortlist_id": identifier,
                    **{f"delta_{metric}": deltas[metric] for metric in deltas},
                    **{f"tolerance_{metric}": tolerances[metric] for metric in tolerances},
                    "parity_pass": passed,
                }
            )
        if not all(row["parity_pass"] for row in development_parity):
            raise ValueError("At least one development checkpoint failed reference parity")

    merged_rows = [rows_by_id[identifier] for identifier in expected_order]
    csv_path = output / "summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(merged_rows[0]))
        writer.writeheader()
        writer.writerows(merged_rows)
    development_parity_path = None
    if development_parity:
        development_parity_path = output / "development_parity.csv"
        with development_parity_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(development_parity[0]))
            writer.writeheader()
            writer.writerows(development_parity)
    validation = {
        "status": "pass",
        "mode": args.mode,
        "shard_count": 3,
        "checkpoint_count": len(merged_rows),
        "checkpoint_order": expected_order,
        "exact_shortlist_partition": True,
        "single_grid_identity_sha256": next(iter(grid_hashes)),
        "all_exact_sample_counts": True,
        "all_exact_mask_counts": True,
        "all_metrics_finite": True,
        "metric_based_checkpoint_selection": False,
        "development_reference_parity": (
            all(row["parity_pass"] for row in development_parity)
            if args.mode == "development"
            else None
        ),
        "source_artifacts": source_artifacts,
        "summary": str(csv_path),
        "summary_sha256": sha256(csv_path),
        "development_parity": (
            str(development_parity_path) if development_parity_path is not None else None
        ),
        "development_parity_sha256": (
            sha256(development_parity_path)
            if development_parity_path is not None
            else None
        ),
        "test_scene_files_opened": args.mode == "final",
        "test_masks_generated": args.mode == "final",
        "final_test_model_forward_executed": args.mode == "final",
    }
    write_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
