"""Validate the completed report-wide test outputs without running a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

IDENTITY_COLUMNS = (
    "sample_index",
    "sample_id",
    "scene_id",
    "mask_id",
    "split",
    "label",
)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def one_file(root: Path, name: str) -> Path:
    matches = sorted(root.glob(f"*/{name}"))
    if len(matches) != 1:
        raise ValueError(f"expected one {name} below {root}, found {matches}")
    return matches[0]


def expected_contract(entry: dict) -> tuple[int, int, int]:
    contract = entry["test_contract"]
    return (
        int(contract["samples"]),
        int(contract["mask_count"]),
        int(contract["scenes_per_mask"]),
    )


def validate_run(entry: dict, run_root: Path) -> dict:
    expected_samples, expected_masks, expected_scenes = expected_contract(entry)
    summary_path = one_file(run_root, "summary.json")
    per_image_path = one_file(run_root, "per_image.csv")
    per_mask_path = one_file(run_root, "per_mask.csv")
    summary = json.loads(summary_path.read_text())
    provenance = summary.get("provenance", {})
    errors = []

    expected_provenance = {
        "report_test_entry": entry["id"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "mode": "test",
        "expected_samples": expected_samples,
        "expected_masks": expected_masks,
        "expected_scenes_per_mask": expected_scenes,
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            errors.append(f"provenance:{key}")
    if summary.get("sample_count") != expected_samples:
        errors.append("summary:sample_count")
    if summary.get("mask_count") != expected_masks:
        errors.append("summary:mask_count")
    if summary.get("samples_per_mask") != [expected_scenes]:
        errors.append("summary:samples_per_mask")

    mask_counts: Counter[str] = Counter()
    scene_ids: set[str] = set()
    sample_ids: set[str] = set()
    metric_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    identity_digest = hashlib.sha256()
    row_count = 0
    metric_columns: list[str] = []
    identity_columns: list[str] = []
    with per_image_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            errors.append("per_image:missing_header")
        else:
            identity_columns = [
                column for column in IDENTITY_COLUMNS if column in reader.fieldnames
            ]
            metric_columns = [
                column for column in reader.fieldnames if column not in IDENTITY_COLUMNS
            ]
        for index, row in enumerate(reader):
            row_count += 1
            if row.get("sample_index") != str(index):
                errors.append("per_image:sample_index")
                break
            if row.get("split") != "test":
                errors.append("per_image:split")
                break
            mask_id = str(row["mask_id"])
            mask_counts[mask_id] += 1
            scene_ids.add(str(row["scene_id"]))
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                errors.append("per_image:duplicate_sample_id")
                break
            sample_ids.add(sample_id)
            identity_digest.update(
                (
                    "\x1f".join(str(row[column]) for column in identity_columns) + "\n"
                ).encode()
            )
            for metric in metric_columns:
                value = float(row[metric])
                if not math.isfinite(value):
                    errors.append(f"per_image:non_finite:{metric}")
                    break
                metric_sums[mask_id][metric] += value
            if errors:
                break

    if row_count != expected_samples:
        errors.append("per_image:row_count")
    if len(sample_ids) != expected_samples:
        errors.append("per_image:sample_id_count")
    if len(mask_counts) != expected_masks:
        errors.append("per_image:mask_count")
    if set(mask_counts.values()) != {expected_scenes}:
        errors.append("per_image:mask_balance")
    expected_unique_scenes = (
        expected_samples if entry["dataset"] == "digicam real" else expected_scenes
    )
    if len(scene_ids) != expected_unique_scenes:
        errors.append("per_image:scene_count")
    if entry["dataset"] != "digicam real":
        expected_mask_ids = {f"test_{index:05d}" for index in range(expected_masks)}
        if set(mask_counts) != expected_mask_ids:
            errors.append("per_image:mask_ids")

    observed_metrics = summary.get("mask_balanced", {})
    if set(observed_metrics) != set(metric_columns):
        errors.append("summary:metric_columns")
    else:
        for metric in metric_columns:
            recomputed = sum(
                metric_sums[mask][metric] / mask_counts[mask] for mask in mask_counts
            ) / len(mask_counts)
            if not math.isclose(
                recomputed,
                float(observed_metrics[metric]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                errors.append(f"summary:metric_recompute:{metric}")

    with per_mask_path.open(newline="") as stream:
        per_mask_rows = list(csv.DictReader(stream))
    if len(per_mask_rows) != expected_masks:
        errors.append("per_mask:row_count")
    if {int(row["sample_count"]) for row in per_mask_rows} != {expected_scenes}:
        errors.append("per_mask:sample_count")

    per_class_path = summary_path.parent / "per_class.csv"
    if entry["dataset"] == "mnist":
        if not per_class_path.is_file():
            errors.append("per_class:missing")
        else:
            with per_class_path.open(newline="") as stream:
                per_class = list(csv.DictReader(stream))
            if {int(row["label"]) for row in per_class} != set(range(10)):
                errors.append("per_class:labels")
            if sum(int(row["sample_count"]) for row in per_class) != expected_samples:
                errors.append("per_class:sample_count")

    return {
        "id": entry["id"],
        "dataset": entry["dataset"],
        "sample_count": row_count,
        "mask_count": len(mask_counts),
        "scene_count": len(scene_ids),
        "identity_columns": identity_columns,
        "identity_sha256": identity_digest.hexdigest(),
        "metric_columns": metric_columns,
        "summary": str(summary_path),
        "errors": sorted(set(errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("runs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    runs_root = Path(args.runs).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for entry in manifest["entries"]:
        run_root = runs_root / entry["id"]
        if not run_root.is_dir():
            rows.append(
                {
                    "id": entry["id"],
                    "dataset": entry["dataset"],
                    "errors": ["run:missing"],
                }
            )
            continue
        rows.append(validate_run(entry, run_root))

    grid_errors = []
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        valid_rows = [row for row in dataset_rows if not row["errors"]]
        if len(valid_rows) != len(dataset_rows):
            continue
        identity_keys = {
            (tuple(row["identity_columns"]), row["identity_sha256"])
            for row in valid_rows
        }
        if len(identity_keys) != 1:
            grid_errors.append(f"identity_grid:{dataset}")

    passed = sum(not row["errors"] for row in rows)
    result = {
        "status": (
            "pass"
            if len(rows) == int(manifest["standard_entry_count"])
            and passed == len(rows)
            and not grid_errors
            else "fail"
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "expected_run_count": int(manifest["standard_entry_count"]),
        "run_count": len(rows),
        "passed_run_count": passed,
        "failed_run_count": len(rows) - passed,
        "grid_errors": grid_errors,
        "model_forward_executed": False,
        "rows": rows,
    }
    save_json(Path(args.output).expanduser().resolve(), result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"},
            indent=2,
        )
    )
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
