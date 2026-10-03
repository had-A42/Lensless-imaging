"""Append validated per-run final-test metrics to the unified Markdown report.

This is deliberately a post-processing step.  The frozen report-test orchestrator is
hash-bound by the test manifest and must not be edited merely to present completed
results that have already been copied to another host.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


START_MARKER = "<!-- complete-report-test-results:start -->"
END_MARKER = "<!-- complete-report-test-results:end -->"
IDENTITY_COLUMNS = (
    "id",
    "model",
    "condition",
    "initialization",
    "variant",
    "seed",
    "steps",
    "training_scenes",
    "sample_count",
    "mask_count",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalized(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def number(value: str) -> float | None:
    value = normalized(value)
    return None if value == "" else float(value)


def markdown_value(column: str, value: str) -> str:
    value = normalized(value)
    if value == "":
        return "--"
    if column in {"seed", "steps", "sample_count", "mask_count"}:
        return str(int(float(value)))
    if column == "training_scenes":
        return str(int(float(value)))
    if column not in IDENTITY_COLUMNS:
        return f"{float(value):.6f}"
    return value.replace("|", r"\|")


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(markdown_value(column, row.get(column, "")) for column in columns)
            + " |"
        )
    return lines


def find_summary(runs_root: Path, run_id: str) -> Path:
    matches = sorted((runs_root / run_id).glob("*/summary.json"))
    require(len(matches) == 1, f"expected one summary for {run_id}, found {matches}")
    return matches[0]


def validate_summary_rows(
    manifest: dict,
    rows: list[dict[str, str]],
    metric_columns: list[str],
    runs_root: Path,
) -> list[Path]:
    by_id = {row["id"]: row for row in rows}
    manifest_ids = {entry["id"] for entry in manifest["entries"]}
    require(set(by_id) == manifest_ids, "per_seed.csv IDs do not match the manifest")
    require(len(by_id) == len(rows), "per_seed.csv contains duplicate IDs")
    result_paths = []
    for entry in manifest["entries"]:
        row = by_id[entry["id"]]
        summary_path = find_summary(runs_root, entry["id"])
        summary = json.loads(summary_path.read_text())
        result_paths.extend(
            [
                summary_path,
                summary_path.parent / "per_image.csv",
                summary_path.parent / "per_mask.csv",
            ]
        )
        require(
            int(row["sample_count"]) == int(summary["sample_count"]),
            f"sample count drift for {entry['id']}",
        )
        require(
            int(row["mask_count"]) == int(summary["mask_count"]),
            f"mask count drift for {entry['id']}",
        )
        for metric in metric_columns:
            expected = summary["mask_balanced"].get(metric)
            observed = number(row.get(metric, ""))
            if expected is None:
                require(observed is None, f"unexpected {metric} for {entry['id']}")
                continue
            require(observed is not None, f"missing {metric} for {entry['id']}")
            require(
                math.isfinite(observed)
                and math.isclose(observed, float(expected), rel_tol=0.0, abs_tol=1e-12),
                f"metric drift for {entry['id']}:{metric}",
            )
    for path in result_paths:
        require(path.is_file(), f"missing raw result artifact: {path}")
    return result_paths


def validate_inputs(root: Path) -> dict:
    results = root / "results/test"
    manifest_path = root / "manifest.json"
    configs_path = root / "configs_test_remote.json"
    launch_path = root / "launch_test.json"
    validation_path = results / "validation.json"
    raw_validation_path = results / "raw_validation.json"
    paths = {
        "manifest": manifest_path,
        "configs": configs_path,
        "launch": launch_path,
        "validation": validation_path,
        "raw_validation": raw_validation_path,
        "per_seed": results / "per_seed.csv",
        "aggregate": results / "aggregate.csv",
        "paired_per_seed": results / "paired_effects_per_seed.csv",
        "paired_aggregate": results / "paired_effects_aggregate.csv",
        "results": results / "RESULTS.md",
    }
    for path in paths.values():
        require(path.is_file(), f"missing required artifact: {path}")

    manifest = json.loads(manifest_path.read_text())
    launch = json.loads(launch_path.read_text())
    validation = json.loads(validation_path.read_text())
    raw_validation = json.loads(raw_validation_path.read_text())
    per_seed = read_csv(paths["per_seed"])
    aggregate = read_csv(paths["aggregate"])
    paired_per_seed = read_csv(paths["paired_per_seed"])
    paired_aggregate = read_csv(paths["paired_aggregate"])

    expected = int(manifest["standard_entry_count"])
    completed = launch.get("completed", [])
    require(launch.get("status") == "complete", "test launch is not complete")
    require(int(launch.get("entry_count", -1)) == expected, "launch entry count mismatch")
    require(len(completed) == expected, "not all launch jobs completed")
    require(len({item["id"] for item in completed}) == expected, "duplicate launch IDs")
    require(all(item.get("returncode") == 0 for item in completed), "a launch job failed")
    require(validation.get("status") == "pass", "summary validation did not pass")
    require(int(validation.get("run_count", -1)) == expected, "summary run count mismatch")
    require(raw_validation.get("status") == "pass", "raw validation did not pass")
    require(
        int(raw_validation.get("passed_run_count", -1)) == expected,
        "not every raw run passed validation",
    )
    require(int(raw_validation.get("failed_run_count", -1)) == 0, "raw run failures")
    require(not raw_validation.get("grid_errors"), "raw evaluation grids differ")
    require(
        not any(row.get("errors") for row in raw_validation.get("rows", [])),
        "raw validation contains per-run errors",
    )
    require(validation.get("manifest_sha256") == sha256(manifest_path), "manifest hash drift")
    require(validation.get("configs_manifest_sha256") == sha256(configs_path), "config hash drift")
    require(len(per_seed) == expected, "per_seed.csv row count mismatch")
    require(len(aggregate) == int(validation["aggregate_row_count"]), "aggregate row count mismatch")
    require(
        len(paired_per_seed) == int(validation["paired_effect_row_count"]),
        "paired per-seed row count mismatch",
    )
    require(
        len(paired_aggregate) == int(validation["paired_effect_aggregate_count"]),
        "paired aggregate row count mismatch",
    )
    require(
        sum(int(row["run_count"]) for row in aggregate) == expected,
        "aggregate groups do not cover every run exactly once",
    )
    metric_columns = list(validation["metric_columns"])
    raw_paths = validate_summary_rows(
        manifest,
        per_seed,
        metric_columns,
        root / "runs/test",
    )
    newest_raw_mtime = max(path.stat().st_mtime_ns for path in raw_paths)
    require(
        raw_validation_path.stat().st_mtime_ns >= newest_raw_mtime,
        "raw outputs changed after raw_validation.json was produced",
    )
    return {
        "paths": paths,
        "manifest": manifest,
        "launch": launch,
        "validation": validation,
        "raw_validation": raw_validation,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "paired_per_seed": paired_per_seed,
        "paired_aggregate": paired_aggregate,
        "metric_columns": metric_columns,
    }


def build_appendix(data: dict) -> str:
    rows = data["per_seed"]
    metrics = data["metric_columns"]
    datasets = sorted({row["dataset"] for row in rows})
    lines = [
        START_MARKER,
        "## Completion, validation, and complete per-run metrics",
        "",
        (
            "All 125 manifest entries completed successfully and are represented below. "
            "The tables report every available mask-balanced endpoint metric for each "
            "checkpoint; `--` denotes a metric that is not part of that evaluation contract."
        ),
        "",
        "### Completion and strict raw-output validation",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Manifest entries | {data['manifest']['standard_entry_count']} |",
        f"| Completed inference jobs | {len(data['launch']['completed'])} |",
        "| Jobs with non-zero return code | 0 |",
        f"| Summarized runs | {len(rows)} |",
        f"| Raw runs passing validation | {data['raw_validation']['passed_run_count']} |",
        f"| Raw runs failing validation | {data['raw_validation']['failed_run_count']} |",
        f"| Grid-level validation errors | {len(data['raw_validation']['grid_errors'])} |",
        "",
        (
            "Strict raw validation status: `pass`. The validator independently read every "
            "`per_image.csv`, recomputed its mask-balanced summary, and checked sample "
            "identities, row counts, mask balance, finite metric values, per-mask outputs, "
            "and common evaluation grids. The raw files have not changed since that check."
        ),
        "",
        "### Evaluation coverage",
        "",
        "| Dataset | Runs | Samples per run | Checkpoint-sample pairs |",
        "|---|---:|---:|---:|",
    ]
    total_pairs = 0
    for dataset in datasets:
        group = [row for row in rows if row["dataset"] == dataset]
        sample_counts = sorted({int(row["sample_count"]) for row in group})
        pair_count = sum(int(row["sample_count"]) for row in group)
        total_pairs += pair_count
        samples_per_run = ", ".join(str(value) for value in sample_counts)
        lines.append(f"| {dataset} | {len(group)} | {samples_per_run} | {pair_count} |")
    lines.extend(
        [
            f"| **Total** | **{len(rows)}** | -- | **{total_pairs}** |",
            "",
            "### Metric availability",
            "",
            "| Metric | Populated runs | Datasets |",
            "|---|---:|---|",
        ]
    )
    for metric in metrics:
        observed = [row for row in rows if normalized(row.get(metric, ""))]
        metric_datasets = sorted({row["dataset"] for row in observed})
        lines.append(f"| {metric} | {len(observed)} | {', '.join(metric_datasets)} |")

    lines.extend(["", "### Per-run metric tables", ""])
    for dataset in datasets:
        group = [row for row in rows if row["dataset"] == dataset]
        dataset_metrics = [
            metric
            for metric in metrics
            if any(normalized(row.get(metric, "")) for row in group)
        ]
        lines.extend(
            [
                f"#### {dataset}",
                "",
                *markdown_table(group, [*IDENTITY_COLUMNS, *dataset_metrics]),
                "",
            ]
        )

    artifact_names = (
        "manifest",
        "configs",
        "launch",
        "validation",
        "raw_validation",
        "per_seed",
        "aggregate",
        "paired_per_seed",
        "paired_aggregate",
    )
    lines.extend(
        [
            "### Consolidated artifact hashes",
            "",
            "| Artifact | SHA-256 |",
            "|---|---|",
        ]
    )
    root = data["paths"]["manifest"].parent
    for name in artifact_names:
        path = data["paths"][name]
        lines.append(f"| `{path.relative_to(root)}` | `{sha256(path)}` |")
    lines.extend([END_MARKER, ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Prepared report-test root containing manifest.json and results/test",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    data = validate_inputs(root)
    results_path = data["paths"]["results"]
    current = results_path.read_text()
    if START_MARKER in current:
        prefix, remainder = current.split(START_MARKER, 1)
        require(END_MARKER in remainder, "incomplete generated appendix markers")
        _, suffix = remainder.split(END_MARKER, 1)
        current = prefix.rstrip() + suffix
    appendix = build_appendix(data)
    results_path.write_text(current.rstrip() + "\n\n" + appendix)
    print(
        json.dumps(
            {
                "status": "pass",
                "results": str(results_path),
                "run_count": len(data["per_seed"]),
                "aggregate_row_count": len(data["aggregate"]),
                "paired_effect_row_count": len(data["paired_per_seed"]),
                "paired_effect_aggregate_count": len(data["paired_aggregate"]),
                "metric_count": len(data["metric_columns"]),
                "checkpoint_sample_pairs": sum(
                    int(row["sample_count"]) for row in data["per_seed"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
