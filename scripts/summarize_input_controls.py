"""Summarize completed E2 input-use controls without loading checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED = {
    "E2-mirflickr-xrest100k-seed42": ("MIRFLICKR", "X-Restormer", 42),
    "E2-celeba-10k-seed42": ("CelebA32", "X-Restormer", 42),
    "E2-celeba-10k-seed52": ("CelebA32", "X-Restormer", 52),
    "E2-celeba-10k-seed62": ("CelebA32", "X-Restormer", 62),
    "E2-celeba-drunet-10k-seed42": ("CelebA32", "DRUNet", 42),
    "E2-celeba-drunet-10k-seed52": ("CelebA32", "DRUNet", 52),
    "E2-celeba-drunet-10k-seed62": ("CelebA32", "DRUNet", 62),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_run(path: Path, dataset: str, model: str, seed: int) -> tuple[list[dict], dict]:
    summary_path = path / "summary.json"
    validation_path = path / "validation.json"
    if not summary_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(f"Incomplete input-control directory: {path}")
    summary = json.loads(summary_path.read_text())
    validation = json.loads(validation_path.read_text())
    if summary.get("status") != "complete" or validation.get("complete") is not True:
        raise ValueError(f"Input-control run is not complete: {path}")
    if int(summary.get("sample_count", -1)) != 1024:
        raise ValueError(f"Unexpected sample count: {path}")
    if int(validation.get("mask_count", -1)) != 32:
        raise ValueError(f"Unexpected mask count: {path}")
    baseline_deltas = validation.get("baseline_deltas", {})
    for metric in ("PSNR", "SSIM", "LPIPS"):
        tolerance = 0.01 if metric == "PSNR" else 0.001
        if metric not in baseline_deltas or abs(float(baseline_deltas[metric])) > tolerance:
            raise ValueError(f"Baseline parity failed for {path}: {metric}")

    rows: list[dict] = []
    for condition, metrics in summary["conditions"].items():
        for metric, value in metrics.items():
            rows.append(
                {
                    "run": path.name,
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "condition": condition,
                    "metric": metric,
                    "value": float(value),
                    "sample_count": int(summary["sample_count"]),
                    "mask_count": int(validation["mask_count"]),
                }
            )
    return rows, {
        "run": path.name,
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "baseline_deltas": baseline_deltas,
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "peak_vram_bytes": summary.get("peak_vram_bytes"),
        "summary": str(summary_path),
        "validation": str(validation_path),
    }


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    correct = {
        (row["run"], row["metric"]): row["value"]
        for row in rows
        if row["condition"] == "Correct"
    }
    effects: list[dict] = []
    for row in rows:
        if row["condition"] == "Correct":
            continue
        baseline = correct[(row["run"], row["metric"])]
        effects.append(
            {
                "run": row["run"],
                "dataset": row["dataset"],
                "model": row["model"],
                "seed": row["seed"],
                "condition": row["condition"],
                "metric": row["metric"],
                "condition_minus_correct": row["value"] - baseline,
            }
        )

    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["condition"], row["metric"])].append(
            row["value"]
        )
    summary_rows: list[dict] = []
    for (dataset, model, condition, metric), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "condition": condition,
                "metric": metric,
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else "",
                "n_training_runs": len(values),
            }
        )
    return effects, summary_rows


def result_markdown(rows: list[dict], effects: list[dict]) -> str:
    lines = [
        "# Input-use controls",
        "",
        "Все результаты получены на development data. Final synthetic test не открывался.",
        "",
    ]
    groups = sorted({(row["dataset"], row["model"]) for row in rows})
    for dataset, model in groups:
        group_rows = [
            row for row in rows if row["dataset"] == dataset and row["model"] == model
        ]
        if not group_rows:
            continue
        metrics = (
            ("PSNR", "SSIM", "LPIPS")
            if dataset == "MIRFLICKR"
            else ("PSNR_coarse", "SSIM_coarse", "LPIPS")
        )
        lines.extend(
            [
                f"## {dataset}: {model}",
                "",
                "| Seed | Condition | " + " | ".join(metrics) + " |",
                "|---:|---|" + "---:|" * len(metrics),
            ]
        )
        by_run_condition: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
        for row in group_rows:
            by_run_condition[(int(row["seed"]), row["condition"])][row["metric"]] = float(
                row["value"]
            )
        order = {"Correct": 0, "Identity": 1, "Shuffled": 2, "Train mean": 3}
        for (seed, condition), values in sorted(
            by_run_condition.items(), key=lambda item: (item[0][0], order[item[0][1]])
        ):
            lines.append(
                f"| {seed} | {condition} | "
                + " | ".join(f"{values[metric]:.6f}" for metric in metrics)
                + " |"
            )
        lines.append("")

        shuffled = [
            row
            for row in effects
            if row["dataset"] == dataset
            and row["model"] == model
            and row["condition"] == "Shuffled"
        ]
        by_metric = defaultdict(list)
        for row in shuffled:
            by_metric[row["metric"]].append(float(row["condition_minus_correct"]))
        lines.append(
            "Shuffled minus Correct: "
            + "; ".join(
                f"{metric} {statistics.mean(by_metric[metric]):+.6f}"
                + (
                    f" ± {statistics.stdev(by_metric[metric]):.6f}"
                    if len(by_metric[metric]) > 1
                    else ""
                )
                for metric in metrics
            )
            + "."
        )
        lines.append("")
    lines.extend(
        [
            "`Train mean` is a measurement-independent prediction of the mean training target, not a mean-measurement input.",
            "For LPIPS, a positive Shuffled-minus-Correct value means worse quality.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-root",
        default="outputs/coursework_completion_20260907/evaluations",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/coursework_status_20260909/input_controls",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    evaluation_root = (repo / args.evaluation_root).resolve()
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    validations: list[dict] = []
    missing: list[str] = []
    for name, (dataset, model, seed) in EXPECTED.items():
        path = evaluation_root / name
        if not path.is_dir():
            missing.append(name)
            continue
        run_rows, validation = load_run(path, dataset, model, seed)
        rows.extend(run_rows)
        validations.append(validation)

    effects, summary_rows = aggregate(rows)
    write_csv(output_dir / "per_run_metrics.csv", rows)
    write_csv(output_dir / "paired_effects.csv", effects)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "RESULTS.md").write_text(result_markdown(rows, effects))
    status = {
        "schema_version": 1,
        "complete": not missing,
        "expected_runs": len(EXPECTED),
        "completed_runs": len(validations),
        "missing": missing,
        "checkpoint_deserialization": False,
        "final_test_accessed": False,
        "validations": validations,
    }
    (output_dir / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
