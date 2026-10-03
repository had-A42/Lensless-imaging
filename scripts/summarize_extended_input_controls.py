"""Aggregate the completed development-only zero and cross-mask controls."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/coursework_post_final_20260911/extended_input_controls_v1"
OUTPUT = ROOT / "results_v1"
METRICS = ("PSNR", "SSIM", "LPIPS")


def classify(name: str) -> tuple[str, str, int]:
    seed = int(name.rsplit("seed", 1)[1])
    if name.startswith("extended-celeba-drunet"):
        return "CelebA", "DRUNet", seed
    if name.startswith("extended-celeba-xrest"):
        return "CelebA", "X-Restormer", seed
    if "xrest100k" in name:
        return "MIRFLICKR", "X-Restormer 100k GoPro", seed
    initialization = "GoPro" if "gopro" in name else "scratch"
    return "MIRFLICKR", f"X-Restormer 50k {initialization}", seed


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    state = json.loads((ROOT / "launcher/launcher_state.json").read_text())
    if state["status"] != "complete" or state["failed_jobs"]:
        raise ValueError("Extended-control launcher is not complete")
    rows = []
    sources = []
    for run in sorted((ROOT / "development_runs").iterdir()):
        if not run.is_dir():
            continue
        validation = json.loads((run / "validation.json").read_text())
        summary = json.loads((run / "summary.json").read_text())
        if validation["status"] != "pass" or summary["status"] != "complete":
            raise ValueError(f"Incomplete run: {run.name}")
        dataset, model, seed = classify(run.name)
        row = {"id": run.name, "dataset": dataset, "model": model, "seed": seed}
        for metric in METRICS:
            correct = summary["conditions"]["Correct"][metric]
            zero = summary["conditions"]["Zero"][metric]
            row[f"correct_{metric}"] = correct
            row[f"zero_{metric}"] = zero
            row[f"zero_minus_correct_{metric}"] = zero - correct
            row[f"other_mask_mean_abs_{metric}"] = summary[
                "paired_other_mask_minus_correct_mean_absolute"
            ][metric]
        row["cross_mask_prediction_MAE"] = summary["cross_mask_consistency_mean"][
            "prediction_pair_MAE"
        ]
        row["cross_mask_prediction_RMSE"] = summary["cross_mask_consistency_mean"][
            "prediction_pair_RMSE"
        ]
        if not all(
            math.isfinite(value)
            for key, value in row.items()
            if key not in {"id", "dataset", "model"}
        ):
            raise ValueError(f"Non-finite values: {run.name}")
        rows.append(row)
        sources.extend([str(run / "summary.json"), str(run / "validation.json")])
    if len(rows) != 11:
        raise ValueError(f"Expected 11 runs, found {len(rows)}")
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "per_run.csv", index=False)
    value_columns = [column for column in frame if column not in {"id", "dataset", "model", "seed"}]
    aggregate_rows = []
    for (dataset, model), group in frame.groupby(["dataset", "model"], sort=True):
        for metric in value_columns:
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "mean": float(group[metric].mean()),
                    "sample_sd": float(group[metric].std(ddof=1)) if len(group) > 1 else "",
                    "run_count": len(group),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUTPUT / "aggregate.csv", index=False)
    validation = {
        "status": "pass",
        "run_count": len(frame),
        "all_source_validations_pass": True,
        "all_metrics_finite": True,
        "data_partition": "development",
        "final_test_accessed": False,
        "sources": sources,
    }
    (OUTPUT / "validation.json").write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Extended input controls",
        "",
        "| Dataset | Model | Runs | Zero - Correct PSNR | Zero - Correct SSIM | Zero - Correct LPIPS | Cross-mask prediction RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (dataset, model), group in frame.groupby(["dataset", "model"], sort=True):
        lines.append(
            f"| {dataset} | {model} | {len(group)} | "
            f"{group['zero_minus_correct_PSNR'].mean():+.4f} | "
            f"{group['zero_minus_correct_SSIM'].mean():+.4f} | "
            f"{group['zero_minus_correct_LPIPS'].mean():+.4f} | "
            f"{group['cross_mask_prediction_RMSE'].mean():.4f} |"
        )
    lines.extend(
        [
            "",
            "Negative PSNR/SSIM and positive LPIPS changes indicate degradation under zero input. Cross-mask RMSE compares reconstructions of the same scene under adjacent unseen masks.",
            "",
            "The aggregate target score for the cyclic other-mask condition equals Correct by construction on a balanced complete grid; the paired absolute changes and prediction-to-prediction distances are the informative quantities.",
        ]
    )
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(validation, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
