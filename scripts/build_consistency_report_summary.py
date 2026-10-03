"""Build report-ready tables from the validated consistency-scaling results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

DR_METRICS = ("PSNR", "SSIM", "LPIPS")
MNIST_METRICS = ("PSNR", "SSIM", "PSNR_32", "SSIM_32", "Dice_loss_32")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_absolute(
    frame: pd.DataFrame,
    keys: list[str],
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for group_values, group in frame.groupby(keys, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        prefix = dict(zip(keys, group_values))
        for metric in metrics:
            values = [float(value) for value in group[metric]]
            rows.append(
                {
                    **prefix,
                    "metric": metric,
                    "mean": statistics.mean(values),
                    "sample_sd": statistics.stdev(values),
                    "run_count": len(values),
                }
            )
    return pd.DataFrame(rows)


def improved_count(values: pd.Series, metric: str) -> int:
    lower_is_better = metric in {"LPIPS", "Dice_loss_32"}
    return int((values < 0).sum() if lower_is_better else (values > 0).sum())


def validate_source_aggregates(bundle: Path) -> dict:
    validation = json.loads((bundle / "results/validation.json").read_text())
    state = json.loads((bundle / "launcher/launcher_state.json").read_text())
    if not (
        validation["status"] == "pass"
        and state["status"] == "complete"
        and state["completed_job_count"] == 20
        and state["failed_job_count"] == 0
        and not state["final_test_accessed"]
    ):
        raise ValueError("Consistency-scaling source bundle is not valid and complete")
    return validation


def compare_aggregate(
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
    keys: list[str],
    metrics: tuple[str, ...],
) -> None:
    for _, row in aggregate.iterrows():
        selected = per_seed
        for key in keys:
            selected = selected[selected[key] == row[key]]
        values = selected[str(row["metric"])]
        if str(row["metric"]) not in metrics or not (
            np.isclose(values.mean(), row["mean_effect"])
            and np.isclose(values.std(ddof=1), row["sample_sd"])
            and len(values) == int(row["run_count"])
        ):
            raise ValueError("Stored aggregate does not match per-seed effects")


def value(
    frame: pd.DataFrame,
    filters: dict[str, object],
    metric: str,
    column: str,
) -> float:
    selected = frame
    for key, expected in filters.items():
        selected = selected[selected[key] == expected]
    row = selected[selected["metric"] == metric]
    if len(row) != 1:
        raise ValueError(f"Expected one aggregate row for {filters}, {metric}")
    return float(row.iloc[0][column])


def plus(value_: float, digits: int) -> str:
    return f"{value_:+.{digits}f}"


def mean_sd(mean: float, sd: float, digits: int) -> str:
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle",
        nargs="?",
        default="outputs/coursework_consistency_scaling_v2_20260912",
    )
    parser.add_argument("--rewrite", action="store_true")
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    output = bundle / "report_ready"
    source_validation = validate_source_aggregates(bundle)

    dr = pd.read_csv(bundle / "results/drunet_endpoints.csv")
    dr_effects = pd.read_csv(bundle / "results/drunet_scene_effects_per_seed.csv")
    dr_stored = pd.read_csv(bundle / "results/drunet_scene_effects_aggregate.csv")
    mnist = pd.read_csv(bundle / "results/mnist_learning_curves.csv")
    mnist_effects = pd.read_csv(bundle / "results/mnist_effects_per_seed.csv")
    mnist_stored = pd.read_csv(bundle / "results/mnist_effects_aggregate.csv")

    if not (
        len(dr) == 24
        and set(dr["seed"]) == {42, 52, 62}
        and len(mnist) == 100
        and set(mnist["seed"]) == {42, 52, 62, 72, 82}
    ):
        raise ValueError("Unexpected endpoint matrix")
    if not (
        np.isfinite(dr[list(DR_METRICS)].to_numpy()).all()
        and np.isfinite(mnist[list(MNIST_METRICS)].to_numpy()).all()
    ):
        raise ValueError("Non-finite report input")

    compare_aggregate(dr_effects, dr_stored, ["condition"], DR_METRICS)
    compare_aggregate(
        mnist_effects,
        mnist_stored,
        ["contrast", "steps"],
        MNIST_METRICS,
    )

    dr_absolute = aggregate_absolute(
        dr,
        ["training_scenes", "condition"],
        DR_METRICS,
    )
    dr_effect_rows = []
    for condition, group in dr_effects.groupby("condition", sort=True):
        for metric in DR_METRICS:
            values = group[metric]
            dr_effect_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "mean_effect": values.mean(),
                    "sample_sd": values.std(ddof=1),
                    "run_count": len(values),
                    "improved_run_count": improved_count(values, metric),
                }
            )
    dr_effect_summary = pd.DataFrame(dr_effect_rows)

    mnist_selected = mnist[mnist["steps"].isin([10000, 50000])]
    mnist_absolute = aggregate_absolute(
        mnist_selected,
        ["steps", "information_regime"],
        MNIST_METRICS,
    )
    sensitivity_rows = []
    populations = {
        "all_five_seeds": mnist_effects,
        "without_seed62_sensitivity": mnist_effects[mnist_effects["seed"] != 62],
    }
    for population, frame in populations.items():
        for steps, group in frame.groupby("steps", sort=True):
            for metric in MNIST_METRICS:
                values = group[metric]
                sensitivity_rows.append(
                    {
                        "population": population,
                        "steps": steps,
                        "metric": metric,
                        "mean_effect": values.mean(),
                        "sample_sd": values.std(ddof=1),
                        "run_count": len(values),
                        "improved_run_count": improved_count(values, metric),
                    }
                )
    mnist_sensitivity = pd.DataFrame(sensitivity_rows)

    output.mkdir(parents=True, exist_ok=bool(args.rewrite))
    dr_absolute.to_csv(output / "drunet_absolute_aggregate.csv", index=False)
    dr_effect_summary.to_csv(output / "drunet_effect_summary.csv", index=False)
    mnist_absolute.to_csv(output / "mnist_absolute_aggregate.csv", index=False)
    mnist_sensitivity.to_csv(output / "mnist_effect_sensitivity.csv", index=False)

    source_files = [
        bundle / "manifest.json",
        bundle / "launcher/launcher_state.json",
        bundle / "results/validation.json",
        bundle / "results/drunet_endpoints.csv",
        bundle / "results/drunet_scene_effects_per_seed.csv",
        bundle / "results/mnist_learning_curves.csv",
        bundle / "results/mnist_effects_per_seed.csv",
    ]
    validation = {
        "status": "pass",
        "source_validation": source_validation["status"],
        "drunet_endpoint_count": len(dr),
        "drunet_repetition_seeds": [42, 52, 62],
        "mnist_curve_rows": len(mnist),
        "mnist_repetition_seeds": [42, 52, 62, 72, 82],
        "stored_aggregates_recomputed": True,
        "seed62_included_in_primary_aggregate": True,
        "seed62_exclusion_used_only_as_sensitivity": True,
        "final_test_accessed": False,
        "source_sha256": {
            str(path.relative_to(bundle)): sha256(path) for path in source_files
        },
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n"
    )

    lines = [
        "# Consistency scaling: report-ready aggregation",
        "",
        "All values use the fixed development grid of 32 masks by 32 scenes. "
        "The primary summaries include every completed endpoint, including the "
        "valid but divergent MNIST seed 62 run.",
        "",
        "## DRUNet scene-count scaling",
        "",
        "| Mask regime | PSNR, 4k scenes | PSNR, 16k scenes | Delta PSNR | Delta SSIM | Delta LPIPS | PSNR improved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    conditions = ("finite100", "finite1000", "finite10000", "streaming")
    for condition in conditions:
        filters4 = {"training_scenes": 4096, "condition": condition}
        filters16 = {"training_scenes": 16384, "condition": condition}
        effect_filter = {"condition": condition}
        psnr4 = mean_sd(
            value(dr_absolute, filters4, "PSNR", "mean"),
            value(dr_absolute, filters4, "PSNR", "sample_sd"),
            4,
        )
        psnr16 = mean_sd(
            value(dr_absolute, filters16, "PSNR", "mean"),
            value(dr_absolute, filters16, "PSNR", "sample_sd"),
            4,
        )
        improved = int(
            value(
                dr_effect_summary,
                effect_filter,
                "PSNR",
                "improved_run_count",
            )
        )
        lines.append(
            f"| {condition} | {psnr4} | {psnr16} | "
            f"{plus(value(dr_effect_summary, effect_filter, 'PSNR', 'mean_effect'), 4)} | "
            f"{plus(value(dr_effect_summary, effect_filter, 'SSIM', 'mean_effect'), 5)} | "
            f"{plus(value(dr_effect_summary, effect_filter, 'LPIPS', 'mean_effect'), 5)} | "
            f"{improved}/3 |"
        )
    lines.extend(
        [
            "",
            "The effect of increasing the scene pool is not uniform across mask "
            "regimes. Mean PSNR is almost unchanged for finite100, improves for "
            "finite1000 and finite10000, and decreases for streaming. The "
            "finite100 and streaming effects also change sign across seeds. "
            "The finite10000 mean is driven mainly by the larger gain in seed 52.",
            "",
            "## MNIST PSF-aware comparison",
            "",
            "| Information regime | PSNR at 50k | PSNR32 at 50k | Dice loss at 50k |",
            "|---|---:|---:|---:|",
        ]
    )
    for regime in ("psf_free", "psf_aware"):
        filters = {"steps": 50000, "information_regime": regime}
        lines.append(
            f"| {regime} | "
            f"{mean_sd(value(mnist_absolute, filters, 'PSNR', 'mean'), value(mnist_absolute, filters, 'PSNR', 'sample_sd'), 4)} | "
            f"{mean_sd(value(mnist_absolute, filters, 'PSNR_32', 'mean'), value(mnist_absolute, filters, 'PSNR_32', 'sample_sd'), 4)} | "
            f"{mean_sd(value(mnist_absolute, filters, 'Dice_loss_32', 'mean'), value(mnist_absolute, filters, 'Dice_loss_32', 'sample_sd'), 5)} |"
        )
    lines.extend(
        [
            "",
            "The paired effects below are PSF-aware minus PSF-free. Negative "
            "Dice-loss values indicate improvement.",
            "",
            "| Steps | Population | Delta PSNR | Delta PSNR32 | Delta Dice loss | PSNR improved |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for steps in (10000, 50000):
        for population in ("all_five_seeds", "without_seed62_sensitivity"):
            filters = {"steps": steps, "population": population}
            run_count = int(value(mnist_sensitivity, filters, "PSNR", "run_count"))
            improved = int(
                value(
                    mnist_sensitivity,
                    filters,
                    "PSNR",
                    "improved_run_count",
                )
            )
            lines.append(
                f"| {steps} | {population} | "
                f"{plus(value(mnist_sensitivity, filters, 'PSNR', 'mean_effect'), 4)} | "
                f"{plus(value(mnist_sensitivity, filters, 'PSNR_32', 'mean_effect'), 4)} | "
                f"{plus(value(mnist_sensitivity, filters, 'Dice_loss_32', 'mean_effect'), 5)} | "
                f"{improved}/{run_count} |"
            )
    lines.extend(
        [
            "",
            "At 50k steps, PSF-aware training does not show a stable average "
            "advantage across the five matched runs: PSNR improves in three of "
            "five seeds, while the all-seed mean is lower. Removing seed 62 "
            "changes the mean direction, but this is a sensitivity analysis, not "
            "a replacement estimand. Seed 62 passed the protocol audit and remains "
            "in the primary aggregate.",
            "",
            "These scores alone do not demonstrate effective use of PSF input. "
            "The separate correct-versus-shuffled PSF replay should be reported "
            "alongside this table.",
        ]
    )
    (output / "RESULTS_FOR_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(validation, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
