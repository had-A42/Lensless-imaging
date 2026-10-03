"""Aggregate the three-seed DRUNet factorial and five-seed MNIST matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import pandas as pd


DR_METRICS = ("PSNR", "SSIM", "LPIPS")
MNIST_METRICS = ("PSNR", "SSIM", "PSNR_32", "SSIM_32", "Dice_loss_32")


def read_mask_means(path: str | Path, metrics: tuple[str, ...]) -> dict[str, float]:
    frame = pd.read_csv(path)
    if len(frame) != 32 or frame["mask_id"].nunique() != 32 or set(frame["sample_count"]) != {32}:
        raise ValueError(f"Invalid 32x32 grid: {path}")
    result = {metric: float(frame[metric].mean()) for metric in metrics}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"Non-finite metric: {path}")
    return result


def dice_curve(log: Path) -> dict[int, float]:
    pattern = re.compile(r"validation_pooled_dice_loss:\s+([0-9.eE+-]+)")
    values = [
        float(match.group(1))
        for line in log.read_text().splitlines()
        if (match := pattern.search(line))
    ]
    if len(values) != 10 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Invalid Dice curve: {log}")
    return {5000 * (index + 1): value for index, value in enumerate(values)}


def aggregate_effects(frame: pd.DataFrame, keys: list[str], metrics: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for group_values, group in frame.groupby(keys, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        prefix = dict(zip(keys, group_values))
        for metric in metrics:
            values = list(group[metric])
            rows.append(
                {
                    **prefix,
                    "metric": metric,
                    "mean_effect": statistics.mean(values),
                    "sample_sd": statistics.stdev(values),
                    "run_count": len(values),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    bundle = root / manifest["bundle"]
    output = bundle / "results"
    output.mkdir(parents=True, exist_ok=False)
    parent = Path(manifest["parent_manifest"]).parent
    parent_manifest = json.loads((parent / "manifest.json").read_text())

    dr_rows = []
    parent_endpoints = pd.read_csv(parent / "results/endpoints.csv")
    parent_dr = parent_endpoints[
        parent_endpoints["experiment"] == "drunet_scene_scaling"
    ]
    for row in parent_dr.to_dict("records"):
        dr_rows.append(
            {
                "seed": 42,
                "training_scenes": int(row["training_scenes"]),
                "condition": row["condition"],
                **{metric: float(row[metric]) for metric in DR_METRICS},
            }
        )
    finite100 = parent_manifest["drunet_protocol"]["historical_controls"]["finite100"]
    dr_rows.append(
        {
            "seed": 42,
            "training_scenes": 4096,
            "condition": "finite100",
            **read_mask_means(finite100["metrics_csv"], DR_METRICS),
        }
    )
    for job in manifest["jobs"]:
        if job["experiment"] != "drunet_scene_mask_factorial":
            continue
        run = root / job["output"]
        complete = json.loads((run / "job_complete.json").read_text())
        if complete["status"] != "complete" or complete["global_step"] != 100000:
            raise ValueError(f"Incomplete DRUNet job: {job['name']}")
        dr_rows.append(
            {
                "seed": job["seed"],
                "training_scenes": job["training_scenes"],
                "condition": job["condition"],
                **read_mask_means(
                    run / "validation_per_mask_epoch0010.csv", DR_METRICS
                ),
            }
        )
    dr = pd.DataFrame(dr_rows).sort_values(["seed", "condition", "training_scenes"])
    if len(dr) != 24:
        raise ValueError(f"Expected 24 DRUNet endpoints, found {len(dr)}")
    dr.to_csv(output / "drunet_endpoints.csv", index=False)
    dr_effect_rows = []
    for (seed, condition), group in dr.groupby(["seed", "condition"]):
        indexed = group.set_index("training_scenes")
        row = {"seed": seed, "condition": condition}
        for metric in DR_METRICS:
            row[metric] = indexed.loc[16384, metric] - indexed.loc[4096, metric]
        dr_effect_rows.append(row)
    dr_effects = pd.DataFrame(dr_effect_rows)
    dr_effects.to_csv(output / "drunet_scene_effects_per_seed.csv", index=False)
    dr_aggregate = aggregate_effects(dr_effects, ["condition"], DR_METRICS)
    dr_aggregate.to_csv(output / "drunet_scene_effects_aggregate.csv", index=False)

    curves = [pd.read_csv(parent / "results/mnist_learning_curves.csv")]
    for job in manifest["jobs"]:
        if job["experiment"] != "mnist_replication_extension":
            continue
        run = root / job["output"]
        complete = json.loads((run / "job_complete.json").read_text())
        if complete["status"] != "complete" or complete["global_step"] != 50000:
            raise ValueError(f"Incomplete MNIST job: {job['name']}")
        dice = dice_curve(run / "info.log")
        rows = []
        for epoch in range(1, 11):
            steps = epoch * 5000
            values = read_mask_means(
                run / f"validation_per_mask_epoch{epoch:04d}.csv",
                MNIST_METRICS[:-1],
            )
            rows.append(
                {
                    "name": job["name"],
                    "information_regime": job["information_regime"],
                    "seed": job["seed"],
                    "steps": steps,
                    **values,
                    "Dice_loss_32": dice[steps],
                }
            )
        curves.append(pd.DataFrame(rows))
    mnist = pd.concat(curves, ignore_index=True).sort_values(
        ["seed", "information_regime", "steps"]
    )
    if len(mnist) != 100 or set(mnist["seed"]) != {42, 52, 62, 72, 82}:
        raise ValueError("Expected a five-seed, two-regime, ten-endpoint MNIST matrix")
    mnist.to_csv(output / "mnist_learning_curves.csv", index=False)
    indexed = mnist.set_index(["seed", "information_regime", "steps"])
    mnist_effect_rows = []
    for seed in sorted(mnist["seed"].unique()):
        for steps in (10000, 50000):
            row = {
                "seed": seed,
                "steps": steps,
                "contrast": "psf_aware_minus_psf_free",
            }
            for metric in MNIST_METRICS:
                row[metric] = (
                    indexed.loc[(seed, "psf_aware", steps), metric]
                    - indexed.loc[(seed, "psf_free", steps), metric]
                )
            mnist_effect_rows.append(row)
    mnist_effects = pd.DataFrame(mnist_effect_rows)
    mnist_effects.to_csv(output / "mnist_effects_per_seed.csv", index=False)
    mnist_aggregate = aggregate_effects(
        mnist_effects, ["contrast", "steps"], MNIST_METRICS
    )
    mnist_aggregate.to_csv(output / "mnist_effects_aggregate.csv", index=False)

    validation = {
        "status": "pass",
        "new_job_count": len(manifest["jobs"]),
        "drunet_endpoint_count": len(dr),
        "drunet_repetition_seeds": [42, 52, 62],
        "mnist_curve_rows": len(mnist),
        "mnist_repetition_seeds": [42, 52, 62, 72, 82],
        "all_fixed_endpoints_complete": True,
        "final_test_accessed": False,
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Consistency scaling results",
        "",
        "## DRUNet: 16,384 minus 4,096 training scenes",
        "",
        "| Mask regime | Delta PSNR | Delta SSIM | Delta LPIPS |",
        "|---|---:|---:|---:|",
    ]
    for condition in manifest["drunet_protocol"]["mask_regimes"]:
        values = dr_aggregate[dr_aggregate["condition"] == condition].set_index("metric")
        lines.append(
            f"| {condition} | {values.loc['PSNR','mean_effect']:+.4f} | "
            f"{values.loc['SSIM','mean_effect']:+.5f} | "
            f"{values.loc['LPIPS','mean_effect']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "## MNIST: PSF-aware minus PSF-free",
            "",
            "| Steps | Metric | Mean effect | SD |",
            "|---:|---|---:|---:|",
        ]
    )
    for _, row in mnist_aggregate.iterrows():
        lines.append(
            f"| {int(row['steps'])} | {row['metric']} | {row['mean_effect']:+.5f} | {row['sample_sd']:.5f} |"
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(validation, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
