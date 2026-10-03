"""Summarize completed DR16k and MNIST PSF-aware training endpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


DR_METRICS = ("PSNR", "SSIM", "LPIPS")
MNIST_METRICS = ("PSNR", "SSIM", "PSNR_32", "SSIM_32", "Dice_loss_32")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_mask_means(path: str | Path, metrics: tuple[str, ...]) -> dict[str, float]:
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 32 or len({row["mask_id"] for row in rows}) != 32:
        raise ValueError(f"Expected 32 balanced masks: {path}")
    if {int(row["sample_count"]) for row in rows} != {32}:
        raise ValueError(f"Expected 32 scenes per mask: {path}")
    values = {
        metric: statistics.mean(float(row[metric]) for row in rows)
        for metric in metrics
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite metrics: {path}")
    return values


def dice_by_epoch(info_log: Path) -> dict[int, float]:
    values = []
    pattern = re.compile(r"validation_pooled_dice_loss:\s+([0-9.eE+-]+)")
    for line in info_log.read_text().splitlines():
        match = pattern.search(line)
        if match:
            values.append(float(match.group(1)))
    if len(values) != 10 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected ten MNIST Dice validation values: {info_log}")
    return {epoch: values[epoch - 1] for epoch in range(1, 11)}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    jobs = {job["name"]: job for job in manifest["jobs"]}
    source_artifacts = []
    endpoints = []
    mnist_curves = []

    for job in jobs.values():
        run = root / job["output"]
        complete_path = run / "job_complete.json"
        complete = json.loads(complete_path.read_text())
        if complete["status"] != "complete" or complete["global_step"] != job["steps"]:
            raise ValueError(f"Incomplete job: {job['name']}")
        source_artifacts.append(
            {"path": str(complete_path), "sha256": sha256(complete_path)}
        )
        if job["experiment"] == "drunet_scene_scaling":
            metrics_path = run / "validation_per_mask_epoch0010.csv"
            values = read_mask_means(metrics_path, DR_METRICS)
            endpoints.append(
                {
                    "experiment": job["experiment"],
                    "name": job["name"],
                    "condition": job["condition"],
                    "training_scenes": job["training_scenes"],
                    "information_regime": "psf_free",
                    "seed": job["seed"],
                    "steps": job["steps"],
                    **values,
                    "PSNR_32": "",
                    "SSIM_32": "",
                    "Dice_loss_32": "",
                }
            )
            source_artifacts.append(
                {"path": str(metrics_path), "sha256": sha256(metrics_path)}
            )
        else:
            dice = dice_by_epoch(run / "info.log")
            for epoch in range(1, 11):
                metrics_path = run / f"validation_per_mask_epoch{epoch:04d}.csv"
                values = read_mask_means(metrics_path, MNIST_METRICS[:-1])
                row = {
                    "name": job["name"],
                    "information_regime": job["information_regime"],
                    "seed": job["seed"],
                    "steps": epoch * 5000,
                    **values,
                    "Dice_loss_32": dice[epoch],
                }
                mnist_curves.append(row)
                if epoch == 10:
                    endpoints.append(
                        {
                            "experiment": job["experiment"],
                            "name": job["name"],
                            "condition": job["condition"],
                            "training_scenes": "",
                            "information_regime": job["information_regime"],
                            "seed": job["seed"],
                            "steps": job["steps"],
                            **values,
                            "LPIPS": "",
                            "Dice_loss_32": dice[epoch],
                        }
                    )
                source_artifacts.append(
                    {"path": str(metrics_path), "sha256": sha256(metrics_path)}
                )

    dr_effects = []
    dr_new = {
        (row["condition"], int(row["training_scenes"])): row
        for row in endpoints
        if row["experiment"] == "drunet_scene_scaling"
    }
    for condition in manifest["drunet_protocol"]["mask_regimes"]:
        new = dr_new[(condition, 16384)]
        record = manifest["drunet_protocol"]["historical_controls"][condition]
        if record is None:
            control = dr_new[(condition, 4096)]
        else:
            values = read_mask_means(record["metrics_csv"], DR_METRICS)
            control = values
            source_artifacts.append(
                {"path": record["metrics_csv"], "sha256": record["metrics_sha256"]}
            )
        for metric in DR_METRICS:
            dr_effects.append(
                {
                    "experiment": "drunet_scene_scaling",
                    "contrast": "scenes16384_minus_scenes4096",
                    "condition": condition,
                    "seed": 42,
                    "steps": 100000,
                    "metric": metric,
                    "effect": float(new[metric]) - float(control[metric]),
                }
            )

    mnist_effects = []
    curve_index = {
        (row["information_regime"], row["seed"], row["steps"]): row
        for row in mnist_curves
    }
    for seed in manifest["mnist_protocol"]["seeds"]:
        for steps in (10000, 50000):
            free = curve_index[("psf_free", seed, steps)]
            aware = curve_index[("psf_aware", seed, steps)]
            for metric in MNIST_METRICS:
                mnist_effects.append(
                    {
                        "experiment": "mnist_long_psf_comparison",
                        "contrast": "psf_aware_minus_psf_free",
                        "endpoint_steps": steps,
                        "information_regime": "paired",
                        "seed": seed,
                        "metric": metric,
                        "effect": aware[metric] - free[metric],
                    }
                )
        for regime in ("psf_free", "psf_aware"):
            short = curve_index[(regime, seed, 10000)]
            long = curve_index[(regime, seed, 50000)]
            for metric in MNIST_METRICS:
                mnist_effects.append(
                    {
                        "experiment": "mnist_long_psf_comparison",
                        "contrast": "steps50000_minus_steps10000",
                        "endpoint_steps": 50000,
                        "information_regime": regime,
                        "seed": seed,
                        "metric": metric,
                        "effect": long[metric] - short[metric],
                    }
                )

    aggregate = []
    keys = sorted(
        {
            (
                row["contrast"],
                row["endpoint_steps"],
                row["information_regime"],
                row["metric"],
            )
            for row in mnist_effects
        }
    )
    for contrast, steps, regime, metric in keys:
        values = [
            row["effect"]
            for row in mnist_effects
            if (
                row["contrast"],
                row["endpoint_steps"],
                row["information_regime"],
                row["metric"],
            )
            == (contrast, steps, regime, metric)
        ]
        aggregate.append(
            {
                "contrast": contrast,
                "endpoint_steps": steps,
                "information_regime": regime,
                "metric": metric,
                "mean_effect": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
                "run_count": len(values),
            }
        )

    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "mnist_learning_curves.csv", mnist_curves)
    write_csv(output / "drunet_scene_effects.csv", dr_effects)
    write_csv(output / "mnist_effects_per_seed.csv", mnist_effects)
    write_csv(output / "mnist_effects_aggregate.csv", aggregate)
    validation = {
        "status": "pass",
        "job_count": len(jobs),
        "drunet_job_count": sum(
            job["experiment"] == "drunet_scene_scaling" for job in jobs.values()
        ),
        "mnist_job_count": sum(
            job["experiment"] == "mnist_long_psf_comparison" for job in jobs.values()
        ),
        "all_fixed_endpoints_complete": True,
        "final_test_accessed": False,
        "post_final_model_selection": False,
        "source_artifacts": source_artifacts,
    }
    save_json(output / "validation.json", validation)
    lines = [
        "# Long DRUNet and MNIST development results",
        "",
        "All models use fixed final endpoints. This queue accesses train and development data only; it does not rerun or inspect the final test.",
        "",
        "## DRUNet scene scaling",
        "",
        "| Mask regime | ΔPSNR | ΔSSIM | ΔLPIPS |",
        "|---|---:|---:|---:|",
    ]
    for condition in manifest["drunet_protocol"]["mask_regimes"]:
        values = {
            row["metric"]: row["effect"]
            for row in dr_effects
            if row["condition"] == condition
        }
        lines.append(
            f"| {condition} | {values['PSNR']:+.4f} | {values['SSIM']:+.5f} | {values['LPIPS']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "Differences are 16,384 minus 4,096 training scenes at 100k steps. Negative LPIPS is better.",
            "",
            "## MNIST matched effects",
            "",
            "| Contrast | Steps | Metric | Mean effect | SD |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in aggregate:
        lines.append(
            f"| {row['contrast']} ({row['information_regime']}) | {row['endpoint_steps']} | {row['metric']} | {row['mean_effect']:+.5f} | {row['sample_sd']:.5f} |"
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(validation, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
