"""Build report-ready metrics for the supervisor PSF-aware experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


METRICS = ("PSNR", "SSIM", "LPIPS")
CONDITIONS = ("Measurement only", "Correct PSF", "Shuffled PSF")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_mask_means(path: Path, condition: str | None = None) -> dict[str, float]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if condition is not None:
        rows = [row for row in rows if row["condition"] == condition]
    if len(rows) != 32 or len({row["mask_id"] for row in rows}) != 32:
        raise ValueError(f"Expected 32 unique masks: {path}, {condition}")
    if {int(row["sample_count"]) for row in rows} != {32}:
        raise ValueError(f"Expected 32 scenes per mask: {path}, {condition}")
    values = {
        metric: statistics.mean(float(row[metric]) for row in rows)
        for metric in METRICS
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite metrics: {path}, {condition}")
    return values


def format_value(mean: float, sd: float, metric: str) -> str:
    digits = 4 if metric == "PSNR" else 5
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def interpretation(effect_rows: list[dict]) -> dict:
    index = {
        (row["contrast"], row["metric"]): float(row["mean_effect"])
        for row in effect_rows
    }
    aware_psnr = index[("correct_psf_minus_measurement_only", "PSNR")]
    shuffle_psnr = index[("shuffled_psf_minus_correct_psf", "PSNR")]
    if aware_psnr > 0 and shuffle_psnr < 0:
        category = "aware_better_and_shuffled_worse"
        statement = (
            "Mean PSNR is higher with the correct PSF than measurement-only, "
            "and cyclic PSF shuffling lowers mean PSNR."
        )
    elif aware_psnr > 0 and shuffle_psnr >= 0:
        category = "aware_better_without_correct_psf_advantage"
        statement = (
            "Mean PSNR is higher for the conditioned branch, but shuffling does "
            "not lower mean PSNR; the gain cannot be attributed to PSF identity."
        )
    elif aware_psnr <= 0:
        category = "aware_not_better"
        statement = (
            "The tested concat-conditioned branch does not improve mean PSNR over "
            "measurement-only DRUNet."
        )
    else:
        category = "unclassified"
        statement = "The observed metric signs require manual interpretation."
    return {
        "category": category,
        "statement": statement,
        "inference_scope": (
            "three matched training seeds; no statistical-significance claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    bundle = root / manifest["bundle"]
    if manifest["final_test_accessed_by_this_queue"] is not False:
        raise ValueError("Final test is forbidden")
    for relative, expected in manifest["program_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Program hash drift: {relative}")

    output = bundle / "results"
    output.mkdir(parents=True, exist_ok=False)
    absolute_rows = []
    effect_rows = []
    consistency_rows = []
    source_artifacts = []
    for job in sorted(manifest["jobs"], key=lambda item: item["seed"]):
        seed = job["seed"]
        control = manifest["controls"][str(seed)]
        run = root / job["output"]
        evaluation = root / job["evaluation_output"]
        complete_path = run / "job_complete.json"
        evaluation_validation_path = evaluation / "validation.json"
        summary_path = evaluation / "summary.json"
        per_mask_path = evaluation / "per_mask.csv"
        for path in (
            complete_path,
            evaluation_validation_path,
            summary_path,
            per_mask_path,
            evaluation / "per_image.csv",
            evaluation / "prediction_consistency.csv",
            evaluation / "qualitative.png",
            evaluation / "provenance.json",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            source_artifacts.append({"path": str(path), "sha256": sha256(path)})
        complete = json.loads(complete_path.read_text())
        validation = json.loads(evaluation_validation_path.read_text())
        summary = json.loads(summary_path.read_text())
        if complete["status"] != "complete" or complete["global_step"] != 100000:
            raise ValueError(f"Incomplete training endpoint: seed {seed}")
        if validation["status"] != "pass" or not validation[
            "correct_replay_within_tolerance"
        ]:
            raise ValueError(f"Invalid correct/shuffled replay: seed {seed}")
        if validation["data_partition"] != "development" or validation[
            "final_test_accessed"
        ]:
            raise ValueError(f"Evaluation partition drift: seed {seed}")

        condition_values = {
            "Measurement only": read_mask_means(Path(control["metrics_csv"])),
            "Correct PSF": read_mask_means(per_mask_path, "Correct PSF"),
            "Shuffled PSF": read_mask_means(per_mask_path, "Shuffled PSF"),
        }
        for condition in CONDITIONS:
            absolute_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "training_steps": 100000,
                    "training_scenes": 16384,
                    "training_masks": 100,
                    **condition_values[condition],
                }
            )

        contrasts = {
            "correct_psf_minus_measurement_only": (
                "Correct PSF",
                "Measurement only",
            ),
            "shuffled_psf_minus_measurement_only": (
                "Shuffled PSF",
                "Measurement only",
            ),
            "shuffled_psf_minus_correct_psf": ("Shuffled PSF", "Correct PSF"),
        }
        for contrast, (left, right) in contrasts.items():
            for metric in METRICS:
                effect_rows.append(
                    {
                        "contrast": contrast,
                        "seed": seed,
                        "metric": metric,
                        "effect": condition_values[left][metric]
                        - condition_values[right][metric],
                    }
                )
        consistency_rows.append(
            {
                "seed": seed,
                **summary["prediction_consistency"],
            }
        )
        source_artifacts.extend(
            [
                {"path": control["checkpoint"], "sha256": control["checkpoint_sha256"]},
                {"path": control["config"], "sha256": control["config_sha256"]},
                {"path": control["metrics_csv"], "sha256": control["metrics_sha256"]},
            ]
        )

    absolute_aggregate = []
    for condition in CONDITIONS:
        rows = [row for row in absolute_rows if row["condition"] == condition]
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            absolute_aggregate.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "mean": statistics.mean(values),
                    "sample_sd": statistics.stdev(values),
                    "run_count": len(values),
                }
            )

    effect_aggregate = []
    for contrast in (
        "correct_psf_minus_measurement_only",
        "shuffled_psf_minus_measurement_only",
        "shuffled_psf_minus_correct_psf",
    ):
        for metric in METRICS:
            rows = [
                row
                for row in effect_rows
                if row["contrast"] == contrast and row["metric"] == metric
            ]
            values = [float(row["effect"]) for row in rows]
            favorable = [
                value > 0 if metric != "LPIPS" else value < 0 for value in values
            ]
            effect_aggregate.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "mean_effect": statistics.mean(values),
                    "sample_sd": statistics.stdev(values),
                    "run_count": len(values),
                    "favorable_direction_count": sum(favorable),
                }
            )

    write_csv(output / "absolute_per_seed.csv", absolute_rows)
    write_csv(output / "absolute_aggregate.csv", absolute_aggregate)
    write_csv(output / "paired_effects_per_seed.csv", effect_rows)
    write_csv(output / "paired_effects_aggregate.csv", effect_aggregate)
    write_csv(output / "prediction_consistency_per_seed.csv", consistency_rows)

    aggregate_index = {
        (row["condition"], row["metric"]): row for row in absolute_aggregate
    }
    lines = [
        "# Matched MIRFLICKR PSF-conditioning results",
        "",
        "All rows use fixed 100k endpoints and the same 32-mask by 32-scene development grid. No final-test sample was accessed.",
        "",
        "| Information | Runs | PSNR mean ± SD | SSIM mean ± SD | LPIPS mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = {metric: aggregate_index[(condition, metric)] for metric in METRICS}
        lines.append(
            f"| {condition} | 3 | "
            f"{format_value(values['PSNR']['mean'], values['PSNR']['sample_sd'], 'PSNR')} | "
            f"{format_value(values['SSIM']['mean'], values['SSIM']['sample_sd'], 'SSIM')} | "
            f"{format_value(values['LPIPS']['mean'], values['LPIPS']['sample_sd'], 'LPIPS')} |"
        )
    lines.extend(
        [
            "",
            "## Paired effects",
            "",
            "| Contrast | Metric | Mean effect | SD | Favorable seeds |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in effect_aggregate:
        lines.append(
            f"| {row['contrast']} | {row['metric']} | {row['mean_effect']:+.6f} | "
            f"{row['sample_sd']:.6f} | {row['favorable_direction_count']}/3 |"
        )
    primary_psnr = [
        row
        for row in effect_rows
        if row["contrast"] == "correct_psf_minus_measurement_only"
        and row["metric"] == "PSNR"
    ]
    lines.extend(
        [
            "",
            "Per-seed paired PSNR effects (Correct PSF minus Measurement only): "
            + ", ".join(
                f"seed{row['seed']} {float(row['effect']):+.4f} dB"
                for row in primary_psnr
            )
            + ".",
            "",
        ]
    )
    result_interpretation = interpretation(effect_aggregate)
    lines.append(result_interpretation["statement"])
    lines.append("")
    lines.append(
        "The comparison contains three training seeds; no statistical-significance claim is made."
    )
    lines.append("")
    (output / "RESULTS.md").write_text("\n".join(lines))

    tex_lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Information & Runs & PSNR, dB $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for condition in CONDITIONS:
        values = {metric: aggregate_index[(condition, metric)] for metric in METRICS}
        tex_lines.append(
            f"{condition} & 3 & "
            f"${values['PSNR']['mean']:.4f}\\pm{values['PSNR']['sample_sd']:.4f}$ & "
            f"${values['SSIM']['mean']:.4f}\\pm{values['SSIM']['sample_sd']:.4f}$ & "
            f"${values['LPIPS']['mean']:.4f}\\pm{values['LPIPS']['sample_sd']:.4f}$ \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (output / "report_table.tex").write_text("\n".join(tex_lines))

    generated = [
        output / "absolute_per_seed.csv",
        output / "absolute_aggregate.csv",
        output / "paired_effects_per_seed.csv",
        output / "paired_effects_aggregate.csv",
        output / "prediction_consistency_per_seed.csv",
        output / "RESULTS.md",
        output / "report_table.tex",
    ]
    validation = {
        "status": "pass",
        "training_endpoint_count": 3,
        "evaluation_count": 3,
        "all_fixed_endpoints_complete": True,
        "all_correct_replays_match_training": True,
        "grid": {"masks": 32, "scenes_per_mask": 32, "pairs_per_seed": 1024},
        "interpretation": result_interpretation,
        "data_partition": "development",
        "final_test_accessed": False,
        "source_artifacts": source_artifacts,
        "generated_artifacts": [
            {"path": str(path), "sha256": sha256(path)} for path in generated
        ],
    }
    save_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
