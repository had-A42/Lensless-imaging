"""Build a row-matched, explicitly contextual PSF-aware comparison table."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("PSNR", "SSIM", "LPIPS")
VIEWS = {
    "inner68": {
        "reference_template": "outputs/ref-psff-real-68/evaluation/seed{seed}/per_row.csv",
        "psf_aware": "outputs/coursework_pre_final_20260910/psf_aware/inner68",
        "samples": 1700,
        "masks": 68,
        "rows_per_mask": 25,
    },
    "outer17": {
        "reference_template": "outputs/ref-psff-outer17-v1/seed{seed}/per_row.csv",
        "psf_aware": "outputs/coursework_pre_final_20260910/psf_aware/outer17",
        "samples": 4250,
        "masks": 17,
        "rows_per_mask": 250,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identities(rows: list[dict]) -> list[tuple[int, int, int]]:
    return sorted(
        (int(row["source_index"]), int(row["mask_id"]), int(row["row_slot"]))
        for row in rows
    )


def aggregate(rows: list[dict], suffix: str = "") -> tuple[list[dict], dict]:
    grouped = defaultdict(lambda: {metric: [] for metric in METRICS})
    for row in rows:
        for metric in METRICS:
            grouped[int(row["mask_id"])][metric].append(float(row[metric + suffix]))
    per_mask = [
        {
            "mask_id": mask_id,
            "sample_count": len(values["PSNR"]),
            **{metric: float(np.mean(values[metric])) for metric in METRICS},
        }
        for mask_id, values in sorted(grouped.items())
    ]
    summary = {
        metric: float(np.mean([row[metric] for row in per_mask])) for metric in METRICS
    }
    return per_mask, summary


def formatted_metric(row: dict, metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    sd = row[f"{metric}_sample_sd"]
    decimals = 4 if metric == "PSNR" else 5
    if sd == "":
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {float(sd):.{decimals}f}"


def results_markdown(rows: list[dict], effects: list[dict]) -> str:
    lines = [
        "# PSF-aware reference on real development rows",
        "",
        "Во всех строках совпадают row identities, crop, orientation, independent peak normalization, metric implementation и mask-balanced aggregation. При этом comparison остается contextual: published model получает PSF на входе и обучался на более широком наборе операторов. Это не matched causal comparison архитектур или training recipes.",
        "",
        "| View | Method | PSF at inference | PSNR, mean ± SD | SSIM, mean ± SD | LPIPS, mean ± SD |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['view']} | {row['method']} | {row['psf_at_inference']} | "
            f"{formatted_metric(row, 'PSNR')} | {formatted_metric(row, 'SSIM')} | "
            f"{formatted_metric(row, 'LPIPS')} |"
        )
    lines.extend(
        [
            "",
            "| View | Contextual difference | ΔPSNR | ΔSSIM | ΔLPIPS |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in effects:
        lines.append(
            f"| {row['view']} | {row['contrast']} | "
            f"{float(row['delta_PSNR']):+.4f} | {float(row['delta_SSIM']):+.5f} | "
            f"{float(row['delta_LPIPS']):+.5f} |"
        )
    lines.extend(
        [
            "",
            "Для LPIPS отрицательная разность означает улучшение. Разности описывают наблюдаемый разрыв на одинаковых evaluation rows, но не изолируют причинный эффект знания PSF.",
            "",
            "Official real test и final synthetic test не открывались.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    output = REPO_ROOT / "outputs/coursework_pre_final_20260910/psf_aware/comparison_v2"
    output.mkdir(parents=True, exist_ok=False)
    summary_rows = []
    per_run_rows = []
    per_mask_rows = []
    effects = []
    sources = []
    checks = []
    for view, contract in VIEWS.items():
        psf_root = REPO_ROOT / contract["psf_aware"]
        psf_path = psf_root / "per_image.csv"
        with psf_path.open(newline="") as stream:
            psf_aware = list(csv.DictReader(stream))
        summary = json.loads((psf_root / "summary.json").read_text())
        validation = json.loads((psf_root / "validation.json").read_text())
        references = {}
        for seed in (42, 52, 62):
            reference_path = REPO_ROOT / contract["reference_template"].format(seed=seed)
            with reference_path.open(newline="") as stream:
                references[seed] = list(csv.DictReader(stream))
            sources.append({"path": str(reference_path), "sha256": sha256(reference_path)})
        identity_match = all(
            identities(reference) == identities(psf_aware)
            for reference in references.values()
        )
        if not identity_match:
            raise ValueError(f"Row identity mismatch for {view}")
        if any(len(reference) != contract["samples"] for reference in references.values()) or len(psf_aware) != contract["samples"]:
            raise ValueError(f"Sample count mismatch for {view}")
        if summary.get("status") != "complete" or validation.get("complete") is not True:
            raise ValueError(f"Incomplete PSF-aware result for {view}")

        method_summaries = {}
        for method, suffix in (
            ("PSF-free, trained on real measurements", "_real"),
            ("PSF-free, trained on matched simulation", "_matched_sim"),
        ):
            run_scores = []
            for seed, reference in sorted(references.items()):
                per_mask, scores = aggregate(reference, suffix=suffix)
                if len(per_mask) != contract["masks"] or {
                    row["sample_count"] for row in per_mask
                } != {contract["rows_per_mask"]}:
                    raise ValueError(f"Mask balance failed for {view}: {method}, seed{seed}")
                run_scores.append(scores)
                per_run_rows.append(
                    {
                        "view": view,
                        "method": method,
                        "training_seed": seed,
                        "psf_at_inference": "false",
                        **scores,
                    }
                )
                for row in per_mask:
                    per_mask_rows.append(
                        {
                            "view": view,
                            "method": method,
                            "training_seed": seed,
                            "psf_at_inference": "false",
                            **row,
                        }
                    )
            score_means = {
                metric: float(np.mean([row[metric] for row in run_scores]))
                for metric in METRICS
            }
            score_sds = {
                metric: float(np.std([row[metric] for row in run_scores], ddof=1))
                for metric in METRICS
            }
            method_summaries[method] = score_means
            summary_rows.append(
                {
                    "view": view,
                    "method": method,
                    "psf_at_inference": "false",
                    "comparison_status": "row-matched contextual; training and PSF access are not matched",
                    "training_run_count": len(run_scores),
                    "sample_count": contract["samples"],
                    "mask_count": contract["masks"],
                    "rows_per_mask": contract["rows_per_mask"],
                    **{
                        f"{metric}_{stat}": values[metric]
                        for stat, values in (("mean", score_means), ("sample_sd", score_sds))
                        for metric in METRICS
                    },
                }
            )
        psf_per_mask, psf_scores = aggregate(psf_aware)
        if len(psf_per_mask) != contract["masks"] or {
            row["sample_count"] for row in psf_per_mask
        } != {contract["rows_per_mask"]}:
            raise ValueError(f"Mask balance failed for {view}: published PSF-aware")
        method_summaries["Published PSF-aware reference"] = psf_scores
        per_run_rows.append(
            {
                "view": view,
                "method": "Published PSF-aware reference",
                "training_seed": "published fixed checkpoint",
                "psf_at_inference": "true",
                **psf_scores,
            }
        )
        summary_rows.append(
            {
                "view": view,
                "method": "Published PSF-aware reference",
                "psf_at_inference": "true",
                "comparison_status": "row-matched contextual; training and PSF access are not matched",
                "training_run_count": 1,
                "sample_count": contract["samples"],
                "mask_count": contract["masks"],
                "rows_per_mask": contract["rows_per_mask"],
                **{
                    item: value
                    for metric in METRICS
                    for item, value in (
                        (f"{metric}_mean", psf_scores[metric]),
                        (f"{metric}_sample_sd", ""),
                    )
                },
            }
        )
        for row in psf_per_mask:
            per_mask_rows.append(
                {
                    "view": view,
                    "method": "Published PSF-aware reference",
                    "training_seed": "published fixed checkpoint",
                    "psf_at_inference": "true",
                    **row,
                }
            )
        for baseline in (
            "PSF-free, trained on real measurements",
            "PSF-free, trained on matched simulation",
        ):
            effects.append(
                {
                    "view": view,
                    "contrast": f"PSF-aware minus {baseline}",
                    **{
                        f"delta_{metric}": psf_scores[metric]
                        - method_summaries[baseline][metric]
                        for metric in METRICS
                    },
                    "comparison_status": "contextual, not a matched causal estimand",
                }
            )
        sources.extend(
            [
                {"path": str(psf_path), "sha256": sha256(psf_path)},
                {
                    "path": str(psf_root / "provenance.json"),
                    "sha256": sha256(psf_root / "provenance.json"),
                },
            ]
        )
        checks.append(
            {
                "view": view,
                "row_identity_match": identity_match,
                "sample_count_match": True,
                "mask_count_match": True,
                "rows_per_mask_match": True,
            }
        )

    write_csv(output / "comparison_summary.csv", summary_rows)
    write_csv(output / "per_run_summary.csv", per_run_rows)
    write_csv(output / "per_mask_comparison.csv", per_mask_rows)
    write_csv(output / "contextual_effects.csv", effects)
    (output / "RESULTS.md").write_text(results_markdown(summary_rows, effects))
    validation = {
        "status": "pass",
        "views": checks,
        "same_crop_orientation_normalization_metrics_aggregation": True,
        "comparison_status": "row-matched contextual; not a matched causal comparison",
        "published_checkpoint_receives_psf": True,
        "published_checkpoint_trained_on_broader_operator_set": True,
        "official_real_test_accessed": False,
        "final_synthetic_test_accessed": False,
        "source_artifacts": sources,
        "generated_artifacts": [
            {
                "path": str(path),
                "sha256": sha256(path),
            }
            for path in (
                output / "comparison_summary.csv",
                output / "per_run_summary.csv",
                output / "per_mask_comparison.csv",
                output / "contextual_effects.csv",
                output / "RESULTS.md",
            )
        ],
    }
    write_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
