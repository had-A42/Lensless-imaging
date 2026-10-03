"""Validate and summarize the frozen revised V4 final-test results."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("PSNR", "SSIM", "LPIPS")
BOOTSTRAP_REPLICATES = 500


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


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity_digest(rows: list[dict]) -> str:
    value = [
        (int(row["sample_index"]), row["mask_id"], row["scene_id"]) for row in rows
    ]
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":")).encode()
    ).hexdigest()


def metric_cube(rows: list[dict], metric: str) -> np.ndarray:
    by_mask = defaultdict(list)
    for row in rows:
        by_mask[row["mask_id"]].append((int(row["sample_index"]), float(row[metric])))
    masks = sorted(by_mask)
    if len(masks) != 100 or {len(by_mask[mask]) for mask in masks} != {256}:
        raise ValueError("Expected a balanced 100-mask x 256-scene grid")
    return np.asarray(
        [[value for _, value in sorted(by_mask[mask])] for mask in masks],
        dtype=np.float64,
    )


def bootstrap_interval(
    effects: list[np.ndarray], *, seed: int
) -> tuple[float, float]:
    stack = np.stack(effects)
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    mask_count, scene_count = stack.shape[1:]
    for index in range(BOOTSTRAP_REPLICATES):
        mask_indices = rng.integers(0, mask_count, size=mask_count)
        scene_indices = rng.integers(
            0, scene_count, size=(mask_count, scene_count)
        )
        selected = stack[:, mask_indices[:, None], scene_indices]
        samples[index] = float(selected.mean())
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def main() -> None:
    root = REPO_ROOT / "outputs/coursework_final_runner_v4_20260910"
    run_root = root / "final_run_revised"
    shortlist_path = root / "revised_shortlist_v1/shortlist.json"
    merged_path = run_root / "merged/summary.csv"
    merged_validation_path = run_root / "merged/validation.json"
    output = root / "final_results_v1"
    output.mkdir(parents=True, exist_ok=False)
    shortlist = json.loads(shortlist_path.read_text())
    merged_validation = json.loads(merged_validation_path.read_text())
    if (
        merged_validation.get("status") != "pass"
        or merged_validation.get("checkpoint_count") != 11
        or merged_validation.get("all_exact_sample_counts") is not True
        or merged_validation.get("all_exact_mask_counts") is not True
        or merged_validation.get("all_metrics_finite") is not True
        or merged_validation.get("metric_based_checkpoint_selection") is not False
    ):
        raise ValueError("Merged final validation is incomplete")
    merged = {row["shortlist_id"]: row for row in read_csv(merged_path)}
    entries = shortlist["entries"]
    if set(merged) != {entry["shortlist_id"] for entry in entries}:
        raise ValueError("Merged summary does not match revised shortlist")
    locations = {}
    for shard_id in (0, 1, 2):
        shard = run_root / f"shard{shard_id}"
        if not shard.is_dir():
            raise FileNotFoundError(shard)
        for entry in entries:
            candidate = shard / entry["shortlist_id"]
            if candidate.is_dir():
                locations[entry["shortlist_id"]] = candidate
    if len(locations) != 11:
        raise ValueError("Could not locate all per-model final outputs")

    per_image = {}
    model_rows = []
    identity_hashes = set()
    source_artifacts = []
    for entry in entries:
        identifier = entry["shortlist_id"]
        directory = locations[identifier]
        per_image_path = directory / "per_image.csv"
        summary_path = directory / "summary.json"
        validation_path = directory / "validation.json"
        rows = read_csv(per_image_path)
        summary = json.loads(summary_path.read_text())
        validation = json.loads(validation_path.read_text())
        if (
            len(rows) != 25_600
            or summary.get("sample_count") != 25_600
            or summary.get("mask_count") != 100
            or summary.get("test_accessed") is not True
            or validation.get("status") != "pass"
            or validation.get("checkpoint_sha256_match") is not True
            or validation.get("checkpoint_endpoint_match") is not True
        ):
            raise ValueError(f"Invalid final model artifact: {identifier}")
        current_identity = identity_digest(rows)
        identity_hashes.add(current_identity)
        per_image[identifier] = rows
        recomputed = {
            metric: float(metric_cube(rows, metric).mean()) for metric in METRICS
        }
        if any(
            abs(recomputed[metric] - float(merged[identifier][metric])) > 1e-10
            for metric in METRICS
        ):
            raise ValueError(f"Merged metric mismatch: {identifier}")
        model_rows.append(
            {
                "shortlist_id": identifier,
                "analysis_role": entry["analysis_role"],
                "training_masks": entry["training_masks"],
                "initialization": entry["initialization"],
                "seed": entry["seed"],
                "training_steps": entry["training_steps"],
                **recomputed,
                "sample_count": len(rows),
                "mask_count": 100,
                "checkpoint_sha256": entry["sha256"],
                "grid_identity_sha256": current_identity,
            }
        )
        source_artifacts.extend(
            {
                "path": str(path),
                "sha256": sha256(path),
            }
            for path in (per_image_path, summary_path, validation_path)
        )
    if len(identity_hashes) != 1:
        raise ValueError("Final models do not share an identical ordered grid")

    primary = {
        (
            int(entry["training_masks"]),
            entry["initialization"],
            int(entry["seed"]),
        ): entry["shortlist_id"]
        for entry in entries
        if entry["analysis_role"] == "primary_matched_matrix"
    }
    expected_primary = {
        (masks, initialization, seed)
        for masks in (100, 1000)
        for initialization in ("scratch", "gopro")
        for seed in (52, 62)
    }
    if set(primary) != expected_primary:
        raise ValueError("Primary revised matrix is incomplete")
    contrast_rows = []
    effect_cubes = defaultdict(lambda: defaultdict(list))

    def add_contrast(
        *, contrast_type: str, group: str, seed: int, first: str, second: str
    ) -> None:
        first_rows = per_image[first]
        second_rows = per_image[second]
        for metric in METRICS:
            effect = metric_cube(first_rows, metric) - metric_cube(second_rows, metric)
            effect_cubes[(contrast_type, group)][metric].append(effect)
            for mask_index, mask_mean in enumerate(effect.mean(axis=1)):
                contrast_rows.append(
                    {
                        "contrast_type": contrast_type,
                        "group": group,
                        "seed": seed,
                        "first": first,
                        "second": second,
                        "mask_index": mask_index,
                        "metric": metric,
                        "effect": float(mask_mean),
                    }
                )

    for initialization in ("scratch", "gopro"):
        for seed in (52, 62):
            add_contrast(
                contrast_type="mask_scale_1000_minus_100",
                group=initialization,
                seed=seed,
                first=primary[(1000, initialization, seed)],
                second=primary[(100, initialization, seed)],
            )
    for masks in (100, 1000):
        for seed in (52, 62):
            add_contrast(
                contrast_type="gopro_minus_scratch",
                group=str(masks),
                seed=seed,
                first=primary[(masks, "gopro", seed)],
                second=primary[(masks, "scratch", seed)],
            )
    write_csv(output / "paired_effects_per_mask.csv", contrast_rows)

    aggregate_rows = []
    for group_index, ((contrast_type, group), metrics) in enumerate(
        sorted(effect_cubes.items())
    ):
        for metric_index, metric in enumerate(METRICS):
            run_effects = [float(cube.mean()) for cube in metrics[metric]]
            low, high = bootstrap_interval(
                metrics[metric], seed=20260911 + 100 * group_index + metric_index
            )
            aggregate_rows.append(
                {
                    "contrast_type": contrast_type,
                    "group": group,
                    "metric": metric,
                    "mean_effect": statistics.mean(run_effects),
                    "sample_sd_across_runs": statistics.stdev(run_effects),
                    "run_effect_1": run_effects[0],
                    "run_effect_2": run_effects[1],
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "inference": "descriptive; no post-test model selection",
                }
            )
    write_csv(output / "paired_effects_aggregate.csv", aggregate_rows)
    write_csv(output / "model_summary.csv", model_rows)

    lookup = {
        (row["contrast_type"], row["group"], row["metric"]): row
        for row in aggregate_rows
    }
    lines = [
        "# Revised V4 final synthetic test",
        "",
        "The final run used the frozen 100-mask x 256-scene grid. Eight models form the primary matched matrix over seeds52/62. Two corrected seed42 finite1000 models are supplementary, and the predeclared 100k model is reported separately. No checkpoint was selected from these results.",
        "",
        "| Contrast | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|",
    ]
    for contrast_type, group, label in (
        ("mask_scale_1000_minus_100", "scratch", "1000 - 100 masks, scratch"),
        ("mask_scale_1000_minus_100", "gopro", "1000 - 100 masks, GoPro"),
        ("gopro_minus_scratch", "100", "GoPro - scratch, 100 masks"),
        ("gopro_minus_scratch", "1000", "GoPro - scratch, 1000 masks"),
    ):
        values = [
            lookup[(contrast_type, group, metric)]["mean_effect"]
            for metric in METRICS
        ]
        lines.append(
            f"| {label} | {values[0]:+.4f} | {values[1]:+.5f} | {values[2]:+.5f} |"
        )
    lines.extend(
        [
            "",
            "For LPIPS, negative values indicate improvement. The mask-scale effect is small and changes sign across initialization for PSNR, whereas GoPro initialization improves all three metrics at both mask counts. These statements describe the frozen primary matrix only.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines))
    generated = [
        output / "model_summary.csv",
        output / "paired_effects_per_mask.csv",
        output / "paired_effects_aggregate.csv",
        output / "RESULTS.md",
    ]
    validation = {
        "status": "pass",
        "checkpoint_count": len(model_rows),
        "primary_matched_count": sum(
            row["analysis_role"] == "primary_matched_matrix" for row in model_rows
        ),
        "supplementary_count": sum(
            row["analysis_role"] == "supplementary_corrected_seed42"
            for row in model_rows
        ),
        "finalist_count": sum(
            row["analysis_role"] == "predeclared_100k_finalist" for row in model_rows
        ),
        "single_grid_identity_sha256": next(iter(identity_hashes)),
        "all_source_validations_pass": True,
        "all_metrics_recomputed": True,
        "post_test_checkpoint_selection": False,
        "source_artifacts": source_artifacts,
        "generated_artifacts": [
            {"path": str(path), "sha256": sha256(path)} for path in generated
        ],
    }
    write_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
