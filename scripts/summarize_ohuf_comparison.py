"""Validate and summarize paired baseline/OHUF reconstruction metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

METRICS = ("PSNR", "SSIM", "LPIPS")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> dict[tuple[str, int], dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sample_id", "mask_id", *METRICS}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"{path} is missing required columns {sorted(required)}")
    output = {}
    for row in rows:
        identity = (str(row["sample_id"]), int(row["mask_id"]))
        if identity in output:
            raise ValueError(f"duplicate identity in {path}: {identity}")
        values = {metric: float(row[metric]) for metric in METRICS}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite metric in {path}: {identity}")
        output[identity] = values
    return output


def gain(metric: str, baseline: float, candidate: float) -> float:
    return baseline - candidate if metric == "LPIPS" else candidate - baseline


def t_interval(values: np.ndarray) -> dict[str, float | None]:
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("at least one mask value is needed")
    mean = float(values.mean())
    if len(values) == 1:
        return {
            "mean": mean,
            "standard_deviation": None,
            "low": None,
            "high": None,
        }
    standard_deviation = float(values.std(ddof=1))
    margin = float(
        t.ppf(0.975, len(values) - 1) * standard_deviation / np.sqrt(len(values))
    )
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "low": mean - margin,
        "high": mean + margin,
    }


def summarize(
    baseline_path: Path,
    candidate_path: Path,
    expected_masks: int | None,
    expected_scenes_per_mask: int | None,
) -> tuple[dict, list[dict], list[dict]]:
    baseline = read_rows(baseline_path)
    candidate = read_rows(candidate_path)
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))[:5]
        extra = sorted(set(candidate) - set(baseline))[:5]
        raise ValueError(f"row identity mismatch; missing={missing}, extra={extra}")

    per_image = []
    grouped: dict[int, list[dict]] = defaultdict(list)
    for sample_id, mask_id in sorted(baseline, key=lambda item: (item[1], item[0])):
        row = {"sample_id": sample_id, "mask_id": mask_id}
        for metric in METRICS:
            baseline_value = baseline[(sample_id, mask_id)][metric]
            candidate_value = candidate[(sample_id, mask_id)][metric]
            row[f"baseline_{metric}"] = baseline_value
            row[f"candidate_{metric}"] = candidate_value
            row[f"gain_{metric}"] = gain(metric, baseline_value, candidate_value)
        per_image.append(row)
        grouped[mask_id].append(row)

    if expected_masks is not None and len(grouped) != expected_masks:
        raise ValueError(f"expected {expected_masks} masks, found {len(grouped)}")
    counts = {mask_id: len(rows) for mask_id, rows in grouped.items()}
    if expected_scenes_per_mask is not None and set(counts.values()) != {
        expected_scenes_per_mask
    }:
        raise ValueError(
            f"expected {expected_scenes_per_mask} scenes per mask, found {counts}"
        )

    per_mask = []
    for mask_id, rows in sorted(grouped.items()):
        per_mask.append(
            {
                "mask_id": mask_id,
                "sample_count": len(rows),
                **{
                    f"gain_{metric}": float(
                        np.mean([row[f"gain_{metric}"] for row in rows])
                    )
                    for metric in METRICS
                },
            }
        )

    intervals = {
        metric: t_interval(np.asarray([row[f"gain_{metric}"] for row in per_mask]))
        for metric in METRICS
    }
    summary = {
        "status": "complete",
        "gain_orientation": {
            "PSNR": "candidate - baseline",
            "SSIM": "candidate - baseline",
            "LPIPS": "baseline - candidate",
        },
        "sample_count": len(per_image),
        "mask_count": len(per_mask),
        "samples_per_mask": sorted(set(counts.values())),
        "mask_mean_t_interval_95": intervals,
        "mask_wins": {
            metric: sum(row[f"gain_{metric}"] > 0 for row in per_mask)
            for metric in METRICS
        },
        "image_wins": {
            metric: sum(row[f"gain_{metric}"] > 0 for row in per_image)
            for metric in METRICS
        },
        "inputs": {
            "baseline": str(baseline_path.resolve()),
            "baseline_sha256": sha256(baseline_path),
            "candidate": str(candidate_path.resolve()),
            "candidate_sha256": sha256(candidate_path),
        },
    }
    return summary, per_image, per_mask


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-masks", type=int)
    parser.add_argument("--expected-scenes-per-mask", type=int)
    args = parser.parse_args()

    summary, per_image, per_mask = summarize(
        args.baseline,
        args.candidate,
        args.expected_masks,
        args.expected_scenes_per_mask,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "per_image_comparison.csv", per_image)
    write_csv(args.output / "per_mask_comparison.csv", per_mask)
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
