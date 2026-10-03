"""Aggregate the three predeclared matched-OHUF confirmation seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

SEEDS = (42, 52, 62)
METRICS = ("PSNR", "SSIM", "LPIPS")


def interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    margin = float(
        t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array))
    )
    return {"mean": mean, "low": mean - margin, "high": mean + margin}


def oriented(metric: str, reference: pd.Series, candidate: pd.Series) -> pd.Series:
    return reference - candidate if metric == "LPIPS" else candidate - reference


def load_seed(root: Path, seed: int) -> tuple[dict, pd.DataFrame, dict]:
    directory = root / f"seed{seed}"
    summary = json.loads((directory / "confirmation" / "summary.json").read_text())
    selection = json.loads((directory / "calibration" / "selection.json").read_text())
    rows = pd.read_csv(directory / "confirmation" / "per_sample.csv")
    if summary.get("status") != "complete" or summary.get("phase") != "confirmation":
        raise ValueError(f"invalid confirmation summary for seed {seed}")
    if selection.get("status") != "frozen" or int(selection.get("seed")) != seed:
        raise ValueError(f"invalid alpha selection for seed {seed}")
    if summary.get("official_test_accessed") is not False:
        raise ValueError("official test must remain closed")
    if int(summary.get("seed")) != seed or len(rows) == 0:
        raise ValueError(f"seed mismatch or empty rows for {seed}")
    if set(rows["mask_id"].unique()) != set(summary["dataset"]["mask_ids"]):
        raise ValueError(f"mask identity mismatch for seed {seed}")
    if set(rows["row_slot"].unique()) != set(range(233, 250)):
        raise ValueError(f"row protocol mismatch for seed {seed}")
    return summary, rows, selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    sources = {seed: load_seed(root, seed) for seed in SEEDS}
    method_sets = [set(rows["method"].unique()) for _, rows, _ in sources.values()]
    if any(methods != method_sets[0] for methods in method_sets[1:]):
        raise ValueError("confirmation methods differ across seeds")
    methods = sorted(method_sets[0])
    if "baseline" not in methods:
        raise ValueError("baseline is missing")

    absolute_rows = []
    seed_mask_rows = []
    contrast_definitions = {
        f"{method}_vs_baseline": (method, "baseline")
        for method in methods
        if method != "baseline"
    }
    contrast_definitions.update(
        {
            "grouped_jitter_k4_vs_prefix_k4": (
                "grouped_jitter_k4",
                "prefix_k4",
            ),
            "jitter_psf_k4_vs_prefix_k4": ("jitter_psf_k4", "prefix_k4"),
            "otf_kcenter_k4_vs_prefix_k4": ("otf_kcenter_k4", "prefix_k4"),
            "axis_shift_k8_vs_prefix_k8": ("axis_shift_k8", "prefix_k8"),
        }
    )
    per_seed_contrast = defaultdict(lambda: defaultdict(dict))

    for seed, (summary, rows, selection) in sources.items():
        del summary
        for method in methods:
            selected = rows[rows["method"] == method]
            for metric in METRICS:
                absolute_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "alpha": (
                            0.0
                            if method == "baseline"
                            else selection["selected_alphas"][method]
                        ),
                        "metric": metric,
                        "mean": float(selected[metric].mean()),
                    }
                )

        indexed = {
            method: rows[rows["method"] == method].set_index(["sample_id", "mask_id"])
            for method in methods
        }
        for contrast, (candidate, reference) in contrast_definitions.items():
            if candidate not in indexed or reference not in indexed:
                raise ValueError(f"missing method for {contrast}")
            if not indexed[candidate].index.equals(indexed[reference].index):
                raise ValueError(
                    f"sample identity mismatch for {contrast}, seed {seed}"
                )
            for metric in METRICS:
                values = oriented(
                    metric,
                    indexed[reference][metric],
                    indexed[candidate][metric],
                )
                mask_values = values.groupby("mask_id").mean()
                per_seed_contrast[contrast][metric][seed] = float(mask_values.mean())
                for mask_id, value in mask_values.items():
                    seed_mask_rows.append(
                        {
                            "seed": seed,
                            "mask_id": int(mask_id),
                            "contrast": contrast,
                            "metric": metric,
                            "gain": float(value),
                        }
                    )

    seed_mask = pd.DataFrame(seed_mask_rows)
    aggregate = {}
    for contrast in sorted(contrast_definitions):
        aggregate[contrast] = {}
        for metric in METRICS:
            seed_values = [per_seed_contrast[contrast][metric][seed] for seed in SEEDS]
            subset = seed_mask[
                (seed_mask["contrast"] == contrast) & (seed_mask["metric"] == metric)
            ]
            mask_values = subset.groupby("mask_id")["gain"].mean().tolist()
            aggregate[contrast][metric] = {
                "seed_mean_t_interval_95": interval(seed_values),
                "mask_mean_t_interval_95_after_seed_average": interval(mask_values),
                "positive_seeds": int(sum(value > 0 for value in seed_values)),
                "positive_masks": int(sum(value > 0 for value in mask_values)),
            }

    with (output / "per_seed_absolute.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(absolute_rows[0]))
        writer.writeheader()
        writer.writerows(absolute_rows)
    seed_mask.to_csv(output / "per_seed_mask_gain.csv", index=False)
    result = {
        "status": "complete",
        "seeds": list(SEEDS),
        "official_test_accessed": False,
        "confirmation_samples_per_seed": 272,
        "confirmation_masks": 16,
        "confirmation_scenes_per_mask": 17,
        "gain_orientation": {
            "PSNR": "candidate - reference",
            "SSIM": "candidate - reference",
            "LPIPS": "reference - candidate",
        },
        "selected_alphas": {
            str(seed): selection["selected_alphas"]
            for seed, (_, _, selection) in sources.items()
        },
        "absolute": absolute_rows,
        "contrasts": aggregate,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
