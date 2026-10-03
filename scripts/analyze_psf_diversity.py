"""Analyze nested 100/1,000/10,000 train-mask banks against development PSFs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


BANK_COUNTS = (100, 1_000, 10_000)
EXPECTED_SEEDS = (42, 52, 62)
NEAR_THRESHOLDS = (1e-6, 1e-4, 1e-3, 1e-2)


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


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"Explicit fields are required for empty CSV: {path}")
        fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_partition(directory: Path, partition: str) -> tuple[dict, dict, Path]:
    validation_path = directory / "validation.json"
    path = directory / f"{partition}_features.npz"
    if not validation_path.is_file() or not path.is_file():
        raise FileNotFoundError(f"Missing validated {partition} features in {directory}")
    validation = json.loads(validation_path.read_text())
    if validation.get("status") != "pass" or validation.get("test_partition_accessed") is not False:
        raise ValueError(f"Invalid or unsafe extractor validation: {directory}")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    required = {
        "mask_id",
        "mask_seed",
        "partition",
        "fourier_feature",
        "radial_energy",
        "pattern_sha256",
        "psf_sha256",
    }
    if set(arrays) != required:
        raise ValueError(f"Unexpected feature schema in {path}: {set(arrays)}")
    if set(arrays["partition"].tolist()) != {partition}:
        raise ValueError(f"Partition mismatch in {path}")
    if not np.isfinite(arrays["fourier_feature"]).all() or not np.isfinite(
        arrays["radial_energy"]
    ).all():
        raise ValueError(f"Non-finite feature in {path}")
    return arrays, validation, path


def parse_seed_dirs(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError("--seed-dir values must have SEED=PATH form")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"Duplicate seed directory: {seed}")
        result[seed] = Path(path_text).resolve()
    if tuple(sorted(result)) != EXPECTED_SEEDS:
        raise ValueError(f"Expected seed directories for {EXPECTED_SEEDS}, got {sorted(result)}")
    return result


def pairwise_distances(features: np.ndarray, *, seed: int) -> tuple[np.ndarray, str]:
    count = len(features)
    if count <= 1_000:
        left, right = np.triu_indices(count, k=1)
        method = "all unordered pairs"
    else:
        pair_count = 250_000
        rng = np.random.default_rng(np.random.SeedSequence([seed, count, 20260910]))
        left = rng.integers(0, count, size=pair_count)
        right = rng.integers(0, count - 1, size=pair_count)
        right += right >= left
        method = f"deterministic sample of {pair_count} ordered draws"
    delta = features[left].astype(np.float64) - features[right].astype(np.float64)
    return np.linalg.norm(delta, axis=1), method


def spectrum_statistics(features: np.ndarray) -> tuple[np.ndarray, dict]:
    centered = features.astype(np.float64) - features.mean(axis=0, dtype=np.float64)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    total = float(eigenvalues.sum())
    if total <= 0:
        raise ValueError("Degenerate Fourier feature covariance")
    probability = eigenvalues / total
    positive = probability > 0
    effective_rank = float(np.exp(-np.sum(probability[positive] * np.log(probability[positive]))))
    participation_ratio = float(total**2 / np.sum(eigenvalues**2))
    cumulative = np.cumsum(probability)
    return eigenvalues, {
        "pca_effective_rank": effective_rank,
        "pca_participation_ratio": participation_ratio,
        "pca_components_90": int(np.searchsorted(cumulative, 0.90) + 1),
        "pca_components_95": int(np.searchsorted(cumulative, 0.95) + 1),
        "pca_components_99": int(np.searchsorted(cumulative, 0.99) + 1),
    }


def nearest_development(
    train: np.ndarray,
    development: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train64 = train.astype(np.float64)
    development64 = development.astype(np.float64)
    squared = (
        np.sum(development64**2, axis=1, keepdims=True)
        + np.sum(train64**2, axis=1)[None, :]
        - 2 * development64 @ train64.T
    )
    squared = np.maximum(squared, 0)
    indices = np.argmin(squared, axis=1)
    distances = np.sqrt(squared[np.arange(len(development)), indices])
    return distances, indices


def duplicate_statistics(values: np.ndarray) -> tuple[int, int]:
    counts = Counter(values.tolist())
    groups = sum(count > 1 for count in counts.values())
    members_beyond_first = sum(count - 1 for count in counts.values() if count > 1)
    return groups, members_beyond_first


def markdown(summary_rows: list[dict], development_rows: list[dict]) -> str:
    lines = [
        "# Exploratory PSF diversity analysis",
        "",
        "Это post-hoc exploratory analysis. Он описывает геометрию сгенерированных PSF и не является заранее запланированным confirmatory test. Final-test masks не создавались и не открывались.",
        "",
        "PSF нормировалась по полной энергии. Признак строился из модуля центрированного FFT: DC-компонента удалялась, `log1p`-спектр сжимался до 8×8 и нормировался по L2.",
        "",
        "## По training seed",
        "",
        "| Seed | Bank | Pairwise median | Dev nearest median | Effective rank | Exact PSF collisions |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    dev_lookup = {
        (int(row["train_seed"]), int(row["bank_count"])): row
        for row in development_rows
    }
    for row in summary_rows:
        dev = dev_lookup[(int(row["train_seed"]), int(row["bank_count"]))]
        lines.append(
            f"| {row['train_seed']} | {row['bank_count']} | "
            f"{float(row['pairwise_q50']):.6f} | "
            f"{float(dev['nearest_median']):.6f} | "
            f"{float(row['pca_effective_rank']):.3f} | "
            f"{row['exact_psf_collision_members']} |"
        )
    lines.extend(
        [
            "",
            "`Dev nearest median` в таблице является медианой по 32 development masks. Банки 100 и 1000 являются точными префиксами соответствующего банка 10000 для того же training seed. Distances сравнимы между строками, поскольку используется один и тот же 64-мерный feature contract.",
            "",
            "Exact collisions проверены исчерпывающе по SHA256 нормированной spatial PSF. Near-collisions для банка 10000 оценены на фиксированной выборке пар, поэтому отсутствие близких пар в этой части не доказывает их полного отсутствия.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", action="append", required=True)
    parser.add_argument("--development-seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seed_dirs = parse_seed_dirs(args.seed_dir)
    if args.development_seed != 42:
        raise ValueError("Frozen development mask base seed must be 42")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_state.json", {"status": "running"})

    train_sources = {}
    source_artifacts = []
    for seed, directory in sorted(seed_dirs.items()):
        arrays, validation, path = load_partition(directory, "train")
        if int(validation["base_seed"]) != seed or len(arrays["mask_id"]) != 10_000:
            raise ValueError(f"Seed/count mismatch in {directory}")
        expected_ids = [f"train_{index:05d}" for index in range(10_000)]
        if arrays["mask_id"].tolist() != expected_ids:
            raise ValueError(f"Training bank is not the canonical ordered prefix: {directory}")
        train_sources[seed] = arrays
        source_artifacts.append(
            {"seed": seed, "path": str(path), "sha256": sha256(path)}
        )

    development, dev_validation, dev_path = load_partition(
        seed_dirs[args.development_seed], "validation"
    )
    if int(dev_validation["base_seed"]) != args.development_seed or len(
        development["mask_id"]
    ) != 32:
        raise ValueError("Development feature contract mismatch")
    if development["mask_id"].tolist() != [
        f"validation_{index:05d}" for index in range(32)
    ]:
        raise ValueError("Development masks are not the frozen ordered bank")
    source_artifacts.append(
        {"seed": args.development_seed, "path": str(dev_path), "sha256": sha256(dev_path)}
    )

    summary_rows = []
    nearest_rows = []
    nearest_summary_rows = []
    spectrum_rows = []
    radial_rows = []
    collision_rows = []
    development_features = development["fourier_feature"]
    for train_seed, arrays in sorted(train_sources.items()):
        for bank_count in BANK_COUNTS:
            features = arrays["fourier_feature"][:bank_count]
            distances, pair_method = pairwise_distances(
                features, seed=train_seed
            )
            eigenvalues, spectrum = spectrum_statistics(features)
            pattern_groups, pattern_members = duplicate_statistics(
                arrays["pattern_sha256"][:bank_count]
            )
            psf_groups, psf_members = duplicate_statistics(
                arrays["psf_sha256"][:bank_count]
            )
            threshold_counts = {
                f"sampled_pairs_le_{threshold:g}": int(np.sum(distances <= threshold))
                for threshold in NEAR_THRESHOLDS
            }
            row = {
                "train_seed": train_seed,
                "bank_count": bank_count,
                "feature_dimension": int(features.shape[1]),
                "pairwise_method": pair_method,
                "pair_count": len(distances),
                "pairwise_mean": float(np.mean(distances)),
                "pairwise_sd": float(np.std(distances, ddof=1)),
                "pairwise_min": float(np.min(distances)),
                "pairwise_q01": float(np.quantile(distances, 0.01)),
                "pairwise_q05": float(np.quantile(distances, 0.05)),
                "pairwise_q50": float(np.quantile(distances, 0.50)),
                "pairwise_q95": float(np.quantile(distances, 0.95)),
                "pairwise_q99": float(np.quantile(distances, 0.99)),
                "exact_pattern_collision_groups": pattern_groups,
                "exact_pattern_collision_members": pattern_members,
                "exact_psf_collision_groups": psf_groups,
                "exact_psf_collision_members": psf_members,
                **threshold_counts,
                **spectrum,
            }
            summary_rows.append(row)
            collision_rows.append(
                {
                    key: row[key]
                    for key in (
                        "train_seed",
                        "bank_count",
                        "pairwise_method",
                        "pair_count",
                        "pairwise_min",
                        "exact_pattern_collision_groups",
                        "exact_pattern_collision_members",
                        "exact_psf_collision_groups",
                        "exact_psf_collision_members",
                        *threshold_counts,
                    )
                }
            )

            total_eigenvalue = float(eigenvalues.sum())
            for component, eigenvalue in enumerate(eigenvalues, start=1):
                spectrum_rows.append(
                    {
                        "train_seed": train_seed,
                        "bank_count": bank_count,
                        "component": component,
                        "eigenvalue": float(eigenvalue),
                        "explained_fraction": float(eigenvalue / total_eigenvalue),
                    }
                )
            for band, (train_values, dev_values) in enumerate(
                zip(
                    arrays["radial_energy"][:bank_count].T,
                    development["radial_energy"].T,
                )
            ):
                radial_rows.append(
                    {
                        "train_seed": train_seed,
                        "bank_count": bank_count,
                        "radial_band": band,
                        "train_mean": float(np.mean(train_values)),
                        "train_sd": float(np.std(train_values, ddof=1)),
                        "development_mean": float(np.mean(dev_values)),
                        "development_sd": float(np.std(dev_values, ddof=1)),
                    }
                )

            dev_distances, nearest_indices = nearest_development(
                features, development_features
            )
            train_psf_hashes = set(arrays["psf_sha256"][:bank_count].tolist())
            exact_cross_collisions = 0
            for index, (distance, nearest_index) in enumerate(
                zip(dev_distances, nearest_indices)
            ):
                exact_collision = development["psf_sha256"][index] in train_psf_hashes
                exact_cross_collisions += int(exact_collision)
                nearest_rows.append(
                    {
                        "train_seed": train_seed,
                        "bank_count": bank_count,
                        "development_mask_id": development["mask_id"][index],
                        "development_mask_seed": int(development["mask_seed"][index]),
                        "nearest_train_mask_id": arrays["mask_id"][nearest_index],
                        "nearest_train_mask_seed": int(arrays["mask_seed"][nearest_index]),
                        "fourier_distance": float(distance),
                        "exact_psf_hash_collision": exact_collision,
                    }
                )
            nearest_summary_rows.append(
                {
                    "train_seed": train_seed,
                    "bank_count": bank_count,
                    "development_mask_count": len(dev_distances),
                    "nearest_median": float(np.median(dev_distances)),
                    "nearest_mean": float(np.mean(dev_distances)),
                    "nearest_sd": float(np.std(dev_distances, ddof=1)),
                    "nearest_min": float(np.min(dev_distances)),
                    "nearest_q05": float(np.quantile(dev_distances, 0.05)),
                    "nearest_q50": float(np.quantile(dev_distances, 0.50)),
                    "nearest_q95": float(np.quantile(dev_distances, 0.95)),
                    "nearest_max": float(np.max(dev_distances)),
                    "exact_cross_partition_psf_collisions": exact_cross_collisions,
                }
            )

    write_csv(output / "diversity_summary.csv", summary_rows)
    write_csv(output / "nearest_train_to_development.csv", nearest_rows)
    write_csv(output / "nearest_train_to_development_summary.csv", nearest_summary_rows)
    write_csv(output / "pca_spectrum.csv", spectrum_rows)
    write_csv(output / "radial_frequency_energy.csv", radial_rows)
    write_csv(output / "collisions_audit.csv", collision_rows)
    (output / "RESULTS.md").write_text(markdown(summary_rows, nearest_summary_rows))

    generated = [
        "diversity_summary.csv",
        "nearest_train_to_development.csv",
        "nearest_train_to_development_summary.csv",
        "pca_spectrum.csv",
        "radial_frequency_energy.csv",
        "collisions_audit.csv",
        "RESULTS.md",
    ]
    validation = {
        "status": "pass",
        "analysis_status": "post-hoc exploratory",
        "train_seeds": list(EXPECTED_SEEDS),
        "bank_counts": list(BANK_COUNTS),
        "development_partition": "validation",
        "development_mask_count": 32,
        "test_partition_accessed": False,
        "nested_bank_prefixes_verified": True,
        "expected_summary_rows": len(EXPECTED_SEEDS) * len(BANK_COUNTS),
        "actual_summary_rows": len(summary_rows),
        "expected_nearest_rows": len(EXPECTED_SEEDS) * len(BANK_COUNTS) * 32,
        "actual_nearest_rows": len(nearest_rows),
        "all_values_finite": bool(
            all(
                math.isfinite(float(value))
                for row in summary_rows + nearest_summary_rows
                for key, value in row.items()
                if key not in {"pairwise_method"}
            )
        ),
        "source_artifacts": source_artifacts,
        "generated_artifacts": [
            {"path": str(output / name), "sha256": sha256(output / name)}
            for name in generated
        ],
    }
    required_checks = (
        validation["actual_summary_rows"] == validation["expected_summary_rows"]
        and validation["actual_nearest_rows"] == validation["expected_nearest_rows"]
        and validation["all_values_finite"]
        and validation["test_partition_accessed"] is False
    )
    if not required_checks:
        validation["status"] = "fail"
    write_json(output / "validation.json", validation)
    summary = {
        "status": validation["status"],
        "analysis_status": "post-hoc exploratory",
        "feature_contract": "8x8 L2-normalized non-DC log Fourier magnitude",
        "normalization": "grayscale PSF divided by total spatial energy",
        "pairwise_100_1000": "all unordered pairs",
        "pairwise_10000": "deterministic 250,000-pair sample per training seed",
        "exact_collision_audit": "exhaustive SHA256 audit for patterns and normalized spatial PSFs",
        "near_collision_audit": "full for banks <=1000; deterministic sampled pairs for bank 10000",
        "test_partition_accessed": False,
        "validation": str(output / "validation.json"),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "run_state.json", {"status": "complete"})
    if validation["status"] != "pass":
        raise RuntimeError("PSF diversity analysis validation failed")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
