"""Extract deterministic Fourier features for train/development PSFs.

The final-test mask namespace is deliberately unavailable in this program.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.digicam_synth.mask_protocol import get_mask_records  # noqa: E402
from src.digicam_synth.pipeline import (  # noqa: E402
    generate_random_pattern,
    pattern_to_psf,
)


RGB_WEIGHTS = np.asarray([0.299, 0.587, 0.114], dtype=np.float64)
ALLOWED_PARTITIONS = {"train", "validation"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def fourier_features(
    psf: np.ndarray,
    *,
    feature_side: int,
    radial_bands: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    psf = np.asarray(psf, dtype=np.float64)
    if psf.ndim != 4 or psf.shape[0] != 1 or psf.shape[-1] not in {1, 3}:
        raise ValueError(f"Expected DHWC PSF, received {psf.shape}")
    spatial = psf[0]
    if spatial.shape[-1] == 3:
        spatial = np.sum(spatial * RGB_WEIGHTS[None, None, :], axis=-1)
    else:
        spatial = spatial[..., 0]
    if not np.isfinite(spatial).all() or float(spatial.min()) < 0:
        raise ValueError("PSF contains invalid values")
    energy = float(spatial.sum())
    if energy <= 0:
        raise ValueError("PSF has no positive energy")
    spatial = spatial / energy
    spatial32 = np.ascontiguousarray(spatial, dtype=np.float32)

    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(spatial)))
    center = (magnitude.shape[0] // 2, magnitude.shape[1] // 2)
    magnitude[center] = 0.0
    log_magnitude = np.log1p(magnitude)
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        torch.from_numpy(log_magnitude.astype(np.float32))[None, None],
        (feature_side, feature_side),
    )[0, 0].numpy()
    feature = pooled.reshape(-1).astype(np.float64)
    feature_norm = float(np.linalg.norm(feature))
    if feature_norm <= 0:
        raise ValueError("Fourier feature has zero norm")
    feature = np.ascontiguousarray(feature / feature_norm, dtype=np.float32)

    yy, xx = np.indices(magnitude.shape)
    radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    radius /= float(radius.max())
    bins = np.minimum((radius * radial_bands).astype(np.int64), radial_bands - 1)
    power = magnitude**2
    radial = np.bincount(
        bins.reshape(-1), weights=power.reshape(-1), minlength=radial_bands
    ).astype(np.float64)
    radial_sum = float(radial.sum())
    if radial_sum <= 0:
        raise ValueError("Non-DC Fourier power is zero")
    radial = np.ascontiguousarray(radial / radial_sum, dtype=np.float32)
    return feature, radial, array_sha256(spatial32)


def extract_partition(
    config,
    *,
    base_seed: int,
    partition: str,
    count: int,
    feature_side: int,
    radial_bands: int,
) -> dict[str, np.ndarray]:
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError(f"Closed mask partition: {partition}")
    records = get_mask_records(base_seed, partition, count)
    features = np.empty((count, feature_side * feature_side), dtype=np.float32)
    radial = np.empty((count, radial_bands), dtype=np.float32)
    ids: list[str] = []
    seeds = np.empty(count, dtype=np.uint64)
    pattern_hashes: list[str] = []
    psf_hashes: list[str] = []
    started = time.monotonic()
    for index, record in enumerate(records):
        mask_seed = int(record["mask_seed"])
        rng = np.random.default_rng(mask_seed)
        pattern = generate_random_pattern(config, rng)
        psf, _, _ = pattern_to_psf(
            config,
            pattern,
            rng,
            create_simulator=False,
        )
        if torch.is_tensor(psf):
            psf = psf.detach().cpu().numpy()
        feature, radial_energy, psf_hash = fourier_features(
            psf,
            feature_side=feature_side,
            radial_bands=radial_bands,
        )
        features[index] = feature
        radial[index] = radial_energy
        ids.append(str(record["mask_id"]))
        seeds[index] = mask_seed
        pattern_hashes.append(array_sha256(np.asarray(pattern, dtype=np.float32)))
        psf_hashes.append(psf_hash)
        if (index + 1) % 100 == 0 or index + 1 == count:
            elapsed = time.monotonic() - started
            print(
                f"{base_seed} {partition}: {index + 1}/{count} "
                f"({elapsed:.1f}s)",
                flush=True,
            )
    return {
        "mask_id": np.asarray(ids),
        "mask_seed": seeds,
        "partition": np.asarray([partition] * count),
        "fourier_feature": features,
        "radial_energy": radial,
        "pattern_sha256": np.asarray(pattern_hashes),
        "psf_sha256": np.asarray(psf_hashes),
    }


def save_partition(output: Path, partition: str, arrays: dict[str, np.ndarray]) -> dict:
    npz_path = output / f"{partition}_features.npz"
    np.savez_compressed(npz_path, **arrays)
    csv_path = output / f"{partition}_features.csv.gz"
    feature_count = arrays["fourier_feature"].shape[1]
    radial_count = arrays["radial_energy"].shape[1]
    fields = [
        "mask_id",
        "mask_seed",
        "partition",
        "pattern_sha256",
        "psf_sha256",
        *[f"fourier_{index:03d}" for index in range(feature_count)],
        *[f"radial_{index:02d}" for index in range(radial_count)],
    ]
    with gzip.open(csv_path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(arrays["mask_id"])):
            row = {
                "mask_id": arrays["mask_id"][index],
                "mask_seed": int(arrays["mask_seed"][index]),
                "partition": arrays["partition"][index],
                "pattern_sha256": arrays["pattern_sha256"][index],
                "psf_sha256": arrays["psf_sha256"][index],
            }
            row.update(
                {
                    f"fourier_{column:03d}": f"{value:.9g}"
                    for column, value in enumerate(arrays["fourier_feature"][index])
                }
            )
            row.update(
                {
                    f"radial_{column:02d}": f"{value:.9g}"
                    for column, value in enumerate(arrays["radial_energy"][index])
                }
            )
            writer.writerow(row)
    return {
        "count": int(len(arrays["mask_id"])),
        "npz": str(npz_path),
        "npz_sha256": sha256(npz_path),
        "csv_gz": str(csv_path),
        "csv_gz_sha256": sha256(csv_path),
        "unique_mask_ids": len(set(arrays["mask_id"].tolist())),
        "unique_mask_seeds": len(set(map(int, arrays["mask_seed"].tolist()))),
        "unique_pattern_hashes": len(set(arrays["pattern_sha256"].tolist())),
        "unique_psf_hashes": len(set(arrays["psf_sha256"].tolist())),
        "features_finite": bool(np.isfinite(arrays["fourier_feature"]).all()),
        "radial_finite": bool(np.isfinite(arrays["radial_energy"]).all()),
        "feature_norm_max_error": float(
            np.max(np.abs(np.linalg.norm(arrays["fourier_feature"], axis=1) - 1))
        ),
        "radial_sum_max_error": float(
            np.max(np.abs(arrays["radial_energy"].sum(axis=1) - 1))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--train-count", type=int, default=10_000)
    parser.add_argument("--include-development", action="store_true")
    parser.add_argument("--development-count", type=int, default=32)
    parser.add_argument("--feature-side", type=int, default=8)
    parser.add_argument("--radial-bands", type=int, default=12)
    parser.add_argument(
        "--simulator-config", default="src/configs/simulator/digicam_article.yaml"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.train_count != 10_000:
        raise ValueError("Extractor must materialize the full nested 10,000-mask bank")
    if args.development_count != 32:
        raise ValueError("Frozen development mask count must be 32")
    if args.feature_side < 2 or args.radial_bands < 2:
        raise ValueError("Feature dimensions are too small")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_state.json", {"status": "running"})
    config_path = Path(args.simulator_config).resolve()
    config = OmegaConf.load(config_path)
    torch.set_num_threads(2)
    partitions = [("train", args.train_count)]
    if args.include_development:
        partitions.append(("validation", args.development_count))

    artifacts = {}
    all_arrays = {}
    started = time.monotonic()
    for partition, count in partitions:
        arrays = extract_partition(
            config,
            base_seed=args.base_seed,
            partition=partition,
            count=count,
            feature_side=args.feature_side,
            radial_bands=args.radial_bands,
        )
        all_arrays[partition] = arrays
        artifacts[partition] = save_partition(output, partition, arrays)

    namespace_disjoint = True
    if "validation" in all_arrays:
        namespace_disjoint = set(all_arrays["train"]["mask_seed"].tolist()).isdisjoint(
            all_arrays["validation"]["mask_seed"].tolist()
        )
    validation = {
        "status": "pass",
        "base_seed": args.base_seed,
        "allowed_partitions": sorted(ALLOWED_PARTITIONS),
        "test_partition_accessed": False,
        "train_count_match": artifacts["train"]["count"] == 10_000,
        "development_count_match": (
            artifacts.get("validation", {}).get("count") == 32
            if args.include_development
            else None
        ),
        "namespace_disjoint": namespace_disjoint,
        "artifacts": artifacts,
    }
    checks = [
        validation["train_count_match"],
        namespace_disjoint,
        *[
            item[key]
            for item in artifacts.values()
            for key in ("features_finite", "radial_finite")
        ],
        *[
            item["unique_mask_ids"] == item["count"]
            and item["unique_mask_seeds"] == item["count"]
            for item in artifacts.values()
        ],
    ]
    if not all(checks):
        validation["status"] = "fail"
    write_json(output / "validation.json", validation)
    provenance = {
        "schema_version": 1,
        "analysis_status": "post-hoc exploratory",
        "base_seed": args.base_seed,
        "simulator_config": str(config_path),
        "simulator_config_sha256": sha256(config_path),
        "feature_contract": {
            "spatial_normalization": "grayscale PSF divided by total energy",
            "fourier_transform": "absolute centered 2D FFT with DC set to zero",
            "fourier_compression": f"log1p magnitude, adaptive average pool to {args.feature_side}x{args.feature_side}, L2 normalization",
            "radial_energy_bands": args.radial_bands,
        },
        "extractor": str(Path(__file__).resolve()),
        "extractor_sha256": sha256(Path(__file__).resolve()),
        "source_hashes": {
            str(path): sha256(path)
            for path in (
                REPO_ROOT / "src/digicam_synth/mask_protocol.py",
                REPO_ROOT / "src/digicam_synth/pipeline.py",
            )
        },
        "elapsed_seconds": time.monotonic() - started,
        "test_partition_accessed": False,
    }
    write_json(output / "provenance.json", provenance)
    write_json(output / "run_state.json", {"status": "complete"})
    if validation["status"] != "pass":
        raise RuntimeError("PSF feature validation failed")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
