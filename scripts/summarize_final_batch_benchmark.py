"""Select the final-runner batch size using a frozen development-only rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_BATCHES = (1, 2, 4, 8)
MAX_PEAK_VRAM_BYTES = 64 * 1024**3


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_root = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    sources = []
    for batch_size in EXPECTED_BATCHES:
        path = input_root / f"batch{batch_size}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text())
        sources.append({"path": str(path), "sha256": sha256(path)})
        row = {
            "batch_size": batch_size,
            "status": value.get("status"),
            "sample_count": value.get("sample_count"),
            "mask_count": value.get("mask_count"),
            "elapsed_seconds": value.get("elapsed_seconds"),
            "samples_per_second": value.get("samples_per_second"),
            "peak_vram_bytes": value.get("peak_vram_bytes"),
            "PSNR_per_image_max_abs": value.get("parity", {})
            .get("per_image_max_abs", {})
            .get("PSNR"),
            "SSIM_per_image_max_abs": value.get("parity", {})
            .get("per_image_max_abs", {})
            .get("SSIM"),
            "LPIPS_per_image_max_abs": value.get("parity", {})
            .get("per_image_max_abs", {})
            .get("LPIPS"),
            "PSNR_mask_balanced_abs": value.get("parity", {})
            .get("mask_balanced_abs", {})
            .get("PSNR"),
            "SSIM_mask_balanced_abs": value.get("parity", {})
            .get("mask_balanced_abs", {})
            .get("SSIM"),
            "LPIPS_mask_balanced_abs": value.get("parity", {})
            .get("mask_balanced_abs", {})
            .get("LPIPS"),
            "vram_limit_pass": int(value.get("peak_vram_bytes", MAX_PEAK_VRAM_BYTES + 1))
            <= MAX_PEAK_VRAM_BYTES,
            "development_only": value.get("test_scene_files_opened") is False
            and value.get("test_masks_generated") is False
            and value.get("final_test_model_forward_executed") is False,
        }
        rows.append(row)
    passing = [
        row
        for row in rows
        if row["status"] == "pass"
        and row["sample_count"] == 1024
        and row["mask_count"] == 32
        and row["vram_limit_pass"]
        and row["development_only"]
    ]
    if not passing:
        raise RuntimeError("No batch size passed the frozen parity and VRAM rule")
    selected = max(passing, key=lambda row: row["batch_size"])
    batch1 = next(row for row in rows if row["batch_size"] == 1)
    for row in rows:
        row["throughput_speedup_vs_batch1"] = (
            float(row["samples_per_second"]) / float(batch1["samples_per_second"])
        )
        row["selected"] = row["batch_size"] == selected["batch_size"]
    csv_path = output / "batch_benchmark.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "pass",
        "selection_rule": "largest tested batch size passing frozen per-image and mask-balanced parity limits with peak VRAM <= 64 GiB",
        "tested_batch_sizes": list(EXPECTED_BATCHES),
        "selected_batch_size": selected["batch_size"],
        "selected_samples_per_second": selected["samples_per_second"],
        "selected_peak_vram_bytes": selected["peak_vram_bytes"],
        "selected_speedup_vs_batch1": selected["throughput_speedup_vs_batch1"],
        "source_summaries": sources,
        "csv": str(csv_path),
        "csv_sha256": sha256(csv_path),
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "final_test_model_forward_executed": False,
    }
    write_json(output / "selection.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
