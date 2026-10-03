"""Check a selected checkpoint against its own BF16 development reference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_final_runner_batch_size import (  # noqa: E402
    build_loader,
    scalar_list,
    sha256,
    write_json,
)
from scripts.evaluate_frozen_final_synthetic import static_checkpoint_metadata  # noqa: E402


METRICS = ("PSNR", "SSIM", "LPIPS")
TOLERANCES = {"PSNR": 0.01, "SSIM": 0.001, "LPIPS": 0.001}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-per-mask", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"0", "1", "2", "3", "4", "5"}:
        raise RuntimeError("Use one physical GPU from 0 through 5")

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from src.metrics.reconstruction import LPIPSMetric, PSNRMetric, SSIMMetric
    from src.utils.init_utils import set_random_seed

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_state.json", {"status": "preflight"})
    checkpoint = Path(args.checkpoint).resolve()
    reference_path = Path(args.reference_per_mask).resolve()
    if not checkpoint.is_file() or not reference_path.is_file():
        raise FileNotFoundError("Checkpoint or reference is missing")
    with reference_path.open(newline="") as stream:
        reference_rows = list(csv.DictReader(stream))
    if len(reference_rows) != 32 or len({row["mask_id"] for row in reference_rows}) != 32:
        raise ValueError("Reference must contain 32 mask rows")
    reference = {
        metric: float(np.mean([float(row[metric]) for row in reference_rows]))
        for metric in METRICS
    }
    endpoint = static_checkpoint_metadata(checkpoint)
    if endpoint.get("global_step") not in {50_000, 100_000}:
        raise ValueError("Checkpoint is not a frozen final endpoint")
    set_random_seed(42)
    torch.set_num_threads(4)
    loader, masks = build_loader(1)
    model_config = OmegaConf.load(REPO_ROOT / "src/configs/model/psf_free_xrestormer.yaml")
    model_config.checkpoint_path = None
    model_config.output_crop = [80, 100, 200, 266]
    model = instantiate(model_config).cuda().eval()
    value = torch.load(str(checkpoint), map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(value["state_dict"], strict=True)
    del value
    metrics = (
        PSNRMetric(name="PSNR", normalize_by_max=True),
        SSIMMetric(name="SSIM", normalize_by_max=True),
        LPIPSMetric(name="LPIPS", net_type="vgg", device="cuda", normalize_by_max=True),
    )
    rows = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    write_json(output / "run_state.json", {"status": "running"})
    with torch.inference_mode():
        for batch in loader:
            measurement = batch["measurement"].cuda()
            target = batch["target"].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(measurement=measurement)["prediction"]
            prediction = prediction.float()
            values = {
                metric.name: metric.per_image(prediction=prediction, target=target)
                .detach()
                .cpu()
                .reshape(-1)
                for metric in metrics
            }
            count = prediction.shape[0]
            mask_ids = scalar_list(batch["mask_id"], count)
            scene_ids = scalar_list(batch["scene_id"], count)
            for offset in range(count):
                rows.append(
                    {
                        "sample_index": len(rows),
                        "mask_id": str(mask_ids[offset]),
                        "scene_id": str(scene_ids[offset]),
                        **{
                            name: float(metric_values[offset])
                            for name, metric_values in values.items()
                        },
                    }
                )
            if len(rows) % 256 == 0:
                print(args.name, len(rows), "/1024", flush=True)
    if len(rows) != 1024:
        raise ValueError("Expected 1024 development rows")
    grouped = defaultdict(lambda: {metric: [] for metric in METRICS})
    for row in rows:
        for metric in METRICS:
            grouped[row["mask_id"]][metric].append(row[metric])
    if len(grouped) != len(masks) or {len(v["PSNR"]) for v in grouped.values()} != {32}:
        raise ValueError("Development grid balance failed")
    observed = {
        metric: float(np.mean([np.mean(values[metric]) for values in grouped.values()]))
        for metric in METRICS
    }
    deltas = {metric: observed[metric] - reference[metric] for metric in METRICS}
    passed = all(abs(deltas[metric]) <= TOLERANCES[metric] for metric in METRICS)
    with (output / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "pass" if passed else "fail",
        "name": args.name,
        "precision": "BF16 autocast",
        "sample_count": len(rows),
        "mask_count": len(grouped),
        "observed": observed,
        "reference": reference,
        "deltas": deltas,
        "tolerances": TOLERANCES,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_endpoint": endpoint,
        "reference_per_mask": str(reference_path),
        "reference_per_mask_sha256": sha256(reference_path),
        "dataset_split": "validation",
        "mask_partition": "validation",
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "final_test_model_forward_executed": False,
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "run_state.json",
        {"status": "complete" if passed else "parity_failed"},
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
