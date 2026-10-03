"""Run one worker for the frozen three-GPU synthetic evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from scripts.evaluate_frozen_final_synthetic import (  # noqa: E402
    load_and_validate_metadata as load_v3_metadata,
    resolve_checkpoint,
    resolve_repo_path,
    scalar_list,
    sha256,
    static_checkpoint_metadata,
    write_json,
)


METRICS = ("PSNR", "SSIM", "LPIPS")
ALLOWED_MODES = ("development", "final")


def load_v4_metadata(path: Path) -> tuple[dict, dict, list[dict]]:
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "locked_pre_execution":
        raise ValueError("V4 manifest is not locked")
    if manifest.get("final_test_accessed") is not False:
        raise ValueError("V4 manifest does not prove an untouched final test")
    if manifest.get("checkpoint_selection_by_test_metrics_forbidden") is not True:
        raise ValueError("Post-test checkpoint selection prohibition is missing")
    programs = manifest.get("program_hashes", {})
    expected_programs = {
        "worker": "scripts/final_runner_v4_worker.py",
        "launcher": "scripts/final_runner_v4_launcher.py",
        "merger": "scripts/final_runner_v4_merge.py",
    }
    if set(programs) != set(expected_programs):
        raise ValueError("Unexpected V4 program hash set")
    for name, relative_path in expected_programs.items():
        program_path = resolve_repo_path(relative_path)
        if not program_path.is_file() or sha256(program_path) != programs[name]:
            raise ValueError(f"V4 {name} hash drift")

    selection_path = resolve_repo_path(manifest["batch_protocol"]["selection"])
    if not selection_path.is_file() or sha256(selection_path) != manifest["batch_protocol"][
        "selection_sha256"
    ]:
        raise ValueError("Batch-size selection artifact drift")
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "pass" or selection.get(
        "selected_batch_size"
    ) != manifest["batch_protocol"]["batch_size"]:
        raise ValueError("Batch-size selection mismatch")
    development_reference_path = resolve_repo_path(
        manifest["development_validation_contract"]["reference"]
    )
    if (
        not development_reference_path.is_file()
        or sha256(development_reference_path)
        != manifest["development_validation_contract"]["reference_sha256"]
    ):
        raise ValueError("Development reference drift")

    parent_path = resolve_repo_path(manifest["parent_protocol"]["manifest"])
    if not parent_path.is_file() or sha256(parent_path) != manifest["parent_protocol"][
        "manifest_sha256"
    ]:
        raise ValueError("Parent V3 manifest drift")
    parent, parent_preflight = load_v3_metadata(parent_path)
    shortlist_path = resolve_repo_path(manifest["shortlist_protocol"]["manifest"])
    if (
        not shortlist_path.is_file()
        or sha256(shortlist_path) != manifest["shortlist_protocol"]["manifest_sha256"]
    ):
        raise ValueError("Revised shortlist manifest drift")
    shortlist = json.loads(shortlist_path.read_text())
    entries = shortlist.get("entries", [])
    if shortlist.get("status") != "frozen_revised" or len(entries) != manifest[
        "execution_protocol"
    ]["checkpoint_count"]:
        raise ValueError("Revised shortlist count or status drift")
    expected_ids = [entry["shortlist_id"] for entry in entries]
    shards = manifest.get("shards", [])
    if len(shards) != 3 or [int(shard["shard_id"]) for shard in shards] != [0, 1, 2]:
        raise ValueError("V4 requires exactly shards 0, 1 and 2")
    mapped_ids = [identifier for shard in shards for identifier in shard["checkpoint_ids"]]
    if len(mapped_ids) != len(set(mapped_ids)) or set(mapped_ids) != set(expected_ids):
        raise ValueError("Shard checkpoint mapping is not an exact partition")
    if [int(shard["physical_gpu"]) for shard in shards] != [0, 1, 2]:
        raise ValueError("Frozen physical GPU mapping must be 0, 1, 2")
    return manifest, parent, entries


def validate_final_authorization(path: Path | None, manifest_path: Path) -> dict:
    if path is None or not path.is_file():
        raise PermissionError("Final mode requires a separate user-created V4 authorization JSON")
    value = json.loads(path.read_text())
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "authorized_by_user": True,
        "mode": "final",
        "manifest_sha256": sha256(manifest_path),
        "worker_sha256": manifest["program_hashes"]["worker"],
        "launcher_sha256": manifest["program_hashes"]["launcher"],
    }
    if value != expected:
        raise PermissionError("V4 authorization does not match the frozen protocol")
    return value


def build_loader(parent: dict, mode: str, batch_size: int):
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from src.datasets.mirflickr import MirFlickrSceneDataset
    from src.datasets.on_the_fly import DigiCamOnTheFlyDataset, DigiCamValidationBatchSampler
    from src.digicam_synth.mask_protocol import get_mask_records

    split_path = resolve_repo_path(parent["scene_protocol"]["source_split_manifest"])
    if mode == "development":
        split = "validation"
        mask_partition = "validation"
        mask_count = 32
        scenes_per_mask = 32
        allow_test = False
        expected_available_scenes = 128
        expected_mask_ids = parent["mask_protocol"]["development_mask_ids"]
    elif mode == "final":
        split = "test"
        mask_partition = "test"
        mask_count = parent["mask_protocol"]["test_mask_count"]
        scenes_per_mask = parent["scene_protocol"]["split_counts"]["test"]
        allow_test = True
        expected_available_scenes = scenes_per_mask
        expected_mask_ids = parent["mask_protocol"]["test_mask_ids"]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    scenes = MirFlickrSceneDataset(
        root_dir=REPO_ROOT / "data/raw/mirflickr25k/extracted",
        splits_path=split_path,
        split=split,
        image_size=None,
        verify_files=True,
    )
    if len(scenes) != expected_available_scenes:
        raise ValueError(f"Unexpected {mode} scene count")
    masks = get_mask_records(
        parent["mask_protocol"]["base_seed"],
        mask_partition,
        mask_count,
        allow_test=allow_test,
    )
    if [record["mask_id"] for record in masks] != expected_mask_ids:
        raise ValueError(f"Frozen {mode} mask IDs drifted")
    simulator = OmegaConf.load(
        resolve_repo_path(parent["evaluation_protocol"]["simulator_config"])
    )
    psf_cache = (
        {
            "mode": "read_only",
            "root_dir": str(REPO_ROOT / "data/psf_cache"),
            "request_modes": ["finite"],
        }
        if mode == "development"
        else {"mode": "off"}
    )
    dataset = DigiCamOnTheFlyDataset(
        scenes,
        simulator,
        measurement_size=None,
        target_size=(200, 266),
        simulation_mode="roi_convolution",
        roi=(80, 100, 200, 266),
        finite_cache_size=2,
        psf_cache=psf_cache,
    )
    sampler = DigiCamValidationBatchSampler(
        scene_count=len(scenes),
        batch_size=batch_size,
        mask_records=masks,
        run_seed=parent["evaluation_protocol"]["scene_selector_seed"],
        scenes_per_mask=scenes_per_mask,
        scene_offset=0,
        scene_selector_salt=parent["evaluation_protocol"]["scene_selector_salt"],
    )
    return (
        DataLoader(dataset, batch_sampler=sampler, num_workers=0),
        masks,
        scenes_per_mask,
    )


def instantiate_model(parent: dict, checkpoint: Path):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    model_config = OmegaConf.load(
        resolve_repo_path(parent["evaluation_protocol"]["model_config"])
    )
    model_config.checkpoint_path = None
    model_config.output_crop = parent["evaluation_protocol"]["roi"]
    model = instantiate(model_config).cuda().eval()
    value = torch.load(str(checkpoint), map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(value["state_dict"], strict=True)
    del value
    return model


def evaluate_one(
    *,
    entry: dict,
    checkpoint: Path,
    parent: dict,
    mode: str,
    batch_size: int,
    output: Path,
    metrics,
) -> dict:
    import torch

    model = instantiate_model(parent, checkpoint)
    loader, masks, scenes_per_mask = build_loader(parent, mode, batch_size)
    rows = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch in loader:
            measurement = batch["measurement"].cuda()
            target = batch["target"].cuda()
            prediction = model(measurement=measurement)["prediction"].float()
            if prediction.shape != target.shape:
                raise ValueError("Prediction/target shape mismatch")
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
                row = {
                    "sample_index": len(rows),
                    "mask_id": str(mask_ids[offset]),
                    "scene_id": str(scene_ids[offset]),
                    **{
                        name: float(metric_values[offset])
                        for name, metric_values in values.items()
                    },
                }
                if not all(np.isfinite(row[metric]) for metric in METRICS):
                    raise FloatingPointError("Non-finite metric")
                rows.append(row)
            if len(rows) % 256 == 0:
                print(entry["shortlist_id"], len(rows), flush=True)
    expected_samples = len(masks) * scenes_per_mask
    if len(rows) != expected_samples:
        raise ValueError("Exact sample count failed")
    grouped = defaultdict(lambda: {metric: [] for metric in METRICS})
    for row in rows:
        for metric in METRICS:
            grouped[row["mask_id"]][metric].append(row[metric])
    if len(grouped) != len(masks) or {
        len(values["PSNR"]) for values in grouped.values()
    } != {scenes_per_mask}:
        raise ValueError("Mask-balanced grid is incomplete")
    per_mask = [
        {
            "mask_id": mask_id,
            "sample_count": len(values["PSNR"]),
            **{metric: float(np.mean(values[metric])) for metric in METRICS},
        }
        for mask_id, values in sorted(grouped.items())
    ]
    with (output / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "per_mask.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_mask[0]))
        writer.writeheader()
        writer.writerows(per_mask)
    elapsed = time.monotonic() - started
    grid_identity_sha256 = hashlib.sha256(
        json.dumps(
            [(row["sample_index"], row["mask_id"], row["scene_id"]) for row in rows],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    summary = {
        "status": "complete",
        "mode": mode,
        "shortlist_id": entry["shortlist_id"],
        "sample_count": len(rows),
        "mask_count": len(per_mask),
        "scenes_per_mask": scenes_per_mask,
        "batch_size": batch_size,
        "mask_balanced": {
            metric: float(np.mean([row[metric] for row in per_mask]))
            for metric in METRICS
        },
        "elapsed_seconds": elapsed,
        "samples_per_second": len(rows) / elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": entry["sha256"],
        "grid_identity_sha256": grid_identity_sha256,
        "dataset_split": "validation" if mode == "development" else "test",
        "mask_partition": "validation" if mode == "development" else "test",
        "test_accessed": mode == "final",
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "validation.json",
        {
            "status": "pass",
            "sample_count_match": True,
            "mask_count_match": True,
            "scenes_per_mask_match": True,
            "metrics_finite": True,
            "checkpoint_sha256_match": True,
            "checkpoint_endpoint_match": static_checkpoint_metadata(checkpoint)
            == {
                key: int(entry[key])
                for key in ("epoch", "global_step", "sampler_step", "T_max")
            },
            "selected_using_test": False,
            "test_accessed": mode == "final",
        },
    )
    del model
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=ALLOWED_MODES, required=True)
    parser.add_argument("--shard-id", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--authorization")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest, parent, entries = load_v4_metadata(manifest_path)
    shard = manifest["shards"][args.shard_id]
    if int(shard["shard_id"]) != args.shard_id:
        raise ValueError("Shard order drift")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(shard["physical_gpu"]) or visible not in {
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
    }:
        raise RuntimeError("Worker physical GPU does not match the frozen shard mapping")
    if args.mode == "final":
        authorization = validate_final_authorization(
            Path(args.authorization).resolve() if args.authorization else None,
            manifest_path,
        )
    else:
        if args.authorization is not None:
            raise ValueError("Development mode does not accept an authorization file")
        authorization = None

    import torch

    from src.metrics.reconstruction import LPIPSMetric, PSNRMetric, SSIMMetric
    from src.utils.init_utils import set_random_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_random_seed(42)
    torch.set_num_threads(4)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "run_state.json",
        {
            "status": "preflight",
            "mode": args.mode,
            "shard_id": args.shard_id,
            "physical_gpu": visible,
        },
    )
    by_id = {entry["shortlist_id"]: entry for entry in entries}
    assigned = [by_id[identifier] for identifier in shard["checkpoint_ids"]]
    checkpoints = {entry["shortlist_id"]: resolve_checkpoint(entry) for entry in assigned}
    write_json(
        output / "preflight.json",
        {
            "status": "pass",
            "mode": args.mode,
            "shard_id": args.shard_id,
            "physical_gpu": visible,
            "batch_size": manifest["batch_protocol"]["batch_size"],
            "checkpoint_ids": shard["checkpoint_ids"],
            "checkpoint_hashes_verified_before_data_access": True,
            "authorization": authorization,
            "test_accessed": False,
        },
    )
    metrics = (
        PSNRMetric(name="PSNR", normalize_by_max=True),
        SSIMMetric(name="SSIM", normalize_by_max=True),
        LPIPSMetric(name="LPIPS", net_type="vgg", device="cuda", normalize_by_max=True),
    )
    summaries = []
    write_json(output / "run_state.json", {"status": "running", "mode": args.mode})
    try:
        for entry in assigned:
            model_output = output / entry["shortlist_id"]
            model_output.mkdir(exist_ok=False)
            write_json(model_output / "run_state.json", {"status": "running"})
            summary = evaluate_one(
                entry=entry,
                checkpoint=checkpoints[entry["shortlist_id"]],
                parent=parent,
                mode=args.mode,
                batch_size=manifest["batch_protocol"]["batch_size"],
                output=model_output,
                metrics=metrics,
            )
            write_json(model_output / "run_state.json", {"status": "complete"})
            summaries.append(summary)
        with (output / "summary.csv").open("w", newline="") as stream:
            fields = [
                "shortlist_id",
                "PSNR",
                "SSIM",
                "LPIPS",
                "sample_count",
                "mask_count",
                "elapsed_seconds",
                "samples_per_second",
                "peak_vram_bytes",
                "grid_identity_sha256",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for summary in summaries:
                writer.writerow(
                    {
                        "shortlist_id": summary["shortlist_id"],
                        **summary["mask_balanced"],
                        **{key: summary[key] for key in fields[4:]},
                    }
                )
        validation = {
            "status": "pass",
            "mode": args.mode,
            "shard_id": args.shard_id,
            "checkpoint_ids": shard["checkpoint_ids"],
            "completed_checkpoint_ids": [row["shortlist_id"] for row in summaries],
            "exact_checkpoint_partition": [row["shortlist_id"] for row in summaries]
            == shard["checkpoint_ids"],
            "all_model_validations_pass": all(
                json.loads((output / row["shortlist_id"] / "validation.json").read_text())[
                    "status"
                ]
                == "pass"
                for row in summaries
            ),
            "test_accessed": args.mode == "final",
            "final_test_model_forward_executed": args.mode == "final",
        }
        write_json(output / "validation.json", validation)
        write_json(
            output / "provenance.json",
            {
                "status": "pass",
                "mode": args.mode,
                "shard_id": args.shard_id,
                "physical_gpu": visible,
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "worker": str(Path(__file__).resolve()),
                "worker_sha256": sha256(Path(__file__).resolve()),
                "checkpoint_ids": shard["checkpoint_ids"],
                "test_accessed": args.mode == "final",
            },
        )
        write_json(output / "run_state.json", {"status": "complete", "mode": args.mode})
        print(json.dumps(validation, indent=2), flush=True)
    except BaseException as error:
        write_json(
            output / "run_state.json",
            {
                "status": "failed_closed",
                "mode": args.mode,
                "error_type": type(error).__name__,
                "error": str(error),
                "automatic_retry": False,
                "checkpoint_substitution": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
