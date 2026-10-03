"""Evaluate a frozen synthetic final-test protocol after explicit authorization.

Without ``--execute-final-test`` this program validates metadata only. Test scenes
and test masks are constructed exclusively inside the authorized execution path.
"""

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

METRICS = ("PSNR", "SSIM", "LPIPS")
EXPECTED_MASK_IDS = tuple(f"test_{index:05d}" for index in range(100))
EXPECTED_SOURCE_PATHS = (
    "src/model/psf_free_xrestormer.py",
    "src/datasets/on_the_fly.py",
    "src/datasets/mirflickr.py",
    "src/digicam_synth/mask_protocol.py",
    "src/digicam_synth/pipeline.py",
    "src/metrics/reconstruction.py",
)


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


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def static_checkpoint_metadata(path: Path) -> dict[str, int]:
    import pickletools
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            name = next(value for value in archive.namelist() if value.endswith("/data.pkl"))
            data = archive.read(name)
    except (OSError, StopIteration, zipfile.BadZipFile):
        return {}
    operations = list(pickletools.genops(data))
    fields = {"epoch", "global_step", "sampler_step", "T_max"}
    ignored = {"BINPUT", "LONG_BINPUT", "MEMOIZE", "PUT"}
    integers = {"BININT", "BININT1", "BININT2", "INT", "LONG1", "LONG4"}
    result = {}
    for index, (opcode, argument, _) in enumerate(operations):
        if opcode.name not in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}:
            continue
        if argument not in fields:
            continue
        cursor = index + 1
        while cursor < len(operations) and operations[cursor][0].name in ignored:
            cursor += 1
        if cursor < len(operations) and operations[cursor][0].name in integers:
            value = int(operations[cursor][1])
            if argument in result and result[argument] != value:
                return {}
            result[argument] = value
    return result


def load_and_validate_metadata(manifest_path: Path) -> tuple[dict, dict]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "locked_pre_execution":
        raise ValueError("Final-test manifest is not locked for pre-execution")
    if manifest.get("model_forward_executed") is not False:
        raise ValueError("Manifest does not prove an untouched final test")
    if manifest.get("checkpoint_selection_by_test_metrics_forbidden") is not True:
        raise ValueError("Checkpoint selection prohibition is missing")
    if manifest.get("evaluator_sha256") != sha256(Path(__file__).resolve()):
        raise ValueError("Evaluator hash drifted after final-test freeze")

    split_path = resolve_repo_path(manifest["scene_protocol"]["source_split_manifest"])
    shortlist_path = resolve_repo_path(manifest["checkpoint_protocol"]["manifest"])
    remote_audit_path = resolve_repo_path(
        manifest["checkpoint_protocol"]["remote_audit_summary"]
    )
    for path in (split_path, shortlist_path, remote_audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(split_path) != manifest["scene_protocol"]["source_split_manifest_sha256"]:
        raise ValueError("Scene split manifest hash drift")
    if sha256(shortlist_path) != manifest["checkpoint_protocol"]["manifest_sha256"]:
        raise ValueError("Frozen checkpoint manifest hash drift")
    source_hashes = manifest.get("source_hashes", {})
    if set(source_hashes) != set(EXPECTED_SOURCE_PATHS):
        raise ValueError("Evaluation source hash set drift")
    for relative_path, expected_hash in source_hashes.items():
        source_path = resolve_repo_path(relative_path)
        if not source_path.is_file() or sha256(source_path) != expected_hash:
            raise ValueError(f"Evaluation source hash drift: {relative_path}")

    evaluation_protocol = manifest["evaluation_protocol"]
    for path_key, hash_key in (
        ("simulator_config", "simulator_config_sha256"),
        ("model_config", "model_config_sha256"),
    ):
        config_path = resolve_repo_path(evaluation_protocol[path_key])
        if not config_path.is_file() or sha256(config_path) != evaluation_protocol[hash_key]:
            raise ValueError(f"Evaluation config hash drift: {path_key}")
    if evaluation_protocol.get("batch_size") != 1 or evaluation_protocol.get(
        "evaluation_precision"
    ) != "FP32":
        raise ValueError("Frozen precision or batch size drift")

    split_payload = json.loads(split_path.read_text())
    splits = split_payload.get("splits", {})
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Scene split schema drift")
    scene_sets = {name: set(values) for name, values in splits.items()}
    if any(
        scene_sets[left] & scene_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Scene split overlap detected")
    expected_counts = manifest["scene_protocol"]["split_counts"]
    if {name: len(values) for name, values in splits.items()} != expected_counts:
        raise ValueError("Scene split count drift")
    if tuple(splits["test"]) != tuple(manifest["scene_protocol"]["test_scene_ids"]):
        raise ValueError("Frozen test scene IDs drifted")
    order = np.random.default_rng(
        np.random.SeedSequence(
            [
                evaluation_protocol["scene_selector_seed"],
                evaluation_protocol["scene_selector_salt"],
            ]
        )
    ).permutation(len(splits["test"]))
    ordered_test_scenes = tuple(splits["test"][int(index)] for index in order)
    if ordered_test_scenes != tuple(manifest["scene_protocol"]["test_scene_order"]):
        raise ValueError("Frozen deterministic test scene order drifted")

    mask_protocol = manifest["mask_protocol"]
    if tuple(mask_protocol["test_mask_ids"]) != EXPECTED_MASK_IDS:
        raise ValueError("Frozen test mask IDs drifted")
    train_ids = set(mask_protocol["train_mask_ids"])
    development_ids = set(mask_protocol["development_mask_ids"])
    test_ids = set(mask_protocol["test_mask_ids"])
    if train_ids & development_ids or train_ids & test_ids or development_ids & test_ids:
        raise ValueError("Mask namespace overlap detected")
    if len(train_ids) != 10_000 or len(development_ids) != 32 or len(test_ids) != 100:
        raise ValueError("Mask counts drifted")
    expected_samples = len(splits["test"]) * len(test_ids)
    if expected_samples != manifest["evaluation_protocol"]["expected_samples_per_checkpoint"]:
        raise ValueError("Expected sample count drifted")
    qualitative_indices = (0, 255, 256, 12_800, 25_599)
    expected_qualitative = [
        {
            "sample_index": index,
            "mask_id": EXPECTED_MASK_IDS[index // len(ordered_test_scenes)],
            "scene_id": ordered_test_scenes[index % len(ordered_test_scenes)],
        }
        for index in qualitative_indices
    ]
    if evaluation_protocol.get("qualitative_examples") != expected_qualitative:
        raise ValueError("Frozen qualitative example IDs drifted")

    shortlist = json.loads(shortlist_path.read_text())
    entries = shortlist.get("entries", [])
    if len(entries) != 13 or shortlist.get("status") != "frozen":
        raise ValueError("Frozen checkpoint shortlist drifted")
    if any(entry.get("selected_using_final_test") is not False for entry in entries):
        raise ValueError("A checkpoint was selected using final-test evidence")
    remote_audit = json.loads(remote_audit_path.read_text())
    if remote_audit.get("status") != "pass" or remote_audit.get("sha256_match_count") != 13:
        raise ValueError("Remote shortlist audit is not complete")
    preflight = {
        "status": "pass",
        "metadata_only": True,
        "scene_split_counts": expected_counts,
        "scene_overlap_count": 0,
        "mask_counts": {"train": 10_000, "development": 32, "test": 100},
        "mask_overlap_count": 0,
        "checkpoint_count": len(entries),
        "expected_samples_per_checkpoint": expected_samples,
        "expected_total_model_forwards": expected_samples * len(entries),
        "evaluator_sha256": manifest["evaluator_sha256"],
        "manifest_sha256": sha256(manifest_path),
        "checkpoint_manifest_sha256": sha256(shortlist_path),
        "checkpoint_selection_by_test_metrics_forbidden": True,
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "model_forward_executed": False,
    }
    return manifest, preflight


def validate_authorization(
    authorization_path: Path | None,
    *,
    manifest_path: Path,
) -> dict:
    if authorization_path is None or not authorization_path.is_file():
        raise PermissionError(
            "Final-test execution needs a separate user-created authorization JSON"
        )
    authorization = json.loads(authorization_path.read_text())
    expected = {
        "authorized_by_user": True,
        "manifest_sha256": sha256(manifest_path),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
    }
    if authorization != expected:
        raise PermissionError("Authorization JSON does not match the frozen protocol")
    return authorization


def resolve_checkpoint(entry: dict) -> Path:
    candidates = (
        Path(entry["checkpoint_remote_path"]),
        Path(entry["checkpoint_local_path"]),
    )
    matches = [path.resolve() for path in candidates if path.is_file()]
    if not matches:
        raise FileNotFoundError(entry["shortlist_id"])
    for path in matches:
        if path.stat().st_size != int(entry["size_bytes"]) or sha256(path) != entry["sha256"]:
            continue
        metadata = static_checkpoint_metadata(path)
        if all(
            int(metadata.get(key, -1)) == int(entry[key])
            for key in ("epoch", "global_step", "sampler_step", "T_max")
        ):
            return path
    raise ValueError(f"No exact checkpoint match for {entry['shortlist_id']}")


def scalar_list(value, count: int) -> list:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * count


def build_loader(manifest: dict, scene_root: Path):
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from src.datasets.mirflickr import MirFlickrSceneDataset
    from src.datasets.on_the_fly import DigiCamOnTheFlyDataset, DigiCamValidationBatchSampler
    from src.digicam_synth.mask_protocol import get_mask_records

    scene_protocol = manifest["scene_protocol"]
    scene_dataset = MirFlickrSceneDataset(
        root_dir=scene_root,
        splits_path=resolve_repo_path(scene_protocol["source_split_manifest"]),
        split="test",
        image_size=None,
        verify_files=True,
    )
    if len(scene_dataset) != scene_protocol["split_counts"]["test"]:
        raise ValueError("Test scene count mismatch after dataset construction")
    mask_records = get_mask_records(
        manifest["mask_protocol"]["base_seed"],
        "test",
        manifest["mask_protocol"]["test_mask_count"],
        allow_test=True,
    )
    if tuple(row["mask_id"] for row in mask_records) != tuple(
        manifest["mask_protocol"]["test_mask_ids"]
    ):
        raise ValueError("Generated test mask IDs do not match the frozen manifest")
    simulator = OmegaConf.load(
        resolve_repo_path(manifest["evaluation_protocol"]["simulator_config"])
    )
    dataset = DigiCamOnTheFlyDataset(
        scene_dataset,
        simulator,
        measurement_size=None,
        target_size=(200, 266),
        simulation_mode="roi_convolution",
        roi=(80, 100, 200, 266),
        finite_cache_size=2,
        psf_cache={"mode": "off"},
    )
    sampler = DigiCamValidationBatchSampler(
        scene_count=len(scene_dataset),
        batch_size=1,
        mask_records=mask_records,
        run_seed=manifest["evaluation_protocol"]["scene_selector_seed"],
        scenes_per_mask=len(scene_dataset),
        scene_offset=0,
        scene_selector_salt=manifest["evaluation_protocol"]["scene_selector_salt"],
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0), mask_records


def evaluate_checkpoint(
    entry: dict,
    *,
    checkpoint: Path,
    loader,
    manifest: dict,
    output: Path,
) -> dict:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from src.metrics.reconstruction import LPIPSMetric, PSNRMetric, SSIMMetric

    model_config = OmegaConf.load(
        resolve_repo_path(manifest["evaluation_protocol"]["model_config"])
    )
    model_config.checkpoint_path = None
    model_config.output_crop = manifest["evaluation_protocol"]["roi"]
    model = instantiate(model_config).cuda().eval()
    checkpoint_value = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    model.load_state_dict(checkpoint_value["state_dict"], strict=True)
    del checkpoint_value
    metrics = (
        PSNRMetric(name="PSNR", normalize_by_max=True),
        SSIMMetric(name="SSIM", normalize_by_max=True),
        LPIPSMetric(name="LPIPS", net_type="vgg", device="cuda", normalize_by_max=True),
    )
    rows = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            measurement = batch["measurement"].cuda()
            target = batch["target"].cuda()
            prediction = model(measurement=measurement)["prediction"].float()
            if prediction.shape != target.shape:
                raise ValueError("Prediction/target shape mismatch")
            values = {
                metric.name: metric.per_image(prediction=prediction, target=target)
                .detach()
                .cpu()
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
                if not all(np.isfinite(row[name]) for name in METRICS):
                    raise FloatingPointError(f"Non-finite metric at sample {row['sample_index']}")
                rows.append(row)
            if (batch_index + 1) % 256 == 0:
                print(
                    entry["shortlist_id"],
                    len(rows),
                    "/",
                    manifest["evaluation_protocol"]["expected_samples_per_checkpoint"],
                    flush=True,
                )
    expected = manifest["evaluation_protocol"]["expected_samples_per_checkpoint"]
    if len(rows) != expected:
        raise ValueError(f"Exact sample count failed: {len(rows)} != {expected}")
    grouped = defaultdict(lambda: {metric: [] for metric in METRICS})
    for row in rows:
        for metric in METRICS:
            grouped[row["mask_id"]][metric].append(row[metric])
    if set(grouped) != set(EXPECTED_MASK_IDS) or {
        len(values["PSNR"]) for values in grouped.values()
    } != {manifest["scene_protocol"]["split_counts"]["test"]}:
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
    summary = {
        "status": "complete",
        "shortlist_id": entry["shortlist_id"],
        "sample_count": len(rows),
        "mask_count": len(per_mask),
        "scenes_per_mask": per_mask[0]["sample_count"],
        "mask_balanced": {
            metric: float(np.mean([row[metric] for row in per_mask]))
            for metric in METRICS
        },
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": entry["sha256"],
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
            "selected_using_final_test": False,
        },
    )
    return summary


def execute(
    manifest_path: Path,
    manifest: dict,
    preflight: dict,
    args: argparse.Namespace,
) -> None:
    import torch

    authorization = validate_authorization(
        Path(args.authorization).resolve() if args.authorization else None,
        manifest_path=manifest_path,
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"0", "1", "2", "3", "4", "5"}:
        raise RuntimeError("Use exactly one physical GPU from 0 through 5")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "run_state.json",
        {
            "status": "authorized_before_test_access",
            "manifest_sha256": preflight["manifest_sha256"],
            "authorization": authorization,
        },
    )
    shortlist = json.loads(
        resolve_repo_path(manifest["checkpoint_protocol"]["manifest"]).read_text()
    )
    checkpoints = {
        entry["shortlist_id"]: resolve_checkpoint(entry)
        for entry in shortlist["entries"]
    }
    write_json(
        output / "preflight.json",
        {
            **preflight,
            "metadata_only": False,
            "checkpoint_hashes_reverified_before_test_access": True,
        },
    )
    try:
        loader, masks = build_loader(manifest, resolve_repo_path(args.scene_root))
        write_json(
            output / "run_state.json",
            {
                "status": "test_access_started",
                "test_scene_files_opened": False,
                "test_masks_generated": True,
                "mask_count": len(masks),
            },
        )
        summaries = []
        for entry in shortlist["entries"]:
            model_output = output / entry["shortlist_id"]
            model_output.mkdir(exist_ok=False)
            write_json(model_output / "run_state.json", {"status": "running"})
            loader, _ = build_loader(manifest, resolve_repo_path(args.scene_root))
            summary = evaluate_checkpoint(
                entry,
                checkpoint=checkpoints[entry["shortlist_id"]],
                loader=loader,
                manifest=manifest,
                output=model_output,
            )
            write_json(model_output / "run_state.json", {"status": "complete"})
            summaries.append(summary)
        with (output / "summary.csv").open("w", newline="") as stream:
            fields = ["shortlist_id", "PSNR", "SSIM", "LPIPS", "sample_count", "mask_count"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for summary in summaries:
                writer.writerow(
                    {
                        "shortlist_id": summary["shortlist_id"],
                        **summary["mask_balanced"],
                        "sample_count": summary["sample_count"],
                        "mask_count": summary["mask_count"],
                    }
                )
        write_json(
            output / "validation.json",
            {
                "status": "pass",
                "checkpoint_count": len(summaries),
                "expected_checkpoint_count": 13,
                "all_exact_sample_counts": True,
                "checkpoint_selection_by_test_metrics": False,
            },
        )
        write_json(output / "run_state.json", {"status": "complete"})
    except BaseException as error:
        write_json(
            output / "run_state.json",
            {
                "status": "failed_closed",
                "error_type": type(error).__name__,
                "error": str(error),
                "automatic_retry": False,
                "checkpoint_substitution": False,
                "settings_changed": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="outputs/coursework_pre_final_20260910/final_test_v3/final_test_manifest.json",
    )
    parser.add_argument("--execute-final-test", action="store_true")
    parser.add_argument("--authorization")
    parser.add_argument("--scene-root", default="data/raw/mirflickr25k/extracted")
    parser.add_argument("--output")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest, preflight = load_and_validate_metadata(manifest_path)
    if not args.execute_final_test:
        print(json.dumps(preflight, indent=2), flush=True)
        return
    if not args.output:
        raise ValueError("--output is required for authorized final-test execution")
    execute(manifest_path, manifest, preflight, args)


if __name__ == "__main__":
    main()
