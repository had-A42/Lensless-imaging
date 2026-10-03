"""Prepare the locked final synthetic-test manifest without opening test files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def symbolic_mask_ids(partition: str, count: int) -> list[str]:
    if partition not in {"train", "validation", "test"}:
        raise ValueError("Unknown symbolic mask namespace")
    return [f"{partition}_{index:05d}" for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-manifest", default="manifests/mirflickr25k_splits.json"
    )
    parser.add_argument(
        "--shortlist-manifest",
        default="outputs/coursework_pre_final_20260910/frozen_shortlist_v2/frozen_shortlist.json",
    )
    parser.add_argument(
        "--shortlist-remote-audit",
        default="outputs/coursework_pre_final_20260910/frozen_shortlist_v2/remote_audit/summary.json",
    )
    parser.add_argument(
        "--evaluator", default="scripts/evaluate_frozen_final_synthetic.py"
    )
    parser.add_argument(
        "--output", default="outputs/coursework_pre_final_20260910/final_test_v3"
    )
    args = parser.parse_args()
    split_path = (REPO_ROOT / args.split_manifest).resolve()
    shortlist_path = (REPO_ROOT / args.shortlist_manifest).resolve()
    remote_audit_path = (REPO_ROOT / args.shortlist_remote_audit).resolve()
    evaluator_path = (REPO_ROOT / args.evaluator).resolve()
    for path in (split_path, shortlist_path, remote_audit_path, evaluator_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    shortlist = json.loads(shortlist_path.read_text())
    remote_audit = json.loads(remote_audit_path.read_text())
    if shortlist.get("status") != "frozen" or len(shortlist.get("entries", [])) != 13:
        raise ValueError("Checkpoint shortlist is not frozen")
    if remote_audit.get("status") != "pass" or remote_audit.get("sha256_match_count") != 13:
        raise ValueError("Remote shortlist verification is incomplete")

    split_payload = json.loads(split_path.read_text())
    if set(split_payload) != {"seed", "splits"} or set(split_payload["splits"]) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("Unexpected scene split schema")
    splits = split_payload["splits"]
    expected_counts = {"train": 4096, "validation": 128, "test": 256}
    if {name: len(values) for name, values in splits.items()} != expected_counts:
        raise ValueError("Scene split counts drifted")
    sets = {name: set(values) for name, values in splits.items()}
    if any(len(sets[name]) != expected_counts[name] for name in sets):
        raise ValueError("Duplicate scene ID inside a split")
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets[
        "validation"
    ] & sets["test"]:
        raise ValueError("Scene split overlap detected")

    train_masks = symbolic_mask_ids("train", 10_000)
    development_masks = symbolic_mask_ids("validation", 32)
    test_masks = symbolic_mask_ids("test", 100)
    if set(train_masks) & set(development_masks) or set(train_masks) & set(
        test_masks
    ) or set(development_masks) & set(test_masks):
        raise ValueError("Symbolic mask namespace overlap detected")

    scene_selector_seed = 52
    scene_selector_salt = 59
    order = np.random.default_rng(
        np.random.SeedSequence([scene_selector_seed, scene_selector_salt])
    ).permutation(len(splits["test"]))
    ordered_test_scenes = [splits["test"][int(index)] for index in order]
    qualitative_indices = (0, 255, 256, 12_800, 25_599)
    qualitative_examples = [
        {
            "sample_index": index,
            "mask_id": test_masks[index // len(ordered_test_scenes)],
            "scene_id": ordered_test_scenes[index % len(ordered_test_scenes)],
        }
        for index in qualitative_indices
    ]
    simulator_path = REPO_ROOT / "src/configs/simulator/digicam_article.yaml"
    model_path = REPO_ROOT / "src/configs/model/psf_free_xrestormer.yaml"
    expected_samples = len(splits["test"]) * len(test_masks)
    source_paths = (
        "src/model/psf_free_xrestormer.py",
        "src/datasets/on_the_fly.py",
        "src/datasets/mirflickr.py",
        "src/digicam_synth/mask_protocol.py",
        "src/digicam_synth/pipeline.py",
        "src/metrics/reconstruction.py",
    )
    manifest = {
        "schema_version": 1,
        "status": "locked_pre_execution",
        "prepared_before_final_test_access": True,
        "checkpoint_selection_by_test_metrics_forbidden": True,
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "model_forward_executed": False,
        "scene_protocol": {
            "dataset": "MIRFLICKR-25K",
            "source_split_manifest": args.split_manifest,
            "source_split_manifest_sha256": sha256(split_path),
            "split_seed": int(split_payload["seed"]),
            "split_counts": expected_counts,
            "test_scene_ids": splits["test"],
            "test_scene_order": ordered_test_scenes,
            "train_test_overlap": 0,
            "development_test_overlap": 0,
            "file_content_read_during_freeze": False,
        },
        "mask_protocol": {
            "base_seed": 42,
            "train_mask_ids": train_masks,
            "development_mask_ids": development_masks,
            "test_mask_ids": test_masks,
            "test_mask_count": len(test_masks),
            "train_test_overlap": 0,
            "development_test_overlap": 0,
            "mask_values_generated_during_freeze": False,
        },
        "checkpoint_protocol": {
            "manifest": args.shortlist_manifest,
            "manifest_sha256": sha256(shortlist_path),
            "remote_audit_summary": args.shortlist_remote_audit,
            "checkpoint_count": len(shortlist["entries"]),
            "all_hashes_and_endpoints_verified": True,
            "selection_locked_before_test": True,
        },
        "evaluation_protocol": {
            "expected_samples_per_checkpoint": expected_samples,
            "expected_total_model_forwards": expected_samples * len(shortlist["entries"]),
            "measurement_shape": [3, 380, 507],
            "target_shape": [3, 200, 266],
            "roi": [80, 100, 200, 266],
            "measurement_orientation": "synthetic simulator native orientation; no post-hoc rotation",
            "normalization": "independent per-image maximum for prediction and target",
            "metrics": [
                {"name": "PSNR", "normalize_by_max": True},
                {"name": "SSIM", "normalize_by_max": True},
                {"name": "LPIPS", "net": "vgg", "normalize_by_max": True},
            ],
            "aggregation": "image -> equal mean within each mask -> equal mean across masks; training runs remain separate",
            "evaluation_precision": "FP32",
            "batch_size": 1,
            "scene_selector_seed": scene_selector_seed,
            "scene_selector_salt": scene_selector_salt,
            "simulator_config": "src/configs/simulator/digicam_article.yaml",
            "simulator_config_sha256": sha256(simulator_path),
            "model_config": "src/configs/model/psf_free_xrestormer.yaml",
            "model_config_sha256": sha256(model_path),
            "qualitative_examples": qualitative_examples,
        },
        "failure_policy": {
            "non_finite_metric": "abort immediately and retain partial artifacts",
            "oom": "abort immediately; do not change batch size automatically",
            "checkpoint_failure": "abort immediately; do not substitute another checkpoint",
            "automatic_retry": False,
            "post_test_model_selection": False,
            "completed_model_overwrite": False,
        },
        "execution_gate": {
            "default_mode": "metadata-only preflight",
            "required_flag": "--execute-final-test",
            "authorization_file_prepared": False,
            "authorization_schema": {
                "authorized_by_user": True,
                "manifest_sha256": "SHA256 of this finalized manifest",
                "evaluator_sha256": sha256(evaluator_path),
            },
        },
        "evaluator": args.evaluator,
        "evaluator_sha256": sha256(evaluator_path),
        "source_hashes": {
            relative_path: sha256(REPO_ROOT / relative_path)
            for relative_path in source_paths
        },
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
    }
    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "final_test_manifest.json"
    write_json(manifest_path, manifest)
    validation = {
        "status": "pass",
        "metadata_only": True,
        "scene_split_counts": expected_counts,
        "scene_overlap_count": 0,
        "mask_counts": {"train": 10_000, "development": 32, "test": 100},
        "mask_overlap_count": 0,
        "expected_samples_per_checkpoint": expected_samples,
        "checkpoint_count": len(shortlist["entries"]),
        "shortlist_remote_verified": True,
        "evaluator_sha256": sha256(evaluator_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "authorization_file_created": False,
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "model_forward_executed": False,
    }
    write_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
