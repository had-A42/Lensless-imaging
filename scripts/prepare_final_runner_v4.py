"""Freeze the three-GPU V4 runner after development-only batch selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_frozen_final_synthetic import (  # noqa: E402
    load_and_validate_metadata,
    sha256,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-selection",
        default="outputs/coursework_final_runner_v4_20260910/batch_benchmark/selection/selection.json",
    )
    parser.add_argument(
        "--parent-manifest",
        default="outputs/coursework_pre_final_20260910/final_test_v3/final_test_manifest.json",
    )
    parser.add_argument(
        "--development-reference",
        default="outputs/coursework_final_runner_v4_20260910/development_reference/reference.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/coursework_final_runner_v4_20260910/final_test_v4",
    )
    args = parser.parse_args()
    selection_path = (REPO_ROOT / args.batch_selection).resolve()
    parent_path = (REPO_ROOT / args.parent_manifest).resolve()
    development_reference_path = (REPO_ROOT / args.development_reference).resolve()
    for path in (selection_path, parent_path, development_reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("status") != "pass"
        or selection.get("test_scene_files_opened") is not False
        or selection.get("test_masks_generated") is not False
        or selection.get("final_test_model_forward_executed") is not False
    ):
        raise ValueError("Batch selection is incomplete or touched final-test data")
    parent, parent_preflight = load_and_validate_metadata(parent_path)
    shortlist_path = (REPO_ROOT / parent["checkpoint_protocol"]["manifest"]).resolve()
    shortlist = json.loads(shortlist_path.read_text())
    entries = shortlist["entries"]
    by_seed = {seed: [] for seed in (42, 52, 62)}
    finalist = None
    for entry in entries:
        if entry["role"] == "predeclared_100k_finalist":
            if finalist is not None:
                raise ValueError("Expected exactly one 100k finalist")
            finalist = entry["shortlist_id"]
        else:
            by_seed[int(entry["seed"])].append(entry["shortlist_id"])
    if finalist is None or any(len(values) != 4 for values in by_seed.values()):
        raise ValueError("Frozen shortlist does not form the expected seed blocks")
    shards = [
        {
            "shard_id": 0,
            "physical_gpu": 0,
            "checkpoint_ids": by_seed[42] + [finalist],
            "mapping_basis": "all retained seed42 core arms plus the predeclared 100k finalist",
        },
        {
            "shard_id": 1,
            "physical_gpu": 1,
            "checkpoint_ids": by_seed[52],
            "mapping_basis": "all retained seed52 core arms",
        },
        {
            "shard_id": 2,
            "physical_gpu": 2,
            "checkpoint_ids": by_seed[62],
            "mapping_basis": "all retained seed62 core arms",
        },
    ]
    mapped = [identifier for shard in shards for identifier in shard["checkpoint_ids"]]
    expected = [entry["shortlist_id"] for entry in entries]
    if len(mapped) != 13 or len(set(mapped)) != 13 or set(mapped) != set(expected):
        raise ValueError("Three-GPU mapping is not an exact shortlist partition")
    program_paths = {
        "worker": REPO_ROOT / "scripts/final_runner_v4_worker.py",
        "launcher": REPO_ROOT / "scripts/final_runner_v4_launcher.py",
        "merger": REPO_ROOT / "scripts/final_runner_v4_merge.py",
    }
    for path in program_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {
        "schema_version": 1,
        "status": "locked_pre_execution",
        "protocol_version": "final-test-runner-v4",
        "prepared_before_final_test_access": True,
        "final_test_accessed": False,
        "checkpoint_selection_by_test_metrics_forbidden": True,
        "parent_protocol": {
            "manifest": args.parent_manifest,
            "manifest_sha256": sha256(parent_path),
            "metadata_preflight_status": parent_preflight["status"],
            "final_scene_count": parent["scene_protocol"]["split_counts"]["test"],
            "final_mask_count": parent["mask_protocol"]["test_mask_count"],
            "final_samples_per_checkpoint": parent["evaluation_protocol"][
                "expected_samples_per_checkpoint"
            ],
        },
        "batch_protocol": {
            "batch_size": int(selection["selected_batch_size"]),
            "selection": args.batch_selection,
            "selection_sha256": sha256(selection_path),
            "selection_rule": selection["selection_rule"],
            "development_only": True,
        },
        "execution_protocol": {
            "concurrent_gpu_processes": 3,
            "allowed_physical_gpus": [0, 1, 2, 3, 4, 5],
            "used_physical_gpus": [0, 1, 2],
            "checkpoint_count": 13,
            "development_samples_per_checkpoint": 1024,
            "final_samples_per_checkpoint": parent["evaluation_protocol"][
                "expected_samples_per_checkpoint"
            ],
            "precision": "FP32",
            "automatic_retry": False,
            "checkpoint_substitution": False,
            "metric_based_checkpoint_selection": False,
            "merge_order": expected,
        },
        "shards": shards,
        "program_hashes": {
            name: sha256(path) for name, path in program_paths.items()
        },
        "authorization_gate": {
            "authorization_file_created": False,
            "required_only_for_mode": "final",
            "development_mode_accepts_authorization": False,
            "schema": {
                "authorized_by_user": True,
                "mode": "final",
                "manifest_sha256": "SHA256 of finalized V4 manifest",
                "worker_sha256": sha256(program_paths["worker"]),
                "launcher_sha256": sha256(program_paths["launcher"]),
            },
        },
        "development_validation_contract": {
            "scene_split": "validation",
            "available_scene_count": 128,
            "scenes_per_mask": 32,
            "mask_partition": "validation",
            "mask_count": 32,
            "expected_samples_per_checkpoint": 1024,
            "psf_source": "read-only historical cache",
            "psf_cache_config_hash": "3a2a4d87833fb3e1a8e9618f3c365255616331ada32f68305272b0bd0f8e82d1",
            "reference": args.development_reference,
            "reference_sha256": sha256(development_reference_path),
            "test_scene_files_opened": False,
            "test_masks_generated": False,
            "final_test_model_forward_executed": False,
        },
        "final_execution_state": {
            "authorization_present": False,
            "test_scene_files_opened": False,
            "test_masks_generated": False,
            "final_test_model_forward_executed": False,
        },
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
    }
    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    validation = {
        "status": "pass",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "shard_count": len(shards),
        "mapped_checkpoint_count": len(mapped),
        "unique_mapped_checkpoint_count": len(set(mapped)),
        "exact_shortlist_partition": set(mapped) == set(expected),
        "batch_selection_status": selection["status"],
        "parent_preflight_status": parent_preflight["status"],
        "authorization_file_created": False,
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "final_test_model_forward_executed": False,
    }
    write_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
