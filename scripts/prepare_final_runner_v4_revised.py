"""Freeze the user-approved revised three-GPU final-test runner."""

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
        "--output",
        default="outputs/coursework_final_runner_v4_20260910/final_test_v4_revised",
    )
    args = parser.parse_args()
    selection_relative = (
        "outputs/coursework_final_runner_v4_20260910/batch_benchmark/selection/selection.json"
    )
    parent_relative = (
        "outputs/coursework_pre_final_20260910/final_test_v3/final_test_manifest.json"
    )
    shortlist_relative = (
        "outputs/coursework_final_runner_v4_20260910/revised_shortlist_v1/shortlist.json"
    )
    development_relative = (
        "outputs/coursework_final_runner_v4_20260910/revised_development_evidence_v2/summary.csv"
    )
    selection_path = REPO_ROOT / selection_relative
    parent_path = REPO_ROOT / parent_relative
    shortlist_path = REPO_ROOT / shortlist_relative
    development_path = REPO_ROOT / development_relative
    selection = json.loads(selection_path.read_text())
    parent, parent_preflight = load_and_validate_metadata(parent_path)
    shortlist = json.loads(shortlist_path.read_text())
    if selection.get("selected_batch_size") != 1 or shortlist.get("status") != "frozen_revised":
        raise ValueError("Revised runner inputs are not frozen")
    entries = shortlist["entries"]
    shard0 = [
        entry["shortlist_id"]
        for entry in entries
        if entry["analysis_role"]
        in {"supplementary_corrected_seed42", "predeclared_100k_finalist"}
    ]
    shard1 = [
        entry["shortlist_id"]
        for entry in entries
        if entry["analysis_role"] == "primary_matched_matrix" and int(entry["seed"]) == 52
    ]
    shard2 = [
        entry["shortlist_id"]
        for entry in entries
        if entry["analysis_role"] == "primary_matched_matrix" and int(entry["seed"]) == 62
    ]
    shards = [
        {"shard_id": 0, "physical_gpu": 0, "checkpoint_ids": shard0},
        {"shard_id": 1, "physical_gpu": 1, "checkpoint_ids": shard1},
        {"shard_id": 2, "physical_gpu": 2, "checkpoint_ids": shard2},
    ]
    mapped = [identifier for shard in shards for identifier in shard["checkpoint_ids"]]
    expected = [entry["shortlist_id"] for entry in entries]
    if [len(shard["checkpoint_ids"]) for shard in shards] != [3, 4, 4]:
        raise ValueError("Revised shard sizes must be 3/4/4")
    if len(mapped) != len(set(mapped)) or set(mapped) != set(expected):
        raise ValueError("Revised shards are not an exact shortlist partition")
    programs = {
        "worker": REPO_ROOT / "scripts/final_runner_v4_worker.py",
        "launcher": REPO_ROOT / "scripts/final_runner_v4_launcher.py",
        "merger": REPO_ROOT / "scripts/final_runner_v4_merge.py",
    }
    manifest = {
        "schema_version": 2,
        "status": "locked_pre_execution",
        "protocol_version": "final-test-runner-v4-revised",
        "prepared_before_final_test_access": True,
        "user_authorization_text": "начинай проводить тест на тех моделях, что есть",
        "final_test_accessed": False,
        "checkpoint_selection_by_test_metrics_forbidden": True,
        "parent_protocol": {
            "manifest": parent_relative,
            "manifest_sha256": sha256(parent_path),
            "metadata_preflight_status": parent_preflight["status"],
            "final_scene_count": 256,
            "final_mask_count": 100,
            "final_samples_per_checkpoint": 25_600,
        },
        "shortlist_protocol": {
            "manifest": shortlist_relative,
            "manifest_sha256": sha256(shortlist_path),
            "checkpoint_count": len(entries),
            "primary_matched_count": 8,
            "supplementary_count": 2,
            "finalist_count": 1,
            "excluded_legacy_count": 2,
        },
        "batch_protocol": {
            "batch_size": 1,
            "selection": selection_relative,
            "selection_sha256": sha256(selection_path),
            "selection_rule": selection["selection_rule"],
            "development_only": True,
        },
        "execution_protocol": {
            "concurrent_gpu_processes": 3,
            "allowed_physical_gpus": [0, 1, 2, 3, 4, 5],
            "used_physical_gpus": [0, 1, 2],
            "checkpoint_count": len(entries),
            "development_samples_per_checkpoint": 1024,
            "final_samples_per_checkpoint": 25_600,
            "precision": "FP32",
            "automatic_retry": False,
            "checkpoint_substitution": False,
            "metric_based_checkpoint_selection": False,
            "merge_order": expected,
        },
        "shards": shards,
        "program_hashes": {name: sha256(path) for name, path in programs.items()},
        "authorization_gate": {
            "authorization_file_created": False,
            "required_only_for_mode": "final",
            "development_mode_accepts_authorization": False,
            "schema": {
                "authorized_by_user": True,
                "mode": "final",
                "manifest_sha256": "SHA256 of this finalized revised manifest",
                "worker_sha256": sha256(programs["worker"]),
                "launcher_sha256": sha256(programs["launcher"]),
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
            "reference": development_relative,
            "reference_sha256": sha256(development_path),
            "source_validation": "outputs/coursework_final_runner_v4_20260910/revised_development_evidence_v2/validation.json",
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
        "shard_sizes": [len(shard["checkpoint_ids"]) for shard in shards],
        "checkpoint_count": len(entries),
        "exact_revised_shortlist_partition": True,
        "batch_size": 1,
        "authorization_file_created": False,
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "final_test_model_forward_executed": False,
    }
    write_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
