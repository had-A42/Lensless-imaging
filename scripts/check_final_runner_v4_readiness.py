"""Validate the three-GPU V4 runner after development-only execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_frozen_final_synthetic import sha256, write_json  # noqa: E402
from scripts.final_runner_v4_worker import load_v4_metadata  # noqa: E402


def load(relative: str) -> tuple[dict, Path]:
    path = (REPO_ROOT / relative).resolve()
    return (json.loads(path.read_text()) if path.is_file() else {}), path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="outputs/coursework_final_runner_v4_20260910/final_runner_v4_readiness.json",
    )
    args = parser.parse_args()
    checks = []
    artifacts = []

    def record(name: str, passed: bool, evidence: str, path: Path | None = None) -> None:
        checks.append({"name": name, "status": "pass" if passed else "fail", "evidence": evidence})
        if path is not None and path.is_file():
            artifacts.append({"path": str(path), "sha256": sha256(path)})

    base, base_path = load(
        "outputs/coursework_pre_final_20260910/pre_final_test_readiness.json"
    )
    record(
        "base_development_evidence_except_shortlist",
        base.get("status") == "fail"
        and all(
            row.get("status") == "pass"
            for row in base.get("checks", [])
            if row.get("name") != "frozen_checkpoint_shortlist"
        )
        and next(
            row.get("status")
            for row in base.get("checks", [])
            if row.get("name") == "frozen_checkpoint_shortlist"
        )
        == "fail"
        and base.get("final_test_model_forward_executed") is False,
        "All prior evidence remains valid; only the newly exposed shortlist mapping is invalidated",
        base_path,
    )
    selection, selection_path = load(
        "outputs/coursework_final_runner_v4_20260910/batch_benchmark/selection/selection.json"
    )
    record(
        "development_batch_selection",
        selection.get("status") == "pass"
        and selection.get("selected_batch_size") in {1, 2, 4, 8}
        and selection.get("test_scene_files_opened") is False
        and selection.get("test_masks_generated") is False
        and selection.get("final_test_model_forward_executed") is False,
        "Largest batch passing frozen per-image, aggregate and VRAM limits on development",
        selection_path,
    )
    manifest_validation, manifest_validation_path = load(
        "outputs/coursework_final_runner_v4_20260910/final_test_v4/validation.json"
    )
    manifest_path = (
        REPO_ROOT
        / "outputs/coursework_final_runner_v4_20260910/final_test_v4/manifest.json"
    ).resolve()
    try:
        manifest, _, entries = load_v4_metadata(manifest_path)
        manifest_metadata_pass = True
    except Exception as error:
        manifest = {}
        entries = []
        manifest_metadata_pass = False
        manifest_error = str(error)
    record(
        "v4_manifest_and_three_shards",
        manifest_metadata_pass
        and manifest_validation.get("status") == "pass"
        and manifest_validation.get("shard_count") == 3
        and manifest_validation.get("mapped_checkpoint_count") == 13
        and manifest_validation.get("unique_mapped_checkpoint_count") == 13
        and manifest_validation.get("exact_shortlist_partition") is True
        and manifest.get("execution_protocol", {}).get("concurrent_gpu_processes") == 3
        and manifest.get("execution_protocol", {}).get("used_physical_gpus") == [0, 1, 2]
        and manifest.get("final_test_accessed") is False,
        "Exact 5/4/4 checkpoint partition on physical GPU0/1/2; metadata error: "
        + ("none" if manifest_metadata_pass else manifest_error),
        manifest_validation_path,
    )
    if manifest_path.is_file():
        artifacts.append({"path": str(manifest_path), "sha256": sha256(manifest_path)})

    launcher, launcher_path = load(
        "outputs/coursework_final_runner_v4_20260910/development_run/launcher_state.json"
    )
    merged, merged_path = load(
        "outputs/coursework_final_runner_v4_20260910/development_run/merged/validation.json"
    )
    record(
        "three_gpu_development_worker_execution",
        launcher.get("status") == "failed_closed_at_merge"
        and launcher.get("mode") == "development"
        and len(launcher.get("results", [])) == 3
        and all(row.get("returncode") == 0 for row in launcher.get("results", []))
        and launcher.get("test_accessed") is False
        and all(
            load(
                f"outputs/coursework_final_runner_v4_20260910/development_run/shard{shard}/validation.json"
            )[0].get("status")
            == "pass"
            for shard in (0, 1, 2)
        ),
        "All three workers returned zero after 13 complete 32-mask x 32-scene evaluations",
        launcher_path,
    )
    integrity, integrity_path = load(
        "outputs/coursework_final_runner_v4_20260910/shortlist_integrity_audit/audit.json"
    )
    record(
        "development_reference_parity_and_shortlist_integrity",
        merged.get("status") == "pass"
        and merged.get("development_reference_parity") is True
        and integrity.get("status") == "pass",
        "Merge must pass all 13 references; current blocker is two missing corrected seed42 checkpoints",
        integrity_path,
    )

    remote, remote_path = load(
        "outputs/coursework_final_runner_v4_20260910/final_test_v4/remote_metadata_preflight.json"
    )
    record(
        "source_compatible_remote_preflight",
        remote.get("status") == "pass"
        and remote.get("metadata_only") is True
        and remote.get("shard_count") == 3
        and remote.get("checkpoint_count") == 13
        and remote.get("test_scene_files_opened") is False
        and remote.get("test_masks_generated") is False
        and remote.get("final_test_model_forward_executed") is False,
        "V4 programs, source dependencies, batch selection and shard mapping match on A800",
        remote_path,
    )
    tests, tests_path = load(
        "outputs/coursework_final_runner_v4_20260910/tests/test_summary.json"
    )
    record(
        "v4_unit_and_contract_tests",
        tests.get("status") == "pass" and tests.get("failed") == 0,
        "Saved V4 metadata, authorization, shard and merge tests",
        tests_path,
    )
    authorization = manifest_path.parent / "authorization.json"
    final_run = REPO_ROOT / "outputs/coursework_final_runner_v4_20260910/final_run"
    record(
        "final_test_still_closed",
        not authorization.exists()
        and not final_run.exists()
        and manifest.get("authorization_gate", {}).get("authorization_file_created") is False
        and manifest.get("final_execution_state", {}).get("test_scene_files_opened") is False
        and manifest.get("final_execution_state", {}).get("test_masks_generated") is False
        and manifest.get("final_execution_state", {}).get(
            "final_test_model_forward_executed"
        )
        is False,
        "No V4 authorization or final-run output exists",
    )
    unique_artifacts = {row["path"]: row for row in artifacts}
    all_passed = all(row["status"] == "pass" for row in checks)
    blocked = integrity.get("status") == "blocked_missing_exact_corrected_checkpoints"
    result = {
        "schema_version": 1,
        "status": "pass" if all_passed else ("blocked" if blocked else "fail"),
        "purpose": "three-GPU V4 readiness after development-only execution",
        "checks": checks,
        "passed_checks": sum(row["status"] == "pass" for row in checks),
        "total_checks": len(checks),
        "artifacts": [unique_artifacts[key] for key in sorted(unique_artifacts)],
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "final_test_model_forward_executed": False,
        "authorization_file_present": authorization.exists(),
        "blocking_condition": (
            "two exact corrected seed42 finite100 checkpoints are missing"
            if blocked
            else None
        ),
    }
    output = (REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
