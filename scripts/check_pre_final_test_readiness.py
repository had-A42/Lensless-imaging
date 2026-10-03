"""Validate every artifact required before the first final synthetic-test access."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_frozen_final_synthetic import load_and_validate_metadata  # noqa: E402


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


def load_json(relative: str) -> tuple[dict, Path]:
    path = (REPO_ROOT / relative).resolve()
    if not path.is_file():
        return {}, path
    return json.loads(path.read_text()), path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="outputs/coursework_pre_final_20260910/pre_final_test_readiness.json",
    )
    args = parser.parse_args()
    checks = []
    artifacts = []

    def record(name: str, passed: bool, evidence: str, path: Path | None = None) -> None:
        checks.append({"name": name, "status": "pass" if passed else "fail", "evidence": evidence})
        if path is not None and path.is_file():
            artifacts.append({"path": str(path), "sha256": sha256(path)})

    remote_inventory, remote_inventory_path = load_json(
        "outputs/coursework_status_20260909/remote_checkpoint_audit_summary.json"
    )
    record(
        "remote_audit_33_checkpoints",
        remote_inventory.get("status") == "complete"
        and remote_inventory.get("requested_paths") == 33
        and remote_inventory.get("existence", {}).get("true") == 33
        and remote_inventory.get("endpoint_matches_declared_steps") == 33
        and remote_inventory.get("checkpoint_deserialization") is False,
        "33/33 existence, endpoint metadata and remote SHA256 required without deserialization",
        remote_inventory_path,
    )

    controls, controls_path = load_json(
        "outputs/coursework_status_20260909/input_controls_all_20260910/status.json"
    )
    completed_names = {row.get("run") for row in controls.get("validations", [])}
    expected_drunet = {f"E2-celeba-drunet-10k-seed{seed}" for seed in (42, 52, 62)}
    record(
        "development_input_use_controls",
        controls.get("complete") is True
        and controls.get("expected_runs") == 7
        and controls.get("completed_runs") == 7
        and expected_drunet <= completed_names
        and controls.get("final_test_accessed") is False,
        "7/7 controls, including exact CelebA DRUNet checkpoints for seeds42/52/62",
        controls_path,
    )
    drunet_hashes = {
        42: "29ed84f7663ad31508dc22d0f6f3df490a07cd39fae2a41c44240c8c5b09fd32",
        52: "cb257ef07dd36f7465a6b5afeecaa2305f538b09f516b6cd05aed371c5c17b55",
        62: "616bcdb18a19535100769ff2755ff8caab409e61fa68d879f80c7c109c185844",
    }
    drunet_provenance_ok = True
    for seed, expected_hash in drunet_hashes.items():
        provenance, provenance_path = load_json(
            f"outputs/coursework_completion_20260907/evaluations/E2-celeba-drunet-10k-seed{seed}/provenance.json"
        )
        drunet_provenance_ok &= provenance.get("checkpoint_sha256") == expected_hash
        if provenance_path.is_file():
            artifacts.append({"path": str(provenance_path), "sha256": sha256(provenance_path)})
    record(
        "drunet_exact_checkpoint_hashes",
        drunet_provenance_ok,
        "Three published run artifacts must retain the hashes verified in remote inventory",
    )

    psf_aware_ok = True
    for view, expected_rows, expected_masks, expected_per_mask in (
        ("inner68", 1700, 68, 25),
        ("outer17", 4250, 17, 250),
    ):
        root = f"outputs/coursework_pre_final_20260910/psf_aware/{view}"
        summary, summary_path = load_json(f"{root}/summary.json")
        validation, validation_path = load_json(f"{root}/validation.json")
        provenance, provenance_path = load_json(f"{root}/provenance.json")
        current = (
            summary.get("status") == "complete"
            and summary.get("sample_count") == expected_rows
            and summary.get("mask_count") == expected_masks
            and summary.get("rows_per_mask") == expected_per_mask
            and validation.get("complete") is True
            and validation.get("row_identity_match") is True
            and validation.get("metrics_finite") is True
            and validation.get("model_checkpoint_sha256")
            == "51ca006404ae6cf54fb82f99c54afa06fbb6a55f09dc929f3882f1a4303f2818"
            and provenance.get("comparison_status")
            == "contextual privileged reference, not a matched causal comparison"
            and provenance.get("crop") == [80, 100, 200, 266]
            and provenance.get("measurement_rotation_degrees") == 180
            and provenance.get("normalization")
            == "independent per-image maximum for prediction and target"
            and summary.get("official_real_test_accessed") is False
            and summary.get("final_synthetic_test_accessed") is False
        )
        psf_aware_ok &= current
        for path in (summary_path, validation_path, provenance_path):
            if path.is_file():
                artifacts.append({"path": str(path), "sha256": sha256(path)})
    comparison, comparison_path = load_json(
        "outputs/coursework_pre_final_20260910/psf_aware/comparison_v2/validation.json"
    )
    psf_aware_ok &= (
        comparison.get("status") == "pass"
        and comparison.get("same_crop_orientation_normalization_metrics_aggregation")
        is True
        and comparison.get("comparison_status")
        == "row-matched contextual; not a matched causal comparison"
        and comparison.get("official_real_test_accessed") is False
        and comparison.get("final_synthetic_test_accessed") is False
    )
    if comparison_path.is_file():
        artifacts.append({"path": str(comparison_path), "sha256": sha256(comparison_path)})
    record(
        "psf_aware_development_reference",
        psf_aware_ok,
        "inner68 and outer17 must be complete on exact PSF-free rows with contextual labeling",
    )

    diversity, diversity_path = load_json(
        "outputs/coursework_pre_final_20260910/psf_diversity/analysis/validation.json"
    )
    diversity_results = (
        REPO_ROOT
        / "outputs/coursework_pre_final_20260910/psf_diversity/analysis/RESULTS.md"
    ).resolve()
    record(
        "exploratory_psf_diversity",
        diversity.get("status") == "pass"
        and diversity.get("analysis_status") == "post-hoc exploratory"
        and diversity.get("train_seeds") == [42, 52, 62]
        and diversity.get("bank_counts") == [100, 1000, 10000]
        and diversity.get("actual_summary_rows") == 9
        and diversity.get("actual_nearest_rows") == 288
        and diversity.get("test_partition_accessed") is False
        and diversity_results.is_file(),
        "Validated Fourier, pairwise, nearest-development, PCA/effective-rank and collision artifacts",
        diversity_path,
    )
    if diversity_results.is_file():
        artifacts.append({"path": str(diversity_results), "sha256": sha256(diversity_results)})

    shortlist, shortlist_path = load_json(
        "outputs/coursework_pre_final_20260910/frozen_shortlist_v2/validation.json"
    )
    shortlist_remote, shortlist_remote_path = load_json(
        "outputs/coursework_pre_final_20260910/frozen_shortlist_v2/remote_audit/summary.json"
    )
    shortlist_integrity, shortlist_integrity_path = load_json(
        "outputs/coursework_final_runner_v4_20260910/shortlist_integrity_audit/audit.json"
    )
    post_freeze_integrity_pass = (
        not shortlist_integrity_path.is_file()
        or shortlist_integrity.get("status") == "pass"
    )
    record(
        "frozen_checkpoint_shortlist",
        shortlist.get("status") == "pass"
        and shortlist.get("entry_count") == 13
        and shortlist.get("core_entry_count") == 12
        and shortlist.get("finalist_entry_count") == 1
        and shortlist_remote.get("status") == "pass"
        and shortlist_remote.get("exists_count") == 13
        and shortlist_remote.get("endpoint_match_count") == 13
        and shortlist_remote.get("sha256_match_count") == 13
        and shortlist_remote.get("checkpoint_files_copied") == 0
        and shortlist_remote.get("final_synthetic_test_accessed") is False
        and post_freeze_integrity_pass,
        "12-run 50k matrix plus one predeclared 100k finalist, including post-freeze development parity",
        shortlist_path,
    )
    if shortlist_integrity_path.is_file():
        artifacts.append(
            {"path": str(shortlist_integrity_path), "sha256": sha256(shortlist_integrity_path)}
        )
    if shortlist_remote_path.is_file():
        artifacts.append({"path": str(shortlist_remote_path), "sha256": sha256(shortlist_remote_path)})

    final_validation, final_validation_path = load_json(
        "outputs/coursework_pre_final_20260910/final_test_v3/validation.json"
    )
    final_manifest_path = (
        REPO_ROOT
        / "outputs/coursework_pre_final_20260910/final_test_v3/final_test_manifest.json"
    ).resolve()
    try:
        _, final_preflight = load_and_validate_metadata(final_manifest_path)
    except Exception as error:
        final_preflight = {"status": "fail", "error": str(error)}
    authorization_path = final_manifest_path.parent / "authorization.json"
    remote_preflight, remote_preflight_path = load_json(
        "outputs/coursework_pre_final_20260910/final_test_v3/remote_metadata_preflight.json"
    )
    execution_outputs = [
        path
        for path in final_manifest_path.parent.iterdir()
        if path.is_dir()
    ] if final_manifest_path.parent.is_dir() else []
    record(
        "final_test_prepared_but_unopened",
        final_validation.get("status") == "pass"
        and final_preflight.get("status") == "pass"
        and final_preflight.get("expected_samples_per_checkpoint") == 25_600
        and final_preflight.get("checkpoint_count") == 13
        and final_preflight.get("checkpoint_selection_by_test_metrics_forbidden") is True
        and final_preflight.get("test_scene_files_opened") is False
        and final_preflight.get("test_masks_generated") is False
        and final_preflight.get("model_forward_executed") is False
        and remote_preflight.get("status") == "pass"
        and remote_preflight.get("manifest_sha256") == final_preflight.get("manifest_sha256")
        and remote_preflight.get("evaluator_sha256") == final_preflight.get("evaluator_sha256")
        and remote_preflight.get("test_scene_files_opened") is False
        and remote_preflight.get("test_masks_generated") is False
        and remote_preflight.get("model_forward_executed") is False
        and not authorization_path.exists()
        and not execution_outputs,
        "Locked 256x100 manifest, evaluator hash, no authorization and no execution directory",
        final_validation_path,
    )
    if final_manifest_path.is_file():
        artifacts.append({"path": str(final_manifest_path), "sha256": sha256(final_manifest_path)})
    if remote_preflight_path.is_file():
        artifacts.append({"path": str(remote_preflight_path), "sha256": sha256(remote_preflight_path)})

    tests, tests_path = load_json(
        "outputs/coursework_pre_final_20260910/tests/test_summary.json"
    )
    record(
        "relevant_unit_and_config_tests",
        tests.get("status") == "pass"
        and tests.get("failed") == 0
        and tests.get("passed", 0) >= 21,
        "Saved commands and passing results for pre-final protocol and related contracts",
        tests_path,
    )

    unique_artifacts = {row["path"]: row for row in artifacts}
    output_value = {
        "schema_version": 1,
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "purpose": "safe readiness gate before any final synthetic-test scene or mask access",
        "checks": checks,
        "passed_checks": sum(row["status"] == "pass" for row in checks),
        "total_checks": len(checks),
        "artifacts": [unique_artifacts[key] for key in sorted(unique_artifacts)],
        "official_real_test_accessed": False,
        "final_test_scene_files_opened": False,
        "final_test_masks_generated": False,
        "final_test_model_forward_executed": False,
        "authorization_file_present": authorization_path.exists(),
    }
    output = (REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, output_value)
    print(json.dumps(output_value, indent=2), flush=True)
    if output_value["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
