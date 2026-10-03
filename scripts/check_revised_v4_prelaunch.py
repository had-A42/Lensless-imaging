"""Fail-closed gate immediately before the authorized revised final run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_frozen_final_synthetic import sha256, write_json  # noqa: E402
from scripts.final_runner_v4_worker import (  # noqa: E402
    load_v4_metadata,
    validate_final_authorization,
)


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="outputs/coursework_final_runner_v4_20260910"
    )
    args = parser.parse_args()
    root = (REPO_ROOT / args.root).resolve()
    manifest_path = root / "final_test_v4_revised/manifest.json"
    authorization_path = root / "final_test_v4_revised/authorization.json"
    manifest, _, entries = load_v4_metadata(manifest_path)
    authorization = validate_final_authorization(authorization_path, manifest_path)
    shortlist = read(root / "revised_shortlist_v1/validation.json")
    development = read(root / "revised_development_evidence_v2/validation.json")
    remote = read(root / "final_test_v4_revised/remote_metadata_preflight.json")
    tests = read(root / "tests/test_summary.json")
    checks = {
        "manifest_metadata": len(entries) == 11
        and manifest["shortlist_protocol"]["primary_matched_count"] == 8
        and manifest["shortlist_protocol"]["supplementary_count"] == 2
        and manifest["shortlist_protocol"]["finalist_count"] == 1,
        "exact_three_shard_partition": [
            len(shard["checkpoint_ids"]) for shard in manifest["shards"]
        ]
        == [3, 4, 4],
        "revised_shortlist": shortlist.get("status") == "pass"
        and shortlist.get("entry_count") == 11,
        "development_evidence": development.get("status") == "pass"
        and development.get("checkpoint_count") == 11
        and development.get("all_model_validations_pass") is True
        and development.get("final_test_model_forward_executed") is False,
        "remote_metadata_preflight": remote.get("status") == "pass"
        and remote.get("checkpoint_count") == 11
        and remote.get("test_scene_files_opened") is False
        and remote.get("test_masks_generated") is False
        and remote.get("final_test_model_forward_executed") is False,
        "contract_tests": tests.get("status") == "pass" and tests.get("failed") == 0,
        "authorization": authorization["manifest_sha256"] == sha256(manifest_path),
        "no_existing_final_output": not (root / "final_run_revised").exists(),
        "post_test_selection_forbidden": manifest[
            "checkpoint_selection_by_test_metrics_forbidden"
        ]
        is True,
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "authorization": str(authorization_path),
        "authorization_sha256": sha256(authorization_path),
        "checkpoint_count": len(entries),
        "shard_sizes": [len(shard["checkpoint_ids"]) for shard in manifest["shards"]],
        "physical_gpus": [shard["physical_gpu"] for shard in manifest["shards"]],
        "test_scene_files_opened_before_launch": False,
        "test_masks_generated_before_launch": False,
        "final_test_model_forward_executed_before_launch": False,
    }
    output = root / "final_test_v4_revised/prelaunch_readiness.json"
    write_json(output, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
