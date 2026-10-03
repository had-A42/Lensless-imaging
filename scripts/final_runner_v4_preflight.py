"""Run metadata-only checks for the frozen V4 three-GPU runner."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest, _, entries = load_v4_metadata(manifest_path)
    mapped = [
        identifier for shard in manifest["shards"] for identifier in shard["checkpoint_ids"]
    ]
    authorization = manifest_path.parent / "authorization.json"
    result = {
        "status": "pass",
        "metadata_only": True,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "batch_size": manifest["batch_protocol"]["batch_size"],
        "shard_count": len(manifest["shards"]),
        "checkpoint_count": len(entries),
        "mapped_checkpoint_count": len(mapped),
        "unique_mapped_checkpoint_count": len(set(mapped)),
        "physical_gpus": [int(shard["physical_gpu"]) for shard in manifest["shards"]],
        "program_hashes": manifest["program_hashes"],
        "authorization_file_present": authorization.exists(),
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "final_test_model_forward_executed": False,
    }
    if (
        authorization.exists()
        or len(entries) != manifest["execution_protocol"]["checkpoint_count"]
        or len(set(mapped)) != len(entries)
    ):
        result["status"] = "fail"
    if args.output:
        write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
