"""Independently audit completion of the supervisor PSF-aware queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_finite(rows: list[dict], metrics: tuple[str, ...]) -> None:
    if not all(
        math.isfinite(float(row[metric])) for row in rows for metric in metrics
    ):
        raise ValueError("A metric CSV contains non-finite values")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    root = Path(manifest["root"])
    bundle = root / manifest["bundle"]
    results = bundle / "results"

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if revision != manifest["source_revision"]:
        raise ValueError("Source revision drift")
    tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
    ).strip()
    if tracked:
        raise ValueError(f"Tracked worktree is dirty: {tracked}")
    for relative, expected in manifest["source_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Source hash drift: {relative}")
    for relative, expected in manifest["program_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Frozen program hash drift: {relative}")
    if sha256(manifest["scene_manifest"]["path"]) != manifest["scene_manifest"][
        "sha256"
    ]:
        raise ValueError("Scene manifest hash drift")

    preflight = load_json(bundle / "preflight.json")
    if preflight["status"] != "pass":
        raise ValueError("Preflight did not pass")
    if preflight["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Preflight/manifest hash mismatch")
    if preflight["optimizer_steps"] != 0 or preflight["final_test_accessed"]:
        raise ValueError("Invalid preflight state")
    launcher = load_json(bundle / "launcher/launcher_state.json")
    if launcher["status"] != "complete":
        raise ValueError("Training launcher is not complete")
    if launcher["completed_job_count"] != 3 or launcher["failed_job_count"] != 0:
        raise ValueError("Training job counts do not prove completion")
    if launcher["final_test_accessed"]:
        raise ValueError("Training launcher reports final-test access")

    endpoint_audits = []
    evaluation_audits = []
    required_evaluation = (
        "per_image.csv",
        "per_mask.csv",
        "prediction_consistency.csv",
        "summary.json",
        "validation.json",
        "qualitative.png",
        "provenance.json",
        "run_state.json",
    )
    for job in manifest["jobs"]:
        seed = job["seed"]
        control = manifest["controls"][str(seed)]
        for field, hash_field in (
            ("config", "config_sha256"),
            ("checkpoint", "checkpoint_sha256"),
            ("metrics_csv", "metrics_sha256"),
        ):
            if sha256(control[field]) != control[hash_field]:
                raise ValueError(f"Control {field} hash drift: seed {seed}")

        run = root / job["output"]
        complete_path = run / "job_complete.json"
        complete = load_json(complete_path)
        if complete["status"] != "complete" or complete["seed"] != seed:
            raise ValueError(f"Incomplete endpoint record: seed {seed}")
        for field, hash_field in (
            ("checkpoint", "checkpoint_sha256"),
            ("config", "config_sha256"),
            ("metrics_csv", "metrics_sha256"),
        ):
            if sha256(complete[field]) != complete[hash_field]:
                raise ValueError(f"Endpoint {field} hash drift: seed {seed}")
        state = torch.load(
            str(complete["checkpoint"]),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if state.get("global_step") != 100000 or state.get("sampler_step") != 100000:
            raise ValueError(f"Endpoint step mismatch: seed {seed}")
        if state.get("lr_scheduler", {}).get("T_max") != 100000:
            raise ValueError(f"Endpoint scheduler mismatch: seed {seed}")
        del state
        endpoint_rows = csv_rows(Path(complete["metrics_csv"]))
        if len(endpoint_rows) != 32 or len(
            {row["mask_id"] for row in endpoint_rows}
        ) != 32:
            raise ValueError(f"Endpoint mask count mismatch: seed {seed}")
        if {int(row["sample_count"]) for row in endpoint_rows} != {32}:
            raise ValueError(f"Endpoint scene count mismatch: seed {seed}")
        require_finite(endpoint_rows, ("PSNR", "SSIM", "LPIPS"))
        endpoint_audits.append(
            {
                "seed": seed,
                "global_step": 100000,
                "sampler_step": 100000,
                "scheduler_T_max": 100000,
                "checkpoint_sha256": complete["checkpoint_sha256"],
                "metrics_sha256": complete["metrics_sha256"],
                "mask_count": 32,
                "scenes_per_mask": 32,
            }
        )

        evaluation = root / job["evaluation_output"]
        for name in required_evaluation:
            path = evaluation / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        validation = load_json(evaluation / "validation.json")
        summary = load_json(evaluation / "summary.json")
        run_state = load_json(evaluation / "run_state.json")
        if validation["status"] != "pass" or run_state["status"] != "complete":
            raise ValueError(f"Evaluation did not pass: seed {seed}")
        if not validation["correct_replay_within_tolerance"]:
            raise ValueError(f"Correct replay parity failed: seed {seed}")
        if any(
            value.get("final_test_accessed") is not False
            for value in (validation, summary, run_state)
        ):
            raise ValueError(f"Evaluation final-test flag drift: seed {seed}")
        per_image = csv_rows(evaluation / "per_image.csv")
        per_mask = csv_rows(evaluation / "per_mask.csv")
        consistency = csv_rows(evaluation / "prediction_consistency.csv")
        if len(per_image) != 2048 or {
            row["condition"] for row in per_image
        } != {"Correct PSF", "Shuffled PSF"}:
            raise ValueError(f"Per-image replay count mismatch: seed {seed}")
        if len(per_mask) != 64:
            raise ValueError(f"Per-mask replay count mismatch: seed {seed}")
        for condition in ("Correct PSF", "Shuffled PSF"):
            rows = [row for row in per_mask if row["condition"] == condition]
            if len(rows) != 32 or {int(row["sample_count"]) for row in rows} != {32}:
                raise ValueError(f"Per-mask balance mismatch: seed {seed}/{condition}")
        if len(consistency) != 1024:
            raise ValueError(f"Prediction consistency count mismatch: seed {seed}")
        require_finite(per_image, ("PSNR", "SSIM", "LPIPS"))
        require_finite(consistency, ("prediction_MAE", "prediction_RMSE"))
        evaluation_audits.append(
            {
                "seed": seed,
                "per_image_rows": 2048,
                "per_mask_rows": 64,
                "prediction_consistency_rows": 1024,
                "correct_replay_max_abs": validation["correct_replay_max_abs"],
                "qualitative_sha256": sha256(evaluation / "qualitative.png"),
                "data_partition": "development",
                "final_test_accessed": False,
            }
        )

    result_validation_path = results / "validation.json"
    result_validation = load_json(result_validation_path)
    if result_validation["status"] != "pass":
        raise ValueError("Aggregate result validation did not pass")
    if result_validation["training_endpoint_count"] != 3:
        raise ValueError("Aggregate endpoint count mismatch")
    if result_validation["evaluation_count"] != 3:
        raise ValueError("Aggregate evaluation count mismatch")
    if result_validation["final_test_accessed"]:
        raise ValueError("Aggregate reports final-test access")
    for record in result_validation["generated_artifacts"]:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Aggregate artifact hash drift: {record['path']}")

    interpretation_path = results / "interpretation_validation.json"
    interpretation = load_json(interpretation_path)
    if interpretation["status"] != "pass" or interpretation[
        "changes_metrics_or_endpoints"
    ]:
        raise ValueError("Corrected interpretation validation failed")
    evidence_files = {
        "paired_effects_aggregate_sha256": results / "paired_effects_aggregate.csv",
        "paired_effects_per_seed_sha256": results / "paired_effects_per_seed.csv",
        "prediction_consistency_per_seed_sha256": results
        / "prediction_consistency_per_seed.csv",
        "first_pass_results_sha256": results / "RESULTS.md",
    }
    for key, path in evidence_files.items():
        if sha256(path) != interpretation["evidence"][key]:
            raise ValueError(f"Interpretation evidence hash drift: {path}")

    audit = {
        "status": "pass",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "source_revision": revision,
        "tracked_worktree_clean": True,
        "preflight_status": "pass",
        "launcher_status": "complete",
        "endpoint_audits": endpoint_audits,
        "evaluation_audits": evaluation_audits,
        "aggregate_validation_sha256": sha256(result_validation_path),
        "interpretation_validation_sha256": sha256(interpretation_path),
        "definition_of_done": {
            "three_100k_endpoints": True,
            "hashes_recorded": True,
            "finite_32x32_metrics": True,
            "correct_replay_matches_training": True,
            "shuffled_replay_complete": True,
            "paired_to_registered_controls": True,
            "report_ready_markdown_and_csv": True,
            "development_partition_recorded": True,
            "interpretation_recorded": True,
            "no_test_selection": True
        },
        "data_partition": "development",
        "final_test_accessed": False,
    }
    save_json(results / "completion_audit.json", audit)
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
