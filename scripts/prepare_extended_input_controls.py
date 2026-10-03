"""Freeze the post-final, development-only four-GPU input-control queue."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/home/hadhad/project/Lensless-imaging-celeba-20260905")
BUNDLE_REL = Path("outputs/coursework_post_final_20260911/extended_input_controls_v1")
BUNDLE = REPO / BUNDLE_REL
REMOTE_BUNDLE = REMOTE_ROOT / BUNDLE_REL
EVALUATOR_REL = Path("scripts/evaluate_extended_input_controls.py")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def remote(path: str | Path) -> str:
    return str(REMOTE_ROOT / path)


def add_common(job: dict, baseline_local: Path, config_sha: str, checkpoint_sha: str) -> dict:
    return {
        **job,
        "root": str(REMOTE_ROOT),
        "output": str(REMOTE_BUNDLE / "development_runs" / job["name"]),
        "expected_samples": 1024,
        "expected_masks": 32,
        "expected_scenes_per_mask": 32,
        "expected_checkpoint_sha256": checkpoint_sha,
        "expected_config_sha256": config_sha,
        "expected_baseline_sha256": sha256(baseline_local),
        "data_partition": "development",
        "final_test_accessed": False,
        "selected_using_final_test": False,
    }


def mir_job(identifier: str, initialization: str, seed: int, shard: int, checkpoint: str, checkpoint_sha: str) -> dict:
    run_name = f"cv1-xrest-{initialization}-finite-100-50000step-seed{seed}-508a878-r2"
    config_local = REPO / "saved" / run_name / "config.yaml"
    baseline_rel = Path(
        f"outputs/coursework_final_runner_v4_20260910/development_run/shard{shard}/{identifier}/per_image.csv"
    )
    return add_common(
        {
            "name": f"extended-{identifier}",
            "config": remote(Path("saved") / run_name / "config.yaml").replace(
                "Lensless-imaging-celeba-20260905/saved",
                "Lensless-imaging-research-508a878/saved",
            ),
            "checkpoint": remote(Path("saved") / run_name / checkpoint).replace(
                "Lensless-imaging-celeba-20260905/saved",
                "Lensless-imaging-research-508a878/saved",
            ),
            "baseline_csv": remote(baseline_rel),
            "dataset": "mirflickr",
            "steps": 50000,
            "seed": seed,
            "evaluation_precision": "fp32",
            "selection_basis": "pre-frozen finite100 matched matrix; selected by design before final-test inspection",
        },
        REPO / baseline_rel,
        sha256(config_local),
        checkpoint_sha,
    )


def existing_job(source_name: str, new_name: str, selection_basis: str) -> dict:
    source = REPO / "outputs/coursework_completion_20260907/evaluations" / source_name
    job = json.loads((source / "job.json").read_text())
    provenance = json.loads((source / "provenance.json").read_text())
    return add_common(
        {
            "name": new_name,
            "config": job["config"],
            "checkpoint": job["checkpoint"],
            "baseline_csv": str(REMOTE_ROOT / source.relative_to(REPO) / "per_image.csv"),
            "dataset": job["dataset"],
            "steps": job["steps"],
            "seed": job["seed"],
            "evaluation_precision": job.get("evaluation_precision", "fp32"),
            "selection_basis": selection_basis,
        },
        source / "per_image.csv",
        provenance["source_config_sha256"],
        provenance["checkpoint_sha256"],
    )


def main() -> None:
    if BUNDLE.exists():
        raise FileExistsError(BUNDLE)
    (BUNDLE / "jobs").mkdir(parents=True)
    jobs = [
        existing_job(
            "E2-mirflickr-xrest100k-seed42",
            "extended-mirflickr-xrest100k-gopro-seed42",
            "predeclared 100k finalist fixed before final-test access",
        ),
        mir_job(
            "xrest50k-m100-scratch-seed52",
            "scratch",
            52,
            1,
            "model_best.pth",
            "23b4e118bf0f24273429f809de2b0113d6b7882f7c3497a4c379f03fd7684db6",
        ),
        mir_job(
            "xrest50k-m100-gopro-seed52",
            "gopro",
            52,
            1,
            "model_best.pth",
            "103428c336286ccb0893b7b5bb5e8104a0d5cabaf1c65fa648dc56a19b950441",
        ),
        mir_job(
            "xrest50k-m100-scratch-seed62",
            "scratch",
            62,
            2,
            "model_best.pth",
            "56c039d302b7b9281203c638158327ed81f97b28b5b618446dd7a977bdacc6aa",
        ),
        mir_job(
            "xrest50k-m100-gopro-seed62",
            "gopro",
            62,
            2,
            "checkpoint-epoch5.pth",
            "c53a4b5b46032110638eede1776da980d7181074a682e525fb6c5639e6beef6a",
        ),
    ]
    for architecture, prefix in (("xrest", "E2-celeba-10k"), ("drunet", "E2-celeba-drunet-10k")):
        for seed in (42, 52, 62):
            jobs.append(
                existing_job(
                    f"{prefix}-seed{seed}",
                    f"extended-celeba-{architecture}-seed{seed}",
                    f"canonical CelebA {architecture} development control; retained seed set fixed before final test",
                )
            )
    for job in jobs:
        relative = BUNDLE_REL / "jobs" / f"{job['name']}.json"
        save_json(REPO / relative, job)
        job["job_file"] = str(relative)
    lanes = [
        {
            "lane_id": 0,
            "physical_gpu": 0,
            "jobs": [
                "extended-mirflickr-xrest100k-gopro-seed42",
                "extended-xrest50k-m100-scratch-seed52",
                "extended-xrest50k-m100-gopro-seed52",
            ],
        },
        {
            "lane_id": 1,
            "physical_gpu": 1,
            "jobs": [
                "extended-xrest50k-m100-scratch-seed62",
                "extended-xrest50k-m100-gopro-seed62",
            ],
        },
        {
            "lane_id": 2,
            "physical_gpu": 2,
            "jobs": [f"extended-celeba-xrest-seed{seed}" for seed in (42, 52, 62)],
        },
        {
            "lane_id": 3,
            "physical_gpu": 3,
            "jobs": [f"extended-celeba-drunet-seed{seed}" for seed in (42, 52, 62)],
        },
    ]
    manifest = {
        "schema_version": 1,
        "status": "frozen_pre_execution",
        "scientific_question": "Does reconstruction depend on nonzero measurements, and how stable is it across unseen masks for the same scene?",
        "root": str(REMOTE_ROOT),
        "evaluator": str(EVALUATOR_REL),
        "evaluator_sha256": sha256(REPO / EVALUATOR_REL),
        "data_partition": "development",
        "final_test_accessed": False,
        "checkpoint_selection_uses_final_results": False,
        "new_training": False,
        "run_output": str(BUNDLE_REL / "launcher"),
        "jobs": [
            {"name": job["name"], "job_file": job["job_file"]} for job in jobs
        ],
        "lanes": lanes,
        "stopping_rule": "run each frozen job once; no automatic retries and no checkpoint substitution",
        "interpretation_note": "same-scene/other-mask aggregate target metrics are permutation-invariant on this balanced grid; report paired absolute changes and prediction consistency",
        "preflight_history": [
            {
                "attempt": 1,
                "status": "failed_before_data_access",
                "reason": "an over-broad substring guard mistook coursework_final_runner/development_run for a final-run baseline",
                "model_forward_executed": False,
            }
        ],
    }
    save_json(BUNDLE / "manifest.json", manifest)
    print(json.dumps({"status": "prepared", "jobs": len(jobs), "lanes": len(lanes)}, indent=2))


if __name__ == "__main__":
    main()
