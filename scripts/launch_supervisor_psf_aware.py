"""Launch the three fixed supervisor PSF-aware DRUNet jobs."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_metrics(path: Path) -> None:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 32 or len({row["mask_id"] for row in rows}) != 32:
        raise ValueError(f"Validation mask count mismatch: {path}")
    if {int(row["sample_count"]) for row in rows} != {32}:
        raise ValueError(f"Validation scene count mismatch: {path}")
    values = [
        float(row[metric])
        for row in rows
        for metric in ("PSNR", "SSIM", "LPIPS")
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite endpoint metrics: {path}")


def validate_endpoint(root: Path, job: dict) -> dict:
    output = root / job["output"]
    checkpoint = output / f"checkpoint-epoch{job['epochs']}.pth"
    metrics = output / f"validation_per_mask_epoch{job['epochs']:04d}.csv"
    config = output / "config.yaml"
    if not checkpoint.is_file() or not metrics.is_file() or not config.is_file():
        raise FileNotFoundError(f"Final endpoint artifacts missing: {job['name']}")
    state = torch.load(
        str(checkpoint), map_location="cpu", weights_only=False, mmap=True
    )
    if state.get("global_step") != job["steps"]:
        raise ValueError(f"Global step mismatch: {job['name']}")
    if state.get("sampler_step") != job["steps"]:
        raise ValueError(f"Sampler step mismatch: {job['name']}")
    scheduler = state.get("lr_scheduler")
    if not isinstance(scheduler, dict) or scheduler.get("T_max") != job["steps"]:
        raise ValueError(f"Scheduler horizon mismatch: {job['name']}")
    validate_metrics(metrics)
    del state
    result = {
        "status": "complete",
        "name": job["name"],
        "seed": job["seed"],
        "global_step": job["steps"],
        "sampler_step": job["steps"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "config": str(config),
        "config_sha256": sha256(config),
        "metrics_csv": str(metrics),
        "metrics_sha256": sha256(metrics),
        "mask_count": 32,
        "scenes_per_mask": 32,
        "data_partition": "development",
        "final_test_accessed": False,
    }
    save_json(output / "job_complete.json", result)
    return result


def validate_launch(manifest_path: Path) -> tuple[dict, Path, Path]:
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    bundle = root / manifest["bundle"]
    if manifest["final_test_accessed_by_this_queue"] is not False:
        raise ValueError("Final test is forbidden")
    if manifest["uses_final_test_for_selection"] is not False:
        raise ValueError("Final test cannot select these jobs")
    for relative, expected in manifest["program_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Program hash drift: {relative}")
    preflight_path = bundle / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    if preflight["status"] != "pass":
        raise ValueError("GPU preflight has not passed")
    if preflight["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Preflight belongs to another manifest")
    if preflight["optimizer_steps"] != 0 or preflight["final_test_accessed"]:
        raise ValueError("Invalid preflight execution state")
    for job in manifest["jobs"]:
        if sha256(root / job["config"]) != job["config_sha256"]:
            raise ValueError(f"Config hash drift: {job['name']}")
        if (root / job["output"]).exists():
            raise FileExistsError(f"Training output exists: {root / job['output']}")
    if len({job["seed"] for job in manifest["jobs"]}) != 3:
        raise ValueError("Expected exactly three training seeds")
    if len({job["gpu"] for job in manifest["jobs"]}) != 3:
        raise ValueError("Expected exactly three GPU lanes")
    return manifest, root, bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest, root, bundle = validate_launch(manifest_path)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "jobs": len(manifest["jobs"]),
                    "optimizer_steps": sum(job["steps"] for job in manifest["jobs"]),
                    "final_test_accessed": False,
                },
                indent=2,
            )
        )
        return

    launcher = bundle / "launcher"
    launcher.mkdir(parents=True, exist_ok=False)
    logs = launcher / "logs"
    logs.mkdir()
    hydra = launcher / "hydra"
    hydra.mkdir()
    state = {
        "status": "running",
        "started_unix": time.time(),
        "job_count": len(manifest["jobs"]),
        "completed_job_count": 0,
        "failed_job_count": 0,
        "jobs": {},
        "automatic_retry": False,
        "data_partitions": ["train", "development"],
        "final_test_accessed": False,
    }
    lock = threading.Lock()
    save_json(launcher / "launcher_state.json", state)

    def record(name: str, value: dict) -> None:
        with lock:
            state["jobs"][name] = value
            values = list(state["jobs"].values())
            state["completed_job_count"] = sum(
                item.get("status") == "complete" for item in values
            )
            state["failed_job_count"] = sum(
                item.get("status") == "failed" for item in values
            )
            save_json(launcher / "launcher_state.json", state)

    def run(job: dict) -> dict:
        name = job["name"]
        log_path = logs / f"{name}.log"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        env["MPLCONFIGDIR"] = str(bundle / ".cache/matplotlib")
        env["HF_HOME"] = str(root / "data/huggingface")
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        running = {
            "status": "running",
            "seed": job["seed"],
            "physical_gpu": job["gpu"],
            "started_unix": time.time(),
            "log": str(log_path),
        }
        record(name, running)
        command = [
            sys.executable,
            str(root / "train.py"),
            "--config-path",
            str((root / job["config"]).parent),
            "--config-name",
            Path(job["config"]).stem,
            f"hydra.run.dir={hydra / name}",
            "hydra.job.chdir=false",
        ]
        started = time.monotonic()
        with log_path.open("w") as log:
            result = subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        completed = {
            **running,
            "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "finished_unix": time.time(),
        }
        if result.returncode == 0:
            try:
                completed["endpoint"] = validate_endpoint(root, job)
                completed["status"] = "complete"
            except Exception as error:
                completed["status"] = "failed"
                completed["validation_error"] = repr(error)
        else:
            completed["status"] = "failed"
        record(name, completed)
        return completed

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, job) for job in manifest["jobs"]]
        results = [future.result() for future in futures]
    failures = [result for result in results if result["status"] != "complete"]
    state["status"] = "failed" if failures else "complete"
    state["finished_unix"] = time.time()
    state["final_test_accessed"] = False
    save_json(launcher / "launcher_state.json", state)
    if failures:
        raise SystemExit(1)
    print(json.dumps(state, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
