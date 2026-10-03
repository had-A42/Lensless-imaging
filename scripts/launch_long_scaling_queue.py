"""Launch the frozen long-training manifest in four sequential GPU lanes."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
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


def validate_endpoint(root: Path, job: dict) -> dict:
    output = root / job["output"]
    checkpoint = output / f"checkpoint-epoch{job['epochs']}.pth"
    metrics = output / f"validation_per_mask_epoch{job['epochs']:04d}.csv"
    if not checkpoint.is_file() or not metrics.is_file():
        raise FileNotFoundError(f"Final endpoint artifacts missing: {job['name']}")
    state = torch.load(
        str(checkpoint), map_location="cpu", weights_only=False, mmap=True
    )
    if state["global_step"] != job["steps"] or state["sampler_step"] != job["steps"]:
        raise ValueError(f"Endpoint step mismatch: {job['name']}")
    if state["lr_scheduler"]["T_max"] != job["steps"]:
        raise ValueError(f"Scheduler horizon mismatch: {job['name']}")
    frame = pd.read_csv(metrics)
    if len(frame) != 32 or frame.mask_id.nunique() != 32:
        raise ValueError(f"Validation mask count mismatch: {job['name']}")
    if set(frame.sample_count) != {32}:
        raise ValueError(f"Validation scene count mismatch: {job['name']}")
    numeric = frame.select_dtypes(include="number").to_numpy(dtype=float)
    if not bool(np.isfinite(numeric).all()):
        raise ValueError(f"Non-finite endpoint metrics: {job['name']}")
    result = {
        "status": "complete",
        "name": job["name"],
        "global_step": state["global_step"],
        "sampler_step": state["sampler_step"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "metrics_csv": str(metrics),
        "metrics_sha256": sha256(metrics),
        "mask_count": 32,
        "scenes_per_mask": 32,
        "final_test_accessed": False,
    }
    save_json(output / "job_complete.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    bundle = root / manifest["bundle"]
    if manifest["final_test_accessed_by_this_queue"] is not False:
        raise ValueError("Final test is forbidden")
    for name, expected in manifest["program_hashes"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Program hash drift: {name}")
    preflight_path = bundle / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    if preflight["status"] != "pass":
        raise ValueError("GPU preflight has not passed")
    if preflight["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Preflight belongs to another manifest")
    jobs = {job["name"]: job for job in manifest["jobs"]}
    assigned = [name for lane in manifest["lanes"] for name in lane["jobs"]]
    if sorted(assigned) != sorted(jobs) or len(assigned) != len(set(assigned)):
        raise ValueError("Lane partition drift")
    if any((root / job["output"]).exists() for job in jobs.values()):
        raise FileExistsError("At least one training output already exists")
    if not args.execute:
        print(json.dumps({"status": "ready", "jobs": len(jobs)}, indent=2))
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
        "job_count": len(jobs),
        "completed_job_count": 0,
        "failed_job_count": 0,
        "data_partitions": ["train", "development"],
        "final_test_accessed": False,
        "lanes": {},
    }
    lock = threading.Lock()
    save_json(launcher / "launcher_state.json", state)

    def update(lane_id: int, lane_state: dict) -> None:
        with lock:
            state["lanes"][str(lane_id)] = lane_state
            records = [
                record
                for lane_value in state["lanes"].values()
                for record in lane_value["jobs"]
            ]
            state["completed_job_count"] = sum(
                record.get("returncode") == 0 for record in records
            )
            state["failed_job_count"] = sum(
                record.get("returncode", 0) != 0 for record in records
            )
            save_json(launcher / "launcher_state.json", state)

    def run_lane(lane: dict) -> dict:
        lane_state = {
            "physical_gpu": lane["physical_gpu"],
            "status": "running",
            "jobs": [],
        }
        update(lane["lane_id"], lane_state)
        for name in lane["jobs"]:
            job = jobs[name]
            config_path = root / job["config"]
            log_path = logs / f"{name}.log"
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(lane["physical_gpu"])
            env["MPLCONFIGDIR"] = str(bundle / ".cache/matplotlib")
            env["HF_HOME"] = str(root / "data/huggingface")
            Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            started = time.time()
            command = [
                sys.executable,
                str(root / "train.py"),
                "--config-path",
                str(config_path.parent),
                "--config-name",
                config_path.stem,
                f"hydra.run.dir={hydra / name}",
                "hydra.job.chdir=false",
            ]
            with log_path.open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            record = {
                "name": name,
                "returncode": result.returncode,
                "elapsed_seconds": time.time() - started,
                "log": str(log_path),
            }
            if result.returncode == 0:
                try:
                    record["endpoint"] = validate_endpoint(root, job)
                except Exception as error:
                    record["returncode"] = 97
                    record["validation_error"] = repr(error)
            lane_state["jobs"].append(record)
            update(lane["lane_id"], lane_state)
            if record["returncode"] != 0:
                lane_state["status"] = "failed"
                update(lane["lane_id"], lane_state)
                return lane_state
        lane_state["status"] = "complete"
        update(lane["lane_id"], lane_state)
        return lane_state

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_lane, lane) for lane in manifest["lanes"]]
        lane_results = [future.result() for future in futures]
    failures = [lane for lane in lane_results if lane["status"] != "complete"]
    state["status"] = "failed" if failures else "complete"
    state["finished_unix"] = time.time()
    state["final_test_accessed"] = False
    save_json(launcher / "launcher_state.json", state)
    if failures:
        raise SystemExit(1)
    summarizer = root / "scripts/summarize_long_scaling_results.py"
    result = subprocess.run(
        [sys.executable, str(summarizer), str(manifest_path)], cwd=root
    )
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
