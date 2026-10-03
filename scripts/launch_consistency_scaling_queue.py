"""Launch the frozen four-lane consistency-scaling training queue."""

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


def gpu_memory() -> dict[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        int(line.split(",")[0]): int(line.split(",")[1].strip())
        for line in output.splitlines()
        if line.strip()
    }


def validate_endpoint(root: Path, job: dict) -> dict:
    output = root / job["output"]
    checkpoint = output / f"checkpoint-epoch{job['epochs']}.pth"
    metrics = output / f"validation_per_mask_epoch{job['epochs']:04d}.csv"
    if not checkpoint.is_file() or not metrics.is_file():
        raise FileNotFoundError(f"Missing endpoint artifacts: {job['name']}")
    state = torch.load(
        str(checkpoint), map_location="cpu", weights_only=False, mmap=True
    )
    if not (
        state["global_step"] == state["sampler_step"] == job["steps"]
        and state["lr_scheduler"]["T_max"] == job["steps"]
    ):
        raise ValueError(f"Endpoint metadata mismatch: {job['name']}")
    frame = pd.read_csv(metrics)
    if len(frame) != 32 or frame["mask_id"].nunique() != 32 or set(frame["sample_count"]) != {32}:
        raise ValueError(f"Development grid mismatch: {job['name']}")
    if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
        raise ValueError(f"Non-finite metrics: {job['name']}")
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
    parser.add_argument("--gpus", nargs=4, type=int)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    bundle = root / manifest["bundle"]
    if manifest["final_test_accessed"]:
        raise ValueError("Final test access is forbidden")
    for relative, expected in manifest["program_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Program hash drift: {relative}")
    preflight = json.loads((bundle / "preflight.json").read_text())
    if preflight["status"] != "pass" or preflight["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("GPU preflight is missing or belongs to another manifest")
    jobs = {job["name"]: job for job in manifest["jobs"]}
    if any((root / job["output"]).exists() for job in jobs.values()):
        raise FileExistsError("At least one output directory already exists")
    gpus = args.gpus or [lane["preferred_gpu"] for lane in manifest["lanes"]]
    if len(set(gpus)) != 4:
        raise ValueError("Four distinct physical GPUs are required")
    memory = gpu_memory()
    occupied = {gpu: memory[gpu] for gpu in gpus if memory[gpu] > 1024}
    if occupied:
        raise RuntimeError(f"Refusing to start on occupied GPUs: {occupied}")
    if not args.execute:
        print(json.dumps({"status": "ready", "jobs": len(jobs), "gpus": gpus}, indent=2))
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
        "physical_gpus": gpus,
        "data_partitions": ["train", "development"],
        "final_test_accessed": False,
        "lanes": {},
    }
    lock = threading.Lock()
    save_json(launcher / "launcher_state.json", state)

    def update(lane_id: int, value: dict) -> None:
        with lock:
            state["lanes"][str(lane_id)] = value
            records = [
                record
                for lane in state["lanes"].values()
                for record in lane["jobs"]
            ]
            state["completed_job_count"] = sum(
                record.get("returncode") == 0 for record in records
            )
            state["failed_job_count"] = sum(
                record.get("returncode", 0) != 0 for record in records
            )
            save_json(launcher / "launcher_state.json", state)

    def run_lane(lane: dict, gpu: int) -> dict:
        value = {"physical_gpu": gpu, "status": "running", "jobs": []}
        update(lane["lane_id"], value)
        for name in lane["jobs"]:
            job = jobs[name]
            config = root / job["config"]
            log_path = logs / f"{name}.log"
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["MPLCONFIGDIR"] = str(bundle / ".cache/matplotlib")
            env["HF_HOME"] = str(root / "data/huggingface")
            Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            started = time.time()
            with log_path.open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(root / "train.py"),
                        "--config-path",
                        str(config.parent),
                        "--config-name",
                        config.stem,
                        f"hydra.run.dir={hydra / name}",
                        "hydra.job.chdir=false",
                    ],
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
            value["jobs"].append(record)
            update(lane["lane_id"], value)
            if record["returncode"] != 0:
                value["status"] = "failed"
                update(lane["lane_id"], value)
                return value
        value["status"] = "complete"
        update(lane["lane_id"], value)
        return value

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda pair: run_lane(pair[1], gpus[pair[0]]),
                enumerate(manifest["lanes"]),
            )
        )
    failures = [value for value in results if value["status"] != "complete"]
    state["status"] = "failed" if failures else "complete"
    state["finished_unix"] = time.time()
    save_json(launcher / "launcher_state.json", state)
    if failures:
        raise SystemExit(1)
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/summarize_consistency_scaling_results.py"),
            str(manifest_path),
        ],
        cwd=root,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
