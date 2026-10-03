"""Run frozen development-only input controls in sequential GPU lanes."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: str | Path, value: object) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--execute-development", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    evaluator = root / manifest["evaluator"]
    if sha256(evaluator) != manifest["evaluator_sha256"]:
        raise ValueError("Evaluator hash drift")
    if manifest["data_partition"] != "development":
        raise ValueError("Only development manifests are accepted")
    if manifest["final_test_accessed"] is not False:
        raise ValueError("Manifest final_test_accessed must be false")
    if len(manifest["lanes"]) > 4:
        raise ValueError("At most four GPU lanes are allowed")

    jobs = {job["name"]: job for job in manifest["jobs"]}
    assigned = [name for lane in manifest["lanes"] for name in lane["jobs"]]
    if sorted(assigned) != sorted(jobs) or len(assigned) != len(set(assigned)):
        raise ValueError("GPU lanes are not an exact partition of jobs")

    # All hashes and paths are checked before the first model forward.
    for name in assigned:
        job_path = root / jobs[name]["job_file"]
        result = subprocess.run(
            [sys.executable, str(evaluator), str(job_path)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode:
            raise RuntimeError(f"Preflight failed for {name}:\n{result.stdout}")
    if not args.execute_development:
        print(json.dumps({"status": "preflight_pass", "jobs": len(jobs)}, indent=2))
        return

    run_root = root / manifest["run_output"]
    run_root.mkdir(parents=True, exist_ok=False)
    log_root = run_root / "logs"
    log_root.mkdir()
    state = {
        "status": "running",
        "started_unix": time.time(),
        "data_partition": "development",
        "final_test_accessed": False,
        "job_count": len(jobs),
        "lanes": {},
    }
    save_json(run_root / "launcher_state.json", state)

    def run_lane(lane: dict) -> dict:
        lane_state = {"physical_gpu": lane["physical_gpu"], "jobs": []}
        for name in lane["jobs"]:
            job_path = root / jobs[name]["job_file"]
            log_path = log_root / f"{name}.log"
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(lane["physical_gpu"])
            started = time.time()
            with log_path.open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(evaluator),
                        str(job_path),
                        "--execute-development",
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
            lane_state["jobs"].append(record)
            state["lanes"][str(lane["lane_id"])] = lane_state
            save_json(run_root / "launcher_state.json", state)
            if result.returncode:
                lane_state["status"] = "failed"
                return lane_state
        lane_state["status"] = "complete"
        return lane_state

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(manifest["lanes"])) as pool:
        futures = {
            pool.submit(run_lane, lane): lane["lane_id"] for lane in manifest["lanes"]
        }
        lane_results = {}
        for future in concurrent.futures.as_completed(futures):
            lane_id = futures[future]
            lane_results[str(lane_id)] = future.result()
            state["lanes"] = lane_results | state["lanes"]
            save_json(run_root / "launcher_state.json", state)
    failed = [
        record
        for lane in lane_results.values()
        for record in lane["jobs"]
        if record["returncode"] != 0
    ]
    state.update(
        {
            "status": "failed" if failed else "complete",
            "finished_unix": time.time(),
            "lanes": lane_results,
            "failed_jobs": [record["name"] for record in failed],
        }
    )
    save_json(run_root / "launcher_state.json", state)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
