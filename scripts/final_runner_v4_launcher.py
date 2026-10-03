"""Launch the frozen V4 checkpoint shards on exactly three physical GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=("development", "final"), required=True)
    parser.add_argument("--authorization")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest, _, _ = load_v4_metadata(manifest_path)
    if args.mode == "final":
        authorization_path = (
            Path(args.authorization).resolve() if args.authorization else None
        )
        validate_final_authorization(authorization_path, manifest_path)
    else:
        if args.authorization is not None:
            raise ValueError("Development mode does not accept authorization")
        authorization_path = None

    shards = manifest["shards"]
    if len(shards) != 3 or [int(shard["physical_gpu"]) for shard in shards] != [0, 1, 2]:
        raise ValueError("Launcher requires the frozen GPU0/GPU1/GPU2 mapping")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "launcher_state.json",
        {
            "status": "preflight",
            "mode": args.mode,
            "manifest_sha256": sha256(manifest_path),
            "process_limit": 3,
            "physical_gpus": [0, 1, 2],
            "automatic_retry": False,
            "test_accessed": False,
        },
    )
    worker = REPO_ROOT / "scripts/final_runner_v4_worker.py"
    merger = REPO_ROOT / "scripts/final_runner_v4_merge.py"
    commands = []
    for shard in shards:
        shard_id = int(shard["shard_id"])
        command = [
            sys.executable,
            str(worker),
            "--manifest",
            str(manifest_path),
            "--mode",
            args.mode,
            "--shard-id",
            str(shard_id),
            "--output",
            str(output / f"shard{shard_id}"),
        ]
        if authorization_path is not None:
            command.extend(["--authorization", str(authorization_path)])
        commands.append(
            {
                "shard_id": shard_id,
                "physical_gpu": int(shard["physical_gpu"]),
                "checkpoint_ids": shard["checkpoint_ids"],
                "command": command,
            }
        )
    write_json(
        output / "launch_plan.json",
        {
            "status": "frozen",
            "mode": args.mode,
            "concurrent_gpu_processes": len(commands),
            "commands": commands,
            "test_accessed": False,
        },
    )

    processes = []
    logs = []
    wait_results = []
    try:
        for command in commands:
            log_path = output / f"shard{command['shard_id']}.log"
            log_stream = log_path.open("w")
            logs.append(log_stream)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(command["physical_gpu"])
            process = subprocess.Popen(
                command["command"],
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((command, process, log_path))
        write_json(
            output / "launcher_state.json",
            {
                "status": "running",
                "mode": args.mode,
                "processes": [
                    {
                        "shard_id": command["shard_id"],
                        "physical_gpu": command["physical_gpu"],
                        "pid": process.pid,
                        "log": str(log_path),
                    }
                    for command, process, log_path in processes
                ],
                "test_accessed": args.mode == "final",
            },
        )
        wait_results = [
            {
                "shard_id": command["shard_id"],
                "physical_gpu": command["physical_gpu"],
                "returncode": process.wait(),
                "log": str(log_path),
            }
            for command, process, log_path in processes
        ]
    finally:
        for stream in logs:
            stream.close()
    results = [
        {**result, "log_sha256": sha256(Path(result["log"]))}
        for result in wait_results
    ]
    if any(result["returncode"] != 0 for result in results):
        write_json(
            output / "launcher_state.json",
            {
                "status": "failed_closed",
                "mode": args.mode,
                "results": results,
                "automatic_retry": False,
                "test_accessed": args.mode == "final",
            },
        )
        raise SystemExit(2)

    merge_command = [
        sys.executable,
        str(merger),
        "--manifest",
        str(manifest_path),
        "--mode",
        args.mode,
        "--input-root",
        str(output),
        "--output",
        str(output / "merged"),
    ]
    completed = subprocess.run(
        merge_command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    (output / "merge.stdout.log").write_text(completed.stdout)
    (output / "merge.stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        write_json(
            output / "launcher_state.json",
            {
                "status": "failed_closed_at_merge",
                "mode": args.mode,
                "results": results,
                "merge_returncode": completed.returncode,
                "automatic_retry": False,
                "test_accessed": args.mode == "final",
            },
        )
        raise SystemExit(3)
    write_json(
        output / "launcher_state.json",
        {
            "status": "complete",
            "mode": args.mode,
            "results": results,
            "merge_returncode": completed.returncode,
            "merged_validation": str(output / "merged/validation.json"),
            "test_accessed": args.mode == "final",
            "final_test_model_forward_executed": args.mode == "final",
        },
    )
    print(json.dumps(json.loads((output / "merged/validation.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
