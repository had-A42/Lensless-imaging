"""Launch fixed-classifier reconstruction evaluation on one to four GPUs."""

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

import pandas as pd


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()
    if not 1 <= len(args.gpus) <= 4 or len(args.gpus) != len(set(args.gpus)):
        raise ValueError("Choose one to four distinct GPUs")
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    if manifest["official_test_accessed"] or manifest["final_test_accessed"]:
        raise ValueError("Only development data are allowed")
    if sha256(root / manifest["evaluator"]) != manifest["evaluator_sha256"]:
        raise ValueError("Evaluator hash drift")
    if sha256(root / manifest["launcher"]) != manifest["launcher_sha256"]:
        raise ValueError("Launcher hash drift")
    if sha256(root / manifest["classifier_checkpoint"]) != manifest["classifier_checkpoint_sha256"]:
        raise ValueError("Classifier checkpoint drift")
    for relative, expected in manifest["source_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Source hash drift: {relative}")
    for entry in manifest["entries"]:
        for field, hash_field in (
            ("config", "config_sha256"),
            ("checkpoint", "checkpoint_sha256"),
        ):
            if sha256(entry[field]) != entry[hash_field]:
                raise ValueError(f"{entry['id']} {field} drift")
        if (root / entry["output"]).exists():
            raise FileExistsError(root / entry["output"])
    memory = gpu_memory()
    occupied = {gpu: memory[gpu] for gpu in args.gpus if memory[gpu] > 1024}
    if occupied:
        raise RuntimeError(f"Refusing to use occupied GPUs: {occupied}")
    if not args.execute:
        print(
            json.dumps(
                {"status": "ready", "entries": len(manifest["entries"]), "gpus": args.gpus},
                indent=2,
            )
        )
        return
    bundle = manifest_path.parent
    logs = bundle / "logs"
    logs.mkdir(exist_ok=False)
    queues = {gpu: [] for gpu in args.gpus}
    for index, entry in enumerate(manifest["entries"]):
        queues[args.gpus[index % len(args.gpus)]].append(entry)
    state = {
        "status": "running",
        "started_unix": time.time(),
        "completed": [],
        "failed": [],
        "gpus": args.gpus,
        "official_test_accessed": False,
        "final_test_accessed": False,
    }
    lock = threading.Lock()
    save_json(bundle / "launcher_state.json", state)

    def run_queue(gpu: int, entries: list[dict]) -> list[dict]:
        results = []
        for entry in entries:
            log_path = logs / f"{entry['id']}.log"
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["MPLCONFIGDIR"] = str(bundle / ".cache/matplotlib")
            env["HF_HOME"] = str(root / "data/huggingface")
            Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            started = time.time()
            with log_path.open("w") as log:
                process = subprocess.run(
                    [
                        sys.executable,
                        str(root / manifest["evaluator"]),
                        str(manifest_path),
                        entry["id"],
                    ],
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            result = {
                "id": entry["id"],
                "gpu": gpu,
                "returncode": process.returncode,
                "elapsed_seconds": time.time() - started,
                "log": str(log_path),
            }
            results.append(result)
            with lock:
                state["completed" if process.returncode == 0 else "failed"].append(result)
                save_json(bundle / "launcher_state.json", state)
            if process.returncode:
                break
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queues)) as pool:
        list(pool.map(lambda item: run_queue(*item), queues.items()))
    if state["failed"]:
        state["status"] = "failed"
        state["finished_unix"] = time.time()
        save_json(bundle / "launcher_state.json", state)
        raise SystemExit(1)
    rows = []
    for entry in manifest["entries"]:
        output = root / entry["output"]
        validation = json.loads((output / "validation.json").read_text())
        summary = json.loads((output / "summary.json").read_text())
        if validation["status"] != "pass":
            raise ValueError(f"Validation failed: {entry['id']}")
        rows.append(summary)
    frame = pd.DataFrame(rows)
    frame.to_csv(bundle / "summary_per_run.csv", index=False)
    aggregate = (
        frame.groupby("information_regime")["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.to_csv(bundle / "summary_aggregate.csv", index=False)
    paired = frame.pivot(index="seed", columns="information_regime", values="accuracy")
    paired["psf_aware_minus_psf_free"] = paired["psf_aware"] - paired["psf_free"]
    paired.reset_index().to_csv(bundle / "paired_effects.csv", index=False)
    state["status"] = "complete"
    state["finished_unix"] = time.time()
    save_json(bundle / "launcher_state.json", state)
    save_json(
        bundle / "validation.json",
        {
            "status": "pass",
            "entry_count": len(rows),
            "all_entry_validations_pass": True,
            "data_partition": "development",
            "official_test_accessed": False,
            "final_test_accessed": False,
        },
    )


if __name__ == "__main__":
    main()
