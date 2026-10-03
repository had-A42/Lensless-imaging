"""Run the frozen three-GPU MNIST PSF-conditioning diagnostic."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    if manifest["data_partition"] != "development" or manifest["final_test_accessed"]:
        raise ValueError("Only development execution is allowed")
    if sha256(root / manifest["evaluator"]) != manifest["evaluator_sha256"]:
        raise ValueError("Evaluator hash drift")
    if sha256(root / manifest["launcher"]) != manifest["launcher_sha256"]:
        raise ValueError("Launcher hash drift")
    if sha256(manifest["parent_manifest"]) != manifest["parent_manifest_sha256"]:
        raise ValueError("Parent manifest drift")
    for relative, expected in manifest["source_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Source hash drift: {relative}")
    for entry in manifest["entries"]:
        for field, hash_field in (
            ("config", "config_sha256"),
            ("checkpoint", "checkpoint_sha256"),
            ("reference_per_mask", "reference_per_mask_sha256"),
        ):
            if sha256(entry[field]) != entry[hash_field]:
                raise ValueError(f"{entry['id']} {field} drift")
        if (root / entry["output"]).exists():
            raise FileExistsError(root / entry["output"])
    if not args.execute:
        print(json.dumps({"status": "ready", "entries": len(manifest["entries"])}, indent=2))
        return
    bundle = manifest_path.parent
    logs = bundle / "logs"
    logs.mkdir(exist_ok=False)
    state = {
        "status": "running",
        "started_unix": time.time(),
        "completed": [],
        "failed": [],
        "final_test_accessed": False,
    }
    save_json(bundle / "launcher_state.json", state)

    def run(entry: dict, gpu: int) -> dict:
        log_path = logs / f"{entry['id']}.log"
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
                    str(root / manifest["evaluator"]),
                    str(manifest_path),
                    entry["id"],
                ],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return {
            "id": entry["id"],
            "gpu": gpu,
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
            "log": str(log_path),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(run, entry, manifest["gpus"][index])
            for index, entry in enumerate(manifest["entries"])
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            key = "completed" if result["returncode"] == 0 else "failed"
            state[key].append(result)
            save_json(bundle / "launcher_state.json", state)
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
        for condition, values in summary["conditions"].items():
            rows.append({"seed": entry["seed"], "condition": condition, **values})
    frame = pd.DataFrame(rows)
    frame.to_csv(bundle / "summary_per_seed.csv", index=False)
    metrics = ["PSNR", "SSIM", "PSNR_32", "SSIM_32", "Dice_loss_32"]
    aggregate = []
    for condition, group in frame.groupby("condition"):
        for metric in metrics:
            aggregate.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "mean": float(group[metric].mean()),
                    "sample_sd": float(group[metric].std(ddof=1)),
                    "run_count": len(group),
                }
            )
    pd.DataFrame(aggregate).to_csv(bundle / "summary_aggregate.csv", index=False)
    state["status"] = "complete"
    state["finished_unix"] = time.time()
    state["final_test_accessed"] = False
    save_json(bundle / "launcher_state.json", state)
    save_json(
        bundle / "validation.json",
        {
            "status": "pass",
            "entry_count": 3,
            "all_entry_validations_pass": True,
            "data_partition": "development",
            "final_test_accessed": False,
        },
    )


if __name__ == "__main__":
    main()
