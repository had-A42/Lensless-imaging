"""Freeze the pre-declared final synthetic-test checkpoint shortlist safely."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.checkpoint_inventory import static_checkpoint_metadata  # noqa: E402


EXPECTED_SEEDS = (42, 52, 62)
EXPECTED_MASK_COUNTS = (100, 1_000)
EXPECTED_INITIALIZATIONS = ("scratch", "gopro")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def remote_core_path(local_path: Path) -> str:
    relative = local_path.resolve().relative_to(REPO_ROOT)
    checkout = (
        "Lensless-imaging-research-508a878"
        if local_path.parent.name.startswith("cv1-")
        else "Lensless-imaging"
    )
    return f"/home/hadhad/project/{checkout}/{relative}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core-inventory",
        default="outputs/coursework_status_20260909/core_xrest50k_checkpoint_inventory.csv",
    )
    parser.add_argument(
        "--checkpoint-inventory",
        default="outputs/coursework_status_20260909/checkpoint_inventory.csv",
    )
    parser.add_argument(
        "--output", default="outputs/coursework_pre_final_20260910/frozen_shortlist"
    )
    args = parser.parse_args()
    core_path = (REPO_ROOT / args.core_inventory).resolve()
    inventory_path = (REPO_ROOT / args.checkpoint_inventory).resolve()
    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    with core_path.open(newline="") as stream:
        core_rows = list(csv.DictReader(stream))
    expected_keys = {
        (masks, initialization, seed)
        for masks in EXPECTED_MASK_COUNTS
        for initialization in EXPECTED_INITIALIZATIONS
        for seed in EXPECTED_SEEDS
    }
    actual_keys = {
        (int(row["masks"]), row["initialization"], int(row["seed"]))
        for row in core_rows
    }
    if actual_keys != expected_keys or len(core_rows) != 12:
        raise ValueError("Core X-Restormer matrix is not exactly the frozen 12-run design")

    entries = []
    for row in sorted(
        core_rows,
        key=lambda item: (
            int(item["masks"]),
            item["initialization"],
            int(item["seed"]),
        ),
    ):
        path = Path(row["selected_final_checkpoint"]).resolve()
        if row["status"] != "available_final" or not path.is_file():
            raise FileNotFoundError(f"Frozen core checkpoint is unavailable: {path}")
        metadata = static_checkpoint_metadata(path)
        expected_metadata = {
            "epoch": 5,
            "global_step": 50_000,
            "sampler_step": 50_000,
            "T_max": 50_000,
        }
        if metadata != expected_metadata:
            raise ValueError(f"Endpoint mismatch for {path}: {metadata}")
        actual_hash = sha256(path)
        if actual_hash != row["selected_sha256"]:
            raise ValueError(f"Checkpoint hash drift: {path}")
        entries.append(
            {
                "shortlist_id": (
                    f"xrest50k-m{row['masks']}-{row['initialization']}-seed{row['seed']}"
                ),
                "role": "core_50k_matrix",
                "architecture": "X-Restormer",
                "training_masks": int(row["masks"]),
                "initialization": row["initialization"],
                "seed": int(row["seed"]),
                "training_steps": 50_000,
                "checkpoint_local_path": str(path),
                "checkpoint_remote_path": remote_core_path(path),
                "size_bytes": path.stat().st_size,
                **metadata,
                "sha256": actual_hash,
                "selection_basis": "all retained runs in the frozen 100/1000 x scratch/GoPro matrix; final 50k endpoint",
                "selected_using_final_test": False,
            }
        )

    with inventory_path.open(newline="") as stream:
        inventory_rows = list(csv.DictReader(stream))
    finalist_rows = [
        row for row in inventory_rows if row["run_name"] == "E2-mirflickr-xrest100k-seed42"
    ]
    if len(finalist_rows) != 1:
        raise ValueError("Expected one pre-declared 100k finalist inventory row")
    finalist_source = finalist_rows[0]
    finalist_path = Path(finalist_source["local_paths"]).resolve()
    if not finalist_path.is_file():
        raise FileNotFoundError(finalist_path)
    finalist_metadata = static_checkpoint_metadata(finalist_path)
    expected_finalist_metadata = {
        "epoch": 10,
        "global_step": 100_000,
        "sampler_step": 100_000,
        "T_max": 100_000,
    }
    if finalist_metadata != expected_finalist_metadata:
        raise ValueError(f"100k finalist endpoint mismatch: {finalist_metadata}")
    finalist_hash = sha256(finalist_path)
    if finalist_hash != finalist_source["actual_sha256"]:
        raise ValueError("100k finalist hash drift")
    entries.append(
        {
            "shortlist_id": "xrest100k-m100-gopro-seed42",
            "role": "predeclared_100k_finalist",
            "architecture": "X-Restormer",
            "training_masks": 100,
            "initialization": "gopro",
            "seed": 42,
            "training_steps": 100_000,
            "checkpoint_local_path": str(finalist_path),
            "checkpoint_remote_path": finalist_source["declared_checkpoint"],
            "size_bytes": finalist_path.stat().st_size,
            **finalist_metadata,
            "sha256": finalist_hash,
            "selection_basis": "single 100k finalist fixed before final-test access from development evidence and the input-use control",
            "selected_using_final_test": False,
        }
    )

    if len(entries) != 13 or len({row["sha256"] for row in entries}) != 13:
        raise ValueError("Frozen shortlist must contain 13 distinct checkpoint hashes")
    csv_path = output / "frozen_shortlist.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "frozen_before_final_test_access": True,
        "checkpoint_selection_by_final_test_metrics_forbidden": True,
        "final_synthetic_test_accessed": False,
        "matrix_contract": {
            "architecture": "X-Restormer",
            "training_mask_counts": list(EXPECTED_MASK_COUNTS),
            "initializations": list(EXPECTED_INITIALIZATIONS),
            "retained_seeds": list(EXPECTED_SEEDS),
            "core_training_steps": 50_000,
            "core_checkpoint_count": 12,
            "additional_100k_finalists": 1,
        },
        "entries": entries,
        "sources": {
            "core_inventory": str(core_path),
            "core_inventory_sha256": sha256(core_path),
            "checkpoint_inventory": str(inventory_path),
            "checkpoint_inventory_sha256": sha256(inventory_path),
            "generator": str(Path(__file__).resolve()),
            "generator_sha256": sha256(Path(__file__).resolve()),
        },
        "csv": str(csv_path),
    }
    json_path = output / "frozen_shortlist.json"
    write_json(json_path, manifest)
    validation = {
        "status": "pass",
        "entry_count": len(entries),
        "core_entry_count": sum(row["role"] == "core_50k_matrix" for row in entries),
        "finalist_entry_count": sum(
            row["role"] == "predeclared_100k_finalist" for row in entries
        ),
        "all_local_files_exist": all(
            Path(row["checkpoint_local_path"]).is_file() for row in entries
        ),
        "all_local_hashes_verified": True,
        "all_endpoints_verified_statically": True,
        "checkpoint_deserialization": False,
        "final_synthetic_test_accessed": False,
        "manifest": str(json_path),
        "manifest_sha256": sha256(json_path),
        "csv": str(csv_path),
        "csv_sha256": sha256(csv_path),
    }
    write_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
