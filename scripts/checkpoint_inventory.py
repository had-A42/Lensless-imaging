"""Build a read-only checkpoint inventory from the canonical run index.

The script never deserializes checkpoint files. It matches declared checkpoint
paths to local files by run-directory and file name, then reads only JSON
sidecars when they are available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickletools
import zipfile
from collections import Counter
from pathlib import Path


CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_files(repo: Path) -> list[Path]:
    files: list[Path] = []
    for root_name in ("saved", "model_weights", "outputs"):
        root = repo / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in CHECKPOINT_SUFFIXES or path.name == "recon_epochBEST":
                files.append(path.resolve())
    return sorted(set(files))


def checkpoint_index(paths: list[Path]) -> dict[tuple[str, str], list[Path]]:
    index: dict[tuple[str, str], list[Path]] = {}
    for path in paths:
        index.setdefault((path.parent.name, path.name), []).append(path)
    return index


def static_checkpoint_metadata(path: Path) -> dict[str, int]:
    """Read scalar endpoint fields from pickle opcodes without deserializing them."""
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_name = next(
                name for name in archive.namelist() if name.endswith("/data.pkl")
            )
            data = archive.read(metadata_name)
    except (OSError, StopIteration, zipfile.BadZipFile):
        return {}

    operations = list(pickletools.genops(data))
    values: dict[str, int] = {}
    fields = {"epoch", "global_step", "sampler_step", "T_max"}
    ignored = {"BINPUT", "LONG_BINPUT", "MEMOIZE", "PUT"}
    integers = {"BININT", "BININT1", "BININT2", "INT", "LONG1", "LONG4"}
    for index, (opcode, argument, _) in enumerate(operations):
        if opcode.name not in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}:
            continue
        if argument not in fields:
            continue
        next_index = index + 1
        while next_index < len(operations) and operations[next_index][0].name in ignored:
            next_index += 1
        if next_index < len(operations) and operations[next_index][0].name in integers:
            value = int(operations[next_index][1])
            if argument in values and values[argument] != value:
                return {}
            values[argument] = value
    return values


def read_completion(path: Path) -> dict:
    candidates = (
        path.parent / "completion.json",
        path.parent / "complete.json",
        path.parent.parent / "completion.json",
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(value, dict):
            value["_path"] = str(candidate)
            return value
    return {}


def first_present(mapping: dict, names: tuple[str, ...]):
    for name in names:
        if mapping.get(name) is not None:
            return mapping[name]
    return None


def build_rows(run_index: Path, repo: Path, hash_unique: bool) -> list[dict[str, str]]:
    local_paths = checkpoint_files(repo)
    local_index = checkpoint_index(local_paths)

    declared: dict[str, dict[str, str]] = {}
    with run_index.open(newline="") as stream:
        for row in csv.DictReader(stream):
            checkpoint = row.get("final_checkpoint", "").strip()
            if not checkpoint:
                continue
            declared.setdefault(checkpoint, row)

    rows: list[dict[str, str]] = []
    for declared_path, source in sorted(declared.items()):
        remote_path = Path(declared_path)
        matches = local_index.get((remote_path.parent.name, remote_path.name), [])
        if not matches:
            availability = "missing_local"
        elif len(matches) == 1:
            availability = "available_unique"
        else:
            availability = "available_multiple"

        completion = read_completion(matches[0]) if len(matches) == 1 else {}
        recorded_hash = first_present(
            completion,
            (
                "checkpoint_sha256",
                "final_checkpoint_sha256",
                "model_sha256",
            ),
        )
        actual_hash = ""
        hash_status = "not_checked"
        if len(matches) == 1 and hash_unique:
            actual_hash = sha256(matches[0])
            hash_status = "computed"
            if recorded_hash:
                hash_status = "match" if actual_hash == recorded_hash else "mismatch"
        elif recorded_hash:
            hash_status = "recorded_only"

        global_step = first_present(
            completion,
            ("global_step", "step", "steps", "final_step"),
        )
        rows.append(
            {
                "experiment_id": source.get("experiment_id", ""),
                "run_name": source.get("run_name", ""),
                "model": source.get("model", ""),
                "dataset": source.get("dataset", ""),
                "mask_count_or_mode": source.get("mask_count_or_mode", ""),
                "seed": source.get("seed", ""),
                "initialization": source.get("initialization", ""),
                "declared_steps": source.get("steps", ""),
                "declared_status": source.get("status", ""),
                "declared_checkpoint": declared_path,
                "availability": availability,
                "local_match_count": str(len(matches)),
                "local_paths": "|".join(str(path) for path in matches),
                "size_bytes": str(matches[0].stat().st_size) if len(matches) == 1 else "",
                "sidecar_global_step": "" if global_step is None else str(global_step),
                "sidecar_path": str(completion.get("_path", "")),
                "recorded_sha256": "" if recorded_hash is None else str(recorded_hash),
                "actual_sha256": actual_hash,
                "hash_status": hash_status,
            }
        )
    return rows


def core_xrest_run_name(masks: int, initialization: str, seed: int) -> str:
    if masks == 100 and seed == 42:
        return f"a800-xrest-{initialization}-bf16-50k-seed42-r3"
    revision = "r2" if masks == 100 else "r3"
    return (
        f"cv1-xrest-{initialization}-finite-{masks}-50000step-"
        f"seed{seed}-508a878-{revision}"
    )


def build_core_xrest_rows(
    endpoints_path: Path,
    repo: Path,
    hash_unique: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with endpoints_path.open(newline="") as stream:
        endpoints = list(csv.DictReader(stream))
    for endpoint in endpoints:
        masks = int(endpoint["masks"])
        seed = int(endpoint["seed"])
        initialization = endpoint["init"]
        run_name = core_xrest_run_name(masks, initialization, seed)
        run_dir = repo / "saved" / run_name
        candidates = sorted(run_dir.glob("*.pth")) if run_dir.is_dir() else []
        candidate_metadata = [
            (path, static_checkpoint_metadata(path)) for path in candidates
        ]
        finals = [
            (path, metadata)
            for path, metadata in candidate_metadata
            if metadata.get("global_step") == 50_000
        ]
        finals.sort(
            key=lambda item: (
                not item[0].name.startswith("checkpoint-epoch5"),
                item[0].name,
            )
        )
        selected = finals[0] if finals else None
        selected_path = selected[0] if selected else None
        selected_metadata = selected[1] if selected else {}
        rows.append(
            {
                "masks": str(masks),
                "initialization": initialization,
                "seed": str(seed),
                "run_name": run_name,
                "endpoint_PSNR": endpoint["PSNR"],
                "endpoint_SSIM": endpoint["SSIM"],
                "endpoint_LPIPS": endpoint["LPIPS"],
                "metric_source": endpoint["source"],
                "run_directory_exists": str(run_dir.is_dir()).lower(),
                "checkpoint_candidate_count": str(len(candidates)),
                "final_candidate_count": str(len(finals)),
                "selected_final_checkpoint": str(selected_path or ""),
                "selected_epoch": str(selected_metadata.get("epoch", "")),
                "selected_global_step": str(selected_metadata.get("global_step", "")),
                "selected_sampler_step": str(selected_metadata.get("sampler_step", "")),
                "selected_T_max": str(selected_metadata.get("T_max", "")),
                "selected_size_bytes": (
                    str(selected_path.stat().st_size) if selected_path else ""
                ),
                "selected_sha256": (
                    sha256(selected_path) if selected_path and hash_unique else ""
                ),
                "status": "available_final" if selected_path else "missing_final",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-index",
        default="outputs/coursework_completion_20260907/run_index.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/coursework_status_20260909",
    )
    parser.add_argument(
        "--xrest-endpoints",
        default="docs/reviews/artifacts/2026-09-05/xrest_50k_endpoints.csv",
    )
    parser.add_argument(
        "--hash-unique",
        action="store_true",
        help="Hash each uniquely matched checkpoint. This can read many gigabytes.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    run_index = (repo / args.run_index).resolve()
    output_dir = (repo / args.output_dir).resolve()
    if not run_index.is_file():
        raise FileNotFoundError(run_index)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(run_index, repo, hash_unique=args.hash_unique)
    fieldnames = list(rows[0]) if rows else []
    csv_path = output_dir / "checkpoint_inventory.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    endpoints_path = (repo / args.xrest_endpoints).resolve()
    core_rows = build_core_xrest_rows(
        endpoints_path,
        repo,
        hash_unique=args.hash_unique,
    )
    core_csv_path = output_dir / "core_xrest50k_checkpoint_inventory.csv"
    with core_csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(core_rows[0]))
        writer.writeheader()
        writer.writerows(core_rows)

    availability = Counter(row["availability"] for row in rows)
    hash_status = Counter(row["hash_status"] for row in rows)
    summary = {
        "schema_version": 1,
        "source_run_index": str(run_index),
        "checkpoint_files_scanned": len(checkpoint_files(repo)),
        "declared_unique_checkpoints": len(rows),
        "availability": dict(sorted(availability.items())),
        "hash_status": dict(sorted(hash_status.items())),
        "hash_unique_requested": bool(args.hash_unique),
        "checkpoint_deserialization": False,
        "csv": str(csv_path),
        "core_xrest50k": {
            "source_endpoints": str(endpoints_path),
            "rows": len(core_rows),
            "available_final": sum(
                row["status"] == "available_final" for row in core_rows
            ),
            "missing_final": sum(row["status"] == "missing_final" for row in core_rows),
            "csv": str(core_csv_path),
        },
    }
    (output_dir / "checkpoint_inventory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
