"""Audit declared remote checkpoints without deserializing their contents."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from collections import Counter
from pathlib import Path


REMOTE_CODE = r'''
import hashlib
import json
import os
import pickletools
import sys
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(path):
    try:
        with zipfile.ZipFile(path) as archive:
            name = next(value for value in archive.namelist() if value.endswith("/data.pkl"))
            data = archive.read(name)
    except Exception:
        return {}
    operations = list(pickletools.genops(data))
    fields = {"epoch", "global_step", "sampler_step", "T_max"}
    ignored = {"BINPUT", "LONG_BINPUT", "MEMOIZE", "PUT"}
    integers = {"BININT", "BININT1", "BININT2", "INT", "LONG1", "LONG4"}
    result = {}
    for index, (opcode, argument, _) in enumerate(operations):
        if opcode.name not in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}:
            continue
        if argument not in fields:
            continue
        cursor = index + 1
        while cursor < len(operations) and operations[cursor][0].name in ignored:
            cursor += 1
        if cursor < len(operations) and operations[cursor][0].name in integers:
            value = int(operations[cursor][1])
            if argument in result and result[argument] != value:
                return {}
            result[argument] = value
    return result


for path in sys.argv[1:]:
    row = {"path": path, "exists": os.path.isfile(path)}
    if row["exists"]:
        stat = os.stat(path)
        row.update({"size_bytes": stat.st_size, "sha256": sha256(path), "metadata": metadata(path)})
    print(json.dumps(row, sort_keys=True), flush=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="a800.sas.yp-c.yandex.net")
    parser.add_argument(
        "--inventory",
        default="outputs/coursework_status_20260909/checkpoint_inventory.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/coursework_status_20260909",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    inventory_path = (repo / args.inventory).resolve()
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with inventory_path.open(newline="") as stream:
        inventory = list(csv.DictReader(stream))
    pending = [row for row in inventory if row["availability"] == "missing_local"]
    paths = sorted({row["declared_checkpoint"] for row in pending})

    remote_command = "python3 -c " + shlex.quote(REMOTE_CODE)
    if paths:
        remote_command += " " + " ".join(shlex.quote(path) for path in paths)
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        args.host,
        remote_command,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    audited = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    by_path = {row["path"]: row for row in audited}
    if set(by_path) != set(paths):
        raise RuntimeError("Remote audit did not return exactly the requested paths")

    output_rows = []
    for source in pending:
        remote = by_path[source["declared_checkpoint"]]
        metadata = remote.get("metadata", {})
        output_rows.append(
            {
                "experiment_id": source["experiment_id"],
                "run_name": source["run_name"],
                "model": source["model"],
                "dataset": source["dataset"],
                "seed": source["seed"],
                "initialization": source["initialization"],
                "declared_steps": source["declared_steps"],
                "remote_path": remote["path"],
                "exists": str(bool(remote["exists"])).lower(),
                "size_bytes": str(remote.get("size_bytes", "")),
                "epoch": str(metadata.get("epoch", "")),
                "global_step": str(metadata.get("global_step", "")),
                "sampler_step": str(metadata.get("sampler_step", "")),
                "T_max": str(metadata.get("T_max", "")),
                "sha256": remote.get("sha256", ""),
                "checkpoint_deserialization": "false",
            }
        )

    csv_path = output_dir / "remote_checkpoint_audit.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    existence = Counter(row["exists"] for row in output_rows)
    endpoint_matches = sum(
        row["global_step"] and row["global_step"] == row["declared_steps"]
        for row in output_rows
    )
    summary = {
        "schema_version": 1,
        "host": args.host,
        "requested_paths": len(paths),
        "inventory_rows": len(output_rows),
        "existence": dict(sorted(existence.items())),
        "endpoint_matches_declared_steps": endpoint_matches,
        "all_requested_paths_returned": set(by_path) == set(paths),
        "checkpoint_deserialization": False,
        "official_real_test_accessed": False,
        "final_synthetic_test_accessed": False,
        "csv": str(csv_path),
    }
    summary["status"] = (
        "complete"
        if len(paths) == len(output_rows) and existence.get("true", 0) == len(output_rows)
        else "complete_with_missing"
    )
    summary_path = output_dir / "remote_checkpoint_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
