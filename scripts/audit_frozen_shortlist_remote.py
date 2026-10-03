"""Verify frozen-shortlist checkpoints on a remote host without deserialization."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
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
        row["size_bytes"] = os.path.getsize(path)
        row["sha256"] = sha256(path)
        row["metadata"] = metadata(path)
    print(json.dumps(row, sort_keys=True), flush=True)
'''


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="a800.sas.yp-c.yandex.net")
    parser.add_argument(
        "--manifest",
        default="outputs/coursework_pre_final_20260910/frozen_shortlist/frozen_shortlist.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/coursework_pre_final_20260910/frozen_shortlist/remote_audit",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest_path = (repo / args.manifest).resolve()
    output = (repo / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "frozen" or manifest.get(
        "final_synthetic_test_accessed"
    ) is not False:
        raise ValueError("Shortlist is not safely frozen")
    entries = manifest["entries"]
    paths = [entry["checkpoint_remote_path"] for entry in entries]
    if len(paths) != 13 or len(set(paths)) != 13:
        raise ValueError("Expected 13 unique remote checkpoint paths")

    remote_command = "python3 -c " + shlex.quote(REMOTE_CODE)
    remote_command += " " + " ".join(shlex.quote(path) for path in paths)
    completed = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            args.host,
            remote_command,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    returned = [
        json.loads(line) for line in completed.stdout.splitlines() if line.strip()
    ]
    by_path = {row["path"]: row for row in returned}
    if set(by_path) != set(paths):
        raise RuntimeError("Remote audit returned a different path set")

    rows = []
    for expected in entries:
        remote = by_path[expected["checkpoint_remote_path"]]
        metadata = remote.get("metadata", {})
        exists = bool(remote.get("exists"))
        row = {
            "shortlist_id": expected["shortlist_id"],
            "remote_path": expected["checkpoint_remote_path"],
            "exists": str(exists).lower(),
            "size_bytes": remote.get("size_bytes", ""),
            "size_match": str(
                exists and int(remote["size_bytes"]) == int(expected["size_bytes"])
            ).lower(),
            "epoch": metadata.get("epoch", ""),
            "global_step": metadata.get("global_step", ""),
            "sampler_step": metadata.get("sampler_step", ""),
            "T_max": metadata.get("T_max", ""),
            "endpoint_match": str(
                exists
                and all(
                    int(metadata.get(key, -1)) == int(expected[key])
                    for key in ("epoch", "global_step", "sampler_step", "T_max")
                )
            ).lower(),
            "sha256": remote.get("sha256", ""),
            "sha256_match": str(
                exists and remote.get("sha256") == expected["sha256"]
            ).lower(),
            "checkpoint_deserialization": "false",
        }
        rows.append(row)
    csv_path = output / "remote_shortlist_audit.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = all(
        row[key] == "true"
        for row in rows
        for key in ("exists", "size_match", "endpoint_match", "sha256_match")
    )
    summary = {
        "status": "pass" if passed else "fail",
        "host": args.host,
        "shortlist_entry_count": len(rows),
        "exists_count": sum(row["exists"] == "true" for row in rows),
        "size_match_count": sum(row["size_match"] == "true" for row in rows),
        "endpoint_match_count": sum(
            row["endpoint_match"] == "true" for row in rows
        ),
        "sha256_match_count": sum(row["sha256_match"] == "true" for row in rows),
        "checkpoint_deserialization": False,
        "checkpoint_files_copied": 0,
        "official_real_test_accessed": False,
        "final_synthetic_test_accessed": False,
        "manifest": str(manifest_path),
        "csv": str(csv_path),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
