"""Verify local and remote registry weights without loading model tensors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
REMOTE_CODE = r"""
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
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(WORKSPACE / "plans/all_test_checkpoint_registry_20260913.csv"),
    )
    parser.add_argument("--host", default="a800.sas.yp-c.yandex.net")
    parser.add_argument(
        "--output",
        default=str(
            WORKSPACE / "plans/all_test_checkpoint_registry_20260913.audit.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        default=str(
            WORKSPACE / "plans/all_test_checkpoint_registry_20260913.audit.json"
        ),
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--rewrite", action="store_true")
    args = parser.parse_args()
    registry_path = Path(args.registry).resolve()
    output = Path(args.output).resolve()
    summary_path = Path(args.summary).resolve()
    if not args.rewrite and (output.exists() or summary_path.exists()):
        raise FileExistsError("Audit output exists; pass --rewrite explicitly")
    with registry_path.open(newline="") as stream:
        registry = list(csv.DictReader(stream))

    expected_by_remote: dict[str, str] = {}
    for row in registry:
        path = row["checkpoint_remote"]
        if not path:
            continue
        previous = expected_by_remote.setdefault(path, row["checkpoint_sha256"])
        if previous != row["checkpoint_sha256"]:
            raise ValueError(f"Conflicting expected hash for {path}")
    remote_paths = sorted(expected_by_remote)
    remote_command = "python3 -c " + shlex.quote(REMOTE_CODE)
    remote_command += " " + " ".join(shlex.quote(path) for path in remote_paths)
    completed = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            args.host,
            remote_command,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    remote_rows = [
        json.loads(line) for line in completed.stdout.splitlines() if line.strip()
    ]
    remote = {row["path"]: row for row in remote_rows}
    if set(remote) != set(remote_paths):
        raise ValueError("Remote audit did not return the requested path set")

    audit_rows = []
    for source in registry:
        expected = source["checkpoint_sha256"]
        local_path = (
            Path(source["checkpoint_local"]) if source["checkpoint_local"] else None
        )
        local_exists = bool(local_path and local_path.is_file())
        local_hash = sha256(local_path) if local_exists else ""
        remote_value = remote.get(source["checkpoint_remote"], {})
        remote_exists = bool(remote_value.get("exists", False))
        remote_hash = remote_value.get("sha256", "")
        metadata = remote_value.get("metadata", {})
        local_match = local_exists and local_hash == expected
        remote_match = remote_exists and remote_hash == expected
        audit_rows.append(
            {
                "registry_id": source["registry_id"],
                "dataset": source["dataset"],
                "model": source["model"],
                "checkpoint_local": source["checkpoint_local"],
                "local_exists": str(local_exists).lower(),
                "local_sha256_match": str(local_match).lower(),
                "checkpoint_remote": source["checkpoint_remote"],
                "remote_exists": str(remote_exists).lower(),
                "remote_sha256_match": str(remote_match).lower(),
                "remote_size_bytes": str(remote_value.get("size_bytes", "")),
                "remote_epoch": str(metadata.get("epoch", "")),
                "remote_global_step": str(metadata.get("global_step", "")),
                "remote_sampler_step": str(metadata.get("sampler_step", "")),
                "remote_T_max": str(metadata.get("T_max", "")),
                "expected_sha256": expected,
                "available_exact": str(local_match or remote_match).lower(),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    temporary.replace(output)
    available_count = sum(row["available_exact"] == "true" for row in audit_rows)
    remote_exists_count = sum(row["remote_exists"] == "true" for row in audit_rows)
    remote_match_count = sum(row["remote_sha256_match"] == "true" for row in audit_rows)
    local_match_count = sum(row["local_sha256_match"] == "true" for row in audit_rows)
    summary = {
        "status": "pass" if available_count == len(audit_rows) else "incomplete",
        "registry_rows": len(audit_rows),
        "unique_remote_paths": len(remote_paths),
        "available_exact_rows": available_count,
        "local_sha256_match_rows": local_match_count,
        "remote_exists_rows": remote_exists_count,
        "remote_sha256_match_rows": remote_match_count,
        "checkpoint_deserialization": False,
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "test_model_forward_executed": False,
        "registry_sha256": sha256(registry_path),
        "audit_csv_sha256": sha256(output),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
