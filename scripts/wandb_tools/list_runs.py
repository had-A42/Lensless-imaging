#!/usr/bin/env python3
"""Inventory offline W&B runs in the local archive."""

import argparse
import json
import re
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

ORIGINAL_STREAM = re.compile(r"run-([a-zA-Z0-9]+)\.wandb$")


def run_name(stream):
    reader = DataStore()
    try:
        reader.open_for_scan(str(stream))
        while True:
            data = reader.scan_data()
            if data is None:
                return "unknown"
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "run":
                continue
            config = {
                item.key: json.loads(item.value_json)
                for item in record.run.config.update
            }
            return config.get("writer", {}).get("run_name", "unknown")
    finally:
        reader.close()


def streams_by_id(root):
    by_id = {}
    for stream in root.rglob("run-*.wandb"):
        match = ORIGINAL_STREAM.fullmatch(stream.name)
        if match is None:
            continue
        run_id = match.group(1)
        segment = stream.parent.name
        rank = ("all-checkouts" in stream.parts, len(stream.parts), str(stream))
        current = by_id.setdefault(run_id, {}).get(segment)
        if current is None or rank < current[0]:
            by_id[run_id][segment] = (rank, stream)
    return {
        run_id: sorted(
            (value[1] for value in segments.values()),
            key=lambda stream: stream.parent.name,
        )
        for run_id, segments in by_id.items()
    }


def local_status(stream):
    marker_prefix = f"{stream.stem}*.wandb"
    for marker, status in (
        ("verified", "verified"),
        ("uploaded", "uploaded"),
        ("synced", "synced"),
    ):
        if any(stream.parent.glob(f"{marker_prefix}.{marker}")):
            return status
    return "local"


def combined_status(streams):
    statuses = [local_status(stream) for stream in streams]
    if all(status == "verified" for status in statuses):
        return "verified"
    if all(status in {"verified", "uploaded"} for status in statuses):
        return "uploaded"
    if all(status != "local" for status in statuses):
        return "synced"
    return "local"


def main():
    parser = argparse.ArgumentParser(description="List locally archived W&B runs.")
    parser.add_argument("--root", type=Path, default=Path("wandb/remote-a800"))
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--unsynced", action="store_true")
    args = parser.parse_args()

    streams = streams_by_id(args.root)
    selected = sorted(args.ids or streams)
    print("run_id\tstatus\tsegments\tsize_mib\trun_name\tpath")
    for run_id in selected:
        segments = streams.get(run_id)
        if segments is None:
            print(f"{run_id}\tmissing\t-\t-\t-\t-")
            continue
        status = combined_status(segments)
        if args.unsynced and status != "local":
            continue
        size = sum(stream.stat().st_size for stream in segments) / 1024**2
        paths = ",".join(str(stream.parent) for stream in segments)
        print(
            f"{run_id}\t{status}\t{len(segments)}\t{size:.1f}\t"
            f"{run_name(segments[-1])}\t{paths}"
        )


if __name__ == "__main__":
    main()
