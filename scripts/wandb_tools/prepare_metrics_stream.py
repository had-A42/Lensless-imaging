#!/usr/bin/env python3
"""Create a lightweight W&B stream that contains metrics but no artifacts."""

import argparse
from collections import Counter
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

SKIPPED_RECORDS = {"artifact", "files", "output_raw"}


def default_output_path(source):
    return source.with_name(f"{source.stem}.metrics.wandb")


def prepare_metrics_stream(source, output, force=False):
    source = Path(source)
    output = Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not force:
        print(f"Metrics stream already exists: {output}")
        return output

    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    reader = DataStore()
    writer = DataStore()
    counts = Counter()
    kept = Counter()

    try:
        reader.open_for_scan(str(source))
        writer.open_for_write(str(temporary))
        while True:
            data = reader.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            record_type = record.WhichOneof("record_type")
            counts[record_type] += 1
            if record_type in SKIPPED_RECORDS:
                continue
            writer.write(record)
            kept[record_type] += 1
    finally:
        reader.close()
        writer.close()

    if output.exists():
        output.unlink()
    temporary.replace(output)
    source_mb = source.stat().st_size / 1024**2
    output_mb = output.stat().st_size / 1024**2
    skipped = sum(counts[name] for name in SKIPPED_RECORDS)
    print(
        f"Prepared {output} ({source_mb:.1f} MiB -> {output_mb:.1f} MiB, "
        f"removed {skipped} artifact/file/console records)"
    )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Create a metrics-only copy of an offline W&B stream."
    )
    parser.add_argument("source", type=Path, help="Original run-RUNID.wandb file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output or default_output_path(args.source)
    prepare_metrics_stream(args.source, output, force=args.force)


if __name__ == "__main__":
    main()
