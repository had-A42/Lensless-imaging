"""Freeze the user-approved 11-model shortlist after excluding legacy entries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_frozen_final_synthetic import sha256, write_json  # noqa: E402


EXCLUDED = {
    "xrest50k-m100-gopro-seed42",
    "xrest50k-m100-scratch-seed42",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="outputs/coursework_final_runner_v4_20260910/revised_shortlist_v1",
    )
    args = parser.parse_args()
    old_path = (
        REPO_ROOT
        / "outputs/coursework_pre_final_20260910/frozen_shortlist_v2/frozen_shortlist.json"
    )
    audit_path = (
        REPO_ROOT
        / "outputs/coursework_final_runner_v4_20260910/shortlist_integrity_audit/audit.json"
    )
    remote_path = (
        REPO_ROOT
        / "outputs/coursework_pre_final_20260910/frozen_shortlist_v2/remote_audit/remote_shortlist_audit.csv"
    )
    old = json.loads(old_path.read_text())
    audit = json.loads(audit_path.read_text())
    if set(audit["affected_shortlist_ids"]) != EXCLUDED:
        raise ValueError("Integrity audit affected set drifted")
    entries = []
    for source in old["entries"]:
        if source["shortlist_id"] in EXCLUDED:
            continue
        entry = dict(source)
        if entry["role"] == "predeclared_100k_finalist":
            entry["analysis_role"] = "predeclared_100k_finalist"
        elif int(entry["seed"]) in {52, 62}:
            entry["analysis_role"] = "primary_matched_matrix"
        elif int(entry["seed"]) == 42 and int(entry["training_masks"]) == 1000:
            entry["analysis_role"] = "supplementary_corrected_seed42"
        else:
            raise ValueError(f"Unexpected retained entry: {entry['shortlist_id']}")
        entry["selected_using_final_test"] = False
        entries.append(entry)
    with remote_path.open(newline="") as stream:
        remote = {row["shortlist_id"]: row for row in csv.DictReader(stream)}
    if any(
        remote[entry["shortlist_id"]][key] != "true"
        for entry in entries
        for key in ("exists", "size_match", "endpoint_match", "sha256_match")
    ):
        raise ValueError("A retained checkpoint failed the prior remote audit")
    counts = {
        "primary_matched_matrix": sum(
            entry["analysis_role"] == "primary_matched_matrix" for entry in entries
        ),
        "supplementary_corrected_seed42": sum(
            entry["analysis_role"] == "supplementary_corrected_seed42"
            for entry in entries
        ),
        "predeclared_100k_finalist": sum(
            entry["analysis_role"] == "predeclared_100k_finalist" for entry in entries
        ),
    }
    if len(entries) != 11 or counts != {
        "primary_matched_matrix": 8,
        "supplementary_corrected_seed42": 2,
        "predeclared_100k_finalist": 1,
    }:
        raise ValueError(f"Unexpected revised shortlist counts: {counts}")
    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "shortlist.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    manifest = {
        "schema_version": 1,
        "status": "frozen_revised",
        "frozen_before_final_test_access": True,
        "user_authorization_text": "начинай проводить тест на тех моделях, что есть",
        "interpretation_of_authorization": "use 8 matched seeds52/62 core models, one 100k finalist, and two valid seed42 finite1000 supplementary models",
        "checkpoint_selection_by_final_test_metrics_forbidden": True,
        "final_test_accessed": False,
        "entry_count": len(entries),
        "role_counts": counts,
        "excluded_entries": sorted(EXCLUDED),
        "exclusion_reason": "legacy PSF convention and incorrect mapping to corrected-run metrics",
        "entries": entries,
        "sources": {
            "superseded_shortlist": str(old_path),
            "superseded_shortlist_sha256": sha256(old_path),
            "integrity_audit": str(audit_path),
            "integrity_audit_sha256": sha256(audit_path),
            "remote_audit": str(remote_path),
            "remote_audit_sha256": sha256(remote_path),
        },
        "csv": str(csv_path),
    }
    manifest_path = output / "shortlist.json"
    write_json(manifest_path, manifest)
    validation = {
        "status": "pass",
        "entry_count": len(entries),
        "role_counts": counts,
        "exact_exclusion_set": sorted(EXCLUDED),
        "all_retained_remote_checks_pass": True,
        "all_selected_using_final_test_false": True,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "csv": str(csv_path),
        "csv_sha256": sha256(csv_path),
        "final_test_accessed": False,
    }
    write_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
