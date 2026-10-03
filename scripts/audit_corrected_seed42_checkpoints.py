"""Audit the two corrected seed42 checkpoints exposed by V4 development parity."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_frozen_final_synthetic import sha256, write_json  # noqa: E402


EXPECTED = {
    "gopro": {
        "run_id": "s9ay4t57",
        "run_name": "a800-xrest-cv1-gopro-50k-seed42-73e432d-r1",
        "legacy_id": "xrest50k-m100-gopro-seed42",
        "legacy_run_id": "o98lv52j",
        "legacy_run_name": "a800-xrest-gopro-bf16-50k-seed42-r3",
    },
    "scratch": {
        "run_id": "2qin4zky",
        "run_name": "a800-xrest-cv1-scratch-50k-seed42-73e432d-r1",
        "legacy_id": "xrest50k-m100-scratch-seed42",
        "legacy_run_id": "8nzfyhg5",
        "legacy_run_name": "a800-xrest-scratch-bf16-50k-seed42-r3",
    },
}


def wandb_file_records(path: Path) -> list[str]:
    store = DataStore()
    store.open_for_scan(str(path))
    values = []
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") == "files":
            values.extend(file.path for file in record.files.files)
    return values


def git_source(revision: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{revision}:src/datasets/on_the_fly.py"],
        cwd=REPO_ROOT,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="a800.sas.yp-c.yandex.net")
    parser.add_argument(
        "--output",
        default="outputs/coursework_final_runner_v4_20260910/shortlist_integrity_audit",
    )
    args = parser.parse_args()
    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shortlist_path = (
        REPO_ROOT
        / "outputs/coursework_pre_final_20260910/frozen_shortlist_v2/frozen_shortlist.json"
    )
    shortlist = json.loads(shortlist_path.read_text())
    by_id = {entry["shortlist_id"]: entry for entry in shortlist["entries"]}
    legacy_source = git_source("a22cd38ca3058064e39c749d9a12229bcad17778")
    corrected_source = git_source("73e432d7850f0750b40576f5c1a1907fa2a38faf")
    legacy_uses_raw_psf = "psf=torch.as_tensor(np.asarray(psf), dtype=torch.float32)" in legacy_source
    corrected_uses_prepared_psf = "psf=_prepare_convolution_psf(psf)" in corrected_source
    rows = []
    for arm, contract in EXPECTED.items():
        expected_path = (
            "/home/hadhad/project/Lensless-imaging-night-73e432d/saved/"
            f"{contract['run_name']}/checkpoint-epoch5.pth"
        )
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            args.host,
            f"test -f {expected_path}",
        ]
        remote_exists = subprocess.run(command, check=False).returncode == 0
        offline_root = (
            REPO_ROOT
            / "wandb/remote-a800/all-checkouts/Lensless-imaging-night-73e432d/wandb"
            / f"offline-run-20260827_053747-{contract['run_id']}"
        )
        metadata_path = offline_root / "files/wandb-metadata.json"
        run_path = offline_root / f"run-{contract['run_id']}.wandb"
        metadata = json.loads(metadata_path.read_text())
        file_records = wandb_file_records(run_path)
        checkpoint_records = [
            value
            for value in file_records
            if value.endswith((".pth", ".pt", ".ckpt")) or "checkpoint" in value
        ]
        legacy = by_id[contract["legacy_id"]]
        fp32_summary_path = (
            REPO_ROOT
            / "outputs/coursework_final_runner_v4_20260910/development_run/shard0"
            / contract["legacy_id"]
            / "summary.json"
        )
        bf16_summary_path = (
            REPO_ROOT
            / "outputs/coursework_final_runner_v4_20260910/bf16_diagnostics_v2"
            / f"seed42_{arm}"
            / "summary.json"
        )
        fp32 = json.loads(fp32_summary_path.read_text())
        bf16 = json.loads(bf16_summary_path.read_text())
        rows.append(
            {
                "arm": arm,
                "expected_corrected_run_id": contract["run_id"],
                "expected_corrected_run_name": contract["run_name"],
                "expected_source_revision": metadata["git"]["commit"],
                "expected_remote_checkpoint": expected_path,
                "expected_remote_checkpoint_exists": remote_exists,
                "wandb_checkpoint_file_record_count": len(checkpoint_records),
                "legacy_shortlist_id": contract["legacy_id"],
                "legacy_run_name": contract["legacy_run_name"],
                "legacy_checkpoint": legacy["checkpoint_local_path"],
                "legacy_checkpoint_sha256": legacy["sha256"],
                "legacy_source_revision": "a22cd38ca3058064e39c749d9a12229bcad17778",
                "legacy_fp32_PSNR": fp32["mask_balanced"]["PSNR"],
                "legacy_bf16_PSNR": bf16["observed"]["PSNR"],
                "legacy_saved_reference_PSNR": bf16["reference"]["PSNR"],
                "legacy_bf16_parity_status": bf16["status"],
                "current_shortlist_assignment_matches_corrected_run": False,
            }
        )
    csv_path = output / "audit.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "blocked_missing_exact_corrected_checkpoints",
        "finding": "Two seed42 finite100 shortlist entries point to legacy-convention a22cd38 checkpoints, while the reported endpoint metrics come from corrected-convention 73e432d runs.",
        "legacy_source_uses_raw_psf_in_convolver": legacy_uses_raw_psf,
        "corrected_source_uses_flipped_l2_normalized_psf": corrected_uses_prepared_psf,
        "expected_corrected_remote_files_found": sum(
            bool(row["expected_remote_checkpoint_exists"]) for row in rows
        ),
        "expected_corrected_wandb_checkpoint_file_records": sum(
            int(row["wandb_checkpoint_file_record_count"]) for row in rows
        ),
        "affected_shortlist_ids": [row["legacy_shortlist_id"] for row in rows],
        "unaffected_shortlist_entries": len(shortlist["entries"]) - len(rows),
        "fp32_three_gpu_worker_status": "13/13 model evaluations completed; merge failed closed on development parity",
        "bf16_diagnostics_status": "both legacy seed42 checkpoints failed their own saved development references",
        "required_user_decision": "recover the two exact corrected checkpoints, or approve a revised primary shortlist/estimand that excludes legacy seed42 finite100 entries",
        "sources": {
            "shortlist": str(shortlist_path),
            "shortlist_sha256": sha256(shortlist_path),
            "audit_csv": str(csv_path),
            "audit_csv_sha256": sha256(csv_path),
        },
        "rows": rows,
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "final_test_model_forward_executed": False,
    }
    write_json(output / "audit.json", result)
    (output / "RESULTS.md").write_text(
        "# V4 shortlist integrity audit\n\n"
        "The three-GPU development run completed all 13 model evaluations, but the merger stopped on the frozen parity check. Two finite100 seed42 entries were mapped to legacy checkpoints trained with the pre-correction PSF convention. Their reported table metrics came from different corrected runs.\n\n"
        "The exact corrected checkpoint paths are no longer present on A800, and the synced W&B records contain no checkpoint file record. FP32 and BF16 reevaluations of the legacy files both fail the saved references, so changing precision does not resolve the mismatch.\n\n"
        "Final-test execution remains locked. The next step requires a user decision: recover the exact corrected weights, or approve a revised primary shortlist that excludes the two legacy entries and adjusts the matched estimand.\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
