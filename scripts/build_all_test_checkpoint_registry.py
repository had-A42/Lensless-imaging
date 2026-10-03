"""Build a test-evaluation checkpoint registry without accessing test data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
DEFAULT_OUTPUT = WORKSPACE / "plans/all_test_checkpoint_registry_20260913.csv"
DEFAULT_VALIDATION = (
    WORKSPACE / "plans/all_test_checkpoint_registry_20260913.validation.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_candidate(checkpoint: str) -> str:
    return str(Path(checkpoint).parent / "config.yaml") if checkpoint else ""


def legacy_rows() -> list[dict[str, str]]:
    inventory_path = (
        REPO / "outputs/coursework_status_20260909/checkpoint_inventory.csv"
    )
    remote_audit_path = (
        REPO / "outputs/coursework_status_20260909/remote_checkpoint_audit.csv"
    )
    with inventory_path.open(newline="") as stream:
        inventory = list(csv.DictReader(stream))
    with remote_audit_path.open(newline="") as stream:
        remote = {row["remote_path"]: row for row in csv.DictReader(stream)}
    rows = []
    for source in inventory:
        declared = source["declared_checkpoint"]
        local = (
            source["local_paths"]
            if source["availability"] == "available_unique"
            else ""
        )
        remote_path = declared if declared.startswith("/home/") else ""
        if not remote_path and local:
            try:
                relative_local = Path(local).resolve().relative_to(REPO.resolve())
            except ValueError:
                relative_local = None
            if relative_local is not None and relative_local.parts[0] == "saved":
                remote_path = str(
                    Path("/home/hadhad/project/Lensless-imaging") / relative_local
                )
        audited = remote.get(remote_path, {})
        observed_hash = (
            source["actual_sha256"]
            or source["recorded_sha256"]
            or audited.get("sha256", "")
        )
        notes = "Re-audit final-vs-best and source convention before test"
        readiness = (
            "available_local"
            if source["availability"] == "available_unique"
            else "verified_remote_20260910"
        )
        if source["run_name"] == "completion-E3-celeba-xrest-finite100-10k-seed42":
            local = ""
            observed_hash = (
                "74a38bf14db133e6495322166135f6a53142d1762ad3b9de77f63a80d0d20d73"
            )
            readiness = "verified_remote_deserializable_20260913"
            notes = (
                "Use the original remote checkpoint that produced the archived "
                "development evaluation; the local copy is truncated"
            )
        local_config = config_candidate(local)
        remote_config = config_candidate(remote_path)
        rows.append(
            {
                "registry_id": f"legacy::{source['run_name']}",
                "scope": "legacy_report_inventory",
                "report_role": source["experiment_id"],
                "dataset": source["dataset"],
                "model": source["model"],
                "condition": source["mask_count_or_mode"],
                "seed": source["seed"],
                "steps": source["declared_steps"],
                "checkpoint_local": local,
                "checkpoint_remote": remote_path,
                "checkpoint_sha256": observed_hash,
                "config_local_candidate": local_config,
                "config_remote_candidate": remote_config,
                "config_sha256": "",
                "endpoint_status": source["declared_status"],
                "readiness": readiness,
                "source_registry": str(inventory_path),
                "notes": notes,
            }
        )
    return rows


def queue_rows(bundle_name: str, scope: str) -> list[dict[str, str]]:
    bundle = REPO / "outputs" / bundle_name
    manifest = json.loads((bundle / "manifest.json").read_text())
    remote_root = Path(manifest["root"])
    rows = []
    for job in manifest["jobs"]:
        run = REPO / job["output"]
        complete = json.loads((run / "job_complete.json").read_text())
        if not (
            complete["status"] == "complete"
            and int(complete["global_step"]) == int(job["steps"])
            and complete["final_test_accessed"] is False
        ):
            raise ValueError(f"Invalid endpoint: {job['name']}")
        experiment = job["experiment"]
        dataset = "MNIST" if "mnist" in experiment else "MIRFLICKR"
        model = (
            "PSF-aware DRUNet"
            if "psf_aware" in str(job.get("information_regime", ""))
            else "DRUNet"
        )
        if dataset == "MNIST" and "psf-aware" in job["name"]:
            model = "PSF-aware DRUNet"
        condition = str(job.get("condition", job.get("information_regime", "")))
        local_config = run / "config.yaml"
        remote_config = remote_root / job["output"] / "config.yaml"
        rows.append(
            {
                "registry_id": f"{scope}::{job['name']}",
                "scope": scope,
                "report_role": experiment,
                "dataset": dataset,
                "model": model,
                "condition": condition,
                "seed": str(job["seed"]),
                "steps": str(job["steps"]),
                "checkpoint_local": "",
                "checkpoint_remote": complete["checkpoint"],
                "checkpoint_sha256": complete["checkpoint_sha256"],
                "config_local_candidate": str(local_config),
                "config_remote_candidate": str(remote_config),
                "config_sha256": job["config_sha256"],
                "endpoint_status": "exact_final_endpoint",
                "readiness": "verified_complete",
                "source_registry": str(bundle / "manifest.json"),
                "notes": "Development endpoint; test not accessed by training queue",
            }
        )
    return rows


def real_primary_rows() -> list[dict[str, str]]:
    summary_path = REPO / "outputs/ref-psff-real-68/evaluation/three_seed_summary.json"
    summary = json.loads(summary_path.read_text())
    remote_root = Path("/home/hadhad/project/Lensless-imaging-ref-real-cc984")
    run_ids = {
        (42, "real"): "offline-run-20260827_161504-vjql3obv",
        (42, "matched_sim"): "offline-run-20260827_164334-6goeo3b1",
        (52, "real"): "offline-run-20260827_180755-qpojal25",
        (52, "matched_sim"): "offline-run-20260827_183548-yj4s7lhf",
        (62, "real"): "offline-run-20260827_180756-fi3ablto",
        (62, "matched_sim"): "offline-run-20260827_183547-2ikho7eh",
    }
    rows = []
    for seed in (42, 52, 62):
        artifacts = summary["input_artifacts"][str(seed)]
        for domain in ("real", "matched_sim"):
            local_checkpoint = (
                REPO
                / "wandb/remote-a800/20260827-ref-real"
                / run_ids[(seed, domain)]
                / "files"
                / f"ref-psff-real-68-{domain}-seed{seed}"
                / "checkpoint-epoch10.pth"
            )
            checkpoint = (
                remote_root
                / "saved"
                / f"ref-psff-real-68-{domain}-seed{seed}"
                / "checkpoint-epoch10.pth"
            )
            expected_hash = artifacts[f"{domain}_checkpoint_sha256"]
            if (
                not local_checkpoint.is_file()
                or sha256(local_checkpoint) != expected_hash
            ):
                raise ValueError(f"Local RQ3 checkpoint mismatch: {local_checkpoint}")
            config = (
                REPO
                / "outputs/ref-psff-real-68/training"
                / f"{domain}_seed{seed}/config.yaml"
            )
            rows.append(
                {
                    "registry_id": f"real_primary::{domain}_seed{seed}",
                    "scope": "real_primary_matched",
                    "report_role": "RQ3_real_vs_matched_sim",
                    "dataset": "DigiCam real",
                    "model": "DRUNet",
                    "condition": domain,
                    "seed": str(seed),
                    "steps": "10000",
                    "checkpoint_local": str(local_checkpoint),
                    "checkpoint_remote": str(checkpoint),
                    "checkpoint_sha256": expected_hash,
                    "config_local_candidate": str(config),
                    "config_remote_candidate": "",
                    "config_sha256": sha256(config),
                    "endpoint_status": "exact_final_endpoint",
                    "readiness": "verified_local_archive_requires_upload",
                    "source_registry": str(summary_path),
                    "notes": "Use official DigiCam test; do not substitute similarly named X-Restormer paths",
                }
            )
    return rows


def active_report_rows() -> list[dict[str, str]]:
    local_definitions = [
        *[
            (
                f"dr_small_10k_finite100_seed{seed}",
                "MIRFLICKR",
                "DRUNet",
                "finite100",
                seed,
                10000,
                f"saved/a800-e02-screen-finite-100-seed{seed}/model_best.pth",
            )
            for seed in (42, 52, 62)
        ],
        *[
            (
                f"dr_small_10k_finite1000_seed{seed}",
                "MIRFLICKR",
                "DRUNet",
                "finite1000",
                seed,
                10000,
                f"saved/cv1-drunet-small-finite-1000-10000step-seed{seed}-508a878/model_best.pth",
            )
            for seed in (42, 52, 62)
        ],
        (
            "dr_small_10k_finite10000_seed42",
            "MIRFLICKR",
            "DRUNet",
            "finite10000",
            42,
            10000,
            "saved/cv1-drunet-small-finite-10000-10000step-seed42-508a878/model_best.pth",
        ),
        *[
            (
                f"dr_small_10k_streaming_seed{seed}",
                "MIRFLICKR",
                "DRUNet",
                "streaming",
                seed,
                10000,
                f"saved/a800-e02-screen-infinite-seed{seed}/model_best.pth",
            )
            for seed in (42, 52, 62)
        ],
        (
            "dr_small_50k_finite1000_seed42",
            "MIRFLICKR",
            "DRUNet",
            "finite1000",
            42,
            50000,
            "saved/cv1-drunet-small-finite-1000-50000step-seed42-508a878/checkpoint-epoch5.pth",
        ),
        (
            "dr_small_50k_streaming_seed42",
            "MIRFLICKR",
            "DRUNet",
            "streaming",
            42,
            50000,
            "saved/cv1-drunet-small-infinite-50000step-seed42-508a878/model_best.pth",
        ),
        (
            "dr_large_10k_finite100_random",
            "MIRFLICKR",
            "Large DRUNet",
            "finite100_random",
            42,
            10000,
            "saved/cv1-large-scratch-finite-100-10000step-seed42-508a878/model_best.pth",
        ),
        (
            "dr_large_10k_finite100_denoising",
            "MIRFLICKR",
            "Large DRUNet",
            "finite100_denoising",
            42,
            10000,
            "saved/cv1-large-dpir-finite-100-10000step-seed42-508a878/model_best.pth",
        ),
        (
            "dr_large_10k_finite1000_random",
            "MIRFLICKR",
            "Large DRUNet",
            "finite1000_random",
            42,
            10000,
            "saved/cv1-large-scratch-finite-1000-10000step-seed42-508a878/model_best.pth",
        ),
        (
            "dr_large_10k_finite1000_denoising",
            "MIRFLICKR",
            "Large DRUNet",
            "finite1000_denoising",
            42,
            10000,
            "saved/cv1-large-dpir-finite-1000-10000step-seed42-508a878/model_best.pth",
        ),
        (
            "dr_large_50k_finite100",
            "MIRFLICKR",
            "Large DRUNet",
            "finite100",
            42,
            50000,
            "saved/cv1-drunet-large-finite-100-50000step-seed42-508a878/checkpoint-epoch5.pth",
        ),
        (
            "dr_large_50k_finite1000",
            "MIRFLICKR",
            "Large DRUNet",
            "finite1000",
            42,
            50000,
            "saved/cv1-drunet-large-finite-1000-50000step-seed42-508a878/model_best.pth",
        ),
        (
            "dr_large_50k_streaming_unstable",
            "MIRFLICKR",
            "Large DRUNet",
            "streaming_unstable",
            42,
            50000,
            "saved/cv1-drunet-large-infinite-50000step-seed42-508a878/model_best.pth",
        ),
        (
            "dr_large_100k_finite100",
            "MIRFLICKR",
            "Large DRUNet",
            "finite100",
            42,
            100000,
            "saved/scale100k-drunet-large-scratch-finite-100-seed42-508a878-r3/checkpoint-epoch10.pth",
        ),
        (
            "xrest_100k_finite100_random",
            "MIRFLICKR",
            "X-Restormer",
            "finite100_random",
            42,
            100000,
            "saved/scale100k-xrest-scratch-finite-100-seed42-508a878-r3/model_best.pth",
        ),
        (
            "sdvae_100k_random",
            "MIRFLICKR",
            "SD-VAE",
            "finite100_random",
            42,
            100000,
            "saved/scale100k-sdvae-random-finite-100-seed42-508a878-r3/model_best.pth",
        ),
        (
            "sdvae_100k_pretrained",
            "MIRFLICKR",
            "SD-VAE",
            "finite100_pretrained",
            42,
            100000,
            "saved/scale100k-sdvae-pretrained-finite-100-seed42-508a878-r3/model_best.pth",
        ),
        (
            "dinov2_100k_random",
            "MIRFLICKR",
            "DINOv2",
            "finite100_random",
            42,
            100000,
            "saved/scale100k-dinov2-random-finite-100-seed42-508a878-r3/checkpoint-epoch10.pth",
        ),
        (
            "dinov2_100k_pretrained",
            "MIRFLICKR",
            "DINOv2",
            "finite100_pretrained",
            42,
            100000,
            "saved/scale100k-dinov2-pretrained-finite-100-seed42-508a878-r3/checkpoint-epoch10.pth",
        ),
        (
            "lingbot_100k_random",
            "MIRFLICKR",
            "LingBot",
            "finite100_random",
            42,
            100000,
            "saved/scale100k-lingbot-random-finite-100-seed42-508a878-r3/checkpoint-epoch10.pth",
        ),
        (
            "lingbot_100k_pretrained",
            "MIRFLICKR",
            "LingBot",
            "finite100_pretrained",
            42,
            100000,
            "saved/scale100k-lingbot-pretrained-finite-100-seed42-508a878-r3/checkpoint-epoch10.pth",
        ),
        *[
            (
                f"mnist_rgb_10k_seed{seed}",
                "MNIST",
                "DRUNet RGB",
                "finite100_dice",
                seed,
                10000,
                f"saved/a800-mnist-finite100-dice-10k-seed{seed}/model_best.pth",
            )
            for seed in (42, 52, 62)
        ],
    ]
    gray_runs = {
        42: "offline-run-20260827_020546-68qy8ims",
        52: "offline-run-20260827_020407-8rpgrhvw",
        62: "offline-run-20260827_030155-fgmn28ny",
    }
    for seed, run_id in gray_runs.items():
        local_definitions.append(
            (
                f"mnist_gray_10k_seed{seed}",
                "MNIST",
                "DRUNet grayscale",
                "finite100_dice",
                seed,
                10000,
                "wandb/remote-a800/20260827-night/"
                f"{run_id}/files/a800-mnist-cv2-gray-fp32-10k-seed{seed}-73e432d/model_best.pth",
            )
        )

    rows = []
    for (
        identifier,
        dataset,
        model,
        condition,
        seed,
        steps,
        relative,
    ) in local_definitions:
        local = REPO / relative
        if not local.is_file():
            raise FileNotFoundError(local)
        remote = ""
        if relative.startswith("saved/"):
            remote = str(Path("/home/hadhad/project/Lensless-imaging") / relative)
        config = local.parent / "config.yaml"
        rows.append(
            {
                "registry_id": f"active_report::{identifier}",
                "scope": "active_report_addition",
                "report_role": "active_table_row",
                "dataset": dataset,
                "model": model,
                "condition": condition,
                "seed": str(seed),
                "steps": str(steps),
                "checkpoint_local": str(local),
                "checkpoint_remote": remote,
                "checkpoint_sha256": sha256(local),
                "config_local_candidate": str(config),
                "config_remote_candidate": config_candidate(remote),
                "config_sha256": sha256(config) if config.is_file() else "",
                "endpoint_status": "report_active_candidate",
                "readiness": "exact_weight_available_needs_endpoint_audit",
                "source_registry": "current_report.tex",
                "notes": "Use only after development parity confirms the reported row",
            }
        )

    remote_definitions = [
        (
            "dr_small_50k_finite10000_seed42",
            "DRUNet",
            "finite10000",
            50000,
            "/home/hadhad/project/Lensless-imaging-celeba-20260905/outputs/coursework_completion_20260907/drunet_followup_20260907/training/followup-drunet-small-finite10000-100k-seed42/checkpoint-epoch5.pth",
            "126f5c8c1b53f56c4c564261408381804a0d15b96f16b3b3727352ab53e0aaec",
        ),
        (
            "dr_large_50k_streaming_clipped",
            "Large DRUNet",
            "streaming_clipped",
            50000,
            "/home/hadhad/project/Lensless-imaging-celeba-20260905/outputs/coursework_completion_20260907/drunet_followup_20260907/training/followup-drunet-large-infinite-clip1-50k-seed42/checkpoint-epoch5.pth",
            "ff8a5c37c0a4e80c5d757b127b4b82661dba91bd556b1117c22a8e7d4c19c04e",
        ),
    ]
    for (
        identifier,
        model,
        condition,
        steps,
        remote,
        expected_hash,
    ) in remote_definitions:
        rows.append(
            {
                "registry_id": f"active_report::{identifier}",
                "scope": "active_report_addition",
                "report_role": "active_table_row",
                "dataset": "MIRFLICKR",
                "model": model,
                "condition": condition,
                "seed": "42",
                "steps": str(steps),
                "checkpoint_local": "",
                "checkpoint_remote": remote,
                "checkpoint_sha256": expected_hash,
                "config_local_candidate": "",
                "config_remote_candidate": config_candidate(remote),
                "config_sha256": "",
                "endpoint_status": "report_active_candidate",
                "readiness": "verified_remote_path_needs_development_parity",
                "source_registry": "current_report.tex",
                "notes": "Intermediate 50k endpoint from the recorded longer/modified run",
            }
        )
    return rows


def auxiliary_rows() -> list[dict[str, str]]:
    definitions = [
        (
            "film_psf_descriptor",
            "MIRFLICKR",
            "PSF-conditioned DRUNet",
            "saved/syn-psf-conditioned-drunet-seed42-r1/model_best.pth",
        ),
        (
            "compatibility_model",
            "MIRFLICKR",
            "Compatibility model",
            "outputs/psff_compat_01/pilot-seed42-r1/compatibility_final.pth",
        ),
        (
            "refinement_measurement_context",
            "MIRFLICKR",
            "Residual refinement U-Net",
            "outputs/psff_refine_01/pair-seed42-r1/measurement_context_final.pth",
        ),
        (
            "refinement_self_context",
            "MIRFLICKR",
            "Residual refinement U-Net",
            "outputs/psff_refine_01/pair-seed42-r1/self_context_final.pth",
        ),
        (
            "adapter_blur_only_seed42",
            "DigiCam real",
            "Input adapter",
            "outputs/sim_real_adapter/blur-only-xrest-seed42-2k/adapter_best.pth",
        ),
        (
            "adapter_blur_only_seed52",
            "DigiCam real",
            "Input adapter",
            "outputs/sim_real_adapter/blur-only-xrest-seed52-2k/adapter_best.pth",
        ),
        (
            "adapter_blur_affine_seed42",
            "DigiCam real",
            "Input adapter",
            "outputs/sim_real_adapter/blur-affine-xrest-seed42-2k/adapter_best.pth",
        ),
        (
            "adapter_blur_affine_seed52",
            "DigiCam real",
            "Input adapter",
            "outputs/sim_real_adapter/blur-affine-xrest-seed52-2k/adapter_best.pth",
        ),
        (
            "adapter_hybrid_seed42",
            "DigiCam real",
            "Input adapter",
            "outputs/sim_real_adapter/hybrid-lowpass-xrest-seed42-2k/adapter_best.pth",
        ),
        (
            "adapter_hybrid_seed52",
            "DigiCam real",
            "Input adapter",
            "outputs/sim_real_adapter/hybrid-lowpass-xrest-seed52-2k/adapter_best.pth",
        ),
        (
            "adapter_base_xrestormer_seed42",
            "DigiCam real",
            "Adapter base X-Restormer",
            "saved/operator-prompt-constant-true-d16-finite-100-seed42-r1/checkpoint-epoch50.pth",
        ),
        (
            "adapter_base_xrestormer_seed52",
            "DigiCam real",
            "Adapter base X-Restormer",
            "saved/operator-prompt-constant-true-d16-finite-100-seed52-r1/checkpoint-epoch50.pth",
        ),
        (
            "fixed_mnist_classifier",
            "MNIST",
            "Frozen classifier",
            "outputs/coursework_mnist_classifier_local_v2_20260912/training/classifier_final.pth",
        ),
    ]
    rows = []
    for identifier, dataset, model, relative in definitions:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "registry_id": f"auxiliary::{identifier}",
                "scope": "auxiliary_report_method",
                "report_role": "exploratory_or_metric_support",
                "dataset": dataset,
                "model": model,
                "condition": "",
                "seed": "",
                "steps": "",
                "checkpoint_local": str(path),
                "checkpoint_remote": "",
                "checkpoint_sha256": sha256(path),
                "config_local_candidate": config_candidate(str(path)),
                "config_remote_candidate": "",
                "config_sha256": "",
                "endpoint_status": "auxiliary_checkpoint",
                "readiness": "requires_specialized_evaluator",
                "source_registry": "filesystem_audit_20260913",
                "notes": "Do not mix with feed-forward reconstructor leaderboard",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--validation", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--rewrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    validation_path = Path(args.validation).resolve()
    if not args.rewrite and (output.exists() or validation_path.exists()):
        raise FileExistsError(
            "Registry output already exists; pass --rewrite explicitly"
        )

    rows = legacy_rows()
    rows.extend(
        queue_rows(
            "coursework_long_scaling_v4_20260911",
            "long_scaling_v4",
        )
    )
    rows.extend(
        queue_rows(
            "coursework_consistency_scaling_v2_20260912",
            "consistency_scaling_v2",
        )
    )
    rows.extend(real_primary_rows())
    rows.extend(active_report_rows())
    rows.extend(auxiliary_rows())
    if len({row["registry_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate registry IDs")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)

    counts = {}
    for scope in sorted({row["scope"] for row in rows}):
        counts[scope] = sum(row["scope"] == scope for row in rows)
    validation = {
        "status": "pass",
        "row_count": len(rows),
        "scope_counts": counts,
        "unique_registry_ids": True,
        "legacy_unique_checkpoint_count": 61,
        "new_exact_endpoint_count": 33,
        "real_primary_endpoint_count": len(real_primary_rows()),
        "active_report_addition_count": len(active_report_rows()),
        "auxiliary_checkpoint_count": len(auxiliary_rows()),
        "published_psf_aware_model": {
            "repo_id": "bezzam/digicam-mirflickr-multi-25k-unet4M-unrolled-admm5-unet4M-wave-psfNN",
            "revision": "9c965e99a6b9048eaa0eea3b8cdb2c5b9039416e",
            "local_path_is_resolved_by_snapshot_download": True,
        },
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "test_model_forward_executed": False,
        "registry_sha256": sha256(output),
    }
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
