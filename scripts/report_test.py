"""Prepare and execute the report-wide reconstruction evaluation.

The existing ``inference.py`` remains the execution engine.  This module only
freezes registry rows, validates exact checkpoints, materializes per-run Hydra
configs and later orchestrates/aggregates those runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT.parent / "plans/all_test_checkpoint_registry_20260913.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/coursework_report_test_v1_20260913"
DEFAULT_REMOTE_ROOT = Path(
    "/home/hadhad/project/Lensless-imaging-report-test-20260913/"
    "outputs/coursework_report_test_v1_20260913"
)

PROGRAM_PATHS = {
    "orchestrator": "scripts/report_test.py",
    "inference": "inference.py",
    "inferencer": "src/trainer/inferencer.py",
    "base_trainer": "src/trainer/base_trainer.py",
    "data_utils": "src/datasets/data_utils.py",
    "dataset_split_manifest": "src/datasets/split_manifest.py",
    "mirflickr_scene_manifest": "manifests/mirflickr25k_splits.json",
    "mnist_scene_manifest": "manifests/mnist_splits.json",
    "celeba_scene_manifest": "manifests/celeba_splits.json",
    "on_the_fly": "src/datasets/on_the_fly.py",
    "metrics": "src/metrics/reconstruction.py",
    "metrics_init": "src/metrics/__init__.py",
    "mnist_metrics": "src/metrics/mnist.py",
    "grid_metrics": "src/metrics/grid_artifacts.py",
    "mirflickr_dataset": "src/datasets/mirflickr.py",
    "mnist_dataset": "src/datasets/mnist.py",
    "celeba_dataset": "src/datasets/celeba.py",
    "digicam_dataset": "src/datasets/digicam.py",
    "drunet_model": "src/model/psf_free_drunet.py",
    "psf_aware_drunet_model": "src/model/psf_aware_drunet.py",
    "xrestormer_model": "src/model/psf_free_xrestormer.py",
    "sdvae_model": "src/model/psf_free_autoencoder_kl.py",
    "ssl_model": "src/model/psf_free_vit_ssl.py",
}

STANDARD_MODELS = {
    "mirflickr": {
        "DRUNet",
        "Large DRUNet",
        "X-Restormer",
        "SD-VAE",
        "DINOv2",
        "LingBot",
    },
    "mnist": {
        "DRUNet",
        "DRUNet RGB",
        "DRUNet grayscale",
        "PSF-aware DRUNet",
    },
    "celeba32": {"DRUNet", "X-Restormer"},
    "digicam real": {"DRUNet"},
}

REUSED_REGISTRY_IDS = {"legacy::E2-mirflickr-xrest100k-seed42"}

LEGACY_UNFLIPPED_REGISTRY_IDS = {
    "active_report::dr_small_10k_finite100_seed42",
    "active_report::dr_small_10k_finite100_seed52",
    "active_report::dr_small_10k_finite100_seed62",
    "active_report::dr_small_10k_streaming_seed42",
    "active_report::dr_small_10k_streaming_seed52",
    "active_report::dr_small_10k_streaming_seed62",
    "active_report::mnist_rgb_10k_seed42",
    "active_report::mnist_rgb_10k_seed52",
    "active_report::mnist_rgb_10k_seed62",
}

REMOTE_SUPPORT_PROGRAMS = {
    "ref_psff_real_dataset": {
        "path": "src/datasets/digicam_matched.py",
        "sha256": "b9049e6458acb2e2d645d2a58c7833812389e057f37b3fcf40444e768b10f59b",
    },
    "ref_psff_real_contract": {
        "path": "src/ref_psff_real.py",
        "sha256": "fe027cd258eb36aaf0429547c9153fde8d710d5570a0159b6b12b64cb183f934",
    },
    "digicam_protocol": {
        "path": "src/digicam_protocol.py",
        "sha256": "02ae71957c78587338d82485c7c439dd7fe13421a34aae477911b0abcb010c7f",
    },
    "remote_datasets_init": {
        "path": "src/datasets/__init__.py",
        "sha256": "d6f6db5cc8e0c2558b9c4f97eb810c9076ce263f7c29c5b8c9b59c6f78087ba2",
    },
    "ref_psff_real_manifest": {
        "path": "manifests/ref_psff_real_68_v1.json",
        "sha256": "d9817c152c75a009062e2c3ef7b4c996218b8bbbda7307fbda402e19f04c9069",
    },
    "adapter_training_contract": {
        "path": "run_tools/train_sim_real_adapter.py",
        "sha256": "72272ef88d3fbc6ac7463e154e4708f3335fd18d242ca46873a14fec62a76812",
    },
    "adapter_transfer_evaluator": {
        "path": "run_tools/evaluate_sim_real_adapter_transfer.py",
        "sha256": "1a9e68f2582f7bf52eab7976d5a7ac27648b92b80f871fc0f975df3a526f8dc7",
    },
    "adapter_model_loader": {
        "path": "run_tools/sim_real_reconstruction_screen.py",
        "sha256": "8e9f58ff16d623a828f009cbdd5393b50ef2d7eaccb04e8618f22427562a7a01",
    },
}

TEST_CONTRACTS = {
    "mirflickr": {
        "scene_split": "test",
        "scene_count": 256,
        "scenes_per_mask": 256,
        "mask_split": "test",
        "mask_count": 100,
        "samples": 25_600,
        "target_shape": [3, 200, 266],
        "roi": [80, 100, 200, 266],
    },
    "mnist": {
        "scene_split": "test",
        "scene_count": 10_000,
        "scenes_per_mask": 10_000,
        "mask_split": "test",
        "mask_count": 32,
        "samples": 320_000,
        "target_shape_rgb": [3, 256, 256],
        "target_shape_grayscale": [1, 256, 256],
        "roi": [62, 125, 256, 256],
    },
    "celeba32": {
        "scene_split": "test",
        "scene_count": 256,
        "scenes_per_mask": 256,
        "mask_split": "test",
        "mask_count": 32,
        "samples": 8_192,
        "target_shape": [3, 256, 256],
        "roi": [62, 125, 256, 256],
    },
    "digicam real": {
        "scene_split": "test",
        "scene_count": 3_750,
        "scenes_per_mask": 250,
        "mask_count": 15,
        "samples": 3_750,
        "target_shape": [3, 200, 266],
        "roi": [80, 100, 200, 266],
        "dataset_revision": "21d82b67662ed1e590a40c98688c32cb3c74f079",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def canonical_dataset(value: str) -> str:
    name = value.strip().lower()
    aliases = {
        "mirflickr": "mirflickr",
        "mnist": "mnist",
        "celeba32": "celeba32",
        "digicam real": "digicam real",
    }
    return aliases.get(name, name)


def is_standard(row: dict[str, str]) -> bool:
    if row["registry_id"] in REUSED_REGISTRY_IDS:
        return False
    dataset = canonical_dataset(row["dataset"])
    return row["model"] in STANDARD_MODELS.get(dataset, set())


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result or "entry"


def entry_priority(row: dict[str, str]) -> tuple[int, int, str]:
    scope_priority = {
        "consistency_scaling_v2": 0,
        "long_scaling_v4": 1,
        "real_primary_matched": 2,
        "active_report_addition": 3,
        "legacy_report_inventory": 4,
    }
    return (
        0 if clean(row.get("config_sha256")) else 1,
        scope_priority.get(row["scope"], 9),
        row["registry_id"],
    )


def training_scene_count(row: dict[str, str]) -> int | None:
    match = re.search(r"scene(4096|16384)", row["registry_id"])
    if match:
        return int(match.group(1))
    if canonical_dataset(row["dataset"]) == "mirflickr" and row["model"] in {
        "DRUNet",
        "Large DRUNet",
        "X-Restormer",
        "SD-VAE",
        "DINOv2",
        "LingBot",
    }:
        return 4096
    return None


def batch_size(row: dict[str, str]) -> int:
    if row["model"] in {"X-Restormer", "SD-VAE"}:
        return 1
    return 4


def initialization(row: dict[str, str]) -> str:
    text = " ".join(str(value).lower() for value in row.values())
    if "completion-e3-celeba" in text:
        return "random"
    if "completion-e4-celeba" in text or "gopro" in text:
        return "gopro"
    if any(token in text for token in ("dpir", "denoising")):
        return "denoising"
    if "pretrained" in text:
        return "pretrained"
    if "xrest-l1-" in text:
        return "gopro"
    if any(token in text for token in ("random", "scratch")):
        return "random"
    if row["model"] in {
        "DRUNet",
        "DRUNet RGB",
        "DRUNet grayscale",
        "Large DRUNet",
        "PSF-aware DRUNet",
    }:
        return "random"
    return "unspecified"


def variant(row: dict[str, str]) -> str:
    identifier = row["registry_id"].lower()
    if "completion-e3-celeba" in identifier:
        return "finite100_random"
    if "completion-e4-celeba" in identifier:
        return "finite100_gopro_l1_from_start"
    if "xrest-l1-0.1" in identifier:
        return "continuation_l1_0.1"
    if "xrest-l1-1.0" in identifier:
        return "continuation_l1_1.0"
    return clean(row.get("condition")) or "default"


def make_standard_entry(
    rows: list[dict[str, str]], audit: dict[str, dict[str, str]]
) -> dict:
    representative = min(rows, key=entry_priority)
    dataset = canonical_dataset(representative["dataset"])
    seed = clean(representative.get("seed"))
    steps = clean(representative.get("steps"))
    condition = clean(representative.get("condition")) or "unspecified"
    identifier = "-".join(
        [
            slug(dataset),
            slug(representative["model"]),
            slug(condition),
            f"seed{int(float(seed))}" if seed is not None else "noseed",
            f"step{int(float(steps))}" if steps is not None else "nostep",
            representative["checkpoint_sha256"][:12],
        ]
    )
    audit_rows = [
        audit[row["registry_id"]] for row in rows if row["registry_id"] in audit
    ]
    audited = next(
        (
            value
            for value in audit_rows
            if clean(value.get("remote_global_step")) is not None
        ),
        audit_rows[0] if audit_rows else {},
    )
    return {
        "id": identifier,
        "registry_id": representative["registry_id"],
        "registry_aliases": sorted(row["registry_id"] for row in rows),
        "scope": representative["scope"],
        "report_roles": sorted({row["report_role"] for row in rows}),
        "dataset": dataset,
        "model": representative["model"],
        "condition": condition,
        "initialization": initialization(representative),
        "variant": variant(representative),
        "seed": int(float(seed)) if seed is not None else None,
        "steps": int(float(steps)) if steps is not None else None,
        "training_scenes": training_scene_count(representative),
        "requires_psf": representative["model"] == "PSF-aware DRUNet",
        "channels": 1 if representative["model"] == "DRUNet grayscale" else 3,
        "batch_size": batch_size(representative),
        "checkpoint_sha256": representative["checkpoint_sha256"],
        "checkpoint_local": clean(representative.get("checkpoint_local")),
        "checkpoint_remote": clean(representative.get("checkpoint_remote")),
        "config_local_candidate": clean(representative.get("config_local_candidate")),
        "config_remote_candidate": clean(representative.get("config_remote_candidate")),
        "config_sha256": clean(representative.get("config_sha256")),
        "endpoint_status": representative["endpoint_status"],
        "readiness": representative["readiness"],
        "audited_metadata": {
            key.removeprefix("remote_"): (
                int(float(audited[key]))
                if clean(audited.get(key)) is not None
                else None
            )
            for key in (
                "remote_epoch",
                "remote_global_step",
                "remote_sampler_step",
                "remote_T_max",
            )
        },
        "test_contract": TEST_CONTRACTS[dataset],
    }


def reference_metrics(path: Path) -> dict[str, float]:
    import pandas as pd

    frame = pd.read_csv(path)
    ignored = {"mask_id", "sample_count", "sample_index", "label"}
    metrics = {}
    for column in frame.columns:
        if column in ignored:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().all():
            metrics[column] = float(values.mean())
    if not metrics:
        raise ValueError(f"no metric columns in development reference: {path}")
    return metrics


def reference_index() -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for path in REPO_ROOT.glob("**/validation_per_mask_epoch*.csv"):
        result.setdefault(path.parent.name, []).append(path.resolve())
    return result


def attach_development_reference(entry: dict, index: dict[str, list[Path]]) -> None:
    if entry["dataset"] == "digicam real":
        path = (
            REPO_ROOT
            / "outputs/ref-psff-real-68/evaluation"
            / f"seed{entry['seed']}"
            / "per_row.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        import pandas as pd

        frame = pd.read_csv(path)
        suffix = "real" if entry["condition"] == "real" else "matched_sim"
        metrics = {}
        for metric in ("PSNR", "SSIM", "LPIPS"):
            column = f"{metric}_{suffix}"
            per_mask = frame.groupby("mask_id", sort=True)[column].mean()
            metrics[metric] = float(per_mask.mean())
        entry["development_reference"] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "metrics": metrics,
            "kind": "real_per_row_mask_balanced",
        }
        return

    names = {
        Path(value).parent.name
        for value in (entry.get("checkpoint_local"), entry.get("checkpoint_remote"))
        if value
    }
    candidates = sorted({path for name in names for path in index.get(name, [])})
    epoch = entry["audited_metadata"].get("epoch")
    if epoch is None:
        match = re.search(
            r"checkpoint-epoch(\d+)\.pth$", entry["checkpoint_remote"] or ""
        )
        if match:
            epoch = int(match.group(1))
    if epoch is not None:
        expected_name = f"validation_per_mask_epoch{int(epoch):04d}.csv"
        candidates = [path for path in candidates if path.name == expected_name]
    elif candidates:
        maximum = max(
            int(re.search(r"epoch(\d+)", path.name).group(1)) for path in candidates
        )
        candidates = [path for path in candidates if f"epoch{maximum:04d}" in path.name]
    if not candidates:
        raise FileNotFoundError(f"development reference missing for {entry['id']}")
    hashes = {sha256(path) for path in candidates}
    if len(hashes) != 1:
        raise ValueError(f"development reference copies disagree for {entry['id']}")
    path = min(candidates, key=lambda value: (len(str(value)), str(value)))
    entry["development_reference"] = {
        "path": str(path),
        "sha256": hashes.pop(),
        "metrics": reference_metrics(path),
        "kind": "training_validation_per_mask",
    }


def assign_remote_paths(
    entries: list[dict],
    specialized: list[dict],
    registry: dict[str, dict[str, str]],
    audit: dict[str, dict[str, str]],
    remote_root: Path,
) -> list[dict]:
    staging = {}
    for entry in [*entries, *specialized]:
        aliases = entry.get("registry_aliases", [entry["registry_id"]])
        remote = None
        for identifier in aliases:
            audited = audit.get(identifier, {})
            source = registry.get(identifier, {})
            if str(audited.get("remote_sha256_match", "")).lower() == "true":
                remote = clean(source.get("checkpoint_remote"))
                if remote:
                    break
        if remote:
            entry["checkpoint_remote"] = remote
            entry["remote_source"] = "existing_exact"
            continue
        local = clean(entry.get("checkpoint_local"))
        if local is None or not Path(local).is_file():
            raise FileNotFoundError(
                f"no exact local checkpoint for {entry['registry_id']}"
            )
        destination = remote_root / "weights" / f"{entry['checkpoint_sha256']}.pth"
        entry["checkpoint_remote"] = str(destination)
        entry["remote_source"] = "content_addressed_upload"
        staging.setdefault(
            entry["checkpoint_sha256"],
            {
                "source": str(Path(local).resolve()),
                "destination": str(destination),
                "sha256": entry["checkpoint_sha256"],
                "size_bytes": Path(local).stat().st_size,
            },
        )
    return sorted(staging.values(), key=lambda value: value["sha256"])


def write_manifest_review(output: Path, manifest: dict) -> None:
    counts = Counter(
        (entry["dataset"], entry["model"]) for entry in manifest["entries"]
    )
    lines = [
        "# Report-wide test launch manifest",
        "",
        "Status: prepared for metadata preflight. No test sample has been loaded and no test model forward has run.",
        "",
        "## Standard reconstruction entries",
        "",
        "| Dataset | Model | Checkpoints |",
        "|---|---|---:|",
    ]
    for (dataset, model), count in sorted(counts.items()):
        lines.append(f"| {dataset} | {model} | {count} |")
    lines.extend(
        [
            "",
            f"Total standard entries: {manifest['standard_entry_count']}.",
            "",
            "## Test grids",
            "",
            "| Dataset | Masks | Scenes per mask | Pairs per checkpoint |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset, contract in manifest["test_contracts"].items():
        lines.append(
            f"| {dataset} | {contract['mask_count']} | {contract['scenes_per_mask']} | {contract['samples']} |"
        )
    lines.extend(
        [
            "",
            "## Staging",
            "",
            f"- Content-addressed uploads before remote search: {manifest['weight_upload_count']} files.",
            f"- Maximum upload volume before hard-link reuse: {manifest['weight_upload_bytes']} bytes.",
            f"- Specialized weights/dependencies: {manifest['specialized_entry_count']}.",
            f"- Registry rows outside the active report matrix: {len(manifest['ignored_registry_ids'])}.",
            "",
            "## Required gates",
            "",
            "1. All checkpoint hashes and endpoint metadata must pass on the execution host.",
            "2. Every standard entry must reproduce its frozen development reference within the declared tolerance.",
            "3. Test configs and this manifest must remain unchanged after parity.",
            "4. Test launch requires a separate authorization JSON bound to the manifest, config manifest and parity hashes.",
        ]
    )
    (output / "MANIFEST_REVIEW.md").write_text("\n".join(lines) + "\n")


def prepare(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry).expanduser().resolve()
    audit_path = Path(args.audit).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.rewrite:
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    output.mkdir(parents=True, exist_ok=True)

    with registry_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with audit_path.open(newline="") as stream:
        audit_rows = list(csv.DictReader(stream))
    audit = {row["registry_id"]: row for row in audit_rows}
    registry = {row["registry_id"]: row for row in rows}
    if not rows or len({row["registry_id"] for row in rows}) != len(rows):
        raise ValueError("registry IDs are empty or duplicated")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    classified = {}
    for row in rows:
        if row["registry_id"] in REUSED_REGISTRY_IDS:
            classified[row["registry_id"]] = "reused_result"
        elif is_standard(row):
            key = (canonical_dataset(row["dataset"]), row["checkpoint_sha256"])
            grouped.setdefault(key, []).append(row)
            classified[row["registry_id"]] = "standard"
        elif (
            row["scope"] == "auxiliary_report_method"
            or row["model"]
            in {
                "Input adapter",
                "Adapter base X-Restormer",
                "Compatibility model",
                "Residual refinement U-Net",
                "PSF-conditioned DRUNet",
                "Frozen classifier",
            }
            or (
                row["model"] == "X-Restormer"
                and canonical_dataset(row["dataset"])
                in {"real", "digicam-real-development"}
            )
        ):
            classified[row["registry_id"]] = "specialized"
        else:
            classified[row["registry_id"]] = "not_in_active_test_matrix"

    entries = sorted(
        (make_standard_entry(group, audit) for group in grouped.values()),
        key=lambda item: (
            item["dataset"],
            item["model"],
            item["condition"],
            item["steps"] or -1,
            item["seed"] or -1,
        ),
    )
    index = reference_index()
    for entry in entries:
        attach_development_reference(entry, index)
    specialized = [
        {
            "registry_id": row["registry_id"],
            "dataset": canonical_dataset(row["dataset"]),
            "model": row["model"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_local": clean(row.get("checkpoint_local")),
            "checkpoint_remote": clean(row.get("checkpoint_remote")),
            "readiness": row["readiness"],
        }
        for row in rows
        if classified[row["registry_id"]] == "specialized"
    ]
    ignored = [
        row["registry_id"]
        for row in rows
        if classified[row["registry_id"]] == "not_in_active_test_matrix"
    ]
    reused_registry_ids = [
        row["registry_id"]
        for row in rows
        if classified[row["registry_id"]] == "reused_result"
    ]
    remote_root = Path(args.remote_root)
    staging = assign_remote_paths(
        entries,
        specialized,
        registry,
        audit,
        remote_root,
    )
    program_hashes = {
        name: sha256(REPO_ROOT / relative) for name, relative in PROGRAM_PATHS.items()
    }
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    frozen_registry = inputs / "registry.csv"
    frozen_audit = inputs / "registry.audit.csv"
    frozen_report = inputs / "current_report.tex"
    shutil.copy2(registry_path, frozen_registry)
    shutil.copy2(audit_path, frozen_audit)
    shutil.copy2(Path(args.report).expanduser().resolve(), frozen_report)
    reused_inputs = inputs / "reused"
    reused_inputs.mkdir(parents=True, exist_ok=True)
    reused_sources = {
        "xrestormer_model_summary": (
            REPO_ROOT
            / "outputs/coursework_final_runner_v4_20260910/final_results_v1/model_summary.csv"
        ),
        "xrestormer_paired_effects": (
            REPO_ROOT
            / "outputs/coursework_final_runner_v4_20260910/final_results_v1/paired_effects_aggregate.csv"
        ),
        "xrestormer_validation": (
            REPO_ROOT
            / "outputs/coursework_final_runner_v4_20260910/final_results_v1/validation.json"
        ),
        "published_psf_aware_real": (
            REPO_ROOT / "saved/ref-psf-aware-real-cpu-20260820-v2/test/summary.json"
        ),
    }
    reused_results = {}
    for name, source in reused_sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = reused_inputs / source.name
        if target.exists() and target.name == "summary.json":
            target = reused_inputs / "published_psf_aware_real_summary.json"
        shutil.copy2(source, target)
        reused_results[name] = {
            "path": str(target.relative_to(output)),
            "sha256": sha256(target),
        }
    manifest = {
        "schema_version": 1,
        "status": "prepared_preflight",
        "purpose": "uniform report-wide test inference using inference.py",
        "registry": "inputs/registry.csv",
        "registry_sha256": sha256(frozen_registry),
        "registry_audit": "inputs/registry.audit.csv",
        "registry_audit_sha256": sha256(frozen_audit),
        "report": "inputs/current_report.tex",
        "report_sha256": sha256(frozen_report),
        "program_paths": PROGRAM_PATHS,
        "program_hashes": program_hashes,
        "remote_support_programs": REMOTE_SUPPORT_PROGRAMS,
        "standard_entry_count": len(entries),
        "specialized_entry_count": len(specialized),
        "registry_row_count": len(rows),
        "classification_counts": dict(sorted(Counter(classified.values()).items())),
        "remote_root": str(remote_root),
        "weight_upload_count": len(staging),
        "weight_upload_bytes": sum(item["size_bytes"] for item in staging),
        "test_contracts": TEST_CONTRACTS,
        "development_parity_tolerances": {
            "PSNR": 0.02,
            "SSIM": 0.002,
            "LPIPS": 0.002,
            "PSNR_32": 0.02,
            "SSIM_32": 0.002,
            "PSNR_coarse": 0.02,
            "SSIM_coarse": 0.002,
            "Dice_loss_32": 0.002,
            "block_residual_rmse": 0.001,
            "periodic16_residual_rmse": 0.001,
        },
        "entries": entries,
        "specialized_entries": specialized,
        "ignored_registry_ids": ignored,
        "reused_registry_ids": reused_registry_ids,
        "reused_results": reused_results,
        "test_model_forward_executed": False,
        "authorization_required": True,
    }
    staging_path = output / "staging.json"
    json_write(
        staging_path,
        {
            "status": "prepared",
            "remote_root": str(remote_root),
            "entry_count": len(staging),
            "size_bytes": sum(item["size_bytes"] for item in staging),
            "entries": staging,
        },
    )
    manifest["staging"] = "staging.json"
    manifest["staging_sha256"] = sha256(staging_path)
    json_write(manifest_path, manifest)
    write_manifest_review(output, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str(manifest_path),
                "registry_rows": len(rows),
                "standard_entries": len(entries),
                "specialized_entries": len(specialized),
                "ignored_rows": len(ignored),
                "weight_upload_count": len(staging),
                "weight_upload_bytes": sum(item["size_bytes"] for item in staging),
            },
            indent=2,
        )
    )


def resolve_candidate(entry: dict, location: str, kind: str) -> Path | None:
    value = entry.get(f"{kind}_{location}")
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.exists() else path


def load_checkpoint(path: Path):
    import torch

    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except (RuntimeError, TypeError, ValueError):
        return torch.load(path, **kwargs)


def checkpoint_metadata(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("checkpoint must be a dictionary")
    state = value.get("state_dict", value)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint does not contain a state dict")
    config = value.get("config")
    model_target = None
    if config is not None:
        try:
            model_target = str(config["model"]["_target_"])
        except (KeyError, TypeError):
            model_target = None
    return {
        "epoch": value.get("epoch"),
        "global_step": value.get("global_step"),
        "sampler_step": value.get("sampler_step"),
        "T_max": (
            value.get("lr_scheduler", {}).get("T_max")
            if isinstance(value.get("lr_scheduler"), dict)
            else None
        ),
        "state_key_count": len(state),
        "embedded_config": config is not None,
        "model_target": model_target,
    }


def validate_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "prepared_preflight":
        raise ValueError("manifest is not in prepared_preflight state")
    registry = manifest_path.parent / manifest["registry"]
    if not registry.is_file() or sha256(registry) != manifest["registry_sha256"]:
        raise ValueError("registry hash drift")
    audit = manifest_path.parent / manifest["registry_audit"]
    if not audit.is_file() or sha256(audit) != manifest["registry_audit_sha256"]:
        raise ValueError("registry audit hash drift")
    report = manifest_path.parent / manifest["report"]
    if not report.is_file() or sha256(report) != manifest["report_sha256"]:
        raise ValueError("report source hash drift")
    staging = manifest_path.parent / manifest["staging"]
    if not staging.is_file() or sha256(staging) != manifest["staging_sha256"]:
        raise ValueError("staging plan hash drift")
    for name, value in manifest["reused_results"].items():
        path = manifest_path.parent / value["path"]
        if not path.is_file() or sha256(path) != value["sha256"]:
            raise ValueError(f"reused result hash drift: {name}")
    for name, relative in manifest["program_paths"].items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256(path) != manifest["program_hashes"][name]:
            raise ValueError(f"program hash drift: {name}")
    return manifest


def preflight(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = validate_manifest(manifest_path)
    rows = []
    for entry in manifest["entries"]:
        path = resolve_candidate(entry, args.location, "checkpoint")
        row = {
            "id": entry["id"],
            "checkpoint": str(path) if path else None,
            "exists": bool(path and path.is_file()),
            "sha256_match": False,
            "metadata": None,
            "errors": [],
        }
        if not row["exists"]:
            row["errors"].append("checkpoint_missing")
            rows.append(row)
            continue
        actual = sha256(path)
        row["actual_sha256"] = actual
        row["sha256_match"] = actual == entry["checkpoint_sha256"]
        if not row["sha256_match"]:
            row["errors"].append("checkpoint_sha256_mismatch")
            rows.append(row)
            continue
        if args.deserialize:
            try:
                metadata = checkpoint_metadata(load_checkpoint(path))
                row["metadata"] = metadata
                expected_steps = entry.get("steps")
                actual_step = metadata.get("global_step")
                if expected_steps is not None and actual_step is not None:
                    if int(actual_step) != int(expected_steps):
                        row["errors"].append("global_step_mismatch")
                if not metadata["embedded_config"]:
                    config_path = resolve_candidate(entry, args.location, "config")
                    if config_path is None or not config_path.is_file():
                        row["errors"].append("config_missing")
            except Exception as error:  # metadata errors belong in the artifact
                row["errors"].append(f"checkpoint_load:{type(error).__name__}:{error}")
        rows.append(row)

    dependencies = []
    for entry in manifest["specialized_entries"]:
        path = resolve_candidate(entry, args.location, "checkpoint")
        row = {
            "registry_id": entry["registry_id"],
            "model": entry["model"],
            "checkpoint": str(path) if path else None,
            "exists": bool(path and path.is_file()),
            "sha256_match": False,
            "errors": [],
        }
        if not row["exists"]:
            row["errors"].append("checkpoint_missing")
        else:
            actual = sha256(path)
            row["actual_sha256"] = actual
            row["sha256_match"] = actual == entry["checkpoint_sha256"]
            if not row["sha256_match"]:
                row["errors"].append("checkpoint_sha256_mismatch")
        dependencies.append(row)

    support_programs = []
    if args.location == "remote":
        for name, expected in manifest["remote_support_programs"].items():
            path = REPO_ROOT / expected["path"]
            actual = sha256(path) if path.is_file() else None
            support_programs.append(
                {
                    "name": name,
                    "path": str(path),
                    "exists": path.is_file(),
                    "sha256_match": actual == expected["sha256"],
                    "actual_sha256": actual,
                }
            )

    passed = sum(not row["errors"] for row in rows)
    dependency_passed = sum(not row["errors"] for row in dependencies)
    support_passed = sum(row["sha256_match"] for row in support_programs)
    complete = (
        passed == len(rows)
        and dependency_passed == len(dependencies)
        and support_passed == len(support_programs)
    )
    result = {
        "status": "pass" if complete else "incomplete",
        "metadata_only": True,
        "location": args.location,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "entry_count": len(rows),
        "passed_count": passed,
        "failed_count": len(rows) - passed,
        "dependency_count": len(dependencies),
        "dependency_passed_count": dependency_passed,
        "dependency_failed_count": len(dependencies) - dependency_passed,
        "support_program_count": len(support_programs),
        "support_program_passed_count": support_passed,
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "test_model_forward_executed": False,
        "entries": rows,
        "dependencies": dependencies,
        "support_programs": support_programs,
    }
    output = manifest_path.parent / f"preflight_{args.location}.json"
    json_write(output, result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"entries", "dependencies"}
            },
            indent=2,
        )
    )
    if result["status"] != "pass" and args.require_all:
        raise SystemExit(2)


def resolve_staging(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = validate_manifest(manifest_path)
    staging_path = manifest_path.parent / manifest["staging"]
    staging = json.loads(staging_path.read_text())
    unresolved = {entry["sha256"]: entry for entry in staging["entries"]}
    resolved = []
    for digest, entry in list(unresolved.items()):
        destination = Path(entry["destination"])
        if destination.is_file() and sha256(destination) == digest:
            resolved.append(
                {**entry, "candidate": str(destination), "action": "already_present"}
            )
            del unresolved[digest]

    hints: dict[str, list[dict]] = {}
    for entry in unresolved.values():
        hints.setdefault(Path(entry["source"]).parent.name, []).append(entry)
    for root, directories, files in os.walk(Path(args.search_root)):
        name = Path(root).name
        if name not in hints:
            continue
        filenames = set(files)
        for entry in list(hints[name]):
            source_name = Path(entry["source"]).name
            if source_name not in filenames:
                continue
            candidate = Path(root) / source_name
            try:
                size = candidate.stat().st_size
            except OSError:
                continue
            if size != int(entry["size_bytes"]):
                continue
            try:
                actual_hash = sha256(candidate)
            except OSError:
                continue
            if actual_hash != entry["sha256"]:
                continue
            action = "found"
            if args.link:
                destination = Path(entry["destination"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(destination)
                os.link(candidate, destination)
                action = "hard_linked"
            resolved.append({**entry, "candidate": str(candidate), "action": action})
            del unresolved[entry["sha256"]]
            hints[name].remove(entry)
        if not unresolved:
            break
    result = {
        "status": "pass" if not unresolved else "incomplete",
        "manifest_sha256": sha256(manifest_path),
        "searched_root": str(Path(args.search_root).resolve()),
        "link_requested": bool(args.link),
        "resolved_count": len(resolved),
        "missing_count": len(unresolved),
        "resolved": sorted(resolved, key=lambda value: value["sha256"]),
        "missing": sorted(unresolved.values(), key=lambda value: value["sha256"]),
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "test_model_forward_executed": False,
    }
    output = manifest_path.parent / "remote_staging_resolution.json"
    json_write(output, result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"resolved", "missing"}
            },
            indent=2,
        )
    )


def embedded_config(value, fallback: Path | None):
    from omegaconf import OmegaConf

    config = value.get("config") if isinstance(value, dict) else None
    if config is not None:
        config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    elif fallback is not None and fallback.is_file():
        config = OmegaConf.load(fallback)
    else:
        raise ValueError("checkpoint has no embedded config or exact config candidate")
    if "model" in config and "checkpoint_path" in config.model:
        config.model.checkpoint_path = None
    return config


def metric_names(metrics) -> set[str]:
    return {str(metric.get("name")) for metric in metrics if metric.get("name")}


def add_metric(metrics, value: dict) -> None:
    if value["name"] not in metric_names(metrics):
        metrics.append(value)


def normalize_model_target(model: dict) -> dict:
    aliases = {
        "src.model.PSFFreeDRUNet": ("src.model.psf_free_drunet.PSFFreeDRUNet"),
    }
    target = model.get("_target_")
    if target in aliases:
        model["_target_"] = aliases[target]
    return model


def augment_metrics(
    config, entry: dict, classifier: Path | None, classifier_hash: str | None
):
    from omegaconf import OmegaConf

    metrics = OmegaConf.to_container(config.metrics, resolve=True)
    metrics["device"] = "auto"
    inference = list(metrics.get("inference", []))
    dataset = entry["dataset"]
    if dataset == "mnist":
        add_metric(
            inference,
            {
                "_target_": "src.metrics.PooledDiceLossMetric",
                "pooling_factor": 8,
                "eps": 1e-8,
                "name": "Dice_loss_32",
            },
        )
        if entry["channels"] == 3:
            if classifier is None or classifier_hash is None:
                raise ValueError("RGB MNIST entries require the frozen classifier")
            common = {
                "_target_": "src.metrics.FixedMNISTClassifierAccuracyMetric",
                "checkpoint_path": str(classifier),
                "expected_sha256": classifier_hash,
                "pooling_factor": 8,
            }
            add_metric(
                inference,
                {**common, "use_target": False, "name": "Classifier_accuracy"},
            )
            add_metric(
                inference,
                {
                    **common,
                    "use_target": True,
                    "name": "Target_classifier_accuracy",
                },
            )
    elif dataset == "celeba32":
        for value in (
            {
                "_target_": "src.metrics.LPIPSMetric",
                "net_type": "vgg",
                "device": "auto",
                "normalize_by_max": False,
                "name": "LPIPS",
            },
            {
                "_target_": "src.metrics.PooledPSNRMetric",
                "pooling_factor": 8,
                "normalize_by_max": False,
                "name": "PSNR_32",
            },
            {
                "_target_": "src.metrics.PooledSSIMMetric",
                "pooling_factor": 8,
                "normalize_by_max": False,
                "name": "SSIM_32",
            },
            {
                "_target_": "src.metrics.grid_artifacts.BlockResidualRMSE",
                "block_size": 8,
                "name": "Within_block_RMSE",
            },
            {
                "_target_": "src.metrics.grid_artifacts.PeriodicResidualRMSE",
                "period": 16,
                "name": "Periodic_16_RMSE",
            },
        ):
            add_metric(inference, value)
    metrics["inference"] = inference
    return metrics


def inference_amp(source) -> dict:
    from omegaconf import OmegaConf

    trainer = source.get("trainer")
    if trainer is None or trainer.get("amp") is None:
        return {"enabled": False}
    return OmegaConf.to_container(trainer.amp, resolve=True)


def absolute_data_paths(dataset: str, value: dict, mode: str) -> dict:
    result = dict(value)
    target_aliases = {
        "src.datasets.MirFlickrSceneDataset": (
            "src.datasets.mirflickr.MirFlickrSceneDataset"
        ),
        "src.datasets.MNISTSceneDataset": "src.datasets.mnist.MNISTSceneDataset",
        "src.datasets.CelebASceneDataset": "src.datasets.celeba.CelebASceneDataset",
    }
    if result.get("_target_") in target_aliases:
        result["_target_"] = target_aliases[result["_target_"]]
    result["split"] = "validation" if mode == "development" else "test"
    if dataset == "mirflickr":
        result["root_dir"] = str(REPO_ROOT / "data/raw/mirflickr25k/extracted")
        result["splits_path"] = str(REPO_ROOT / "manifests/mirflickr25k_splits.json")
    elif dataset == "mnist":
        result["root_dir"] = str(REPO_ROOT / "data/raw/mnist")
        if mode == "test" or "splits_path" in result:
            result["splits_path"] = str(REPO_ROOT / "manifests/mnist_splits.json")
        result["download"] = False
        result["max_samples"] = None
    elif dataset == "celeba32":
        result["root_dir"] = str(REPO_ROOT / "data/raw/celeba")
        result["splits_path"] = str(REPO_ROOT / "manifests/celeba_splits.json")
        result["max_samples"] = 256 if mode == "test" else None
    return result


def synthetic_config(
    source,
    entry: dict,
    mode: str,
    checkpoint: Path,
    save_path: Path,
    classifier: Path | None,
    classifier_hash: str | None,
):
    from omegaconf import OmegaConf

    dataset = entry["dataset"]
    source_dataset = OmegaConf.to_container(source.datasets.validation, resolve=True)
    source_builder = OmegaConf.to_container(source.dataloader_builder, resolve=True)
    if source_builder.get("_target_") == "src.datasets.build_on_the_fly_dataloaders":
        source_builder["_target_"] = (
            "src.datasets.on_the_fly.build_on_the_fly_dataloaders"
        )
    contract = entry["test_contract"]
    if mode == "development":
        scene_count = int(source_builder.get("validation_scenes_per_mask", 32))
        mask_count = int(source_builder.get("validation_mask_count", 32))
        mask_split = str(source_builder.get("validation_mask_split", "validation"))
    else:
        scene_count = int(contract["scene_count"])
        mask_count = int(contract["mask_count"])
        mask_split = str(contract["mask_split"])

    source_builder.update(
        {
            "evaluation_only": True,
            "validation_mask_split": mask_split,
            "allow_test": mode == "test",
            "validation_mask_count": mask_count,
            "validation_scenes_per_mask": scene_count,
            "batch_size": int(entry["batch_size"]),
            "num_workers": 4,
            "persistent_workers": False,
        }
    )
    if source_builder.get("psf_cache") is not None:
        # Parity must not depend on the completeness of an untracked cache.
        source_builder["psf_cache"] = {
            "mode": "off",
            "root_dir": str(REPO_ROOT / "data/psf_cache"),
            "request_modes": [],
            "warmup": False,
        }
    source_builder["return_psf"] = bool(entry["requires_psf"])
    trained_with_legacy_unflipped = (
        entry["registry_id"] in LEGACY_UNFLIPPED_REGISTRY_IDS
    )
    legacy_unflipped = mode == "development" and trained_with_legacy_unflipped
    source_builder["flip_convolution_psf"] = not legacy_unflipped

    model = normalize_model_target(OmegaConf.to_container(source.model, resolve=True))
    if "checkpoint_path" in model:
        model["checkpoint_path"] = None
    model["output_crop"] = contract["roi"]
    metrics = augment_metrics(source, entry, classifier, classifier_hash)
    device_tensors = ["measurement", "target"]
    if entry["requires_psf"]:
        device_tensors.append("psf")
    expected_samples = mask_count * scene_count
    config = {
        "model": model,
        "writer": None,
        "metrics": metrics,
        "datasets": {
            "validation": absolute_data_paths(dataset, source_dataset, mode),
        },
        "dataloader_builder": source_builder,
        "simulator": OmegaConf.to_container(source.simulator, resolve=True),
        "inferencer": {
            "device_tensors": device_tensors,
            "device": "cuda",
            "seed": int(entry["seed"]),
            "output_type": "reconstruction",
            "save_path": str(save_path),
            "from_pretrained": str(checkpoint),
            "skip_model_load": False,
            "override": False,
            "example_indices": [],
            "amp": inference_amp(source),
        },
        "provenance": {
            "report_test_entry": entry["id"],
            "registry_id": entry["registry_id"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "mode": mode,
            "expected_samples": expected_samples,
            "expected_masks": mask_count,
            "expected_scenes_per_mask": scene_count,
            "legacy_unflipped_psf": legacy_unflipped,
            "checkpoint_trained_with_legacy_unflipped_psf": (
                trained_with_legacy_unflipped
            ),
        },
        "hydra": {"run": {"dir": str(save_path / "hydra")}},
    }
    if dataset == "mnist" and mode == "test":
        config["provenance"]["expected_classes"] = 10
    return OmegaConf.create(config)


def real_test_config(source, entry: dict, checkpoint: Path, save_path: Path):
    from omegaconf import OmegaConf

    contract = entry["test_contract"]
    model = normalize_model_target(OmegaConf.to_container(source.model, resolve=True))
    if "checkpoint_path" in model:
        model["checkpoint_path"] = None
    model["output_crop"] = contract["roi"]
    return OmegaConf.create(
        {
            "model": model,
            "writer": None,
            "metrics": OmegaConf.to_container(source.metrics, resolve=True),
            "datasets": {
                "test": {
                    "_target_": "src.datasets.digicam.DigiCamRealDataset",
                    "repo_id": "bezzam/DigiCam-Mirflickr-MultiMask-25K",
                    "revision": contract["dataset_revision"],
                    "split": "test",
                    "cache_dir": str(REPO_ROOT / "data/huggingface"),
                    "force_rgb": True,
                    "rotate_measurement": True,
                    "measurement_downsample": 1.0,
                    "target_size": [200, 266],
                    "target_resize_mode": "bilinear",
                    "return_psf": False,
                    "simulator_config": OmegaConf.to_container(
                        source.simulator, resolve=True
                    ),
                    "expected_mask_count": 15,
                    "expected_scenes_per_mask": 250,
                }
            },
            "dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": int(entry["batch_size"]),
                "num_workers": 4,
                "pin_memory": True,
                "prefetch_factor": 2,
                "persistent_workers": False,
            },
            "transforms": {"batch_transforms": {"inference": None}},
            "inferencer": {
                "device_tensors": ["measurement", "target"],
                "device": "cuda",
                "seed": int(entry["seed"]),
                "output_type": "reconstruction",
                "save_path": str(save_path),
                "from_pretrained": str(checkpoint),
                "skip_model_load": False,
                "override": False,
                "example_indices": [],
                "amp": inference_amp(source),
            },
            "provenance": {
                "report_test_entry": entry["id"],
                "registry_id": entry["registry_id"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "mode": "test",
                "dataset_revision": contract["dataset_revision"],
                "expected_samples": 3750,
                "expected_masks": 15,
                "expected_scenes_per_mask": 250,
            },
            "hydra": {"run": {"dir": str(save_path / "hydra")}},
        }
    )


def real_development_config(source, entry: dict, checkpoint: Path, save_path: Path):
    from omegaconf import OmegaConf

    model = normalize_model_target(OmegaConf.to_container(source.model, resolve=True))
    if "checkpoint_path" in model:
        model["checkpoint_path"] = None
    config = {
        "model": model,
        "writer": None,
        "metrics": OmegaConf.to_container(source.metrics, resolve=True),
        "datasets": {
            "validation": OmegaConf.to_container(
                source.datasets.validation, resolve=True
            )
        },
        "dataloader": OmegaConf.to_container(source.dataloader, resolve=True),
        "transforms": OmegaConf.to_container(source.transforms, resolve=True),
        "simulator": OmegaConf.to_container(source.simulator, resolve=True),
        "inferencer": {
            "device_tensors": ["measurement", "target"],
            "device": "cuda",
            "seed": int(entry["seed"]),
            "output_type": "reconstruction",
            "save_path": str(save_path),
            "from_pretrained": str(checkpoint),
            "skip_model_load": False,
            "override": False,
            "example_indices": [],
            "amp": inference_amp(source),
        },
        "provenance": {
            "report_test_entry": entry["id"],
            "registry_id": entry["registry_id"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "mode": "development",
        },
        "hydra": {"run": {"dir": str(save_path / "hydra")}},
    }
    return OmegaConf.create(config)


def classifier_dependency(
    manifest: dict, location: str
) -> tuple[Path | None, str | None]:
    matches = [
        entry
        for entry in manifest["specialized_entries"]
        if entry["model"] == "Frozen classifier"
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one frozen classifier registry row")
    entry = matches[0]
    value = entry.get(f"checkpoint_{location}")
    path = Path(value).expanduser() if value else None
    if path is None or not path.is_file():
        return None, entry["checkpoint_sha256"]
    if sha256(path) != entry["checkpoint_sha256"]:
        raise ValueError("frozen classifier checkpoint hash mismatch")
    return path.resolve(), entry["checkpoint_sha256"]


def materialize(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = validate_manifest(manifest_path)
    preflight_path = manifest_path.parent / f"preflight_{args.location}.json"
    if not preflight_path.is_file():
        raise FileNotFoundError(preflight_path)
    preflight = json.loads(preflight_path.read_text())
    if preflight["status"] != "pass" and not args.allow_partial:
        raise ValueError("materialization requires a complete preflight")
    allowed_ids = set(args.id or [])
    preflight_rows = {row["id"]: row for row in preflight["entries"]}
    classifier, classifier_hash = classifier_dependency(manifest, args.location)
    generated = []
    config_dir = manifest_path.parent / "configs" / args.mode
    for entry in manifest["entries"]:
        if allowed_ids and entry["id"] not in allowed_ids:
            continue
        row = preflight_rows[entry["id"]]
        if row["errors"]:
            if args.allow_partial:
                continue
            raise ValueError(f"preflight failed for {entry['id']}")
        checkpoint = Path(row["checkpoint"]).resolve()
        value = load_checkpoint(checkpoint)
        fallback = resolve_candidate(entry, args.location, "config")
        source = embedded_config(value, fallback)
        save_path = manifest_path.parent / "runs" / args.mode / entry["id"]
        if entry["dataset"] == "digicam real":
            if args.mode == "test":
                config = real_test_config(source, entry, checkpoint, save_path)
            else:
                config = real_development_config(source, entry, checkpoint, save_path)
        else:
            config = synthetic_config(
                source,
                entry,
                args.mode,
                checkpoint,
                save_path,
                classifier,
                classifier_hash,
            )
        path = config_dir / f"{entry['id']}.yaml"
        if path.exists() and not args.rewrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, path, resolve=True)
        generated.append(
            {
                "id": entry["id"],
                "config": str(path),
                "config_sha256": sha256(path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "mode": args.mode,
                "output": str(save_path),
            }
        )
    result = {
        "status": "materialized",
        "mode": args.mode,
        "location": args.location,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "entry_count": len(generated),
        "entries": generated,
        "test_model_forward_executed": False,
    }
    output = manifest_path.parent / f"configs_{args.mode}_{args.location}.json"
    json_write(output, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "entries"}, indent=2
        )
    )


def load_configs_manifest(path: Path, manifest_path: Path, mode: str) -> dict:
    value = json.loads(path.read_text())
    if value.get("status") != "materialized" or value.get("mode") != mode:
        raise ValueError("generated config manifest has the wrong status or mode")
    if value.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("generated configs refer to another launch manifest")
    for entry in value["entries"]:
        config = Path(entry["config"])
        checkpoint = Path(entry["checkpoint"])
        if not config.is_file() or sha256(config) != entry["config_sha256"]:
            raise ValueError(f"generated config hash drift: {entry['id']}")
        if not checkpoint.is_file() or sha256(checkpoint) != entry["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash drift before launch: {entry['id']}")
    return value


def validate_test_authorization(
    path: Path | None,
    manifest_path: Path,
    configs_path: Path,
    parity_path: Path,
) -> dict:
    if path is None or not path.is_file():
        raise PermissionError("test mode requires a separate authorization JSON")
    if not parity_path.is_file():
        raise PermissionError("test mode requires development_parity.json")
    parity = json.loads(parity_path.read_text())
    if parity.get("status") != "pass":
        raise PermissionError("development parity has not passed")
    expected = {
        "authorized_by_user": True,
        "mode": "test",
        "manifest_sha256": sha256(manifest_path),
        "configs_manifest_sha256": sha256(configs_path),
        "development_parity_sha256": sha256(parity_path),
    }
    actual = json.loads(path.read_text())
    if actual != expected:
        raise PermissionError("authorization does not match the frozen launch inputs")
    return actual


def launch(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = validate_manifest(manifest_path)
    configs_path = (
        Path(args.configs).expanduser().resolve()
        if args.configs
        else manifest_path.parent / f"configs_{args.mode}_{args.location}.json"
    )
    configs = load_configs_manifest(configs_path, manifest_path, args.mode)
    selected = set(args.id or [])
    entries = [
        entry for entry in configs["entries"] if not selected or entry["id"] in selected
    ]
    if selected != {entry["id"] for entry in entries} and selected:
        missing = sorted(selected - {entry["id"] for entry in entries})
        raise ValueError(f"unknown launch IDs: {missing}")
    if not entries:
        raise ValueError("launch selection is empty")
    if args.mode == "test":
        authorization = (
            Path(args.authorization).expanduser().resolve()
            if args.authorization
            else None
        )
        validate_test_authorization(
            authorization,
            manifest_path,
            configs_path,
            manifest_path.parent / "development_parity.json",
        )
    elif args.authorization:
        raise ValueError("development mode does not accept an authorization file")

    for entry in entries:
        output = Path(entry["output"])
        if output.exists():
            raise FileExistsError(f"refusing to reuse output directory: {output}")

    state_path = manifest_path.parent / (
        f"launch_{args.mode}.json"
        if not selected
        else f"launch_{args.mode}_subset.json"
    )
    if state_path.exists():
        raise FileExistsError(f"refusing to overwrite launch state: {state_path}")
    gpu_ids = [int(value) for value in args.gpus]
    if len(gpu_ids) != len(set(gpu_ids)) or not gpu_ids:
        raise ValueError("GPU IDs must be a non-empty unique list")

    pending = list(entries)
    available = list(gpu_ids)
    running = {}
    completed = []
    failed = False
    json_write(
        state_path,
        {
            "status": "running",
            "mode": args.mode,
            "manifest_sha256": sha256(manifest_path),
            "configs_manifest_sha256": sha256(configs_path),
            "entry_count": len(entries),
            "gpus": gpu_ids,
            "automatic_retry": False,
            "test_model_forward_executed": args.mode == "test",
            "completed": [],
        },
    )
    try:
        while pending or running:
            while pending and available and not failed:
                entry = pending.pop(0)
                gpu = available.pop(0)
                config = Path(entry["config"])
                log = manifest_path.parent / "logs" / args.mode / f"{entry['id']}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open("x")
                command = [
                    sys.executable,
                    str(REPO_ROOT / "inference.py"),
                    "--config-path",
                    str(config.parent),
                    "--config-name",
                    config.stem,
                ]
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running[process.pid] = {
                    "entry": entry,
                    "gpu": gpu,
                    "process": process,
                    "stream": stream,
                    "log": log,
                    "started": time.time(),
                }
            if not running:
                break
            time.sleep(1)
            for pid, item in list(running.items()):
                returncode = item["process"].poll()
                if returncode is None:
                    continue
                item["stream"].close()
                result = {
                    "id": item["entry"]["id"],
                    "gpu": item["gpu"],
                    "returncode": returncode,
                    "elapsed_seconds": time.time() - item["started"],
                    "log": str(item["log"]),
                    "log_sha256": sha256(item["log"]),
                }
                completed.append(result)
                available.append(item["gpu"])
                del running[pid]
                if returncode != 0:
                    failed = True
            json_write(
                state_path,
                {
                    "status": "failed_closed" if failed else "running",
                    "mode": args.mode,
                    "manifest_sha256": sha256(manifest_path),
                    "configs_manifest_sha256": sha256(configs_path),
                    "entry_count": len(entries),
                    "pending_count": len(pending),
                    "running": [item["entry"]["id"] for item in running.values()],
                    "completed": completed,
                    "automatic_retry": False,
                    "test_model_forward_executed": args.mode == "test",
                },
            )
        if failed:
            raise SystemExit(2)
        json_write(
            state_path,
            {
                "status": "complete",
                "mode": args.mode,
                "manifest_sha256": sha256(manifest_path),
                "configs_manifest_sha256": sha256(configs_path),
                "entry_count": len(entries),
                "completed": completed,
                "automatic_retry": False,
                "test_model_forward_executed": args.mode == "test",
            },
        )
    finally:
        for item in running.values():
            item["stream"].close()
    print(json.dumps(json.loads(state_path.read_text()), indent=2))


def result_summary_path(entry: dict) -> Path:
    output = Path(entry["output"])
    matches = sorted(output.glob("*/summary.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one summary.json for {entry['id']}, found {matches}"
        )
    return matches[0]


def parity(args: argparse.Namespace) -> None:
    import numpy as np

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = validate_manifest(manifest_path)
    configs_path = (
        Path(args.configs).expanduser().resolve()
        if args.configs
        else manifest_path.parent / f"configs_development_{args.location}.json"
    )
    configs = load_configs_manifest(configs_path, manifest_path, "development")
    expected_entries = {entry["id"]: entry for entry in manifest["entries"]}
    observed_ids = {entry["id"] for entry in configs["entries"]}
    if observed_ids != set(expected_entries) and not args.allow_partial:
        raise ValueError("development parity requires every standard entry")

    rows = []
    for run in configs["entries"]:
        entry = expected_entries[run["id"]]
        summary_path = result_summary_path(run)
        summary = json.loads(summary_path.read_text())
        observed = summary["mask_balanced"]
        expected = entry["development_reference"]["metrics"]
        comparisons = {}
        errors = []
        for metric, expected_value in expected.items():
            if metric not in observed:
                errors.append(f"missing_metric:{metric}")
                continue
            actual_value = float(observed[metric])
            delta = actual_value - float(expected_value)
            tolerance = manifest["development_parity_tolerances"].get(metric)
            if tolerance is None:
                errors.append(f"missing_tolerance:{metric}")
                continue
            if not np.isfinite(actual_value) or abs(delta) > float(tolerance):
                errors.append(f"metric_mismatch:{metric}")
            comparisons[metric] = {
                "expected": float(expected_value),
                "actual": actual_value,
                "delta": delta,
                "tolerance": float(tolerance),
            }
        rows.append(
            {
                "id": run["id"],
                "summary": str(summary_path),
                "development_reference_sha256": entry["development_reference"][
                    "sha256"
                ],
                "comparisons": comparisons,
                "errors": errors,
            }
        )
    passed = sum(not row["errors"] for row in rows)
    complete = len(rows) == len(expected_entries)
    result = {
        "status": "pass" if complete and passed == len(rows) else "incomplete",
        "manifest_sha256": sha256(manifest_path),
        "configs_manifest_sha256": sha256(configs_path),
        "expected_entry_count": len(expected_entries),
        "evaluated_entry_count": len(rows),
        "passed_entry_count": passed,
        "failed_entry_count": len(rows) - passed,
        "test_scene_files_opened": False,
        "test_masks_generated": False,
        "test_model_forward_executed": False,
        "entries": rows,
    }
    output = manifest_path.parent / (
        "development_parity.json"
        if not args.allow_partial
        else "development_parity_partial.json"
    )
    json_write(output, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "entries"}, indent=2
        )
    )
    if result["status"] != "pass" and not args.allow_partial:
        raise SystemExit(2)


def markdown_table(frame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def paired_effect_rows(frame, metric_columns: list[str]) -> list[dict]:
    rows = []

    def add(contrast, left, right, join):
        left_frame = frame.loc[left].copy()
        right_frame = frame.loc[right].copy()
        if left_frame.empty or right_frame.empty:
            return
        if left_frame.duplicated(join).any() or right_frame.duplicated(join).any():
            raise ValueError(f"paired contrast is not one-to-one: {contrast}")
        merged = left_frame.merge(
            right_frame,
            on=join,
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        for _, item in merged.iterrows():
            for metric in metric_columns:
                left_name = f"{metric}_left"
                right_name = f"{metric}_right"
                if left_name not in item or right_name not in item:
                    continue
                if (
                    item[left_name] != item[left_name]
                    or item[right_name] != item[right_name]
                ):
                    continue
                rows.append(
                    {
                        "contrast": contrast,
                        **{key: item[key] for key in join},
                        "metric": metric,
                        "effect": float(item[right_name]) - float(item[left_name]),
                    }
                )

    add(
        "DRUNet: 16k minus 4k training scenes",
        (frame["dataset"] == "mirflickr")
        & (frame["model"] == "DRUNet")
        & (frame["steps"] == 100000)
        & (frame["training_scenes"] == 4096),
        (frame["dataset"] == "mirflickr")
        & (frame["model"] == "DRUNet")
        & (frame["steps"] == 100000)
        & (frame["training_scenes"] == 16384),
        ["seed", "condition"],
    )
    add(
        "MNIST: PSF-conditioned minus measurement-only",
        (frame["dataset"] == "mnist")
        & (frame["model"] == "DRUNet")
        & (frame["steps"] == 50000),
        (frame["dataset"] == "mnist")
        & (frame["model"] == "PSF-aware DRUNet")
        & (frame["steps"] == 50000),
        ["seed", "condition"],
    )
    add(
        "DigiCam: real minus matched-simulation training",
        (frame["dataset"] == "digicam real") & (frame["condition"] == "matched_sim"),
        (frame["dataset"] == "digicam real") & (frame["condition"] == "real"),
        ["seed", "model", "steps"],
    )
    add(
        "CelebA: GoPro minus random initialization",
        (frame["dataset"] == "celeba32")
        & (frame["model"] == "X-Restormer")
        & (frame["variant"] == "finite100_random"),
        (frame["dataset"] == "celeba32")
        & (frame["model"] == "X-Restormer")
        & (frame["variant"] == "100")
        & (frame["initialization"] == "gopro"),
        ["seed", "steps"],
    )
    add(
        "CelebA continuation: L1 1.0 minus 0.1",
        (frame["dataset"] == "celeba32") & (frame["variant"] == "continuation_l1_0.1"),
        (frame["dataset"] == "celeba32") & (frame["variant"] == "continuation_l1_1.0"),
        ["seed", "model", "steps"],
    )
    return rows


def summarize(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = validate_manifest(manifest_path)
    configs_path = (
        Path(args.configs).expanduser().resolve()
        if args.configs
        else manifest_path.parent / f"configs_{args.mode}_{args.location}.json"
    )
    configs = load_configs_manifest(configs_path, manifest_path, args.mode)
    by_id = {entry["id"]: entry for entry in manifest["entries"]}
    rows = []
    for run in configs["entries"]:
        summary_path = result_summary_path(run)
        summary = json.loads(summary_path.read_text())
        entry = by_id[run["id"]]
        if summary.get("sample_count") != summary.get("provenance", {}).get(
            "expected_samples", summary.get("sample_count")
        ):
            raise ValueError(f"sample count mismatch: {run['id']}")
        row = {
            "id": run["id"],
            "dataset": entry["dataset"],
            "model": entry["model"],
            "condition": entry["condition"],
            "initialization": entry["initialization"],
            "variant": entry["variant"],
            "seed": entry["seed"],
            "steps": entry["steps"],
            "training_scenes": entry["training_scenes"],
            "sample_count": summary["sample_count"],
            "mask_count": summary["mask_count"],
        }
        for name, value in summary["mask_balanced"].items():
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"non-finite metric {name}: {run['id']}")
            row[name] = value
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    output = manifest_path.parent / "results" / args.mode
    output.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output / "per_seed.csv", index=False)

    keys = [
        "dataset",
        "model",
        "condition",
        "initialization",
        "variant",
        "steps",
        "training_scenes",
    ]
    metric_columns = [
        column
        for column in per_seed.columns
        if column
        not in {
            "id",
            *keys,
            "seed",
            "sample_count",
            "mask_count",
        }
    ]
    aggregate_rows = []
    for values, group in per_seed.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        row["run_count"] = len(group)
        for metric in metric_columns:
            observed = group[metric].dropna().astype(float)
            if observed.empty:
                continue
            row[f"{metric}_mean"] = observed.mean()
            row[f"{metric}_sample_sd"] = (
                observed.std(ddof=1) if len(observed) > 1 else np.nan
            )
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output / "aggregate.csv", index=False)
    paired = pd.DataFrame(paired_effect_rows(per_seed, metric_columns))
    paired.to_csv(output / "paired_effects_per_seed.csv", index=False)
    paired_aggregate = pd.DataFrame()
    if not paired.empty:
        paired_aggregate = (
            paired.groupby(["contrast", "metric"], sort=True)["effect"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"std": "sample_sd", "count": "run_count"})
        )
    paired_aggregate.to_csv(output / "paired_effects_aggregate.csv", index=False)

    reused = manifest["reused_results"]
    xrest_validation = json.loads(
        (manifest_path.parent / reused["xrestormer_validation"]["path"]).read_text()
    )
    if (
        xrest_validation.get("status") != "pass"
        or xrest_validation.get("all_metrics_recomputed") is not True
        or xrest_validation.get("post_test_checkpoint_selection") is not False
    ):
        raise ValueError("reused X-Restormer final validation is not acceptable")
    xrest = pd.read_csv(
        manifest_path.parent / reused["xrestormer_model_summary"]["path"]
    )
    xrest_effects = pd.read_csv(
        manifest_path.parent / reused["xrestormer_paired_effects"]["path"]
    )
    published_real = json.loads(
        (manifest_path.parent / reused["published_psf_aware_real"]["path"]).read_text()
    )
    if (
        published_real.get("sample_count") != 3750
        or published_real.get("mask_count") != 15
        or published_real.get("samples_per_mask") != [250]
    ):
        raise ValueError("reused published real reference has the wrong grid")
    xrest.to_csv(output / "reused_xrestormer_model_summary.csv", index=False)
    xrest_effects.to_csv(output / "reused_xrestormer_paired_effects.csv", index=False)
    json_write(output / "reused_published_psf_aware_real.json", published_real)

    lines = [
        f"# Unified {args.mode} metrics for the coursework report",
        "",
        f"Runs: {len(per_seed)}. All values below are mask-balanced endpoint metrics.",
        "",
    ]
    display_columns = [
        column
        for column in [
            "dataset",
            "model",
            "condition",
            "initialization",
            "variant",
            "steps",
            "training_scenes",
            "run_count",
            *[f"{metric}_mean" for metric in metric_columns],
            *[f"{metric}_sample_sd" for metric in metric_columns],
        ]
        if column in aggregate.columns
    ]
    lines.extend(markdown_table(aggregate, display_columns))
    if not paired_aggregate.empty:
        lines.extend(["", "## Paired effects", ""])
        lines.extend(
            markdown_table(
                paired_aggregate,
                ["contrast", "metric", "mean", "sample_sd", "run_count"],
            )
        )
    lines.extend(["", "## Reused frozen X-Restormer final results", ""])
    lines.extend(
        markdown_table(
            xrest,
            [
                "shortlist_id",
                "analysis_role",
                "training_masks",
                "initialization",
                "seed",
                "training_steps",
                "PSNR",
                "SSIM",
                "LPIPS",
            ],
        )
    )
    lines.extend(["", "### X-Restormer paired effects", ""])
    lines.extend(
        markdown_table(
            xrest_effects,
            [
                "contrast_type",
                "group",
                "metric",
                "mean_effect",
                "sample_sd_across_runs",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Reused published PSF-aware DigiCam reference",
            "",
            "| Samples | Masks | PSNR | SSIM | LPIPS |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {published_real['sample_count']} | {published_real['mask_count']} | "
                f"{published_real['mask_balanced']['PSNR']:.6f} | "
                f"{published_real['mask_balanced']['SSIM']:.6f} | "
                f"{published_real['mask_balanced']['LPIPS']:.6f} |"
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Manifest SHA-256: `{sha256(manifest_path)}`",
            f"- Generated-config manifest SHA-256: `{sha256(configs_path)}`",
            "- Automatic retry: disabled.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    validation = {
        "status": "pass",
        "mode": args.mode,
        "run_count": len(per_seed),
        "aggregate_row_count": len(aggregate),
        "paired_effect_row_count": len(paired),
        "paired_effect_aggregate_count": len(paired_aggregate),
        "reused_xrestormer_checkpoint_count": len(xrest),
        "reused_published_real_samples": published_real["sample_count"],
        "metric_columns": metric_columns,
        "manifest_sha256": sha256(manifest_path),
        "configs_manifest_sha256": sha256(configs_path),
        "test_model_forward_executed": args.mode == "test",
    }
    json_write(output / "validation.json", validation)
    print(json.dumps(validation, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    prepare_parser.add_argument(
        "--audit",
        default=str(DEFAULT_REGISTRY).replace(".csv", ".audit.csv"),
    )
    prepare_parser.add_argument(
        "--report", default=str(REPO_ROOT.parent / "current_report.tex")
    )
    prepare_parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    prepare_parser.add_argument("--remote-root", default=str(DEFAULT_REMOTE_ROOT))
    prepare_parser.add_argument("--rewrite", action="store_true")
    prepare_parser.set_defaults(function=prepare)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("manifest")
    preflight_parser.add_argument(
        "--location", choices=("local", "remote"), required=True
    )
    preflight_parser.add_argument("--deserialize", action="store_true")
    preflight_parser.add_argument("--require-all", action="store_true")
    preflight_parser.set_defaults(function=preflight)

    staging_parser = subparsers.add_parser("resolve-staging")
    staging_parser.add_argument("manifest")
    staging_parser.add_argument("--search-root", default="/home/hadhad/project")
    staging_parser.add_argument("--link", action="store_true")
    staging_parser.set_defaults(function=resolve_staging)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("manifest")
    materialize_parser.add_argument(
        "--location", choices=("local", "remote"), required=True
    )
    materialize_parser.add_argument(
        "--mode", choices=("development", "test"), required=True
    )
    materialize_parser.add_argument("--id", action="append")
    materialize_parser.add_argument("--allow-partial", action="store_true")
    materialize_parser.add_argument("--rewrite", action="store_true")
    materialize_parser.set_defaults(function=materialize)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("manifest")
    launch_parser.add_argument("--location", choices=("local", "remote"), required=True)
    launch_parser.add_argument("--mode", choices=("development", "test"), required=True)
    launch_parser.add_argument("--configs")
    launch_parser.add_argument("--authorization")
    launch_parser.add_argument("--gpus", nargs="+", required=True)
    launch_parser.add_argument("--id", action="append")
    launch_parser.set_defaults(function=launch)

    parity_parser = subparsers.add_parser("parity")
    parity_parser.add_argument("manifest")
    parity_parser.add_argument("--location", choices=("local", "remote"), required=True)
    parity_parser.add_argument("--configs")
    parity_parser.add_argument("--allow-partial", action="store_true")
    parity_parser.set_defaults(function=parity)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("manifest")
    summarize_parser.add_argument(
        "--location", choices=("local", "remote"), required=True
    )
    summarize_parser.add_argument(
        "--mode", choices=("development", "test"), required=True
    )
    summarize_parser.add_argument("--configs")
    summarize_parser.set_defaults(function=summarize)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
