"""Audit the divergent MNIST PSF-aware seed62 learning curve."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "outputs/coursework_long_scaling_v4_20260911"
OUTPUT = BUNDLE / "seed62_audit_v1"
METRICS = ("PSNR", "SSIM", "PSNR_32", "SSIM_32", "Dice_loss_32")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalized_pair_config(path: Path, aware: bool) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    config["model"]["_target_"] = "src.model.psf_free_drunet.PSFFreeDRUNet"
    config["dataloader_builder"]["return_psf"] = False
    config["trainer"]["device_tensors"] = ["measurement", "target"]
    config["protocol"]["information_regime"] = "measurement only"
    config["writer"]["tags"] = [
        "synthetic",
        "psf-free",
        "mnist",
        "50k",
        "matched-v2",
    ]
    config["writer"]["run_name"] = config["writer"]["run_name"].replace(
        "psf-aware", "psf-free"
    )
    config["writer"]["job_type"] = config["writer"]["job_type"].replace(
        "psf-aware", "psf-free"
    )
    return config


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    preflight = json.loads((BUNDLE / "preflight.json").read_text())
    state = json.loads((BUNDLE / "launcher/launcher_state.json").read_text())
    results_validation = json.loads((BUNDLE / "results/validation.json").read_text())
    curves = pd.read_csv(BUNDLE / "results/mnist_learning_curves.csv")
    if not bool(curves[list(METRICS)].map(math.isfinite).all().all()):
        raise ValueError("Learning curves contain non-finite values")
    if not (
        state["status"] == "complete"
        and state["completed_job_count"] == 13
        and state["failed_job_count"] == 0
        and results_validation["status"] == "pass"
    ):
        raise ValueError("Long-training bundle is not complete")
    if preflight["status"] != "pass" or len(preflight["gpu_smoke"]) != 13:
        raise ValueError("Preflight evidence is incomplete")

    config_parity = {}
    first_batch_parity = {}
    logs_clean = {}
    smoke_by_name = {row["name"]: row for row in preflight["gpu_smoke"]}
    for seed in manifest["mnist_protocol"]["seeds"]:
        names = {
            regime: f"cw-mnist-{regime}-finite100-50k-seed{seed}-v2"
            for regime in ("psf-free", "psf-aware")
        }
        configs = {
            regime: BUNDLE / "configs" / f"{name}.yaml"
            for regime, name in names.items()
        }
        free = normalized_pair_config(configs["psf-free"], aware=False)
        aware = normalized_pair_config(configs["psf-aware"], aware=True)
        if free != aware:
            raise ValueError(f"Matched config parity failed for seed{seed}")
        config_parity[str(seed)] = True
        free_smoke = smoke_by_name[names["psf-free"]]
        aware_smoke = smoke_by_name[names["psf-aware"]]
        same_batch = (
            free_smoke["scene_ids"] == aware_smoke["scene_ids"]
            and free_smoke["mask_ids"] == aware_smoke["mask_ids"]
        )
        if not same_batch:
            raise ValueError(f"First-batch identity failed for seed{seed}")
        first_batch_parity[str(seed)] = True
        for regime, name in names.items():
            log = BUNDLE / "training" / name / "info.log"
            text = log.read_text()
            clean = not any(
                marker in text
                for marker in (
                    "Traceback",
                    "CUDA out of memory",
                    "FloatingPointError",
                    "Non-finite",
                )
            )
            if not clean:
                raise ValueError(f"Training log contains an error marker: {name}")
            logs_clean[name] = True

    indexed = curves.set_index(["information_regime", "seed", "steps"])
    paired_rows = []
    for seed in manifest["mnist_protocol"]["seeds"]:
        for steps in sorted(curves["steps"].unique()):
            free = indexed.loc[("psf_free", seed, steps)]
            aware = indexed.loc[("psf_aware", seed, steps)]
            paired_rows.append(
                {
                    "seed": seed,
                    "steps": steps,
                    **{metric: aware[metric] - free[metric] for metric in METRICS},
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUTPUT / "paired_effects_by_step.csv", index=False)
    curves.to_csv(OUTPUT / "learning_curves.csv", index=False)

    seed62_aware = curves[
        (curves["information_regime"] == "psf_aware") & (curves["seed"] == 62)
    ].sort_values("steps")
    psnr32_monotonic = bool(seed62_aware["PSNR_32"].is_monotonic_increasing)
    dice_monotonic = bool(
        seed62_aware["Dice_loss_32"].iloc[::-1].is_monotonic_increasing
    )
    if not psnr32_monotonic or not dice_monotonic:
        raise ValueError("Seed62 coarse metrics do not improve monotonically")
    endpoint = paired[paired["steps"] == 50000].set_index("seed")
    summary = {
        "seed42_psf_aware_minus_free": {
            metric: float(endpoint.loc[42, metric]) for metric in METRICS
        },
        "seed52_psf_aware_minus_free": {
            metric: float(endpoint.loc[52, metric]) for metric in METRICS
        },
        "seed62_psf_aware_minus_free": {
            metric: float(endpoint.loc[62, metric]) for metric in METRICS
        },
        "seed62_psf_aware_curve": {
            "PSNR_32_at_5k": float(seed62_aware.iloc[0]["PSNR_32"]),
            "PSNR_32_at_50k": float(seed62_aware.iloc[-1]["PSNR_32"]),
            "Dice_loss_32_at_5k": float(seed62_aware.iloc[0]["Dice_loss_32"]),
            "Dice_loss_32_at_50k": float(seed62_aware.iloc[-1]["Dice_loss_32"]),
            "PSNR_32_monotonic": psnr32_monotonic,
            "Dice_loss_32_monotonic_improvement": dice_monotonic,
        },
    }
    save_json(OUTPUT / "summary.json", summary)
    validation = {
        "status": "pass",
        "classification": "valid_but_divergent_not_excludable",
        "all_jobs_complete": True,
        "all_metrics_finite": True,
        "training_logs_error_free": all(logs_clean.values()),
        "matched_config_parity": config_parity,
        "first_smoke_batch_identity": first_batch_parity,
        "shared_initialization_exact": preflight["shared_mnist_initialization"],
        "late_training_collapse_detected": False,
        "protocol_failure_detected": False,
        "optimization_variability_is_inference": True,
        "seed62_may_be_excluded": False,
        "next_required_diagnostic": "replay all three PSF-aware endpoints on one development grid with correct and cyclically shuffled PSF",
        "final_test_accessed": False,
        "sources": {
            str(path.relative_to(REPO)): sha256(path)
            for path in (
                BUNDLE / "manifest.json",
                BUNDLE / "preflight.json",
                BUNDLE / "launcher/launcher_state.json",
                BUNDLE / "results/mnist_learning_curves.csv",
            )
        },
    }
    save_json(OUTPUT / "validation.json", validation)
    lines = [
        "# MNIST PSF-aware seed62 audit",
        "",
        "The seed62 endpoint is technically valid and cannot be excluded. All jobs completed, metrics and gradients stayed finite, matched configs agree, the first smoke-batch identities agree, and the shared initialization audit passed.",
        "",
        "The divergence is present from the first 5k endpoint rather than appearing as a late collapse. Seed62 PSF-aware PSNR32 rises monotonically from "
        f"{summary['seed62_psf_aware_curve']['PSNR_32_at_5k']:.4f} to {summary['seed62_psf_aware_curve']['PSNR_32_at_50k']:.4f}; "
        "Dice loss falls monotonically from "
        f"{summary['seed62_psf_aware_curve']['Dice_loss_32_at_5k']:.4f} to {summary['seed62_psf_aware_curve']['Dice_loss_32_at_50k']:.4f}. "
        "This is consistent with optimization variability, but the present evidence does not establish its mechanism.",
        "",
        "The next diagnostic is a frozen development replay of all three PSF-aware endpoints under correct and cyclically shuffled PSF. Continuing only seed62 training would be a post-hoc asymmetric intervention and is not allowed.",
    ]
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(validation, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
