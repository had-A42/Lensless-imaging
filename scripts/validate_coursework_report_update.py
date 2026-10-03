"""Validate the evidence-backed inserts in the canonical coursework report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO.parent / "new_version.tex"
OUTPUT = REPO / "outputs/coursework_report_update_20260912"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(text: str, value: str) -> None:
    if value not in text:
        raise ValueError(f"Report value is missing: {value}")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    text = REPORT.read_text()
    sources = []

    final_path = REPO / "outputs/coursework_final_runner_v4_20260910/final_results_v1/model_summary.csv"
    final_validation = REPO / "outputs/coursework_final_runner_v4_20260910/final_results_v1/validation.json"
    final = pd.read_csv(final_path)
    primary = final[final["analysis_role"] == "primary_matched_matrix"]
    for (masks, initialization), group in primary.groupby(
        ["training_masks", "initialization"]
    ):
        for metric, digits in (("PSNR", 4), ("SSIM", 4), ("LPIPS", 4)):
            token = f"{group[metric].mean():.{digits}f}\\pm{group[metric].std(ddof=1):.{digits}f}"
            require(text, token)
    sources.extend([final_path, final_validation])

    long_root = REPO / "outputs/coursework_long_scaling_v4_20260911/results"
    dr = pd.read_csv(long_root / "drunet_scene_effects.csv")
    for value in dr[dr["metric"] == "PSNR"]["effect"]:
        require(text, f"{value:+.4f}")
    mnist = pd.read_csv(long_root / "endpoints.csv")
    mnist = mnist[mnist["experiment"] == "mnist_long_psf_comparison"]
    for _, group in mnist.groupby("information_regime"):
        for metric in ("PSNR", "SSIM", "PSNR_32", "SSIM_32", "Dice_loss_32"):
            require(text, f"{group[metric].mean():.4f}\\pm{group[metric].std(ddof=1):.4f}")
    sources.extend(
        [
            long_root / "drunet_scene_effects.csv",
            long_root / "endpoints.csv",
            long_root / "validation.json",
        ]
    )

    psf_root = REPO / "outputs/coursework_mnist_psf_conditioning_v2_20260912"
    psf = pd.read_csv(psf_root / "summary_per_seed.csv")
    for seed, expected in ((42, 0.0066), (52, -0.0063), (62, 0.0081)):
        group = psf[psf["seed"] == seed].set_index("condition")
        effect = group.loc["Shuffled PSF", "PSNR_32"] - group.loc["Correct PSF", "PSNR_32"]
        if abs(effect - expected) > 5e-5:
            raise ValueError(f"PSF shuffle value drift: seed{seed}")
        require(text, f"{expected:+.4f}")
    sources.extend([psf_root / "summary_per_seed.csv", psf_root / "validation.json"])

    accuracy_root = REPO / "outputs/coursework_mnist_reconstruction_accuracy_v3_20260912"
    accuracy = pd.read_csv(accuracy_root / "paired_effects.csv")
    for seed in (42, 52, 62):
        row = accuracy[accuracy["seed"] == seed].iloc[0]
        require(text, f"{100 * row['psf_free']:.2f}\\%")
        require(text, f"{100 * row['psf_aware']:.2f}\\%")
    sources.extend(
        [accuracy_root / "paired_effects.csv", accuracy_root / "validation.json"]
    )

    controls_root = REPO / "outputs/coursework_post_final_20260911/extended_input_controls_v1/results_v1"
    controls = pd.read_csv(controls_root / "aggregate.csv")
    checks = [
        ("MIRFLICKR", "X-Restormer 100k GoPro"),
        ("CelebA", "X-Restormer"),
        ("CelebA", "DRUNet"),
    ]
    for dataset, model in checks:
        group = controls[(controls["dataset"] == dataset) & (controls["model"] == model)]
        for metric in (
            "zero_minus_correct_PSNR",
            "zero_minus_correct_SSIM",
            "zero_minus_correct_LPIPS",
            "cross_mask_prediction_RMSE",
        ):
            value = float(group[group["metric"] == metric]["mean"].iloc[0])
            require(text, f"{value:+.4f}" if metric != "cross_mask_prediction_RMSE" else f"{value:.4f}")
    sources.extend([controls_root / "aggregate.csv", controls_root / "validation.json"])

    for view, expected in (
        ("inner68", (18.9227, 0.5215, 0.4536)),
        ("outer17", (19.0538, 0.5202, 0.4528)),
    ):
        path = REPO / f"outputs/coursework_pre_final_20260910/psf_aware/{view}/summary.json"
        data = json.loads(path.read_text())
        values = data["scores"]
        actual = (values["PSNR"], values["SSIM"], values["LPIPS"])
        if any(abs(left - right) > 5e-5 for left, right in zip(actual, expected)):
            raise ValueError(f"PSF-aware value drift: {view}")
        for value in expected:
            require(text, f"{value:.4f}")
        sources.append(path)

    split_expectations = {
        "manifests/mirflickr25k_splits_16k.json": (16384, 128, 256),
        "manifests/mnist_splits.json": (55000, 5000, 10000),
        "manifests/celeba_splits.json": (4096, 128, 19962),
    }
    for relative, expected in split_expectations.items():
        path = REPO / relative
        data = json.loads(path.read_text())
        actual = tuple(len(data["splits"][key]) for key in ("train", "validation", "test"))
        if actual != expected:
            raise ValueError(f"Split count drift: {relative}: {actual}")
        sources.append(path)

    labels = (
        "cw:data-splits",
        "cw:drunet-scene-scale",
        "cw:xrest-final",
        "cw:mnist-psf-aware",
        "cw:mnist-psf-seeds",
        "cw:mnist-classifier",
        "cw:input-use-controls",
    )
    if any(text.count(f"\\label{{{label}}}") != 1 for label in labels):
        raise ValueError("A new report label is missing or duplicated")
    if "they do not constitute an untouched final evaluation" in text:
        raise ValueError("Obsolete final-test limitation remains in the report")
    validation = {
        "status": "pass",
        "report": str(REPORT),
        "report_sha256": sha256(REPORT),
        "evidence_values_match": True,
        "split_counts_match": True,
        "new_labels_unique": True,
        "lightweight_scope": "numeric/source consistency; full LaTeX compilation unavailable",
        "sources": {
            str(path.relative_to(REPO)): sha256(path) for path in sources
        },
    }
    (OUTPUT / "validation.json").write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(validation, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
