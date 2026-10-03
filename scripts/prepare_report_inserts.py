"""Generate separate, reviewable report inserts from validated evidence tables."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "input_controls": "outputs/coursework_status_20260909/input_controls_all_20260910/summary.csv",
    "psf_aware": "outputs/coursework_pre_final_20260910/psf_aware/comparison_v2/comparison_summary.csv",
    "psf_diversity": "outputs/coursework_pre_final_20260910/psf_diversity/analysis/diversity_summary.csv",
    "psf_nearest": "outputs/coursework_pre_final_20260910/psf_diversity/analysis/nearest_train_to_development_summary.csv",
    "readiness": "outputs/coursework_pre_final_20260910/pre_final_test_readiness.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def metric_text(mean: float, sd: str | float, digits: int) -> str:
    if sd == "" or sd is None:
        return f"{mean:.{digits}f}"
    return f"${mean:.{digits}f}\\pm{float(sd):.{digits}f}$"


def controls_table(control_rows: list[dict]) -> str:
    lookup = {
        (row["dataset"], row["model"], row["condition"], row["metric"]): row
        for row in control_rows
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Input-use controls on development data. Scene-shuffled measurements are permuted within each mask. Values are mask-balanced means; uncertainty is the sample SD across retained training runs when available.}",
        r"\label{cw:input-use-controls}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\linewidth}{llXccc}",
        r"\toprule",
        "Dataset & Model & Input condition & PSNR & SSIM & LPIPS \\\\",
        r"\midrule",
    ]
    groups = (
        ("MIRFLICKR", "X-Restormer", ("PSNR", "SSIM", "LPIPS")),
        ("CelebA32", "X-Restormer", ("PSNR_coarse", "SSIM_coarse", "LPIPS")),
        ("CelebA32", "DRUNet", ("PSNR_coarse", "SSIM_coarse", "LPIPS")),
    )
    labels = {
        "Correct": "Correct measurement",
        "Shuffled": "Scene-shuffled measurement",
        "Train mean": "Mean training target",
    }
    for group_index, (dataset, model, metrics) in enumerate(groups):
        for condition in ("Correct", "Shuffled", "Train mean"):
            values = []
            for metric, digits in zip(metrics, (4, 4, 4)):
                row = lookup[(dataset, model, condition, metric)]
                values.append(
                    metric_text(
                        float(row["mean"]),
                        "" if condition == "Train mean" else row["sample_sd"],
                        digits,
                    )
                )
            lines.append(
                f"{dataset} & {model} & {labels[condition]} & "
                + " & ".join(values)
                + r" \\"
            )
        if group_index + 1 != len(groups):
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    return "\n".join(lines)


def psf_aware_table(psf_rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{PSF-free models and the published PSF-aware reference on identical real development rows. The evaluation rows and metric pipeline are shared, but the method comparison is contextual because training and PSF access are not matched.}",
        r"\label{cw:psf-aware-development}",
        r"\begin{tabularx}{\linewidth}{Xlccc}",
        r"\toprule",
        "Development view and method & PSF at inference & PSNR, dB & SSIM & LPIPS \\\\",
        r"\midrule",
    ]
    view_labels = {"inner68": "68-mask inner view", "outer17": "17-mask outer view"}
    method_labels = {
        "PSF-free, trained on real measurements": "PSF-free, real training",
        "PSF-free, trained on matched simulation": "PSF-free, matched-sim training",
        "Published PSF-aware reference": "Published PSF-aware reference",
    }
    for index, row in enumerate(psf_rows):
        values = [
            metric_text(
                float(row[f"{metric}_mean"]),
                row[f"{metric}_sample_sd"],
                4,
            )
            for metric in ("PSNR", "SSIM", "LPIPS")
        ]
        lines.append(
            f"{view_labels[row['view']]}, {method_labels[row['method']]} & "
            f"{'yes' if row['psf_at_inference'] == 'true' else 'no'} & "
            + " & ".join(values)
            + r" \\"
        )
        if index == 2:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    return "\n".join(lines)


def diversity_table(diversity_rows: list[dict], nearest_rows: list[dict]) -> str:
    nearest = {
        (int(row["train_seed"]), int(row["bank_count"])): float(row["nearest_median"])
        for row in nearest_rows
    }
    grouped = defaultdict(lambda: defaultdict(list))
    for row in diversity_rows:
        bank = int(row["bank_count"])
        seed = int(row["train_seed"])
        grouped[bank]["pairwise"].append(float(row["pairwise_q50"]))
        grouped[bank]["nearest"].append(nearest[(seed, bank)])
        grouped[bank]["rank"].append(float(row["pca_effective_rank"]))
        grouped[bank]["collisions"].append(int(row["exact_psf_collision_members"]))
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Post-hoc exploratory PSF-diversity statistics. Fourier distances use the frozen 64-dimensional feature contract. Values are means and sample SDs across the retained mask-bank instantiations.}",
        r"\label{cw:psf-effective-diversity}",
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        "Training masks & Pairwise median & Dev nearest median & Effective rank & Exact collisions \\\\",
        r"\midrule",
    ]
    for bank in (100, 1000, 10_000):
        values = grouped[bank]
        lines.append(
            f"{bank:,}".replace(",", r"\,")
            + " & "
            + metric_text(statistics.mean(values["pairwise"]), statistics.stdev(values["pairwise"]), 4)
            + " & "
            + metric_text(statistics.mean(values["nearest"]), statistics.stdev(values["nearest"]), 4)
            + " & "
            + metric_text(statistics.mean(values["rank"]), statistics.stdev(values["rank"]), 2)
            + " & "
            + str(sum(values["collisions"]))
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    source_paths = {name: (REPO_ROOT / relative).resolve() for name, relative in SOURCES.items()}
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    readiness = json.loads(source_paths["readiness"].read_text())
    if readiness.get("status") != "pass":
        raise ValueError("Report inserts require a passing pre-final readiness gate")
    output = REPO_ROOT / "outputs/coursework_pre_final_20260910/report_inserts_v2"
    output.mkdir(parents=True, exist_ok=False)

    control_rows = rows(source_paths["input_controls"])
    psf_rows = rows(source_paths["psf_aware"])
    diversity_rows = rows(source_paths["psf_diversity"])
    nearest_rows = rows(source_paths["psf_nearest"])
    diversity_by_bank = defaultdict(lambda: defaultdict(list))
    nearest_lookup = {
        (int(row["train_seed"]), int(row["bank_count"])): float(row["nearest_median"])
        for row in nearest_rows
    }
    for row in diversity_rows:
        bank = int(row["bank_count"])
        seed = int(row["train_seed"])
        diversity_by_bank[bank]["pairwise"].append(float(row["pairwise_q50"]))
        diversity_by_bank[bank]["nearest"].append(nearest_lookup[(seed, bank)])
        diversity_by_bank[bank]["rank"].append(float(row["pca_effective_rank"]))
    controls_tex = controls_table(control_rows)
    psf_tex = psf_aware_table(psf_rows)
    diversity_tex = diversity_table(diversity_rows, nearest_rows)
    (output / "input_use_controls_table.tex").write_text(controls_tex)
    (output / "psf_aware_development_table.tex").write_text(psf_tex)
    (output / "psf_diversity_table.tex").write_text(diversity_tex)

    inner = next(
        row
        for row in psf_rows
        if row["view"] == "inner68" and row["method"] == "Published PSF-aware reference"
    )
    outer = next(
        row
        for row in psf_rows
        if row["view"] == "outer17" and row["method"] == "Published PSF-aware reference"
    )
    proposed = f"""# Proposed report inserts

These fragments are prepared for review. `current_report_for_analysis.tex` was not edited.

## Methodology: input-use control

To check whether a reconstructor uses scene-specific information in the measurement, we evaluated the fixed final checkpoints under three input conditions. The Correct condition uses the original measurement-target pair. In the Scene-shuffled condition, measurements are cyclically permuted among scenes recorded with the same mask, with no fixed points, while targets remain unchanged. The Train-mean condition predicts the mean training target and therefore does not use the measurement. All conditions use the same development rows and metric implementation as the corresponding main evaluation.

## Results: input-use control

For X-Restormer, within-mask scene shuffling reduced PSNR by 6.86 dB on MIRFLICKR and by 6.90 dB on CelebA. The direction was the same for SSIM and LPIPS. The model therefore uses information about the particular scene rather than producing only a dataset-level prior. The effect was weaker for CelebA DRUNet: shuffling reduced pooled PSNR by 0.97 dB and pooled SSIM by 0.013 on average, while the LPIPS change was small and varied between retained runs. The DRUNet result is evidence of input use, but not of an equally strong effect under every metric.

Suggested table: `input_use_controls_table.tex`.

## Methodology: PSF-aware development reference

We evaluated the fixed published PSF-aware checkpoint from~\\cite{{bezzam2025towards}} on the same real development rows used for the PSF-free evaluation. Row identities, the $[80,100,200,266]$ crop, the $180^\\circ$ measurement rotation, independent per-image peak normalization, PSNR/SSIM/LPIPS implementations, and mask-balanced aggregation were kept unchanged. The reference receives the PSF at inference. It is still a contextual comparison because the published checkpoint was trained with a different recipe and had access to a broader set of operators.

## Results: PSF-aware development reference

The published PSF-aware model reached {float(inner['PSNR_mean']):.4f} dB PSNR, {float(inner['SSIM_mean']):.4f} SSIM, and {float(inner['LPIPS_mean']):.4f} LPIPS on the 68-mask inner development view. On the 17-mask outer view, it reached {float(outer['PSNR_mean']):.4f} dB, {float(outer['SSIM_mean']):.4f}, and {float(outer['LPIPS_mean']):.4f}. The PSNR gap from the mean PSF-free model trained on real measurements was 6.70 dB and 6.89 dB, respectively. Because PSF access and training are not matched, these gaps describe the practical separation between the evaluated systems; they do not estimate the causal effect of supplying a PSF.

Suggested table: `psf_aware_development_table.tex`.

## Methodology and results: effective PSF diversity

As a post-hoc exploratory analysis, each generated PSF was converted to grayscale, normalized by total spatial energy, and represented by the non-DC magnitude of its centered Fourier transform. The `log1p` spectrum was pooled to $8\\times8$ and L2-normalized. We compared pairwise distances within the nested 100, 1,000, and 10,000-mask banks, nearest-training distances for the fixed development masks, the PCA spectrum and effective rank, radial-frequency energy, and exact SHA256 collisions. Near-collisions in the 10,000-mask bank were audited on a fixed sample of pairs, so their absence cannot establish that no close pair exists anywhere in the bank.

The typical pairwise distance changed little as the bank grew: the mean of the bank-level medians was {statistics.mean(diversity_by_bank[100]['pairwise']):.4f}, {statistics.mean(diversity_by_bank[1000]['pairwise']):.4f}, and {statistics.mean(diversity_by_bank[10000]['pairwise']):.4f}. In contrast, the median distance from a development PSF to its nearest training PSF fell from {statistics.mean(diversity_by_bank[100]['nearest']):.4f} to {statistics.mean(diversity_by_bank[1000]['nearest']):.4f} and {statistics.mean(diversity_by_bank[10000]['nearest']):.4f}. Effective rank increased from {statistics.mean(diversity_by_bank[100]['rank']):.2f} to {statistics.mean(diversity_by_bank[1000]['rank']):.2f} and {statistics.mean(diversity_by_bank[10000]['rank']):.2f}. No exact pattern or PSF collision was found. Thus, larger banks cover the same feature space more densely and add some effective directions, but they do not make individual PSFs progressively farther apart. This helps interpret the weak 100-versus-1,000 reconstruction difference under fixed compute: additional coverage is real, yet each mask is shown less often and the geometric gain is modest after 1,000 masks.

Suggested table: `psf_diversity_table.tex`.

## Final-test protocol (future Methods paragraph)

Before accessing the reserved synthetic test, we froze the 12-checkpoint X-Restormer matrix for 100/1,000 masks and scratch/GoPro initialization, together with one previously selected 100k-step finalist. The manifest fixes 256 scene IDs, 100 mask IDs, preprocessing, metrics, aggregation, qualitative examples, checkpoint hashes, endpoint metadata, source hashes, and the evaluator hash. The evaluator defaults to metadata-only preflight and refuses execution without a separate authorization file. No test scene file was opened, no test mask was generated, and no model forward was performed during this freeze.
"""
    (output / "PROPOSED_INSERTS.md").write_text(proposed)
    generated = sorted(path for path in output.iterdir() if path.is_file())
    provenance = {
        "status": "pass",
        "report_edited": False,
        "source_artifacts": [
            {"path": str(path), "sha256": sha256(path)}
            for path in source_paths.values()
        ],
        "generated_artifacts": [
            {"path": str(path), "sha256": sha256(path)} for path in generated
        ],
        "final_synthetic_test_accessed": False,
        "official_real_test_accessed": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
