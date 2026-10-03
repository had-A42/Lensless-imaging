"""Plot the selected MNIST repeats with English labels; preserve old assets."""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "outputs/mnist_seed100_20260907/report"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mnist-seed100-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = (42, 100, 62)
COLORS = {42: "#0072B2", 100: "#D55E00", 62: "#009E73"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    with (OUTPUT / "all_curves.csv").open() as f:
        curves = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.8))
    for seed in SEEDS:
        rows = [r for r in curves if r["mode"] == "rgb" and int(r["seed"]) == seed]
        assert len(rows) == 4
        for ax, metric in zip(axes, ("PSNR_32", "SSIM_32")):
            ax.plot([int(r["steps"]) / 1000 for r in rows], [float(r[metric]) for r in rows],
                    "-o", color=COLORS[seed], label=f"Seed {seed}")
    for ax, title, ylabel in zip(axes, ("Pixel accuracy", "Structural similarity"),
                                 ("PSNR at 32 × 32 (dB)", "SSIM at 32 × 32")):
        ax.set(title=title, xlabel="Training steps (thousands)", ylabel=ylabel, xticks=[2.5, 5, 7.5, 10])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "mnist_rgb_learning_en")

    predictions, targets, evidence = {}, None, []
    for seed, run in ((42, "offline-run-20260826_153214-a6i6cnc2"),
                      (62, "offline-run-20260826_173030-3vm246so")):
        folder = REPO / "wandb/remote-a800/20260826-phy" / run / "files"
        summary = json.loads((folder / "wandb-summary.json").read_text())
        assert summary["_step"] == 10000
        entry = summary["validation/prediction_target"]
        path = folder / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        pixels = np.array(Image.open(path).convert("RGB"))
        assert pixels.shape == (1034, 518, 3)
        predictions[seed] = np.stack([pixels[2+i*258:258+i*258, 2:258] for i in range(4)])
        target = np.stack([pixels[2+i*258:258+i*258, 260:516] for i in range(4)])
        if targets is None:
            targets = target
        else:
            assert np.array_equal(targets, target)
        evidence.append({"seed": seed, "source": str(path.relative_to(REPO)), "sha256": entry["sha256"]})
    path = OUTPUT.parent / "remote_results/rgb/saved/mnist-rgb-10k-seed100/examples-step10000.pth"
    panel = torch.load(str(path), map_location="cpu", weights_only=True)
    assert panel["seed"] == 100 and panel["step"] == 10000
    assert panel["scene_id"] == ["mnist_train_05243", "mnist_train_32198", "mnist_train_57859", "mnist_train_18032"]
    assert set(panel["mask_id"]) == {"validation_00031"}
    target = (panel["target"].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    assert np.array_equal(targets, target), "New repeat must use the same four targets"
    predictions[100] = (panel["prediction"].clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    evidence.append({"seed": 100, "source": str(path.relative_to(REPO)),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "selection": panel["selection"]})
    fig, axes = plt.subplots(4, 4, figsize=(8.2, 8.5))
    for row in range(4):
        for col, pixels in enumerate([targets[row]] + [predictions[s][row] for s in SEEDS]):
            axes[row, col].imshow(pixels, interpolation="nearest")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(["Ground truth", "Seed 42", "Seed 100", "Seed 62"][col])
    fig.subplots_adjust(left=.01, right=.99, top=.95, bottom=.01, wspace=.025, hspace=.025)
    save(fig, "mnist_rgb_reconstructions_en")
    (OUTPUT / "figure_evidence.json").write_text(json.dumps({
        "seeds": list(SEEDS), "same_targets": True, "numeric_metrics_from_images": False,
        "example_selection": "same last logged validation batch for every repeat", "sources": evidence,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
