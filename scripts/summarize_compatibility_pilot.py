#!/usr/bin/env python3
"""Plot recorded PSFF-COMPAT-01 artifacts without fitting or selecting a model."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    summary = json.loads((directory / "summary.json").read_text())
    rows = list(csv.DictReader((directory / "per_sample.csv").open()))
    masks = 1 + max(int(r["mask_order"]) for r in rows)
    scenes = 1 + max(int(r["scene_order"]) for r in rows)
    confirmation = [r for r in rows if int(r["mask_order"]) >= masks//2 and int(r["scene_order"]) >= scenes//2]
    etas = sorted({float(r["eta"]) for r in rows})
    arms = ("correct", "wrong_scene", "negative", "random")
    colors = ("#157F70", "#D26C26", "#7963A6", "#777777")
    labels = ("Correct measurement", "Wrong scene, same mask", "Negative gradient", "Random direction")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric in zip(axes, ("PSNR", "SSIM", "LPIPS")):
        base = {r["sample_id"]: float(r[metric]) for r in confirmation if r["arm"] == "correct" and float(r["eta"]) == 0}
        for arm, color, label in zip(arms, colors, labels):
            means = []
            for eta in etas:
                selected = [r for r in confirmation if r["arm"] == arm and float(r["eta"]) == eta]
                means.append(np.mean([float(r[metric])-base[r["sample_id"]] for r in selected]))
            axis.plot(etas, means, marker="o", label=label, color=color)
        axis.axhline(0, color="black", linewidth=.7)
        axis.axvline(summary["selected_eta"], color="black", linestyle=":", linewidth=1, label="Selected on calibration")
        axis.set_xscale("symlog", linthresh=.001)
        axis.set_xlim(-.0001, max(etas) * 1.08)
        axis.set_xlabel("RMS-normalized step")
        axis.set_title(f"Change in {metric} ({'lower' if metric == 'LPIPS' else 'higher'} is better)")
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("PSFF-COMPAT-01: confirmation quadrant (development data)")
    figure.tight_layout(rect=(0, .16, 1, .93))
    figure.savefig(directory / "gradient_step_comparison.png", dpi=170)
    figure.savefig(directory / "gradient_step_comparison.pdf")
    plt.close(figure)
    records = [json.loads(line) for line in (directory / "training.jsonl").read_text().splitlines()]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot([r["step"] for r in records], [r["loss"] for r in records], color=colors[0])
    axes[0].set_ylabel("Symmetric contrastive loss")
    axes[1].plot([r["step"] for r in records], [r["retrieval"] for r in records], color=colors[0])
    axes[1].set_ylabel("Training retrieval (single logged batch)")
    for axis in axes:
        axis.set_xlabel("Optimizer step"); axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(directory / "training.png", dpi=170)
    plt.close(figure)
    print(json.dumps({"directory": str(directory), "selected_eta": summary["selected_eta"],
                      "go": summary["exploratory_go"], "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
