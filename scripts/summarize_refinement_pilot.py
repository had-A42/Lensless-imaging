#!/usr/bin/env python3
"""Render recorded REFINE-01 metrics and predetermined examples on the laptop."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parents[1]/".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    directory=args.directory
    summary=json.loads((directory/"summary.json").read_text())
    names=[name for name in summary["metrics"] if name!="baseline"]
    labels={"self_context":"Refiner: initial image only","measurement_context":"Refiner: full measurement",
            "wrong_context":"Refiner: wrong measurement","continued":"Continued X-Restormer","baseline":"Frozen X-Restormer","target":"Target"}
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for axis,metric in zip(axes,("PSNR","SSIM","LPIPS")):
        estimates=[summary["contrasts"][f"{name}_minus_baseline"][metric] for name in names]
        means=np.array([e["mean"] for e in estimates])
        errors=np.array([[e["mean"]-e["lo"] for e in estimates],[e["hi"]-e["mean"] for e in estimates]])
        axis.errorbar(range(len(names)),means,yerr=errors,fmt="o",color="#157f70",capsize=4)
        axis.axhline(0,color="black",linewidth=.7)
        axis.set_xticks(range(len(names)),[labels[n] for n in names],rotation=25,ha="right",fontsize=8)
        axis.set_title(f"Delta {metric}: {'lower' if metric=='LPIPS' else 'higher'} is better")
        axis.grid(axis="y",alpha=.2)
    fig.suptitle("PSFF-REFINE-01: development grid; paired two-way bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(directory/"comparison.png",dpi=170)
    fig.savefig(directory/"comparison.pdf")
    plt.close(fig)
    examples=torch.load(directory/"qualitative_fixed.pt",map_location="cpu",weights_only=False)
    if examples:
        columns=["target","baseline"]+[n for n in names if n!="wrong_context"]
        fig,axes=plt.subplots(len(examples),len(columns),figsize=(3*len(columns),2.5*len(examples)),squeeze=False)
        for row,example in enumerate(examples):
            for col,name in enumerate(columns):
                x=example[name].float()
                x=(x/x.max().clamp_min(1e-8)).clamp(0,1).permute(1,2,0).numpy()
                axes[row,col].imshow(x)
                axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
                if row==0: axes[row,col].set_title(labels[name])
                if col==0:
                    meta=example["metadata"]
                    axes[row,col].set_ylabel(f"mask {meta['mask_order']}, scene {meta['scene_order']}")
        fig.suptitle("Fixed examples selected before results; display clips to [0,1]")
        fig.tight_layout(rect=(0,0,1,.98))
        fig.savefig(directory/"qualitative_fixed.png",dpi=130)
        plt.close(fig)
    records=[json.loads(line) for line in (directory/"training.jsonl").read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for name in names:
        if name=="wrong_context":continue
        for axis,metric in zip(axes,("mse_loss","lpips_loss")):
            axis.plot([r["step"] for r in records],[r[f"{name}/{metric}"] for r in records],label=labels[name],alpha=.8)
            axis.set_title(f"Training {metric} (logged batches)")
            axis.set_xlabel("Optimizer step"); axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(directory/"training.png",dpi=160); plt.close(fig)
    print(json.dumps(summary["metrics"],indent=2))


if __name__=="__main__":main()
