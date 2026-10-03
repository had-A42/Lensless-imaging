#!/usr/bin/env python3
"""Plot recorded oracle gradient diagnostics; no parameter fitting."""
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


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("directory",type=Path)
    args=parser.parse_args(); directory=args.directory
    rows=list(csv.DictReader((directory/"per_sample.csv").open()))
    summary=json.loads((directory/"summary.json").read_text())
    masks=1+max(int(r["mask_order"]) for r in rows); scenes=1+max(int(r["scene_order"]) for r in rows)
    rows=[r for r in rows if int(r["mask_order"])>=masks//2 and int(r["scene_order"])>=scenes//2]
    etas=sorted({float(r["eta"]) for r in rows})
    arms=("true_psf","mean_train_psf","wrong_psf")
    labels=("True PSF (oracle)","Mean training PSF (fixed)","Wrong PSF (control)")
    colors=("#157f70","#2e71bb","#d16e24")
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for axis,metric in zip(axes,("PSNR","SSIM","LPIPS")):
        baseline={r["sample_id"]:float(r[metric]) for r in rows if r["arm"]=="true_psf" and float(r["eta"])==0.}
        for arm,label,color in zip(arms,labels,colors):
            means=[np.mean([float(r[metric])-baseline[r["sample_id"]] for r in rows if r["arm"]==arm and float(r["eta"])==eta]) for eta in etas]
            axis.plot(etas,means,"o-",color=color,label=label)
        axis.axhline(0,color="black",linewidth=.7); axis.grid(alpha=.2)
        axis.set_xscale("symlog",linthresh=.001); axis.set_xlim(-.0001,.0324)
        axis.set_xlabel("RMS-normalized step")
        axis.set_title(f"Delta {metric}: {'lower' if metric=='LPIPS' else 'higher'} is better")
    fig.suptitle("Physical-gradient diagnostic: previously used development quadrant")
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="lower center",ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.1,1,.93))
    fig.savefig(directory/"oracle_gradient_comparison.png",dpi=170)
    fig.savefig(directory/"oracle_gradient_comparison.pdf")
    plt.close(fig)
    print(json.dumps({k:summary[k] for k in ("selected_eta","confirmation","true_minus_wrong","signal")},indent=2))


if __name__=="__main__":main()
