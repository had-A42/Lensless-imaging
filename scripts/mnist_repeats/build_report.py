"""Build the combined MNIST table from completed fixed-10k endpoints.

Keeps the original seed-52 results in the evidence and computes both the
requested 42/100/62 summary and the four-run sensitivity summary.
Writes reviewable artifacts; applying the report update is a separate step.
"""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "outputs/mnist_seed100_20260907"
ARCHIVE = REPO / "wandb/remote-a800/cleanup-metadata-20260831"
METRICS = ("PSNR", "SSIM", "PSNR_32", "SSIM_32")
SELECTED = (42, 100, 62)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def per_mask(path):
    rows = read_csv(path)
    assert len(rows) == 32
    assert len({r["mask_id"] for r in rows}) == 32
    assert {int(r["sample_count"]) for r in rows} == {100}
    values = {m: statistics.mean(float(r[m]) for r in rows) for m in METRICS}
    assert all(math.isfinite(x) for x in values.values())
    return values


def run_dir(mode, seed):
    if seed == 100:
        return OUTPUT / "remote_results" / mode / "saved" / f"mnist-{mode}-10k-seed100"
    if mode == "rgb":
        return ARCHIVE / "Lensless-imaging-phy/saved" / f"a800-mnist-cv1-corrected-rgb-10k-seed{seed}"
    return ARCHIVE / "Lensless-imaging-night-73e432d/saved" / f"a800-mnist-cv2-gray-fp32-10k-seed{seed}-73e432d"


def aggregate(rows, seeds):
    chosen = [r for r in rows if r["seed"] in seeds]
    assert len(chosen) == len(seeds)
    return {m: {"mean": statistics.mean(r[m] for r in chosen),
                "sample_sd": statistics.stdev(r[m] for r in chosen)} for m in METRICS}


def main():
    out = OUTPUT / "report"
    out.mkdir(parents=True, exist_ok=True)
    rows, curves, sources = [], [], {}
    for mode in ("rgb", "gray"):
        for seed in (42, 52, 62, 100):
            folder = run_dir(mode, seed)
            if seed == 100:
                complete = json.loads((folder / "complete.json").read_text())
                assert complete["status"] == "complete" and not complete["technical_smoke"]
                assert complete["steps"] == complete["sampler_steps"] == complete["schedule_horizon"] == 10000
                assert complete["mode"] == mode and complete["seed"] == seed
            for epoch in range(1, 5):
                p = folder / f"validation_per_mask_epoch{epoch:04d}.csv"
                values = per_mask(p)
                sources[str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()
                curves.append({"mode": mode, "seed": seed, "steps": epoch * 2500, **values})
                if epoch == 4:
                    rows.append({"mode": mode, "seed": seed, **values})
                    if seed == 100:
                        for m, value in values.items():
                            assert abs(value - complete["metrics"][f"validation_{m}_mask_balanced"]) < 1e-9
    summaries = {}
    for mode in ("rgb", "gray"):
        selected_rows = [r for r in rows if r["mode"] == mode]
        summaries[mode] = {"selected_three": aggregate(selected_rows, SELECTED),
                           "all_four": aggregate(selected_rows, (42, 52, 62, 100))}
    for name, data in (("all_endpoints.csv", rows), ("all_curves.csv", curves)):
        with (out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    evidence = {"selected_seeds": list(SELECTED), "previous_seeds": [42, 52, 62],
                "selection_after_inspection": True, "seed52_technical_failure_found": False,
                "aggregation": "mean of 32 per-mask means (100 scenes each), then mean and sample SD across training runs; ddof=1",
                "summaries": summaries, "all_endpoints": rows, "source_sha256": sources}
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{MNIST: RGB and grayscale reconstruction after 10k steps, with 100 training masks and 32 unseen validation masks paired with 100 held-out scenes. Entries are mean $\pm$ sample SD across training seeds 42/100/62 ($n=3$).}",
             r"\label{cw:mnist-rgb-results}", r"\label{cw:mnist-gray-results}",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Channels & PSNR$_{256}$, dB & SSIM$_{256}$ & PSNR$_{32}$, dB & SSIM$_{32}$ \\", r"\midrule"]
    for mode, label in (("rgb", "RGB"), ("gray", "Grayscale")):
        data = summaries[mode]["selected_three"]
        lines.append(label + " & " + " & ".join(f"${data[m]['mean']:.4f}\\pm{data[m]['sample_sd']:.4f}$" for m in METRICS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "combined_table.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
