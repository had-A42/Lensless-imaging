#!/usr/bin/env python3
"""Bounded exploratory compatibility/gradient experiment; see PSFF-COMPAT-01.

Run from the repository root. Imports project simulation and metrics unchanged.
The candidate gradient has no access to ground truth or PSF. All selection is
post-training on an explicitly marked development quadrant, never a test split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ETAS = (0.0, 0.001, 0.003, 0.01, 0.03)
ARMS = ("correct", "wrong_scene", "negative", "random")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def peak(value):
    return value / value.amax((1, 2, 3), keepdim=True).clamp_min(1e-8)


def rms_unit(value):
    return value / value.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12)


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers, previous = [], 3
        for channels in (24, 48, 96, 128):
            layers.extend([
                nn.Conv2d(previous, channels, 3, stride=2, padding=1),
                nn.GroupNorm(8, channels), nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels), nn.SiLU(),
            ])
            previous = channels
        self.body = nn.Sequential(*layers)
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8 * 12, 512), nn.LayerNorm(512))

    def forward(self, value):
        value = F.adaptive_avg_pool2d(self.body(value), (8, 12))
        return F.normalize(self.project(value), dim=-1)


class Compatibility(nn.Module):
    def __init__(self):
        super().__init__()
        self.measurement_encoder = SpatialEncoder()
        self.image_encoder = SpatialEncoder()

    def logits(self, measurement, candidate):
        return self.measurement_encoder(measurement) @ self.image_encoder(candidate).T / 0.07


def score_gradient(model, measurement, candidate):
    """Inputs deliberately exclude target and operator metadata."""
    with torch.no_grad():
        measured = model.measurement_encoder(measurement).detach()
    variable = candidate.detach().clone().requires_grad_(True)
    score = (measured * model.image_encoder(variable)).sum(-1)
    gradient, = torch.autograd.grad(score.sum(), variable)
    if not torch.isfinite(gradient).all():
        raise RuntimeError("Nonfinite candidate gradient")
    return gradient.detach(), score.detach()


def two_way_interval(matrix, repeats=2000):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Expected a complete finite mask-by-scene grid")
    rng = np.random.default_rng(20260906)
    masks = rng.integers(0, matrix.shape[0], (repeats, matrix.shape[0]))
    scenes = rng.integers(0, matrix.shape[1], (repeats, matrix.shape[1]))
    means = matrix[masks[:, :, None], scenes[:, None, :]].mean((1, 2))
    return {"mean": float(matrix.mean()), "lo": float(np.quantile(means, .025)),
            "hi": float(np.quantile(means, .975))}


def summarize(rows, masks, scenes):
    # Same predeclared rectangle for all arms; no per-image best-of selection.
    def grid(arm, eta, metric, part):
        mh, sh = masks // 2, scenes // 2
        selected = [r for r in rows if r["arm"] == arm and r["eta"] == eta
                    and ((r["mask_order"] < mh and r["scene_order"] < sh) if part == "calibration"
                         else (r["mask_order"] >= mh and r["scene_order"] >= sh))]
        selected.sort(key=lambda r: (r["mask_order"], r["scene_order"]))
        shape = (mh, sh) if part == "calibration" else (masks - mh, scenes - sh)
        return np.array([r[metric] for r in selected]).reshape(shape)

    table = []
    for eta in ETAS:
        deltas = {m: float((grid("correct", eta, m, "calibration") -
                           grid("correct", 0., m, "calibration")).mean())
                  for m in ("PSNR", "SSIM", "LPIPS")}
        table.append({"eta": eta, **deltas,
                      "eligible": deltas["SSIM"] >= -.002 and deltas["LPIPS"] <= .005})
    chosen = max((r for r in table if r["eligible"]), key=lambda r: (r["PSNR"], -r["eta"]))["eta"]
    confirmation = {}
    for arm in ARMS:
        confirmation[arm] = {
            m: two_way_interval(grid(arm, chosen, m, "confirmation") -
                                grid("correct", 0., m, "confirmation"))
            for m in ("PSNR", "SSIM", "LPIPS")}
    correct, wrong = confirmation["correct"], confirmation["wrong_scene"]
    go = (chosen > 0 and correct["PSNR"]["mean"] >= .05 and correct["PSNR"]["lo"] > 0
          and correct["PSNR"]["mean"] > wrong["PSNR"]["mean"]
          and correct["SSIM"]["mean"] >= -.002 and correct["LPIPS"]["mean"] <= .005)
    baseline = {m: float(np.mean([r[m] for r in rows if r["arm"] == "correct" and r["eta"] == 0]))
                for m in ("PSNR", "SSIM", "LPIPS")}
    return {"status": "complete", "selected_eta": chosen, "exploratory_go": bool(go),
            "baseline_full_grid": baseline, "calibration": table, "confirmation": confirmation,
            "interpretation": "Development pilot, not a closed-test result or full CL-DPS reproduction."}


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    # Numerical directional derivative: catches score sign / detached input errors.
    model = Compatibility().eval()
    y, x = torch.rand(2, 3, 64, 96), torch.rand(2, 3, 48, 64)
    for p in model.parameters():
        p.requires_grad_(False)
    gradient, original = score_gradient(model, y, x)
    direction = rms_unit(gradient)
    with torch.no_grad():
        after = (model.measurement_encoder(y) * model.image_encoder(x + 1e-5 * direction)).sum(-1)
    assert (after > original).all(), (original, after)
    assert torch.isfinite(direction).all() and gradient.abs().sum() > 0
    # Paired bootstrap must preserve a known constant treatment effect.
    interval = two_way_interval(np.full((4, 5), .25))
    assert interval == {"mean": .25, "lo": .25, "hi": .25}
    # Check selection: deliberately attractive confirmation cannot pick eta.
    rows = []
    for m in range(4):
        for s in range(4):
            for arm in ARMS:
                for eta in ETAS:
                    effect = (1.0 if eta == .003 else -1.0) if m < 2 and s < 2 else 0.2
                    if eta == 0:
                        effect = 0
                    rows.append(dict(mask_order=m, scene_order=s, arm=arm, eta=eta,
                                     PSNR=15 + effect, SSIM=.3, LPIPS=.5))
    assert summarize(rows, 4, 4)["selected_eta"] == .003
    print("PASS: gradient direction, finite gradients, paired bootstrap, calibration-only selection", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--base-run", type=Path, default=Path("saved/scale100k-xrest-gopro-finite-100-seed42-508a878-r3"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-masks", type=int, default=32)
    parser.add_argument("--val-scenes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None or min(args.steps, args.batch_size) < 1:
        parser.error("Provide a new --output directory and positive steps/batch size")
    if args.val_masks < 2 or args.val_scenes < 2 or args.batch_size < 2:
        parser.error("At least two masks/scenes and batch size 2 are needed")
    if args.val_masks > 32 or args.val_scenes > 32:
        parser.error("Pilot is limited to the existing 32x32 development grid")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA device through CUDA_VISIBLE_DEVICES")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output
    save_json(output / "status.json", {"status": "starting", "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}})
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from src.datasets.data_utils import get_dataloaders
    from src.metrics.reconstruction import PSNRMetric, SSIMMetric, LPIPSMetric

    config_path = args.base_run / "config.yaml"
    checkpoint_path = args.base_run / "checkpoint-epoch10.pth"
    config = OmegaConf.load(config_path)
    # The completed full checkpoint contains all weights, no external download.
    config.initialization.checkpoint_path = None
    config.model.checkpoint_path = None
    config.trainer.total_steps = args.steps
    config.trainer.seed = args.seed
    config.dataloader_builder.batch_size = args.batch_size
    config.dataloader_builder.num_workers = args.workers
    config.dataloader_builder.validation_mask_count = args.val_masks
    config.dataloader_builder.validation_scenes_per_mask = args.val_scenes
    config.dataloader_builder.train_mask_seed = 42
    config.dataloader_builder.evaluation_mask_seed = 42
    config.dataloader_builder.psf_cache.warmup = False
    resolved = OmegaConf.to_container(config, resolve=True)
    if resolved["protocol"]["version"] != "corrected-v1" or resolved["dataloader_builder"]["simulation_mode"] != "roi_convolution":
        raise ValueError("Wrong simulation contract")
    (output / "source_config.yaml").write_text(config_path.read_text())
    OmegaConf.save(OmegaConf.create(resolved), output / "resolved_config.yaml")
    provenance = {
        "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "source_config_sha256": file_hash(config_path), "checkpoint_sha256": file_hash(checkpoint_path),
        "script_sha256": file_hash(Path(__file__)), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "etas": ETAS, "arms": ARMS, "train_examples": args.steps * args.batch_size,
        "source_hashes": {str(p): file_hash(ROOT / p) for p in (
            Path("src/datasets/on_the_fly.py"), Path("src/model/psf_free_xrestormer.py"),
            Path("src/metrics/reconstruction.py"), Path("src/loss/reconstruction.py"),
            Path("manifests/mirflickr25k_splits.json"))},
    }
    save_json(output / "provenance.json", provenance)
    loaders, transforms = get_dataloaders(config, "cuda")
    if transforms:
        raise RuntimeError("Unexpected transforms; audit before using them")
    model = Compatibility().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    provenance["trainable_parameters"] = sum(p.numel() for p in model.parameters())
    save_json(output / "provenance.json", provenance)
    writer = None
    try:
        import wandb
        writer = wandb.init(project="lensless-imaging", entity="had-2005-hse-university", mode="offline",
                            name=output.name, group="psff-compat-01", dir=str(output), config=provenance)
    except ImportError:
        pass
    save_json(output / "status.json", {"status": "training"})
    model.train()
    with (output / "training.jsonl").open("w", buffering=1) as stream:
        for step, batch in enumerate(loaders["train"], start=1):
            if len(set(batch["scene_id"])) != len(batch["scene_id"]):
                raise RuntimeError("False negative: duplicate scene within contrastive batch")
            y, target = batch["measurement"].cuda(), peak(batch["target"].cuda())
            if tuple(y.shape[1:]) != (3,380,507) or tuple(target.shape[1:]) != (3,200,266):
                raise ValueError("Wrong sensor/ROI shapes")
            noise = torch.rand(len(y), 1, 1, 1, device="cuda") * .1
            candidate = (target + noise * torch.randn_like(target)).clamp(0, 1)
            logits = model.logits(y, candidate)
            labels = torch.arange(len(y), device="cuda")
            loss = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 50 == 0 or step == args.steps:
                record = {"step": step, "loss": float(loss.detach()),
                          "retrieval": float((logits.argmax(-1) == labels).float().mean()),
                          "seconds": time.monotonic() - started}
                stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                if writer:
                    writer.log(record, step=step)
    torch.save({"state_dict": model.state_dict(), "steps": args.steps, "provenance": provenance}, output / "compatibility_final.pth")
    model.eval().requires_grad_(False)
    del optimizer
    save_json(output / "status.json", {"status": "evaluating"})
    base = instantiate(config.model).cuda().eval().requires_grad_(False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base.load_state_dict(state["state_dict"], strict=True)
    del state
    metrics = {"PSNR": PSNRMetric(normalize_by_max=True), "SSIM": SSIMMetric(normalize_by_max=True),
               "LPIPS": LPIPSMetric(net_type="vgg", device="cuda", normalize_by_max=True)}
    rows, diagnostics, mask_order, scene_order = [], [], {}, {}
    batch_index = 0
    with (output / "per_sample.csv").open("w", newline="") as stream:
        csv_writer = None
        for batch in loaders["validation"]:
            batch_index += 1
            y, target = batch["measurement"].cuda(), peak(batch["target"].cuda())
            if len(y) < 2 or len(set(batch["mask_id"])) != 1:
                raise ValueError("Wrong-scene control needs >=2 scenes of the same mask")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                raw_base = torch.cat([base(measurement=item[None])["prediction"].float() for item in y])
            b = peak(raw_base)
            gradient, _ = score_gradient(model, y, b)
            wrong, _ = score_gradient(model, y.roll(1, 0), b)
            directions = {"correct": rms_unit(gradient), "wrong_scene": rms_unit(wrong),
                          "negative": -rms_unit(gradient), "random": rms_unit(torch.randn_like(b))}
            with torch.no_grad():
                retrieval = model.logits(y, target)
                retrieval_hit = (retrieval.argmax(-1) == torch.arange(len(y), device="cuda")).cpu().tolist()
                alignment = F.cosine_similarity(gradient.flatten(1), (target-b).flatten(1), dim=-1).cpu().tolist()
            metadata = []
            for i, sample_id in enumerate(batch["sample_id"]):
                mid, sid = batch["mask_id"][i], batch["scene_id"][i]
                mask_order.setdefault(mid, len(mask_order)); scene_order.setdefault(sid, len(scene_order))
                meta = dict(sample_id=sample_id, mask_id=mid, scene_id=sid, source_index=int(batch["source_index"][i]),
                            mask_order=mask_order[mid], scene_order=scene_order[sid],
                            measurement_sha256=tensor_hash(y[i]), target_sha256=tensor_hash(batch["target"][i]),
                            base_sha256=tensor_hash(raw_base[i]))
                metadata.append(meta)
                diagnostics.append({**meta, "gradient_alignment": alignment[i], "retrieval_hit": retrieval_hit[i],
                                    "retrieval_candidates": len(y)})
            # Baseline evaluated once; identical eta=0 rows for all direction controls.
            cached_baseline = None
            for arm, direction in directions.items():
                for eta in ETAS:
                    candidate = b + eta * direction
                    if eta == 0 and cached_baseline is not None:
                        values = cached_baseline
                    else:
                        with torch.no_grad():
                            values = {name: metric.per_image(candidate, target).cpu().tolist() for name,metric in metrics.items()}
                            values["MSE"] = (peak(candidate)-target).square().flatten(1).mean(-1).cpu().tolist()
                        if eta == 0:
                            cached_baseline = values
                    for i, meta in enumerate(metadata):
                        row = {**meta, "arm": arm, "eta": eta, **{name:value[i] for name,value in values.items()}}
                        if not all(np.isfinite(row[name]) for name in values):
                            raise RuntimeError("Nonfinite evaluation metric")
                        if csv_writer is None:
                            csv_writer = csv.DictWriter(stream, fieldnames=list(row)); csv_writer.writeheader()
                        csv_writer.writerow(row)
                        rows.append(row)
            stream.flush()
            print(json.dumps({"evaluation_batch": batch_index, "samples": len(diagnostics),
                              "seconds": time.monotonic()-started}), flush=True)
    if len(diagnostics) != args.val_masks * args.val_scenes or len({r["sample_id"] for r in diagnostics}) != len(diagnostics):
        raise RuntimeError("Incomplete or duplicate evaluation grid")
    result = summarize(rows, args.val_masks, args.val_scenes)
    result.update({"retrieval": float(np.mean([r["retrieval_hit"] for r in diagnostics])),
                   "gradient_alignment_mean": float(np.mean([r["gradient_alignment"] for r in diagnostics])),
                   "seconds": time.monotonic()-started,
                   "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                   "technical_smoke": args.steps != 2000 or args.val_masks != 32 or args.val_scenes != 32})
    if not result["technical_smoke"]:
        reference = {"PSNR":14.923134836368263, "SSIM":.3443276314283139, "LPIPS":.5560499958810396}
        differences = {m: result["baseline_full_grid"][m]-v for m,v in reference.items()}
        result["baseline_parity_delta"] = differences
        result["baseline_parity_pass"] = all(abs(differences[m]) < tol for m,tol in {"PSNR":.01,"SSIM":.001,"LPIPS":.001}.items())
        if not result["baseline_parity_pass"]:
            result["exploratory_go"] = False
            result["status"] = "requires_baseline_audit"
    save_json(output / "diagnostics.json", diagnostics)
    save_json(output / "summary.json", result)
    save_json(output / "status.json", {"status": result["status"], "exploratory_go": result["exploratory_go"]})
    if writer:
        writer.summary.update(result)
        writer.finish()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
