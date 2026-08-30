import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from lensless import ADMM
from omegaconf import DictConfig, OmegaConf

from src.datasets.mirflickr import MirFlickrSceneDataset
from src.datasets.on_the_fly import (
    DigiCamOnTheFlyDataset,
    _prepare_convolution_psf,
    build_digicam_mask,
)
from src.digicam_synth.mask_protocol import get_mask_records
from src.digicam_synth.psf_cache import PSFCache
from src.metrics.reconstruction import LPIPSMetric, PSNRMetric, SSIMMetric


def _to_dhwc(image):
    return image.movedim(0, -1).unsqueeze(0).contiguous()


def _to_chw(image):
    return image.squeeze(0).movedim(-1, 0).contiguous()


def shift_psf(psf, dy=0, dx=0):
    """Translate a CHW PSF with zero padding instead of circular wrap."""
    shifted = torch.zeros_like(psf)
    height, width = psf.shape[-2:]
    source_y = slice(max(-dy, 0), height - max(dy, 0))
    target_y = slice(max(dy, 0), height - max(-dy, 0))
    source_x = slice(max(-dx, 0), width - max(dx, 0))
    target_x = slice(max(dx, 0), width - max(-dx, 0))
    shifted[..., target_y, target_x] = psf[..., source_y, source_x]
    return shifted / shifted.norm().clamp_min(1e-12)


def paired_two_way_ci(values, samples=10_000, seed=20260827):
    """Mean and paired bootstrap interval for a complete mask-by-scene grid."""
    values = np.asarray(values, dtype=np.float64)
    mask_count, scene_count = values.shape
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, samples, 500):
        count = min(500, samples - start)
        masks = rng.integers(0, mask_count, size=(count, mask_count))
        scenes = rng.integers(0, scene_count, size=(count, scene_count))
        resampled = values[masks[:, :, None], scenes[:, None, :]]
        means.append(resampled.mean(axis=(1, 2)))
    means = np.concatenate(means)
    return {
        "mean": float(values.mean()),
        "ci_lower": float(np.quantile(means, 0.025)),
        "ci_upper": float(np.quantile(means, 0.975)),
    }


def _load_raw_psf(cache, simulator, record):
    seed = int(record["mask_seed"])
    raw = cache.load(seed, "finite")
    if raw is None:
        raw, _, _ = build_digicam_mask(simulator, seed, create_simulator=False)
        cache.save(seed, "finite", raw)
    return raw


def _prepared_psf(cache, simulator, record):
    return (
        _prepare_convolution_psf(_load_raw_psf(cache, simulator, record))
        .squeeze(0)
        .movedim(-1, 0)
        .contiguous()
    )


def _axial_psf(simulator, record, distance):
    changed = OmegaConf.create(OmegaConf.to_container(simulator, resolve=True))
    changed.optics.mask2sensor_min = float(distance)
    changed.optics.mask2sensor_max = float(distance)
    raw, _, _ = build_digicam_mask(
        changed,
        int(record["mask_seed"]),
        create_simulator=False,
    )
    return _prepare_convolution_psf(raw).squeeze(0).movedim(-1, 0).contiguous()


def _mean_psf(cache, simulator, records):
    psfs = [_prepared_psf(cache, simulator, record) for record in records]
    mean = torch.stack(psfs).mean(dim=0)
    return mean / mean.norm().clamp_min(1e-12)


def _arms(experiment, correct_psf, correct_record, wrong_psf, mean_psf, simulator):
    if experiment == "identity":
        return {
            "correct": (correct_psf, "identity", 0.0, "correct"),
            "wrong": (wrong_psf, "identity", 1.0, "wrong"),
            "mean": (mean_psf, "identity", 1.0, "mean"),
        }

    arms = {"nominal": (correct_psf, "nominal", 0.0, "nominal")}
    nominal_distance = float(simulator.optics.mask2sensor_min)
    for level in (0.01, 0.025, 0.05, 0.10):
        for sign, direction in ((-1, "negative"), (1, "positive")):
            name = f"axial_{'m' if sign < 0 else 'p'}{level:g}"
            arms[name] = (
                _axial_psf(
                    simulator,
                    correct_record,
                    nominal_distance * (1 + sign * level),
                ),
                "axial",
                level,
                direction,
            )
    for pixels in (1, 2, 4):
        for direction, (dy, dx) in {
            "up": (-pixels, 0),
            "down": (pixels, 0),
            "left": (0, -pixels),
            "right": (0, pixels),
        }.items():
            name = f"shift_{pixels:02d}px_{direction}"
            arms[name] = (
                shift_psf(correct_psf, dy=dy, dx=dx),
                "registration",
                float(pixels),
                direction,
            )
    arms["foreign"] = (wrong_psf, "foreign", 1.0, "wrong")
    return arms


def _metrics(device):
    common = dict(data_range=1.0, normalize_by_max=True, normalization_eps=1e-8)
    return {
        "PSNR": PSNRMetric(**common),
        "SSIM": SSIMMetric(**common),
        "LPIPS": LPIPSMetric(
            net_type="vgg",
            device=str(device),
            normalize_by_max=True,
            normalization_eps=1e-8,
        ),
    }


def _metric_values(metrics, prediction, target):
    prediction = prediction.unsqueeze(0)
    target = target.unsqueeze(0)
    return {
        name: float(metric.per_image(prediction, target)[0])
        for name, metric in metrics.items()
    }


def _solver(psf, config, device):
    return ADMM(
        _to_dhwc(psf).to(device),
        dtype=str(config.solver.dtype),
        n_iter=int(config.solver.n_iter),
        mu1=float(config.solver.mu1),
        mu2=float(config.solver.mu2),
        mu3=float(config.solver.mu3),
        tau=float(config.solver.tau),
        pad=False,
        norm="backward",
        denoiser=None,
    )


def _reconstruct(solver, measurement, roi, iterations):
    solver.set_data(_to_dhwc(measurement))
    prediction = solver.apply(
        n_iter=int(iterations),
        disp_iter=-1,
        plot=False,
        save=False,
        reset=True,
    )
    top, left, height, width = (int(value) for value in roi)
    return _to_chw(prediction)[:, top : top + height, left : left + width]


def _selected_scenes(config):
    dataset = MirFlickrSceneDataset(
        root_dir=to_absolute_path(str(config.data.root_dir)),
        splits_path=to_absolute_path(str(config.data.splits_path)),
        split="validation",
        verify_files=bool(config.data.verify_files),
    )
    rng = np.random.default_rng(np.random.SeedSequence([52, 59]))
    indices = rng.permutation(len(dataset))[: int(config.data.scene_count)]
    return [dataset[int(index)] for index in indices]


def _summarize(frame, config):
    index = ["mask_index", "scene_position"]
    summary = {"experiment": str(config.experiment), "comparisons": {}}
    metrics = ("PSNR", "SSIM", "LPIPS")

    if config.experiment == "identity":
        for control in ("wrong", "mean"):
            comparison = {}
            for metric in metrics:
                pivot = frame.pivot(index=index, columns="arm", values=metric)
                delta = (
                    pivot[control] - pivot["correct"]
                    if metric == "LPIPS"
                    else pivot["correct"] - pivot[control]
                )
                matrix = delta.to_numpy().reshape(
                    int(config.data.mask_count), int(config.data.scene_count)
                )
                comparison[metric] = paired_two_way_ci(
                    matrix,
                    samples=int(config.statistics.bootstrap_samples),
                    seed=int(config.statistics.bootstrap_seed),
                )
            summary["comparisons"][f"correct_vs_{control}"] = comparison
        return summary

    nominal = frame[frame.arm == "nominal"].set_index(index)
    for family in ("axial", "registration", "foreign"):
        family_rows = frame[frame.family == family]
        for level, rows in family_rows.groupby("level"):
            key = f"{family}_{level:g}"
            comparison = {}
            for metric in metrics:
                perturbed = rows.groupby(index)[metric].mean()
                delta = (
                    perturbed - nominal[metric]
                    if metric == "LPIPS"
                    else nominal[metric] - perturbed
                )
                matrix = delta.to_numpy().reshape(
                    int(config.data.mask_count), int(config.data.scene_count)
                )
                comparison[metric] = paired_two_way_ci(
                    matrix,
                    samples=int(config.statistics.bootstrap_samples),
                    seed=int(config.statistics.bootstrap_seed),
                )
            summary["comparisons"][key] = comparison
    return summary


def run(config):
    device = torch.device(str(config.device))
    simulator = config.simulator
    cache = PSFCache(config.data.psf_cache, simulator)
    scenes = _selected_scenes(config)
    dataset = DigiCamOnTheFlyDataset(
        scenes=scenes,
        simulator_config=simulator,
        measurement_size=[380, 507],
        target_size=[200, 266],
        simulation_mode="roi_convolution",
        roi=config.roi,
        psf_cache=config.data.psf_cache,
    )

    correct_records = get_mask_records(42, "validation", int(config.data.mask_count))
    train_records = get_mask_records(42, "train", 132)
    wrong_records = train_records[100:132]
    mean_psf = (
        _mean_psf(cache, simulator, train_records[:100])
        if config.experiment == "identity"
        else None
    )
    metrics = _metrics(device)
    rows = []

    for mask_index, (correct_record, wrong_record) in enumerate(
        zip(correct_records, wrong_records)
    ):
        samples = []
        for scene_position in range(len(scenes)):
            samples.append(
                dataset[
                    {
                        "scene_index": scene_position,
                        "mask_id": correct_record["mask_id"],
                        "mask_seed": correct_record["mask_seed"],
                        "sample_seed": 0,
                        "step": scene_position,
                        "mode": "finite",
                        "return_psf": True,
                    }
                ]
            )
        correct_psf = samples[0]["psf"]
        wrong_psf = _prepared_psf(cache, simulator, wrong_record)
        arms = _arms(
            str(config.experiment),
            correct_psf,
            correct_record,
            wrong_psf,
            mean_psf,
            simulator,
        )

        for arm, (psf, family, level, direction) in arms.items():
            solver = _solver(psf, config, device)
            for scene_position, sample in enumerate(samples):
                measurement = sample["measurement"].to(device)
                target = sample["target"].to(device)
                prediction = _reconstruct(
                    solver,
                    measurement,
                    config.roi,
                    config.solver.n_iter,
                )
                rows.append(
                    {
                        "mask_index": mask_index,
                        "mask_id": correct_record["mask_id"],
                        "scene_position": scene_position,
                        "scene_id": sample["scene_id"],
                        "arm": arm,
                        "family": family,
                        "level": level,
                        "direction": direction,
                        **_metric_values(metrics, prediction, target),
                    }
                )
            del solver
            if device.type == "cuda":
                torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    output_dir = Path(to_absolute_path(str(config.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_row.csv", index=False)
    summary = _summarize(frame, config)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="operator_sensitivity",
)
def main(config: DictConfig):
    run(config)


if __name__ == "__main__":
    main()
