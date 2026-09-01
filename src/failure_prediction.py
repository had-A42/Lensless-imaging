import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.loss.reconstruction import normalize_per_image_max

MEASUREMENT_FEATURE_NAMES = [
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "quantile_10",
    "quantile_30",
    "quantile_50",
    "quantile_75",
    "quantile_90",
    "quantile_95",
    "quantile_99",
    "dark_fraction",
    "saturated_fraction",
    "horizontal_gradient",
    "vertical_gradient",
    "laplacian_energy",
    "spectrum_0",
    "spectrum_1",
    "spectrum_2",
    "spectrum_3",
    "spectrum_4",
    "spectrum_5",
]

ENSEMBLE_FEATURE_NAMES = [
    "prediction_variance",
    "prediction_variance_p90",
    "edge_disagreement",
    "prediction_contrast",
    "prediction_edge_energy",
]


def _spectral_features(measurement, bands=6, size=64):
    grayscale = measurement.mean(dim=1, keepdim=True)
    grayscale = F.adaptive_avg_pool2d(grayscale, (size, size)).squeeze(1)
    magnitude = torch.fft.rfft2(grayscale, norm="ortho").abs().log1p()

    vertical = torch.fft.fftfreq(size, device=measurement.device).abs()
    horizontal = torch.fft.rfftfreq(size, device=measurement.device)
    radius = torch.sqrt(vertical[:, None].square() + horizontal[None, :].square())
    boundaries = torch.linspace(0, radius.max() + 1e-6, bands + 1, device=radius.device)
    features = []
    for index in range(bands):
        selected = (radius >= boundaries[index]) & (radius < boundaries[index + 1])
        features.append(magnitude[:, selected].mean(dim=1))
    return torch.stack(features, dim=1)


def measurement_features(measurement):
    if measurement.ndim != 4 or measurement.shape[1] != 3:
        raise ValueError("measurement must have N3HW shape")
    flattened = measurement.flatten(1)
    means = measurement.mean(dim=(-2, -1))
    standard_deviations = measurement.std(dim=(-2, -1), unbiased=False)
    quantiles = torch.quantile(
        flattened,
        flattened.new_tensor([0.1, 0.3, 0.5, 0.75, 0.9, 0.95, 0.99]),
        dim=1,
    ).transpose(0, 1)
    dark = (flattened <= 1 / 255).float().mean(dim=1, keepdim=True)
    saturated = (flattened >= 254 / 255).float().mean(dim=1, keepdim=True)
    horizontal = (measurement[..., 1:] - measurement[..., :-1]).abs()
    vertical = (measurement[..., 1:, :] - measurement[..., :-1, :]).abs()
    horizontal_energy = horizontal.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    vertical_energy = vertical.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    laplacian = (
        -4 * measurement[..., 1:-1, 1:-1]
        + measurement[..., :-2, 1:-1]
        + measurement[..., 2:, 1:-1]
        + measurement[..., 1:-1, :-2]
        + measurement[..., 1:-1, 2:]
    )
    laplacian_energy = laplacian.abs().mean(dim=(1, 2, 3)).unsqueeze(1)
    return torch.cat(
        (
            means,
            standard_deviations,
            quantiles,
            dark,
            saturated,
            horizontal_energy,
            vertical_energy,
            laplacian_energy,
            _spectral_features(measurement),
        ),
        dim=1,
    )


def ensemble_features(predictions):
    if predictions.ndim != 5 or predictions.shape[0] < 2:
        raise ValueError("predictions must have model,batch,channel,height,width shape")
    variance = predictions.var(dim=0, unbiased=False)
    flattened_variance = variance.flatten(1)
    variance_mean = flattened_variance.mean(dim=1, keepdim=True)
    variance_p90 = torch.quantile(flattened_variance, 0.9, dim=1).unsqueeze(1)

    horizontal = predictions[..., 1:] - predictions[..., :-1]
    vertical = predictions[..., 1:, :] - predictions[..., :-1, :]
    edge_disagreement = (
        horizontal.var(dim=0, unbiased=False).mean(dim=(1, 2, 3))
        + vertical.var(dim=0, unbiased=False).mean(dim=(1, 2, 3))
    ).unsqueeze(1)

    prediction = predictions.mean(dim=0)
    contrast = prediction.std(dim=(1, 2, 3), unbiased=False).unsqueeze(1)
    prediction_edge = (
        (prediction[..., 1:] - prediction[..., :-1]).abs().mean(dim=(1, 2, 3))
        + (prediction[..., 1:, :] - prediction[..., :-1, :]).abs().mean(dim=(1, 2, 3))
    ).unsqueeze(1)
    return torch.cat(
        (variance_mean, variance_p90, edge_disagreement, contrast, prediction_edge),
        dim=1,
    )


def reconstruction_targets(prediction, target):
    prediction = normalize_per_image_max(prediction.float())
    target = normalize_per_image_max(target.float())
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    psnr = 10 * torch.log10(mse.new_tensor(1.0) / mse.clamp_min(1e-12))
    return mse, psnr


def collect_failure_data(models, dataloader, device, split):
    rows = []
    measurement_values = []
    ensemble_values = []
    risks = []
    psnrs = []

    for batch in tqdm(dataloader, desc=f"failure-{split}"):
        measurement = batch["measurement"].to(device)
        target = batch["target"].to(device)
        with torch.no_grad():
            predictions = torch.stack(
                [model(measurement=measurement)["prediction"] for model in models]
            )
            prediction = predictions.mean(dim=0)
            measurement_batch = measurement_features(measurement)
            ensemble_batch = ensemble_features(predictions)
            risk, psnr = reconstruction_targets(prediction, target)

        measurement_values.append(measurement_batch.cpu().numpy())
        ensemble_values.append(ensemble_batch.cpu().numpy())
        risks.append(risk.cpu().numpy())
        psnrs.append(psnr.cpu().numpy())
        for index in range(measurement.shape[0]):
            rows.append(
                {
                    "split": split,
                    "sample_id": str(batch["sample_id"][index]),
                    "scene_id": str(batch["scene_id"][index]),
                    "mask_id": str(batch["mask_id"][index]),
                }
            )

    return {
        "rows": rows,
        "measurement": np.concatenate(measurement_values),
        "ensemble": np.concatenate(ensemble_values),
        "risk": np.concatenate(risks),
        "psnr": np.concatenate(psnrs),
    }


def collect_measurement_reference(dataloader, device):
    values = []
    for batch in tqdm(dataloader, desc="failure-reference"):
        measurement = batch["measurement"].to(device)
        values.append(measurement_features(measurement).cpu().numpy())
    return np.concatenate(values)


def _rank(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def spearman_correlation(first, second):
    first_ranks = _rank(first)
    second_ranks = _rank(second)
    if first_ranks.std() == 0 or second_ranks.std() == 0:
        return 0.0
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def binary_auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    positive_rank_sum = _rank(scores)[labels].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    )


def fit_ridge(features, targets, regularization):
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1
    standardized = (features - mean) / scale
    design = np.column_stack((np.ones(len(standardized)), standardized))
    penalty = np.eye(design.shape[1]) * float(regularization)
    penalty[0, 0] = 0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return mean, scale, weights


def predict_ridge(features, model):
    mean, scale, weights = model
    standardized = (features - mean) / scale
    design = np.column_stack((np.ones(len(standardized)), standardized))
    return design @ weights


def risk_coverage(scores, risk, psnr):
    order = np.argsort(scores)
    ordered_risk = risk[order]
    ordered_psnr = psnr[order]
    count = np.arange(1, len(order) + 1)
    return pd.DataFrame(
        {
            "coverage": count / len(order),
            "mean_mse": np.cumsum(ordered_risk) / count,
            "mean_psnr": np.cumsum(ordered_psnr) / count,
        }
    )


def evaluate_failure_predictors(
    calibration,
    test,
    reference_measurements=None,
    regularization=1.0,
):
    if reference_measurements is None:
        reference_measurements = calibration["measurement"]
    measurement_mean = reference_measurements.mean(axis=0)
    measurement_scale = reference_measurements.std(axis=0)
    measurement_scale[measurement_scale < 1e-8] = 1

    def distance(values):
        standardized = (values - measurement_mean) / measurement_scale
        return np.sqrt(np.mean(np.square(standardized), axis=1))

    calibration_distance = distance(calibration["measurement"])
    test_distance = distance(test["measurement"])
    calibration_features = np.column_stack(
        (calibration["measurement"], calibration["ensemble"], calibration_distance)
    )
    test_features = np.column_stack(
        (test["measurement"], test["ensemble"], test_distance)
    )
    ridge = fit_ridge(calibration_features, calibration["risk"], regularization)

    scores = {
        "combined": predict_ridge(test_features, ridge),
        "measurement_distance": test_distance,
        "ensemble_disagreement": test["ensemble"][:, 0],
    }
    failure_threshold = float(np.quantile(test["risk"], 0.75))
    labels = test["risk"] >= failure_threshold
    summary = {
        "reference_samples": len(reference_measurements),
        "calibration_samples": len(calibration["risk"]),
        "test_samples": len(test["risk"]),
        "calibration_masks": len({row["mask_id"] for row in calibration["rows"]}),
        "test_masks": len({row["mask_id"] for row in test["rows"]}),
        "test_mean_psnr": float(test["psnr"].mean()),
        "failure_mse_threshold": failure_threshold,
        "predictors": {},
    }
    curves = []
    for name, values in scores.items():
        curve = risk_coverage(values, test["risk"], test["psnr"])
        curve.insert(0, "predictor", name)
        curves.append(curve)
        summary["predictors"][name] = {
            "spearman": spearman_correlation(values, test["risk"]),
            "worst_quartile_auroc": binary_auroc(labels, values),
            "aurc_mse": float(np.trapz(curve["mean_mse"], curve["coverage"])),
        }
    return scores, ridge, summary, pd.concat(curves, ignore_index=True)


def save_failure_results(output_dir, calibration, test, scores, ridge, summary, curves):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split_data in (calibration, test):
        for index, metadata in enumerate(split_data["rows"]):
            row = dict(metadata)
            row["mse"] = float(split_data["risk"][index])
            row["psnr"] = float(split_data["psnr"][index])
            rows.append(row)
    per_image = pd.DataFrame(rows)
    test_start = len(calibration["rows"])
    for name, values in scores.items():
        per_image.loc[test_start:, f"predicted_risk_{name}"] = values
    per_image.to_csv(output_dir / "per_image.csv", index=False)
    curves.to_csv(output_dir / "risk_coverage.csv", index=False)

    feature_names = (
        MEASUREMENT_FEATURE_NAMES + ENSEMBLE_FEATURE_NAMES + ["measurement_distance"]
    )
    pd.DataFrame(
        {
            "feature": ["intercept"] + feature_names,
            "coefficient": ridge[2],
        }
    ).to_csv(output_dir / "combined_coefficients.csv", index=False)
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
