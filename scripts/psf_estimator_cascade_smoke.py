"""Smoke-test measurement-to-PSF estimation on held-out DigiCam masks.

The experiment stays on the upstream ``train`` split.  It trains a compact
PSF estimator on the 68-mask development partition and evaluates it on the
17 masks held out from the coursework PSF-free training.  A published
PSF-aware reconstructor is then evaluated with true, predicted, mean, wrong,
and resolution-matched oracle PSFs.

This is a diagnostic pilot, not a final benchmark: the published reconstructor
has seen all 85 upstream-train masks, so only the PSF estimator is evaluated on
unseen masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn import functional as F

DATASET_REPO = "bezzam/DigiCam-Mirflickr-MultiMask-25K"
DATASET_REVISION = "21d82b67662ed1e590a40c98688c32cb3c74f079"
MODEL_REPO = (
    "bezzam/digicam-mirflickr-multi-25k-" "unet4M-unrolled-admm5-unet4M-wave-psfNN"
)
MODEL_REVISION = "9c965e99a6b9048eaa0eea3b8cdb2c5b9039416e"
SENSOR_SIZE = (380, 507)
TARGET_SIZE = (200, 266)
ROI = (80, 100, 200, 266)
FEATURE_SIZE = (95, 127)
PSF_WORK_SIZE = (190, 254)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def source_index(mask_id: int, row_slot: int) -> int:
    """The upstream train split is row-slot-major over masks 15..99."""

    return int(row_slot) * 85 + int(mask_id) - 15


def normalize_l2(psf: Tensor) -> Tensor:
    if psf.ndim == 3:
        denominator = psf.square().sum().sqrt().clamp_min(1e-12)
    elif psf.ndim == 4:
        denominator = psf.square().sum(dim=(1, 2, 3), keepdim=True).sqrt()
        denominator = denominator.clamp_min(1e-12)
    else:
        raise ValueError("PSF must be CHW or NCHW")
    return psf.clamp_min(0) / denominator


def resize_psf(psf: Tensor, size: tuple[int, int]) -> Tensor:
    squeeze = psf.ndim == 3
    if squeeze:
        psf = psf.unsqueeze(0)
    if psf.ndim != 4:
        raise ValueError("PSF must be CHW or NCHW")
    resized = F.interpolate(psf, size=size, mode="bilinear", align_corners=False)
    resized = normalize_l2(resized)
    return resized[0] if squeeze else resized


def measurement_features(measurement: Tensor) -> Tensor:
    """Build low-cost spatial and spectral features from one measurement."""

    if measurement.shape != (3, *SENSOR_SIZE):
        raise ValueError(f"unexpected measurement shape: {measurement.shape}")
    measurement = measurement / measurement.amax().clamp_min(1e-8)
    spatial = F.interpolate(
        measurement.unsqueeze(0),
        size=FEATURE_SIZE,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    gray = 0.299 * spatial[0:1] + 0.587 * spatial[1:2] + 0.114 * spatial[2:3]
    spectrum = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1)).abs()
    spectrum = torch.log1p(spectrum)
    spectrum = (spectrum - spectrum.mean()) / spectrum.std().clamp_min(1e-6)
    spectrum = spectrum.clamp(-4, 4) / 8 + 0.5
    return torch.cat((spatial, spectrum), dim=0).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.body(value)


class PSFBasis:
    def __init__(
        self,
        mean: Tensor,
        components: Tensor,
        coefficient_scale: Tensor,
        explained_variance: float,
    ) -> None:
        self.mean = mean
        self.components = components
        self.coefficient_scale = coefficient_scale
        self.explained_variance = float(explained_variance)

    @property
    def latent_dim(self) -> int:
        return int(self.components.shape[0])

    def encode(self, psfs: Tensor) -> Tensor:
        squeeze = psfs.ndim == 3
        if squeeze:
            psfs = psfs.unsqueeze(0)
        centered = psfs.flatten(1) - self.mean.flatten().unsqueeze(0)
        coefficients = centered @ self.components.flatten(1).T
        coefficients = coefficients / self.coefficient_scale.unsqueeze(0)
        return coefficients[0] if squeeze else coefficients

    def decode(self, coefficients: Tensor) -> Tensor:
        squeeze = coefficients.ndim == 1
        if squeeze:
            coefficients = coefficients.unsqueeze(0)
        coefficients = coefficients * self.coefficient_scale.unsqueeze(0)
        residual = coefficients @ self.components.flatten(1)
        psfs = self.mean.unsqueeze(0) + residual.reshape(-1, *self.mean.shape)
        psfs = normalize_l2(psfs)
        return psfs[0] if squeeze else psfs

    def project(self, psfs: Tensor) -> Tensor:
        return self.decode(self.encode(psfs))


def fit_psf_basis(psfs: Tensor, latent_dim: int) -> PSFBasis:
    if psfs.ndim != 4 or psfs.shape[1:] != (3, *PSF_WORK_SIZE):
        raise ValueError(f"unexpected PSF bank shape: {tuple(psfs.shape)}")
    if not 0 < latent_dim < psfs.shape[0]:
        raise ValueError("latent_dim must be positive and smaller than the PSF bank")
    matrix = psfs.flatten(1)
    mean_flat = matrix.mean(dim=0)
    centered = matrix - mean_flat
    gram = centered @ centered.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)[:latent_dim]
    selected_values = eigenvalues[order].clamp_min(1e-12)
    selected_vectors = eigenvectors[:, order]
    components_flat = selected_vectors.T @ centered
    components_flat = components_flat / selected_values.sqrt().unsqueeze(1)
    coefficients = centered @ components_flat.T
    coefficient_scale = coefficients.std(dim=0, correction=1).clamp_min(1e-6)
    explained_variance = selected_values.sum() / eigenvalues.clamp_min(0).sum()
    return PSFBasis(
        mean=mean_flat.reshape(psfs.shape[1:]),
        components=components_flat.reshape(latent_dim, *psfs.shape[1:]),
        coefficient_scale=coefficient_scale,
        explained_variance=float(explained_variance),
    )


class CompactPSFEstimator(nn.Module):
    """Predict normalized coefficients of a train-PSF residual basis."""

    def __init__(self, latent_dim: int, width: int = 24) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, width, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(width, 2 * width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResidualBlock(2 * width),
            nn.Conv2d(2 * width, 4 * width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResidualBlock(4 * width),
            nn.Conv2d(4 * width, 4 * width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResidualBlock(4 * width),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
            nn.Linear(4 * width * 3 * 4, 8 * width),
            nn.GELU(),
            nn.Linear(8 * width, latent_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.head(self.encoder(value))


def load_psfs(
    train_masks: list[int],
    evaluation_masks: list[int],
    bundle_path: Path,
    pattern_dir: Path,
    simulator_path: Path,
    cache_path: Path,
) -> dict[int, Tensor]:
    from src.digicam_synth.pipeline import pattern_to_psf

    psfs: dict[int, Tensor] = {}
    with np.load(bundle_path, allow_pickle=False) as bundle:
        for mask_id in train_masks:
            key = f"mask_{mask_id}"
            if key not in bundle:
                raise KeyError(f"{key} missing from prepared PSF bundle")
            psf = torch.from_numpy(np.asarray(bundle[key])).squeeze(0).movedim(-1, 0)
            psfs[mask_id] = normalize_l2(psf.float().contiguous())

    cached: dict[str, np.ndarray] = {}
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as bundle:
            cached = {key: np.asarray(bundle[key]) for key in bundle.files}

    simulator = OmegaConf.load(simulator_path)
    generated: dict[str, np.ndarray] = dict(cached)
    cache_changed = False
    for offset, mask_id in enumerate(evaluation_masks, start=1):
        key = f"mask_{mask_id}"
        cache_shape = (1, *SENSOR_SIZE, 3)
        if key in cached and cached[key].shape == cache_shape:
            psf = torch.from_numpy(cached[key]).squeeze(0).movedim(-1, 0)
        else:
            pattern_path = pattern_dir / f"mask_{mask_id}.npy"
            if not pattern_path.is_file():
                raise FileNotFoundError(pattern_path)
            pattern = np.load(pattern_path)
            raw_psf, _, _ = pattern_to_psf(
                simulator,
                pattern,
                np.random.default_rng(0),
                create_simulator=False,
            )
            psf = torch.flip(torch.as_tensor(raw_psf[0]), dims=(-3, -2))
            psf = psf.movedim(-1, 0)
            psf = normalize_l2(psf.float().contiguous())
            generated[key] = psf.movedim(0, -1).unsqueeze(0).numpy()
            cache_changed = True
            print(
                f"generated held-out PSF {offset}/{len(evaluation_masks)}: {mask_id}",
                flush=True,
            )
        psfs[mask_id] = normalize_l2(psf.float().contiguous())

    if cache_changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **generated)
    if set(psfs) != set(train_masks + evaluation_masks):
        raise ValueError("PSF map does not match the requested mask split")
    return psfs


def load_examples(
    source_dataset,
    masks: list[int],
    row_slots: list[int],
    *,
    keep_full_images: bool,
) -> list[dict]:
    from src.datasets.digicam import DigiCamRealDataset

    identities = sorted(
        (source_index(mask_id, row_slot), mask_id, row_slot)
        for row_slot in row_slots
        for mask_id in masks
    )
    dataset = DigiCamRealDataset(
        repo_id=DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        source_dataset=source_dataset,
        indices=[identity[0] for identity in identities],
        force_rgb=True,
        rotate_measurement=True,
        measurement_downsample=1.0,
        target_size=list(TARGET_SIZE),
        target_resize_mode="bilinear",
    )
    examples = []
    for offset, (_, expected_mask, row_slot) in enumerate(identities):
        sample = dataset[offset]
        mask_id = int(sample["mask_id"])
        if mask_id != expected_mask:
            raise ValueError(f"row identity mismatch: {mask_id} != {expected_mask}")
        example = {
            "mask_id": mask_id,
            "row_slot": row_slot,
            "features": measurement_features(sample["measurement"]),
        }
        if keep_full_images:
            example["measurement"] = sample["measurement"]
            example["target"] = sample["target"]
        examples.append(example)
        if (offset + 1) % 256 == 0 or offset + 1 == len(identities):
            print(f"loaded examples: {offset + 1}/{len(identities)}", flush=True)
    return examples


def stack_examples(
    examples: list[dict], mask_to_coefficients: dict[int, Tensor]
) -> tuple[Tensor, Tensor, Tensor]:
    features = torch.stack([example["features"] for example in examples])
    mask_ids = torch.tensor([example["mask_id"] for example in examples])
    targets = torch.stack([mask_to_coefficients[int(mask_id)] for mask_id in mask_ids])
    return features, targets, mask_ids


def train_estimator(
    model: nn.Module,
    features: Tensor,
    targets: Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> list[dict]:
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    curve = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        indices = torch.randint(len(features), (batch_size,), generator=generator)
        batch_features = features[indices].to(device)
        batch_targets = targets[indices].to(device)
        prediction = model(batch_features)
        loss = F.smooth_l1_loss(prediction, batch_targets)
        coefficient_rmse = (prediction - batch_targets).square().mean().sqrt()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "coefficient_rmse": float(coefficient_rmse.detach().cpu()),
                "elapsed_seconds": time.monotonic() - started,
            }
            curve.append(row)
            print(
                f"step {step}/{steps}: loss={row['loss']:.5f}, "
                f"coefficient_rmse={row['coefficient_rmse']:.5f}",
                flush=True,
            )
    return curve


@torch.inference_mode()
def predict_coefficients(
    model: nn.Module,
    examples: list[dict],
    device: torch.device,
    batch_size: int,
) -> Tensor:
    model.eval()
    predictions = []
    for start in range(0, len(examples), batch_size):
        batch = torch.stack(
            [example["features"] for example in examples[start : start + batch_size]]
        ).to(device)
        predictions.append(model(batch).cpu())
    return torch.cat(predictions)


def cosine_per_image(first: Tensor, second: Tensor) -> Tensor:
    return F.cosine_similarity(first.flatten(1), second.flatten(1), dim=1)


def evaluate_psfs(
    predictions: Tensor,
    examples: list[dict],
    work_psfs: dict[int, Tensor],
    evaluation_masks: list[int],
    basis: PSFBasis,
) -> tuple[dict, list[dict], dict[int, Tensor], dict[int, Tensor]]:
    targets = torch.stack([work_psfs[example["mask_id"]] for example in examples])
    train_mean = work_psfs[-1]
    means = train_mean.unsqueeze(0).expand_as(targets)

    grouped_predictions: dict[int, list[Tensor]] = defaultdict(list)
    for prediction, example in zip(predictions, examples):
        grouped_predictions[example["mask_id"]].append(prediction)
    aggregate = {
        mask_id: normalize_l2(torch.stack(values).mean(dim=0))
        for mask_id, values in grouped_predictions.items()
    }
    aggregate_predictions = torch.stack(
        [aggregate[example["mask_id"]] for example in examples]
    )
    wrong_targets = torch.stack(
        [
            work_psfs[
                evaluation_masks[
                    (evaluation_masks.index(example["mask_id"]) + 1)
                    % len(evaluation_masks)
                ]
            ]
            for example in examples
        ]
    )
    oracle_pca_predictions = basis.project(targets)
    oracle_by_mask = {
        mask_id: basis.project(work_psfs[mask_id]) for mask_id in evaluation_masks
    }

    arms = {
        "oracle_train_pca": oracle_pca_predictions,
        "predicted_single": predictions,
        "predicted_aggregate": aggregate_predictions,
        "mean_train_psf": means,
        "wrong_true_psf": wrong_targets,
    }
    rows = []
    summary = {}
    for arm, values in arms.items():
        cosine = cosine_per_image(values, targets)
        l1 = (values - targets).abs().mean(dim=(1, 2, 3))
        summary[arm] = {
            "cosine_mean": float(cosine.mean()),
            "cosine_std": float(cosine.std()),
            "l1_mean": float(l1.mean()),
        }
        for example, cosine_value, l1_value in zip(examples, cosine, l1):
            rows.append(
                {
                    "mask_id": example["mask_id"],
                    "row_slot": example["row_slot"],
                    "arm": arm,
                    "cosine": float(cosine_value),
                    "l1": float(l1_value),
                }
            )

    candidate_targets = torch.stack(
        [work_psfs[mask_id] for mask_id in evaluation_masks]
    )
    candidate_flat = F.normalize(candidate_targets.flatten(1), dim=1)
    retrieval = {}
    for arm, values in {
        "predicted_single": predictions,
        "predicted_aggregate": aggregate_predictions,
    }.items():
        similarities = F.normalize(values.flatten(1), dim=1) @ candidate_flat.T
        predicted_indices = similarities.argmax(dim=1)
        expected_indices = torch.tensor(
            [evaluation_masks.index(example["mask_id"]) for example in examples]
        )
        retrieval[arm] = float((predicted_indices == expected_indices).float().mean())
    summary["retrieval_accuracy_among_held_out_psfs"] = retrieval
    return summary, rows, aggregate, oracle_by_mask


def independent_max_normalize(images: Tensor) -> Tensor:
    images = images.clamp_min(0)
    return images / images.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)


def psnr_per_image(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = independent_max_normalize(prediction)
    target = independent_max_normalize(target)
    mse = (prediction - target).square().mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse.clamp_min(1e-12))


@torch.inference_mode()
def evaluate_reconstruction(
    examples: list[dict],
    predictions: Tensor,
    aggregate_predictions: dict[int, Tensor],
    oracle_pca_predictions: dict[int, Tensor],
    full_psfs: dict[int, Tensor],
    work_psfs: dict[int, Tensor],
    evaluation_masks: list[int],
    cache_dir: Path,
    scenes_per_mask: int,
) -> tuple[dict, list[dict]]:
    from src.model.psf_aware_lensless import PSFAwareLenslessModel

    model = PSFAwareLenslessModel(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        cache_dir=str(cache_dir),
        output_crop=list(ROI),
    ).eval()
    train_mean = full_psfs[-1]
    chosen_counts: dict[int, int] = defaultdict(int)
    rows = []
    arm_names = (
        "true_psf",
        "oracle_half_resolution",
        "oracle_train_pca",
        "predicted_single",
        "predicted_aggregate",
        "mean_train_psf",
        "wrong_true_psf",
    )
    started = time.monotonic()
    for example, single_prediction in zip(examples, predictions):
        mask_id = example["mask_id"]
        if chosen_counts[mask_id] >= scenes_per_mask:
            continue
        chosen_counts[mask_id] += 1
        wrong_mask = evaluation_masks[
            (evaluation_masks.index(mask_id) + 1) % len(evaluation_masks)
        ]
        psf_batch = torch.stack(
            (
                full_psfs[mask_id],
                resize_psf(work_psfs[mask_id], SENSOR_SIZE),
                resize_psf(oracle_pca_predictions[mask_id], SENSOR_SIZE),
                resize_psf(single_prediction, SENSOR_SIZE),
                resize_psf(aggregate_predictions[mask_id], SENSOR_SIZE),
                train_mean,
                full_psfs[wrong_mask],
            )
        )
        psnr_parts = []
        for start in range(0, len(arm_names), 2):
            stop = min(start + 2, len(arm_names))
            measurement = (
                example["measurement"].unsqueeze(0).expand(stop - start, -1, -1, -1)
            )
            target = example["target"].unsqueeze(0).expand(stop - start, -1, -1, -1)
            prediction = model(
                measurement=measurement,
                psf=psf_batch[start:stop],
            )["prediction"].float()
            psnr_parts.append(psnr_per_image(prediction, target))
        psnr = torch.cat(psnr_parts)
        for arm, value in zip(arm_names, psnr):
            rows.append(
                {
                    "mask_id": mask_id,
                    "row_slot": example["row_slot"],
                    "arm": arm,
                    "PSNR": float(value),
                }
            )
        processed = sum(chosen_counts.values())
        print(
            f"reconstruction: {processed}/{len(evaluation_masks) * scenes_per_mask}",
            flush=True,
        )

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row["PSNR"])
    summary = {
        arm: {
            "PSNR_mean": float(np.mean(values)),
            "PSNR_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "sample_count": len(values),
        }
        for arm, values in grouped.items()
    }
    summary["elapsed_seconds"] = time.monotonic() - started
    if set(grouped) != set(arm_names):
        raise ValueError("not all reconstruction arms were evaluated")
    return summary, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-cache", default="data/huggingface", type=Path)
    parser.add_argument(
        "--prepared-psfs",
        default="data/ref_psff_real/prepared_psfs_v1.npz",
        type=Path,
    )
    parser.add_argument(
        "--pattern-dir",
        default="data/hf/DigiCam-Mirflickr-MultiMask-1K/masks",
        type=Path,
    )
    parser.add_argument(
        "--simulator-config",
        default="src/configs/simulator/digicam_article.yaml",
        type=Path,
    )
    parser.add_argument(
        "--psf-cache",
        default="outputs/psf_estimator_smoke_cache/outer17_psfs.npz",
        type=Path,
    )
    parser.add_argument("--train-row-slots", type=int, default=12)
    parser.add_argument("--eval-row-slots", type=int, default=4)
    parser.add_argument("--eval-row-start", type=int, default=20)
    parser.add_argument("--reconstruction-scenes-per-mask", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache/matplotlib"))
    os.environ.setdefault("HF_HOME", str((REPO_ROOT / args.dataset_cache).resolve()))
    from datasets import load_dataset

    from src.digicam_protocol import build_digicam_mask_split

    set_seed(args.seed)
    torch.set_num_threads(4)
    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output / "run_state.json", {"status": "running"})

    train_masks, evaluation_masks = build_digicam_mask_split()
    paths = {
        "dataset_cache": (REPO_ROOT / args.dataset_cache).resolve(),
        "prepared_psfs": (REPO_ROOT / args.prepared_psfs).resolve(),
        "pattern_dir": (REPO_ROOT / args.pattern_dir).resolve(),
        "simulator_config": (REPO_ROOT / args.simulator_config).resolve(),
        "psf_cache": (REPO_ROOT / args.psf_cache).resolve(),
    }
    full_psfs = load_psfs(
        train_masks,
        evaluation_masks,
        paths["prepared_psfs"],
        paths["pattern_dir"],
        paths["simulator_config"],
        paths["psf_cache"],
    )
    train_mean_full = normalize_l2(
        torch.stack([full_psfs[mask_id] for mask_id in train_masks]).mean(dim=0)
    )
    full_psfs[-1] = train_mean_full
    work_psfs = {
        mask_id: resize_psf(psf, PSF_WORK_SIZE) for mask_id, psf in full_psfs.items()
    }
    basis = fit_psf_basis(
        torch.stack([work_psfs[mask_id] for mask_id in train_masks]),
        args.latent_dim,
    )
    work_psfs[-1] = basis.decode(torch.zeros(basis.latent_dim))
    coefficients_by_mask = {
        mask_id: basis.encode(psf) for mask_id, psf in work_psfs.items()
    }
    print(
        f"PSF basis: dim={basis.latent_dim}, "
        f"explained_variance={basis.explained_variance:.5f}",
        flush=True,
    )

    source_dataset = load_dataset(
        DATASET_REPO,
        revision=DATASET_REVISION,
        split="train",
        cache_dir=str(paths["dataset_cache"]),
    )
    train_slots = list(range(args.train_row_slots))
    evaluation_slots = list(
        range(args.eval_row_start, args.eval_row_start + args.eval_row_slots)
    )
    train_examples = load_examples(
        source_dataset,
        train_masks,
        train_slots,
        keep_full_images=False,
    )
    evaluation_examples = load_examples(
        source_dataset,
        evaluation_masks,
        evaluation_slots,
        keep_full_images=True,
    )
    train_features, train_targets, _ = stack_examples(
        train_examples, coefficients_by_mask
    )

    model = CompactPSFEstimator(latent_dim=args.latent_dim, width=args.width)
    curve = train_estimator(
        model,
        train_features,
        train_targets,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        seed=args.seed,
    )
    predicted_coefficients = predict_coefficients(
        model, evaluation_examples, device, args.batch_size
    )
    predictions = basis.decode(predicted_coefficients)
    psf_summary, psf_rows, aggregate, oracle_pca = evaluate_psfs(
        predictions,
        evaluation_examples,
        work_psfs,
        evaluation_masks,
        basis,
    )
    reconstruction_summary, reconstruction_rows = evaluate_reconstruction(
        evaluation_examples,
        predictions,
        aggregate,
        oracle_pca,
        full_psfs,
        work_psfs,
        evaluation_masks,
        paths["dataset_cache"],
        args.reconstruction_scenes_per_mask,
    )

    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "width": args.width,
            "latent_dim": args.latent_dim,
            "feature_size": FEATURE_SIZE,
            "psf_work_size": PSF_WORK_SIZE,
            "basis_mean": basis.mean,
            "basis_components": basis.components,
            "basis_coefficient_scale": basis.coefficient_scale,
        },
        output / "estimator.pth",
    )
    write_csv(output / "training_curve.csv", curve)
    write_csv(output / "psf_per_sample.csv", psf_rows)
    write_csv(output / "reconstruction_per_sample.csv", reconstruction_rows)
    summary = {
        "status": "complete",
        "purpose": "diagnostic smoke test, not a final benchmark",
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "split": "train",
            "official_test_accessed": False,
            "train_masks": train_masks,
            "held_out_masks": evaluation_masks,
            "train_row_slots": train_slots,
            "evaluation_row_slots": evaluation_slots,
        },
        "interpretation": {
            "estimator_unseen_masks": True,
            "published_reconstructor_unseen_masks": False,
            "reason": (
                "the published PSF-aware checkpoint was trained on all 85 "
                "upstream-train masks"
            ),
        },
        "configuration": {
            "seed": args.seed,
            "device": str(device),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "width": args.width,
            "latent_dim": args.latent_dim,
            "basis_explained_variance": basis.explained_variance,
            "feature_size": FEATURE_SIZE,
            "psf_work_size": PSF_WORK_SIZE,
            "reconstruction_scenes_per_mask": args.reconstruction_scenes_per_mask,
        },
        "psf": psf_summary,
        "reconstruction": reconstruction_summary,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "run_state.json",
        {
            "status": "complete",
            "summary": str(output / "summary.json"),
            "elapsed_seconds": summary["elapsed_seconds"],
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
