"""Calibration-free fusion over a fixed bank of operator hypotheses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path
from torch import Tensor, nn
from torch.nn import functional as F


def normalize_nonnegative_max(image: Tensor, eps: float = 1e-8) -> Tensor:
    if image.ndim != 4:
        raise ValueError("image must be an NCHW tensor")
    if not image.is_floating_point():
        raise TypeError("image must be floating point")
    if eps <= 0:
        raise ValueError("normalization epsilon must be positive")
    image = image.clamp_min(0)
    maximum = image.amax(dim=(1, 2, 3), keepdim=True)
    return image / maximum.clamp_min(eps)


def reflected_box_blur(image: Tensor, kernel_size: int) -> Tensor:
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if kernel_size >= min(image.shape[-2:]):
        raise ValueError("kernel_size must fit inside the image")
    padding = kernel_size // 2
    return F.avg_pool2d(
        F.pad(image, (padding, padding, padding, padding), mode="reflect"),
        kernel_size=kernel_size,
        stride=1,
    )


def _normalize_psf(psf: Tensor) -> Tensor:
    if psf.ndim != 3:
        raise ValueError("each PSF must be a CHW tensor")
    if not psf.is_floating_point():
        psf = psf.float()
    if not torch.isfinite(psf).all():
        raise ValueError("PSF contains non-finite values")
    if psf.amin().item() < 0:
        raise ValueError("PSF must be non-negative")
    norm = psf.square().sum().sqrt()
    if norm.item() <= 0:
        raise ValueError("PSF must contain positive energy")
    return (psf / norm).contiguous()


def load_psf_hypothesis_bank(
    path: str | Path,
    keys: list[str | int],
) -> tuple[Tensor, tuple[str, ...]]:
    if len(keys) < 2:
        raise ValueError("OHUF needs at least two PSF hypotheses")
    normalized_keys = tuple(
        str(key) if str(key).startswith("mask_") else f"mask_{key}" for key in keys
    )
    if len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError("PSF hypothesis keys must be unique")

    bundle_path = Path(to_absolute_path(str(path))).expanduser()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"PSF hypothesis bundle not found: {bundle_path}")
    hypotheses = []
    with np.load(bundle_path, allow_pickle=False) as bundle:
        for key in normalized_keys:
            if key not in bundle:
                raise KeyError(f"PSF hypothesis {key!r} is missing from {bundle_path}")
            array = np.asarray(bundle[key])
            if array.ndim == 4 and array.shape[0] == 1 and array.shape[-1] in (1, 3):
                psf = torch.from_numpy(array).squeeze(0).movedim(-1, 0)
            elif array.ndim == 3 and array.shape[0] in (1, 3):
                psf = torch.from_numpy(array)
            else:
                raise ValueError(
                    f"PSF {key!r} must have 1HWC or CHW layout, got {array.shape}"
                )
            hypotheses.append(_normalize_psf(psf))

    shapes = {tuple(psf.shape) for psf in hypotheses}
    if len(shapes) != 1:
        raise ValueError(f"PSF hypotheses have inconsistent shapes: {sorted(shapes)}")
    return torch.stack(hypotheses), normalized_keys


class OperatorHypothesisUncertaintyFusion(nn.Module):
    """Fuse PSF-free and PSF-conditioned predictions without a test PSF.

    The PSF-conditioned reconstructor is evaluated for a fixed bank of training
    PSFs. Disagreement across these predictions defines a spatial confidence
    map. Only the confidence-gated, low-frequency ensemble residual is added to
    the measurement-only proposal.

    Any ``psf`` supplied in ``batch`` is deliberately ignored. This prevents an
    evaluation dataset from silently turning OHUF into an oracle method.
    """

    def __init__(
        self,
        proposal: nn.Module,
        psf_reconstructor: nn.Module,
        psf_bank_path: str | Path,
        psf_keys: list[str | int],
        beta: float = 4.0,
        fusion_kernel: int = 65,
        fusion_alpha: float = 1.0,
        hypothesis_batch_size: int = 1,
        normalization_eps: float = 1e-8,
        freeze_backbones: bool = True,
    ) -> None:
        super().__init__()
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if fusion_alpha < 0:
            raise ValueError("fusion_alpha must be non-negative")
        if hypothesis_batch_size <= 0:
            raise ValueError("hypothesis_batch_size must be positive")
        if normalization_eps <= 0:
            raise ValueError("normalization_eps must be positive")
        if fusion_kernel <= 0 or fusion_kernel % 2 == 0:
            raise ValueError("fusion_kernel must be a positive odd integer")

        bank, normalized_keys = load_psf_hypothesis_bank(psf_bank_path, psf_keys)
        self.proposal = proposal
        self.psf_reconstructor = psf_reconstructor
        self.register_buffer("psf_bank", bank, persistent=True)
        self.psf_keys = normalized_keys
        self.beta = float(beta)
        self.fusion_kernel = int(fusion_kernel)
        self.fusion_alpha = float(fusion_alpha)
        self.hypothesis_batch_size = int(hypothesis_batch_size)
        self.normalization_eps = float(normalization_eps)
        self.freeze_backbones = bool(freeze_backbones)
        self.load_seconds = 0.0
        if self.freeze_backbones:
            self.proposal.requires_grad_(False)
            self.psf_reconstructor.requires_grad_(False)

    @property
    def num_hypotheses(self) -> int:
        return int(self.psf_bank.shape[0])

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _prediction(output: dict[str, Tensor], name: str) -> Tensor:
        if not isinstance(output, dict) or "prediction" not in output:
            raise TypeError(f"{name} must return a dict containing 'prediction'")
        prediction = output["prediction"]
        if not isinstance(prediction, Tensor) or prediction.ndim != 4:
            raise ValueError(f"{name} prediction must be an NCHW tensor")
        if not torch.isfinite(prediction).all():
            raise ValueError(f"{name} prediction contains non-finite values")
        return prediction

    def _validate_measurement(self, measurement: Tensor) -> None:
        if not isinstance(measurement, Tensor) or measurement.ndim != 4:
            raise ValueError("measurement must be an NCHW tensor")
        if not measurement.is_floating_point():
            raise TypeError("measurement must be floating point")
        if not torch.isfinite(measurement).all():
            raise ValueError("measurement contains non-finite values")
        if tuple(measurement.shape[1:]) != tuple(self.psf_bank.shape[1:]):
            raise ValueError(
                "measurement and PSF-bank shapes must match after the batch axis: "
                f"{tuple(measurement.shape[1:])} != {tuple(self.psf_bank.shape[1:])}"
            )

    def _hypothesis_predictions(self, measurement: Tensor) -> Tensor:
        batch_size = measurement.shape[0]
        chunks = []
        bank = self.psf_bank.to(dtype=measurement.dtype)
        for start in range(0, self.num_hypotheses, self.hypothesis_batch_size):
            psfs = bank[start : start + self.hypothesis_batch_size]
            hypothesis_count = psfs.shape[0]
            repeated_measurement = (
                measurement.unsqueeze(0)
                .expand(hypothesis_count, -1, -1, -1, -1)
                .reshape(hypothesis_count * batch_size, *measurement.shape[1:])
            )
            repeated_psfs = (
                psfs.unsqueeze(1)
                .expand(-1, batch_size, -1, -1, -1)
                .reshape(hypothesis_count * batch_size, *psfs.shape[1:])
            )
            output = self.psf_reconstructor(
                measurement=repeated_measurement,
                psf=repeated_psfs,
            )
            prediction = self._prediction(output, "psf_reconstructor")
            prediction = prediction.reshape(
                hypothesis_count, batch_size, *prediction.shape[1:]
            )
            chunks.append(
                normalize_nonnegative_max(
                    prediction.flatten(0, 1), eps=self.normalization_eps
                ).reshape_as(prediction)
            )
        return torch.cat(chunks, dim=0)

    def forward(
        self,
        measurement: Tensor,
        return_diagnostics: bool = False,
        **batch: Tensor,
    ) -> dict[str, Tensor]:
        del batch
        self._validate_measurement(measurement)
        proposal_output = self.proposal(measurement=measurement)
        proposal = self._prediction(proposal_output, "proposal")
        proposal = normalize_nonnegative_max(proposal, eps=self.normalization_eps)

        hypotheses = self._hypothesis_predictions(measurement)
        if hypotheses.shape[1:] != proposal.shape:
            raise ValueError(
                "proposal and hypothesis predictions must have matching shapes, got "
                f"{tuple(proposal.shape)} and {tuple(hypotheses.shape[1:])}"
            )
        marginal = hypotheses.mean(dim=0)
        variance = hypotheses.var(dim=0, correction=1).mean(dim=1, keepdim=True)
        variance_scale = variance.mean(dim=(2, 3), keepdim=True).clamp_min(
            self.normalization_eps
        )
        confidence = torch.exp(-self.beta * variance / variance_scale)
        guide = proposal + confidence * (marginal - proposal)
        guide = normalize_nonnegative_max(guide, eps=self.normalization_eps)
        correction = reflected_box_blur(
            guide - proposal,
            kernel_size=self.fusion_kernel,
        )
        prediction = proposal + self.fusion_alpha * correction
        self.load_seconds = float(
            getattr(self.proposal, "load_seconds", 0.0)
            + getattr(self.psf_reconstructor, "load_seconds", 0.0)
        )

        output = {"prediction": prediction}
        if return_diagnostics:
            output.update(
                {
                    "proposal_prediction": proposal,
                    "operator_marginal_prediction": marginal,
                    "operator_variance": variance,
                    "operator_confidence": confidence,
                    "operator_correction": correction,
                }
            )
        return output


__all__ = [
    "OperatorHypothesisUncertaintyFusion",
    "load_psf_hypothesis_bank",
    "normalize_nonnegative_max",
    "reflected_box_blur",
]
