import torch
from lensless.recon.rfft_convolve import RealFFTConvolve2D
from torch import Tensor, nn

from src.loss.reconstruction import ReconstructionLoss, normalize_per_image_max


class _ROIReprojector:
    def __init__(self, psf: Tensor, roi: list[int]) -> None:
        if psf.ndim != 4:
            raise ValueError("psf must be an NCHW tensor")
        if len(roi) != 4:
            raise ValueError("roi must contain [top, left, height, width]")

        self.top, self.left, self.height, self.width = (
            int(value) for value in roi
        )
        self.batch_size, self.channels = psf.shape[:2]
        self.sensor_height, self.sensor_width = psf.shape[-2:]
        if (
            self.top < 0
            or self.left < 0
            or self.top + self.height > self.sensor_height
            or self.left + self.width > self.sensor_width
        ):
            raise ValueError("ROI exceeds the sensor canvas")

        psf_dhwc = psf.float().movedim(1, -1).unsqueeze(1)
        self.convolver = RealFFTConvolve2D(psf=psf_dhwc, dtype=torch.float32)

    def __call__(self, prediction: Tensor) -> Tensor:
        if prediction.ndim != 4:
            raise ValueError("prediction must be an NCHW tensor")
        if prediction.shape[:2] != (self.batch_size, self.channels):
            raise ValueError("prediction and psf batch/channel dimensions must match")
        if prediction.shape[-2:] != (self.height, self.width):
            raise ValueError("prediction size must match the configured ROI")

        prediction = prediction.float()
        canvas = prediction.new_zeros(
            self.batch_size,
            1,
            self.sensor_height,
            self.sensor_width,
            self.channels,
        )
        canvas[
            :,
            :,
            self.top : self.top + self.height,
            self.left : self.left + self.width,
        ] = prediction.movedim(1, -1).unsqueeze(1)
        measurement = self.convolver.convolve(canvas).clamp_min(0)
        measurement = measurement.squeeze(1).movedim(-1, 1).contiguous()
        return normalize_per_image_max(measurement)


def reproject_roi(prediction: Tensor, psf: Tensor, roi: list[int]) -> Tensor:
    return _ROIReprojector(psf, roi)(prediction)


class CrossMaskPhysicsLoss(nn.Module):
    component_names = (
        "self_loss",
        "cross_loss",
        "view_loss",
        "vicreg_loss",
        "invariance_loss",
        "variance_loss",
        "covariance_loss",
        "embedding_std",
        "collapsed_fraction",
        "effective_rank",
    )

    def __init__(
        self,
        roi: list[int],
        view_weight: float = 0.1,
        vicreg_weight: float = 0.01,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        charbonnier_epsilon: float = 1e-6,
        variance_epsilon: float = 1e-4,
        collapse_threshold: float = 0.1,
        mse_weight: float = 1.0,
        ssim_weight: float = 0.0,
        lpips_weight: float = 1.0,
        lpips_net: str = "vgg",
        normalize_by_max: bool = True,
    ) -> None:
        super().__init__()
        weights = (
            view_weight,
            vicreg_weight,
            invariance_weight,
            variance_weight,
            covariance_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("loss weights must be non-negative")
        if charbonnier_epsilon <= 0 or variance_epsilon <= 0:
            raise ValueError("loss epsilons must be positive")
        if collapse_threshold <= 0:
            raise ValueError("collapse_threshold must be positive")

        self.roi = tuple(int(value) for value in roi)
        self.view_weight = float(view_weight)
        self.vicreg_weight = float(vicreg_weight)
        self.invariance_weight = float(invariance_weight)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.charbonnier_epsilon = float(charbonnier_epsilon)
        self.variance_epsilon = float(variance_epsilon)
        self.collapse_threshold = float(collapse_threshold)
        self.validation_loss = ReconstructionLoss(
            mse_weight=mse_weight,
            ssim_weight=ssim_weight,
            lpips_weight=lpips_weight,
            lpips_net=lpips_net,
            normalize_by_max=normalize_by_max,
        )

    def _distance(self, first: Tensor, second: Tensor) -> Tensor:
        return torch.sqrt(
            (first.float() - second.float()).square() + self.charbonnier_epsilon
        ).mean()

    def _vicreg(self, first: Tensor, second: Tensor) -> dict[str, Tensor]:
        if first.shape != second.shape or first.ndim != 3:
            raise ValueError("embeddings must have matching [batch, cells, dim] shape")
        with torch.autocast(device_type=first.device.type, enabled=False):
            first = first.float().flatten(0, 1)
            second = second.float().flatten(0, 1)
            if first.shape[0] < 2:
                raise ValueError("VICReg needs at least two spatial embeddings")

            invariance = (first - second).square().mean()
            centered_first = first - first.mean(dim=0)
            centered_second = second - second.mean(dim=0)
            std_first = torch.sqrt(first.var(dim=0) + self.variance_epsilon)
            std_second = torch.sqrt(second.var(dim=0) + self.variance_epsilon)
            variance = 0.5 * (
                torch.relu(1 - std_first).mean()
                + torch.relu(1 - std_second).mean()
            )

            count = first.shape[0] - 1
            cov_first = centered_first.T @ centered_first / count
            cov_second = centered_second.T @ centered_second / count
            off_diagonal_first = cov_first - torch.diag(torch.diagonal(cov_first))
            off_diagonal_second = cov_second - torch.diag(torch.diagonal(cov_second))
            covariance = 0.5 * (
                off_diagonal_first.square().sum()
                + off_diagonal_second.square().sum()
            ) / first.shape[1]

            mean_std = 0.5 * (std_first.mean() + std_second.mean())
            collapsed_fraction = 0.5 * (
                (std_first < self.collapse_threshold).float().mean()
                + (std_second < self.collapse_threshold).float().mean()
            )
            mean_covariance = 0.5 * (cov_first + cov_second)
            eigenvalues = torch.linalg.eigvalsh(mean_covariance.detach()).clamp_min(0)
            total_variance = eigenvalues.sum()
            probabilities = eigenvalues / total_variance.clamp_min(1e-12)
            effective_rank = torch.where(
                total_variance > 0,
                torch.exp(
                    -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
                ),
                total_variance,
            )
        return {
            "invariance_loss": invariance,
            "variance_loss": variance,
            "covariance_loss": covariance,
            "embedding_std": mean_std.detach(),
            "collapsed_fraction": collapsed_fraction.detach(),
            "effective_rank": effective_rank.detach(),
        }

    @staticmethod
    def _zero(reference: Tensor) -> Tensor:
        return reference.new_zeros(())

    def _validation_forward(self, prediction: Tensor, target: Tensor):
        result = self.validation_loss(prediction=prediction, target=target)
        zero = self._zero(result["loss"])
        result.update({name: zero for name in self.component_names})
        return result

    def forward(
        self,
        prediction: Tensor | None = None,
        target: Tensor | None = None,
        prediction_a: Tensor | None = None,
        prediction_b: Tensor | None = None,
        measurement_a: Tensor | None = None,
        measurement_b: Tensor | None = None,
        psf_a: Tensor | None = None,
        psf_b: Tensor | None = None,
        embedding_a: Tensor | None = None,
        embedding_b: Tensor | None = None,
        **batch,
    ) -> dict[str, Tensor]:
        del batch
        if prediction is not None or target is not None:
            if prediction is None or target is None:
                raise ValueError("validation needs both prediction and target")
            return self._validation_forward(prediction, target)

        paired_tensors = (
            prediction_a,
            prediction_b,
            measurement_a,
            measurement_b,
            psf_a,
            psf_b,
            embedding_a,
            embedding_b,
        )
        if any(value is None for value in paired_tensors):
            raise ValueError("cross-mask training requires both views, PSFs and embeddings")

        reproject_a = _ROIReprojector(psf_a, self.roi)
        reproject_b = _ROIReprojector(psf_b, self.roi)
        predicted_aa = reproject_a(prediction_a)
        predicted_ab = reproject_b(prediction_a)
        predicted_bb = reproject_b(prediction_b)
        predicted_ba = reproject_a(prediction_b)
        measurement_a = normalize_per_image_max(measurement_a.float())
        measurement_b = normalize_per_image_max(measurement_b.float())

        self_loss = 0.5 * (
            self._distance(predicted_aa, measurement_a)
            + self._distance(predicted_bb, measurement_b)
        )
        cross_loss = 0.5 * (
            self._distance(predicted_ab, measurement_b)
            + self._distance(predicted_ba, measurement_a)
        )
        view_loss = self._distance(
            normalize_per_image_max(prediction_a.float()),
            normalize_per_image_max(prediction_b.float()),
        )
        vicreg = self._vicreg(embedding_a, embedding_b)
        vicreg_loss = (
            self.invariance_weight * vicreg["invariance_loss"]
            + self.variance_weight * vicreg["variance_loss"]
            + self.covariance_weight * vicreg["covariance_loss"]
        )
        total_loss = (
            self_loss
            + cross_loss
            + self.view_weight * view_loss
            + self.vicreg_weight * vicreg_loss
        )
        zero = self._zero(total_loss)
        return {
            "loss": total_loss,
            "self_loss": self_loss,
            "cross_loss": cross_loss,
            "view_loss": view_loss,
            "vicreg_loss": vicreg_loss,
            **vicreg,
            "mse_loss": zero,
            "lpips_loss": zero,
        }


__all__ = ["CrossMaskPhysicsLoss", "reproject_roi"]
