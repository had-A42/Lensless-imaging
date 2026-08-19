import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from src.loss.reconstruction import (
    _validate_image_pair,
    normalize_per_image_max,
    structural_similarity,
)
from src.metrics.base_metric import BaseMetric


def _prepare_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalize_by_max: bool,
    normalization_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_image_pair(prediction, target)
    if not normalize_by_max:
        return prediction, target
    return (
        normalize_per_image_max(prediction, eps=normalization_eps),
        normalize_per_image_max(target, eps=normalization_eps),
    )


class PSNRMetric(BaseMetric):
    def __init__(
        self,
        data_range: float = 1.0,
        normalize_by_max: bool = True,
        normalization_eps: float = 1e-8,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if data_range <= 0:
            raise ValueError("data_range must be positive")
        if normalization_eps <= 0:
            raise ValueError("normalization_eps must be positive")
        self.data_range = float(data_range)
        self.normalize_by_max = bool(normalize_by_max)
        self.normalization_eps = float(normalization_eps)

    def per_image(
        self, prediction: torch.Tensor, target: torch.Tensor, **batch
    ) -> torch.Tensor:
        prediction, target = _prepare_pair(
            prediction,
            target,
            normalize_by_max=self.normalize_by_max,
            normalization_eps=self.normalization_eps,
        )
        mse_per_image = (prediction - target).square().flatten(1).mean(dim=1)
        peak_sq = prediction.new_tensor(self.data_range**2)
        return (10 * torch.log10(peak_sq / mse_per_image)).detach()

    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, **batch
    ) -> torch.Tensor:
        return self.per_image(prediction, target, **batch).mean()


class SSIMMetric(BaseMetric):
    def __init__(
        self,
        data_range: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
        normalize_by_max: bool = True,
        normalization_eps: float = 1e-8,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if normalization_eps <= 0:
            raise ValueError("normalization_eps must be positive")
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.normalize_by_max = bool(normalize_by_max)
        self.normalization_eps = float(normalization_eps)
        self._metric = None

    def _build_metric(self):
        return StructuralSimilarityIndexMeasure(
            data_range=self.data_range,
            kernel_size=self.window_size,
            sigma=self.sigma,
            reduction="none",
        )

    def per_image(
        self, prediction: torch.Tensor, target: torch.Tensor, **batch
    ) -> torch.Tensor:
        prediction, target = _prepare_pair(
            prediction,
            target,
            normalize_by_max=self.normalize_by_max,
            normalization_eps=self.normalization_eps,
        )
        if min(prediction.shape[-2:]) <= self.window_size:
            values = []
            for index in range(prediction.shape[0]):
                values.append(
                    structural_similarity(
                        prediction[index : index + 1],
                        target[index : index + 1],
                        data_range=self.data_range,
                        window_size=self.window_size,
                        sigma=self.sigma,
                    )
                )
            return torch.stack(values).detach()
        if self._metric is None:
            self._metric = self._build_metric()
        self._metric = self._metric.to(prediction.device)
        self._metric.reset()
        try:
            values = self._metric(prediction, target)
        finally:
            self._metric.reset()
        return values.detach()

    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, **batch
    ) -> torch.Tensor:
        return self.per_image(prediction, target, **batch).mean()


class LPIPSMetric(BaseMetric):
    def __init__(
        self,
        net_type: str = "alex",
        device: str = "auto",
        normalize_by_max: bool = True,
        normalization_eps: float = 1e-8,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if net_type not in {"alex", "vgg", "squeeze"}:
            raise ValueError("net_type must be one of: alex, vgg, squeeze")
        if normalization_eps <= 0:
            raise ValueError("normalization_eps must be positive")
        self.net_type = net_type
        self.device = device
        self.normalize_by_max = bool(normalize_by_max)
        self.normalization_eps = float(normalization_eps)
        self._metric = None

    def _build_metric(self):
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "LPIPSMetric requires torchmetrics and torchvision. "
                "Install the project requirements to enable it."
            ) from exc

        return LearnedPerceptualImagePatchSimilarity(
            net_type=self.net_type,
            reduction="mean",
            normalize=True,
        )

    def per_image(
        self, prediction: torch.Tensor, target: torch.Tensor, **batch
    ) -> torch.Tensor:
        prediction, target = _prepare_pair(
            prediction,
            target,
            normalize_by_max=self.normalize_by_max,
            normalization_eps=self.normalization_eps,
        )
        if self._metric is None:
            self._metric = self._build_metric()

        if self.device == "auto":
            metric_device = prediction.device
        else:
            metric_device = torch.device(self.device)
            if prediction.device != metric_device:
                raise ValueError(
                    f"LPIPS inputs are on {prediction.device}, configured "
                    f"metric device is {metric_device}"
                )

        self._metric = self._metric.to(metric_device)
        self._metric.reset()
        try:
            if hasattr(self._metric, "net"):
                values = self._metric.net(prediction, target, normalize=True)
            else:
                values = self._metric(prediction, target)
        finally:
            self._metric.reset()
        if values.ndim == 0:
            values = values.repeat(prediction.shape[0])
        else:
            values = values.flatten(1).mean(dim=1)
        return values.detach()

    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, **batch
    ) -> torch.Tensor:
        return self.per_image(prediction, target, **batch).mean()
