import torch
from torch import nn
from torch.nn import functional as F


def normalize_per_image_max(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError("image must be an NCHW tensor")
    if not image.is_floating_point():
        raise TypeError("image must be a floating point tensor")
    if eps <= 0:
        raise ValueError("normalization eps must be positive")
    if any(size == 0 for size in image.shape[1:]):
        raise ValueError("image channel and spatial axes must be non-empty")

    image_max = image.amax(dim=(1, 2, 3), keepdim=True)
    return image / image_max.clamp_min(eps)


def _validate_image_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("prediction and target must be NCHW tensors")
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")


def structural_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    _validate_image_pair(prediction, target)
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    height, width = prediction.shape[-2:]
    effective_window = min(window_size, height, width)
    if effective_window % 2 == 0:
        effective_window -= 1
    if effective_window < 1:
        raise ValueError("prediction and target must have non-empty spatial axes")

    coordinates = torch.arange(
        effective_window, dtype=prediction.dtype, device=prediction.device
    )
    coordinates = coordinates - (effective_window - 1) / 2
    gaussian = torch.exp(-(coordinates**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    kernel_2d = gaussian[:, None] * gaussian[None, :]
    channels = prediction.shape[1]
    kernel = kernel_2d.expand(channels, 1, -1, -1)
    padding = effective_window // 2

    def local_mean(image: torch.Tensor) -> torch.Tensor:
        return F.conv2d(image, kernel, padding=padding, groups=channels)

    prediction_mean = local_mean(prediction)
    target_mean = local_mean(target)
    prediction_mean_sq = prediction_mean.square()
    target_mean_sq = target_mean.square()
    mean_product = prediction_mean * target_mean

    prediction_variance = (
        local_mean(prediction.square()) - prediction_mean_sq
    ).clamp_min(0)
    target_variance = (local_mean(target.square()) - target_mean_sq).clamp_min(0)
    covariance = local_mean(prediction * target) - mean_product

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2 * mean_product + c1) * (2 * covariance + c2)
    denominator = (prediction_mean_sq + target_mean_sq + c1) * (
        prediction_variance + target_variance + c2
    )
    return (numerator / denominator).mean()


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        mse_weight: float = 1.0,
        ssim_weight: float = 0.0,
        lpips_weight: float = 0.0,
        lpips_net: str = "vgg",
        data_range: float = 1.0,
        ssim_window_size: int = 11,
        ssim_sigma: float = 1.5,
        normalize_by_max: bool = True,
        normalization_eps: float = 1e-8,
    ):
        super().__init__()
        if mse_weight < 0 or ssim_weight < 0 or lpips_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if mse_weight == 0 and ssim_weight == 0 and lpips_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        if lpips_net not in {"alex", "vgg", "squeeze"}:
            raise ValueError("lpips_net must be alex, vgg or squeeze")
        if normalization_eps <= 0:
            raise ValueError("normalization_eps must be positive")

        self.mse_weight = float(mse_weight)
        self.ssim_weight = float(ssim_weight)
        self.lpips_weight = float(lpips_weight)
        self.lpips_net = str(lpips_net)
        self.data_range = float(data_range)
        self.ssim_window_size = int(ssim_window_size)
        self.ssim_sigma = float(ssim_sigma)
        self.normalize_by_max = bool(normalize_by_max)
        self.normalization_eps = float(normalization_eps)
        self.lpips_model = None

    def _build_lpips(self):
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        except (ImportError, ModuleNotFoundError) as error:
            raise ImportError(
                "LPIPS loss requires torchmetrics and torchvision"
            ) from error
        return LearnedPerceptualImagePatchSimilarity(
            net_type=self.lpips_net,
            normalize=True,
            reduction="mean",
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **batch,
    ) -> dict[str, torch.Tensor]:
        _validate_image_pair(prediction, target)
        if self.normalize_by_max:
            prediction = normalize_per_image_max(prediction, eps=self.normalization_eps)
            target = normalize_per_image_max(target, eps=self.normalization_eps)

        mse_loss = F.mse_loss(prediction, target)
        total_loss = self.mse_weight * mse_loss
        losses = {"mse_loss": mse_loss}

        if self.ssim_weight > 0:
            ssim_loss = 1 - structural_similarity(
                prediction,
                target,
                data_range=self.data_range,
                window_size=self.ssim_window_size,
                sigma=self.ssim_sigma,
            )
            total_loss = total_loss + self.ssim_weight * ssim_loss
            losses["ssim_loss"] = ssim_loss

        if self.lpips_weight > 0:
            if self.lpips_model is None:
                self.lpips_model = self._build_lpips().to(prediction.device)
            self.lpips_model.eval()
            lpips_loss = self.lpips_model.net(
                prediction,
                target,
                normalize=True,
            ).mean()
            total_loss = total_loss + self.lpips_weight * lpips_loss
            losses["lpips_loss"] = lpips_loss

        losses["loss"] = total_loss
        return losses
