from pathlib import Path

import torch
from diffusers import AutoencoderKL
from torch import Tensor, nn
from torch.nn import functional as F


class PSFFreeAutoencoderKL(nn.Module):
    def __init__(
        self,
        local_snapshot_path: str | Path | None,
        pretrained: bool,
        channels: int = 3,
        pad_multiple: int = 8,
        output_crop: list[int] | None = None,
        autoencoder: AutoencoderKL | None = None,
    ) -> None:
        super().__init__()
        if channels != 3:
            raise ValueError("AutoencoderKL expects three-channel measurements")
        if pad_multiple <= 0:
            raise ValueError("pad_multiple must be positive")
        if output_crop is not None:
            if len(output_crop) != 4:
                raise ValueError("output_crop must contain [top, left, height, width]")
            output_crop = tuple(int(value) for value in output_crop)
            if (
                output_crop[0] < 0
                or output_crop[1] < 0
                or output_crop[2] <= 0
                or output_crop[3] <= 0
            ):
                raise ValueError(
                    "output_crop top/left must be non-negative and "
                    "height/width must be positive"
                )

        self.channels = channels
        self.pad_multiple = int(pad_multiple)
        self.output_crop = output_crop
        self.autoencoder = autoencoder or self._load_autoencoder(
            local_snapshot_path,
            pretrained=pretrained,
        )

    @staticmethod
    def _load_autoencoder(
        local_snapshot_path: str | Path | None,
        pretrained: bool,
    ) -> AutoencoderKL:
        if local_snapshot_path is None:
            raise ValueError("local_snapshot_path is required")
        snapshot_path = Path(local_snapshot_path).expanduser()
        if not snapshot_path.is_dir():
            raise FileNotFoundError(
                f"Local AutoencoderKL snapshot not found: {snapshot_path}"
            )

        if pretrained:
            return AutoencoderKL.from_pretrained(
                snapshot_path,
                local_files_only=True,
                use_safetensors=True,
            )
        config = AutoencoderKL.load_config(snapshot_path, local_files_only=True)
        return AutoencoderKL.from_config(config)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_measurement(self, measurement: Tensor) -> None:
        if not isinstance(measurement, Tensor):
            raise TypeError("measurement must be a torch.Tensor")
        if measurement.ndim != 4:
            raise ValueError("measurement must have NCHW shape")
        if measurement.shape[1] != self.channels:
            raise ValueError(
                f"measurement must have {self.channels} channels, got "
                f"{measurement.shape[1]}"
            )
        if not measurement.is_floating_point():
            raise TypeError("measurement must have a floating-point dtype")
        if not torch.isfinite(measurement).all():
            raise ValueError("measurement must contain only finite values")
        if measurement.numel() and (
            measurement.detach().amin().item() < 0
            or measurement.detach().amax().item() > 1
        ):
            raise ValueError("measurement values must be in [0, 1]")

    def forward(self, measurement: Tensor, **batch: Tensor) -> dict[str, Tensor]:
        del batch
        self._validate_measurement(measurement)

        height, width = measurement.shape[-2:]
        pad_height = (-height) % self.pad_multiple
        pad_width = (-width) % self.pad_multiple
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left

        measurement_max = measurement.amax(dim=(1, 2, 3), keepdim=True)
        normalized = torch.where(
            measurement_max > 0,
            measurement / measurement_max.clamp_min(1e-12),
            measurement,
        )
        padded = F.pad(normalized, (left, right, top, bottom), value=0.0)
        encoded = self.autoencoder.encode(padded.mul(2).sub(1))
        latent = encoded.latent_dist.mode()
        prediction = self.autoencoder.decode(latent).sample.add(1).div(2)
        prediction = prediction[..., top : top + height, left : left + width]

        prediction = prediction.clamp_min(0)
        prediction_max = prediction.amax(dim=(1, 2, 3), keepdim=True)
        prediction = torch.where(
            prediction_max > 0,
            prediction / prediction_max.clamp_min(1e-12),
            prediction,
        )

        if self.output_crop is not None:
            crop_top, crop_left, crop_height, crop_width = self.output_crop
            if crop_top + crop_height > height or crop_left + crop_width > width:
                raise ValueError("output_crop exceeds the reconstructed image bounds")
            prediction = prediction[
                ...,
                crop_top : crop_top + crop_height,
                crop_left : crop_left + crop_width,
            ]
        return {"prediction": prediction}


__all__ = ["PSFFreeAutoencoderKL"]
