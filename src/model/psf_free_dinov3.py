from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel


class _DecoderBlock(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )


class _PixelDecoder(nn.Module):
    def __init__(
        self,
        feature_channels: list[int],
        channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(in_channels, channels, 1) for in_channels in feature_channels
        )
        self.refine = nn.ModuleList(_DecoderBlock(channels) for _ in feature_channels)
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
            nn.Conv2d(channels, output_channels, 3, padding=1),
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        output = self.refine[-1](self.lateral[-1](features[-1]))
        for index in range(len(features) - 2, -1, -1):
            output = F.interpolate(
                output,
                size=features[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            output = self.refine[index](output + self.lateral[index](features[index]))
        return self.output(output)


class PSFFreeDINOv3(nn.Module):
    def __init__(
        self,
        local_snapshot_path: str | Path | None,
        pretrained_encoder: bool,
        freeze_encoder: bool = True,
        channels: int = 3,
        feature_channels: list[int] | None = None,
        decoder_channels: int = 128,
        decoder_seed: int = 52,
        pad_multiple: int = 32,
        output_crop: list[int] | None = None,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if feature_channels is None:
            feature_channels = [96, 192, 384, 768]
        if channels != 3:
            raise ValueError("DINOv3 ConvNeXt expects three-channel measurements")
        if len(feature_channels) != 4 or any(value <= 0 for value in feature_channels):
            raise ValueError("feature_channels must contain four positive values")
        if decoder_channels <= 0 or decoder_channels % 8 != 0:
            raise ValueError("decoder_channels must be a positive multiple of eight")
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
        self.feature_channels = tuple(int(value) for value in feature_channels)
        self.freeze_encoder = bool(freeze_encoder)
        self.pad_multiple = int(pad_multiple)
        self.output_crop = output_crop
        self.encoder = encoder or self._load_encoder(
            local_snapshot_path,
            pretrained=pretrained_encoder,
        )
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(decoder_seed))
            self.decoder = _PixelDecoder(
                list(self.feature_channels),
                int(decoder_channels),
                channels,
            )

        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _load_encoder(
        local_snapshot_path: str | Path | None,
        pretrained: bool,
    ) -> DINOv3ConvNextModel:
        if local_snapshot_path is None:
            raise ValueError("local_snapshot_path is required")
        snapshot_path = Path(local_snapshot_path).expanduser()
        if not snapshot_path.is_dir():
            raise FileNotFoundError(f"Local DINOv3 snapshot not found: {snapshot_path}")
        config = DINOv3ConvNextConfig.from_pretrained(
            snapshot_path,
            local_files_only=True,
        )
        config.output_hidden_states = True
        if pretrained:
            return DINOv3ConvNextModel.from_pretrained(
                snapshot_path,
                config=config,
                local_files_only=True,
            )
        return DINOv3ConvNextModel(config)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def train(self, mode: bool = True) -> "PSFFreeDINOv3":
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

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

    def _features(self, measurement: Tensor) -> list[Tensor]:
        context = torch.no_grad() if self.freeze_encoder else nullcontext()
        with context:
            output = self.encoder(
                pixel_values=(measurement - self.image_mean) / self.image_std,
                output_hidden_states=True,
                return_dict=True,
            )
        features = list(output.hidden_states[-4:])
        channels = tuple(feature.shape[1] for feature in features)
        if channels != self.feature_channels:
            raise ValueError(
                f"DINOv3 feature channels must be {self.feature_channels}, got {channels}"
            )
        height, width = measurement.shape[-2:]
        expected_shapes = tuple(
            (height // stride, width // stride) for stride in (4, 8, 16, 32)
        )
        shapes = tuple(feature.shape[-2:] for feature in features)
        if shapes != expected_shapes:
            raise ValueError(
                f"DINOv3 feature shapes must be {expected_shapes}, got {shapes}"
            )
        return features

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
        padded = F.pad(normalized, (left, right, top, bottom), mode="reflect")
        prediction = torch.sigmoid(self.decoder(self._features(padded)))
        prediction = prediction[..., top : top + height, left : left + width]

        if self.output_crop is not None:
            top, left, crop_height, crop_width = self.output_crop
            if top + crop_height > height or left + crop_width > width:
                raise ValueError("output_crop exceeds the reconstructed image bounds")
            prediction = prediction[
                ...,
                top : top + crop_height,
                left : left + crop_width,
            ]
        return {"prediction": prediction}


__all__ = ["PSFFreeDINOv3"]
