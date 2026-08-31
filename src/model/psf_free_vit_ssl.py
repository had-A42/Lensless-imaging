from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import Dinov2Config, Dinov2Model


class _DecoderBlock(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )


class _ViTDecoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        feature_count: int,
        channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.project = nn.Conv2d(feature_dim * feature_count, channels, 1)
        self.blocks = nn.ModuleList(_DecoderBlock(channels) for _ in range(4))
        self.output = nn.Conv2d(channels, output_channels, 3, padding=1)

    def forward(self, features: list[Tensor], output_size: tuple[int, int]) -> Tensor:
        output = self.project(torch.cat(features, dim=1))
        for block in self.blocks:
            output = F.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=False
            )
            output = block(output)
        output = F.interpolate(
            output, size=output_size, mode="bilinear", align_corners=False
        )
        return self.output(output)


class PSFFreeViTSSL(nn.Module):
    """Frozen external SSL backbone with a small, matched reconstruction head."""

    def __init__(
        self,
        encoder_kind: str,
        local_snapshot_path: str | Path | None,
        pretrained_encoder: bool,
        feature_dim: int = 384,
        layer_indices: list[int] | None = None,
        patch_size: int = 16,
        freeze_encoder: bool = True,
        channels: int = 3,
        decoder_channels: int = 128,
        decoder_seed: int = 52,
        output_crop: list[int] | None = None,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if layer_indices is None:
            layer_indices = [2, 5, 8, 11]
        if encoder_kind not in {"lingbot", "dinov2"}:
            raise ValueError("encoder_kind must be lingbot or dinov2")
        if channels != 3:
            raise ValueError("SSL backbones expect three-channel measurements")

        self.encoder_kind = encoder_kind
        self.feature_dim = int(feature_dim)
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.patch_size = int(patch_size)
        self.freeze_encoder = bool(freeze_encoder)
        self.channels = int(channels)
        self.output_crop = tuple(output_crop) if output_crop is not None else None

        self.encoder = encoder or self._load_encoder(
            local_snapshot_path,
            pretrained=pretrained_encoder,
        )
        self.encoder.requires_grad_(not self.freeze_encoder)
        if self.freeze_encoder:
            self.encoder.eval()

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(decoder_seed))
            self.decoder = _ViTDecoder(
                self.feature_dim,
                len(self.layer_indices),
                int(decoder_channels),
                self.channels,
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

    def _load_encoder(
        self,
        local_snapshot_path: str | Path | None,
        pretrained: bool,
    ) -> nn.Module:
        if self.encoder_kind == "lingbot":
            return self._load_lingbot(local_snapshot_path, pretrained)
        return self._load_dinov2(local_snapshot_path, pretrained)

    @staticmethod
    def _load_lingbot(
        local_snapshot_path: str | Path | None,
        pretrained: bool,
    ) -> nn.Module:
        try:
            from lingbot_vision import load_config, load_pretrained_backbone
            from lingbot_vision.build import build_backbone_from_cfg
        except ImportError as error:
            raise ImportError(
                "LingBot-Vision is required for the LingBot probe; install requirements.txt"
            ) from error

        if pretrained:
            if local_snapshot_path is None:
                raise ValueError(
                    "local_snapshot_path is required for pretrained LingBot"
                )
            snapshot_path = Path(local_snapshot_path).expanduser()
            encoder, _ = load_pretrained_backbone(
                repo_id_or_path=snapshot_path,
                variant="small",
                device="cpu",
                dtype=torch.float32,
                local_files_only=True,
                verbose=False,
            )
            return encoder

        config = load_config("configs/lbot_vision_vits.yaml")
        encoder, _ = build_backbone_from_cfg(config)
        return encoder

    @staticmethod
    def _load_dinov2(
        local_snapshot_path: str | Path | None,
        pretrained: bool,
    ) -> Dinov2Model:
        if local_snapshot_path is None:
            raise ValueError("local_snapshot_path is required for DINOv2")
        snapshot_path = Path(local_snapshot_path).expanduser()
        config = Dinov2Config.from_pretrained(snapshot_path, local_files_only=True)
        config.output_hidden_states = True
        if pretrained:
            return Dinov2Model.from_pretrained(
                snapshot_path,
                config=config,
                local_files_only=True,
            )
        return Dinov2Model(config)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def train(self, mode: bool = True) -> "PSFFreeViTSSL":
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _lingbot_features(self, measurement: Tensor) -> list[Tensor]:
        return list(
            self.encoder.get_intermediate_layers(
                measurement,
                n=self.layer_indices,
                reshape=True,
            )
        )

    def _dinov2_features(self, measurement: Tensor) -> list[Tensor]:
        output = self.encoder(
            pixel_values=measurement,
            output_hidden_states=True,
            return_dict=True,
        )
        height = measurement.shape[-2] // self.patch_size
        width = measurement.shape[-1] // self.patch_size
        features = []
        for index in self.layer_indices:
            tokens = self.encoder.layernorm(output.hidden_states[index + 1])[:, 1:]
            features.append(
                tokens.reshape(tokens.shape[0], height, width, self.feature_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        return features

    def _features(self, measurement: Tensor) -> list[Tensor]:
        context = torch.no_grad() if self.freeze_encoder else nullcontext()
        with context:
            if self.encoder_kind == "lingbot":
                return self._lingbot_features(measurement)
            return self._dinov2_features(measurement)

    def forward(self, measurement: Tensor, **batch: Tensor) -> dict[str, Tensor]:
        del batch
        height, width = measurement.shape[-2:]
        pad_height = (-height) % self.patch_size
        pad_width = (-width) % self.patch_size
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
        encoder_input = (padded - self.image_mean) / self.image_std
        prediction = torch.sigmoid(
            self.decoder(self._features(encoder_input), padded.shape[-2:])
        )
        prediction = prediction[..., top : top + height, left : left + width]

        if self.output_crop is not None:
            crop_top, crop_left, crop_height, crop_width = self.output_crop
            prediction = prediction[
                ...,
                crop_top : crop_top + crop_height,
                crop_left : crop_left + crop_width,
            ]
        return {"prediction": prediction}


__all__ = ["PSFFreeViTSSL"]
