from pathlib import Path

import torch
from lensless.recon.drunet.network_unet import UNetRes
from torch import Tensor, nn
from torch.nn import functional as F

UNET8M_CHANNELS = (32, 64, 128, 256)
UNET8M_DEPTH = 4


class PSFFreeDRUNet(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        nc: list[int] = UNET8M_CHANNELS,
        depth: int = UNET8M_DEPTH,
        noise_level: float = 1.0,
        trainable_noise_level: bool = True,
        checkpoint_path: str | Path | None = None,
        strict_checkpoint: bool = True,
        output_crop: list[int] | None = None,
    ) -> None:
        super().__init__()
        nc = tuple(int(value) for value in nc)
        if channels <= 0:
            raise ValueError("channels must be positive")
        if len(nc) != 4 or any(value <= 0 for value in nc):
            raise ValueError("nc must contain four positive channel counts")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if noise_level <= 0 or noise_level > 255:
            raise ValueError("noise_level must be in (0, 255]")

        self.channels = int(channels)
        self.nc = nc
        self.depth = int(depth)
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
        self.output_crop = output_crop
        self.network = UNetRes(
            in_nc=self.channels + 1,
            out_nc=self.channels,
            nc=list(self.nc),
            nb=self.depth,
            act_mode="R",
            downsample_mode="strideconv",
            upsample_mode="convtranspose",
        )
        self.noise_level = nn.Parameter(
            torch.tensor([float(noise_level)]),
            requires_grad=trainable_noise_level,
        )

        if checkpoint_path is not None:
            self.load_local_checkpoint(checkpoint_path, strict=strict_checkpoint)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _checkpoint_file(path: str | Path) -> Path:
        checkpoint_path = Path(path).expanduser()
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "recon_epochBEST"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Local checkpoint not found: {checkpoint_path}. "
                "Pass either a checkpoint file or a model directory containing "
                "`recon_epochBEST`; this adapter never downloads checkpoints."
            )
        return checkpoint_path

    @staticmethod
    def _load_tensor_mapping(path: Path) -> dict[str, Tensor]:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")

        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint must contain a state-dict mapping")
        for wrapper_key in ("state_dict", "model_state_dict", "model"):
            wrapped = checkpoint.get(wrapper_key)
            if isinstance(wrapped, dict):
                checkpoint = wrapped
                break
        if not all(isinstance(key, str) for key in checkpoint):
            raise TypeError("Checkpoint state-dict keys must be strings")
        return checkpoint

    @staticmethod
    def _strip_parallel_prefix(key: str) -> str:
        while key.startswith("module."):
            key = key[len("module.") :]
        return key

    @classmethod
    def _extract_checkpoint_state(
        cls, checkpoint: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor | None]:
        network_state: dict[str, Tensor] = {}
        loaded_noise_level: Tensor | None = None
        network_markers = (
            "post_process_model.module.",
            "post_process_model.",
            "network.",
            "backbone.",
        )

        for original_key, value in checkpoint.items():
            if not isinstance(value, Tensor):
                continue
            key = cls._strip_parallel_prefix(original_key)

            if key.endswith("post_process_param") or key == "noise_level":
                loaded_noise_level = value
                continue

            extracted_key = key
            for marker in network_markers:
                if marker in key:
                    extracted_key = key.split(marker, maxsplit=1)[1]
                    break
            if extracted_key.startswith(
                ("m_head.", "m_down", "m_body.", "m_up", "m_tail.")
            ):
                network_state[extracted_key] = value

        if not network_state:
            raise ValueError(
                "Checkpoint contains no UnetRes post-processor weights. Supported formats are "
                "a bare UNetRes state dict, this adapter's state dict, or LenslessPiCam's full "
                "`recon_epochBEST` state dict."
            )
        return network_state, loaded_noise_level

    def load_local_checkpoint(
        self, checkpoint_path: str | Path, strict: bool = True
    ) -> nn.modules.module._IncompatibleKeys:
        path = self._checkpoint_file(checkpoint_path)
        checkpoint = self._load_tensor_mapping(path)
        network_state, loaded_noise_level = self._extract_checkpoint_state(checkpoint)
        incompatible = self.network.load_state_dict(network_state, strict=strict)

        if loaded_noise_level is not None:
            loaded_noise_level = loaded_noise_level.detach().reshape(-1)
            if loaded_noise_level.numel() != 1:
                raise ValueError(
                    "Checkpoint noise level must contain exactly one value"
                )
            value = loaded_noise_level.to(
                device=self.noise_level.device,
                dtype=self.noise_level.dtype,
            )
            if value.item() <= 0 or value.item() > 255:
                raise ValueError("Checkpoint noise level must be in (0, 255]")
            with torch.no_grad():
                self.noise_level.copy_(value)

        return incompatible

    def _validate_measurement(self, measurement: Tensor) -> None:
        if not isinstance(measurement, Tensor):
            raise TypeError("measurement must be a torch.Tensor")
        if measurement.ndim != 4:
            raise ValueError("measurement must have NCHW shape")
        if measurement.shape[1] != self.channels:
            raise ValueError(
                f"measurement must have {self.channels} channels, got {measurement.shape[1]}"
            )
        if not measurement.is_floating_point():
            raise TypeError("measurement must have a floating-point dtype")
        if not torch.isfinite(measurement).all():
            raise ValueError("measurement must contain only finite values")
        if measurement.numel() and (
            measurement.detach().amin().item() < 0.0
            or measurement.detach().amax().item() > 1.0
        ):
            raise ValueError("measurement values must be in [0, 1]")

    def forward(self, measurement: Tensor, **batch: Tensor) -> dict[str, Tensor]:
        del batch
        self._validate_measurement(measurement)

        height, width = measurement.shape[-2:]
        pad_height = (-height) % 8
        pad_width = (-width) % 8
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left

        scale = measurement.amax(dim=(1, 2, 3), keepdim=True) + 1e-6
        normalized = measurement / scale
        normalized = F.pad(normalized, (left, right, top, bottom), value=0.0)

        if (
            self.noise_level.detach().item() <= 0
            or self.noise_level.detach().item() > 255
        ):
            raise ValueError("learned noise_level must remain in (0, 255]")
        noise_map = (self.noise_level / 255.0).to(normalized)
        noise_map = noise_map.reshape(1, 1, 1, 1).expand(
            normalized.shape[0], 1, normalized.shape[2], normalized.shape[3]
        )
        network_input = torch.cat((normalized, noise_map), dim=1)

        prediction = self.network(network_input)
        prediction = prediction[..., top : top + height, left : left + width]
        prediction = prediction.clamp_min(0.0) * scale
        prediction_max = prediction.amax(dim=(1, 2, 3), keepdim=True)
        prediction = torch.where(
            prediction_max > 0,
            prediction / prediction_max.clamp_min(1e-12),
            prediction,
        )
        if self.output_crop is not None:
            crop_top, crop_left, crop_height, crop_width = self.output_crop
            crop_bottom = crop_top + crop_height
            crop_right = crop_left + crop_width
            if crop_bottom > height or crop_right > width:
                raise ValueError(
                    "output_crop exceeds the reconstructed image bounds: "
                    f"crop={self.output_crop}, image={(height, width)}"
                )
            prediction = prediction[
                ...,
                crop_top:crop_bottom,
                crop_left:crop_right,
            ]
        return {"prediction": prediction}


__all__ = [
    "PSFFreeDRUNet",
    "UNET8M_CHANNELS",
    "UNET8M_DEPTH",
]
