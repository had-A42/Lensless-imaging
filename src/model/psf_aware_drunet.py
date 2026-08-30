import torch
from lensless.recon.drunet.network_unet import UNetRes
from torch import Tensor
from torch.nn import functional as F

from src.model.psf_free_drunet import PSFFreeDRUNet


class PSFAwareDRUNet(PSFFreeDRUNet):
    """DRUNet conditioned on the corresponding full-resolution PSF.

    This arm intentionally differs from the measurement-only baseline only in
    the first convolution: the normalized PSF is concatenated to the sensor
    measurement. It measures whether access to the operator helps under the
    same data, training schedule and reconstruction backbone.
    """

    def __init__(self, *args, **kwargs) -> None:
        if kwargs.get("checkpoint_path") is not None:
            raise ValueError(
                "PSF-aware DRUNet currently supports training from scratch"
            )
        super().__init__(*args, **kwargs)
        baseline_network = self.network
        conditioned_network = UNetRes(
            in_nc=2 * self.channels + 1,
            out_nc=self.channels,
            nc=list(self.nc),
            nb=self.depth,
            act_mode="R",
            downsample_mode="strideconv",
            upsample_mode="convtranspose",
        )
        self._copy_baseline_initialization(baseline_network, conditioned_network)
        self.network = conditioned_network

    def _copy_baseline_initialization(
        self,
        baseline: UNetRes,
        conditioned: UNetRes,
    ) -> None:
        """Share the scratch initialization and initially ignore the PSF."""

        baseline_state = baseline.state_dict()
        conditioned_state = conditioned.state_dict()
        with torch.no_grad():
            for name, value in baseline_state.items():
                if name != "m_head.weight":
                    conditioned_state[name].copy_(value)

            baseline_head = baseline_state["m_head.weight"]
            conditioned_head = conditioned_state["m_head.weight"]
            conditioned_head.zero_()
            conditioned_head[:, : self.channels].copy_(
                baseline_head[:, : self.channels]
            )
            conditioned_head[:, -1:].copy_(baseline_head[:, -1:])

    def _validate_psf(self, psf: Tensor, measurement: Tensor) -> None:
        if not isinstance(psf, Tensor):
            raise TypeError("psf must be a torch.Tensor")
        if psf.shape != measurement.shape:
            raise ValueError("psf and measurement must have the same NCHW shape")
        if not psf.is_floating_point():
            raise TypeError("psf must have a floating-point dtype")
        if not torch.isfinite(psf).all():
            raise ValueError("psf must contain only finite values")
        if psf.detach().amin().item() < 0.0:
            raise ValueError("psf must be non-negative")
        if torch.any(psf.amax(dim=(1, 2, 3)) <= 0):
            raise ValueError("each psf must contain positive energy")

    def forward(
        self,
        measurement: Tensor,
        psf: Tensor,
        return_features: bool = False,
        **batch: Tensor,
    ) -> dict[str, Tensor]:
        del batch
        self._validate_measurement(measurement)
        self._validate_psf(psf, measurement)

        height, width = measurement.shape[-2:]
        padding = self._symmetric_padding(height, width)
        scale = measurement.amax(dim=(1, 2, 3), keepdim=True) + 1e-6
        normalized_measurement = F.pad(measurement / scale, padding, value=0.0)
        normalized_psf = psf / psf.amax(dim=(1, 2, 3), keepdim=True)
        normalized_psf = F.pad(normalized_psf, padding, value=0.0)
        network_input = torch.cat(
            (
                normalized_measurement,
                normalized_psf,
                self._noise_map(normalized_measurement),
            ),
            dim=1,
        )
        prediction, feature = self._network_forward(network_input)
        return self._format_output(
            prediction,
            feature,
            scale,
            image_size=(height, width),
            padding=padding,
            return_features=return_features,
        )


__all__ = ["PSFAwareDRUNet"]
