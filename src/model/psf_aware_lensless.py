import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from hydra.utils import to_absolute_path
from torch import nn

from src.digicam_synth.pipeline import _load_optics_backend

_load_optics_backend()

import lensless  # noqa: E402
from lensless.recon.sv_deconvnet import SVDeconvNet  # noqa: E402

if not hasattr(lensless, "SVDeconvNet"):
    lensless.SVDeconvNet = SVDeconvNet

from lensless.recon.model_dict import load_model  # noqa: E402
from lensless.recon.admm import ADMM  # noqa: E402
from lensless.recon.unrolled_admm import UnrolledADMM  # noqa: E402
from lensless.recon.utils import create_process_network  # noqa: E402


def _to_ndhwc(image):
    if image.ndim != 4:
        raise ValueError(f"expected NCHW tensor, got {tuple(image.shape)}")
    return image.movedim(1, -1).unsqueeze(1)


def _to_nchw(image):
    if image.ndim != 5 or image.shape[1] != 1:
        raise ValueError(f"expected NDHWC tensor, got {tuple(image.shape)}")
    return image[:, 0].movedim(-1, 1).contiguous()


def _crop_nchw(image, output_crop):
    if output_crop is None:
        return image
    top, left, height, width = output_crop
    if top < 0 or left < 0 or height <= 0 or width <= 0:
        raise ValueError("output_crop must be [top, left, height, width]")
    if top + height > image.shape[-2] or left + width > image.shape[-1]:
        raise ValueError("output_crop exceeds the reconstruction canvas")
    return image[..., top : top + height, left : left + width]


def _validate_measurement_psf(measurement, psf, require_unit_psf, tolerance):
    if measurement.ndim != 4 or psf.ndim != 4:
        raise ValueError("measurement and psf must be NCHW tensors")
    if measurement.shape != psf.shape:
        raise ValueError(
            "measurement and psf must have identical NCHW shapes, got "
            f"{tuple(measurement.shape)} and {tuple(psf.shape)}"
        )
    if measurement.shape[1] not in (1, 3):
        raise ValueError("measurement and psf must have one or three channels")
    if not torch.isfinite(psf).all():
        raise ValueError("psf contains NaN or infinity")
    if torch.any(psf < 0):
        raise ValueError("psf must be non-negative")
    norms = psf.flatten(1).norm(dim=1)
    if torch.any(norms <= 0):
        raise ValueError("psf must have positive energy")
    if require_unit_psf and torch.any(torch.abs(norms - 1.0) > tolerance):
        raise ValueError(
            "psf must use global L2 normalization; observed norms "
            f"{norms.detach().cpu().tolist()}"
        )


def _placeholder_psf(channels, measurement_shape):
    height, width = (int(value) for value in measurement_shape)
    channels = int(channels)
    if channels not in (1, 3):
        raise ValueError("channels must be one or three")
    if height <= 0 or width <= 0:
        raise ValueError("measurement_shape must contain positive values")
    psf = torch.zeros((1, height, width, channels), dtype=torch.float32)
    psf[:, height // 2, width // 2, :] = channels**-0.5
    return psf


class PSFAwareLenslessModel(nn.Module):
    def __init__(
        self,
        repo_id,
        revision,
        cache_dir="data/huggingface",
        output_crop=None,
    ):
        super().__init__()
        if not repo_id:
            raise ValueError("repo_id is required")
        if not revision:
            raise ValueError("revision is required")

        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = str(Path(to_absolute_path(str(cache_dir))).expanduser())
        self.output_crop = tuple(output_crop) if output_crop is not None else None
        self.reconstruction = None
        self.load_seconds = 0.0

    def _load(self, psf):
        start_time = time.perf_counter()
        model_path = snapshot_download(
            repo_id=self.repo_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
        )
        self.reconstruction = load_model(
            model_path=model_path,
            psf=psf,
            device=str(psf.device),
            verbose=True,
        )
        self.load_seconds = time.perf_counter() - start_time
        self.reconstruction.eval()

    @staticmethod
    def _to_ndhwc(image):
        return _to_ndhwc(image)

    @staticmethod
    def _to_nchw(image):
        return _to_nchw(image)

    def forward(self, measurement, psf, **batch):
        del batch
        measurement = self._to_ndhwc(measurement)
        psf = self._to_ndhwc(psf)

        if self.reconstruction is None:
            self._load(psf[0])

        prediction = self.reconstruction.forward(batch=measurement, psfs=psf)
        prediction = self._to_nchw(prediction)

        if self.output_crop is not None:
            top, left, height, width = self.output_crop
            prediction = prediction[..., top : top + height, left : left + width]

        return {"prediction": prediction}


class ClassicalPSFAwareADMMModel(nn.Module):
    """Per-sample classical ADMM reference using the supplied physical PSF."""

    def __init__(
        self,
        n_iter=50,
        mu1=1e-6,
        mu2=1e-5,
        mu3=4e-5,
        tau=1e-4,
        pad=False,
        norm="backward",
        output_crop=None,
        output_normalization=None,
        require_unit_psf=True,
        psf_norm_tolerance=1e-3,
    ):
        super().__init__()
        if int(n_iter) < 1:
            raise ValueError("n_iter must be positive")
        self.n_iter = int(n_iter)
        self.admm_kwargs = {
            "mu1": float(mu1),
            "mu2": float(mu2),
            "mu3": float(mu3),
            "tau": float(tau),
            "pad": bool(pad),
            "norm": str(norm),
        }
        self.output_crop = tuple(output_crop) if output_crop is not None else None
        if output_normalization not in {None, "positive_max", "min_max"}:
            raise ValueError("output_normalization must be null, positive_max or min_max")
        self.output_normalization = output_normalization
        self.require_unit_psf = bool(require_unit_psf)
        self.psf_norm_tolerance = float(psf_norm_tolerance)

    def forward(self, measurement, psf, **batch):
        del batch
        _validate_measurement_psf(
            measurement,
            psf,
            self.require_unit_psf,
            self.psf_norm_tolerance,
        )
        measurements = _to_ndhwc(measurement)
        psfs = _to_ndhwc(psf)
        predictions = []
        for index in range(measurements.shape[0]):
            reconstruction = ADMM(psfs[index], **self.admm_kwargs)
            reconstruction.set_data(measurements[index])
            predictions.append(
                reconstruction.apply(n_iter=self.n_iter, plot=False)
            )
        prediction = _to_nchw(torch.stack(predictions, dim=0))
        prediction = _crop_nchw(prediction, self.output_crop)
        if self.output_normalization is not None:
            if self.output_normalization == "min_max":
                prediction = prediction - prediction.amin(dim=(1, 2, 3), keepdim=True)
            else:
                prediction = prediction.clamp_min(0.0)
            peak = prediction.amax(dim=(1, 2, 3), keepdim=True)
            prediction = torch.where(
                peak > 0,
                prediction / peak.clamp_min(1e-12),
                prediction,
            )
        return {"prediction": prediction}


class TrainablePSFAwareLenslessModel(nn.Module):
    """Fresh U-Net + unrolled ADMM + U-Net physical reconstruction model."""

    def __init__(
        self,
        channels=3,
        measurement_shape=(380, 507),
        output_crop=None,
        n_iter=5,
        mu1=1e-6,
        mu2=1e-5,
        mu3=4e-5,
        tau=1e-4,
        pad=False,
        norm="backward",
        processor_network="UnetRes",
        processor_depth=4,
        processor_channels=(64, 128, 256, 512),
        pre_processor_channels=None,
        post_processor_channels=None,
        use_pre_process=True,
        use_post_process=True,
        require_unit_psf=True,
        psf_norm_tolerance=1e-3,
    ):
        super().__init__()
        if int(n_iter) < 1:
            raise ValueError("n_iter must be positive")
        if not use_pre_process and not use_post_process:
            processor_network = None

        processor_channels = list(processor_channels)
        pre_processor_channels = list(
            pre_processor_channels or processor_channels
        )
        post_processor_channels = list(
            post_processor_channels or processor_channels
        )
        process_kwargs = {
            "network": processor_network,
            "device": "cpu",
            "depth": int(processor_depth),
        }
        pre_process = None
        post_process = None
        if use_pre_process:
            pre_process, _ = create_process_network(
                **process_kwargs,
                nc=pre_processor_channels,
            )
        if use_post_process:
            post_process, _ = create_process_network(
                **process_kwargs,
                nc=post_processor_channels,
            )

        placeholder = _placeholder_psf(channels, measurement_shape)
        self.reconstruction = UnrolledADMM(
            placeholder,
            n_iter=int(n_iter),
            mu1=float(mu1),
            mu2=float(mu2),
            mu3=float(mu3),
            tau=float(tau),
            pad=bool(pad),
            norm=str(norm),
            pre_process=pre_process,
            post_process=post_process,
        )
        self.channels = int(channels)
        self.measurement_shape = tuple(int(value) for value in measurement_shape)
        self.output_crop = tuple(output_crop) if output_crop is not None else None
        self.require_unit_psf = bool(require_unit_psf)
        self.psf_norm_tolerance = float(psf_norm_tolerance)

    def _predict(self, measurement, psf):
        _validate_measurement_psf(
            measurement,
            psf,
            self.require_unit_psf,
            self.psf_norm_tolerance,
        )
        prediction = self.reconstruction(
            batch=_to_ndhwc(measurement),
            psfs=_to_ndhwc(psf),
        )
        return _crop_nchw(_to_nchw(prediction), self.output_crop)

    def forward(self, measurement, psf, replay_psf=None, **batch):
        del batch
        if tuple(measurement.shape[1:]) != (
            self.channels,
            *self.measurement_shape,
        ):
            raise ValueError(
                "measurement does not match configured [channels, height, width]: "
                f"{tuple(measurement.shape)}"
            )
        output = {"prediction": self._predict(measurement, psf)}
        if replay_psf is not None:
            output["replay_prediction"] = self._predict(measurement, replay_psf)
        return output
