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
        if image.ndim != 4:
            raise ValueError(f"expected NCHW tensor, got {tuple(image.shape)}")
        return image.movedim(1, -1).unsqueeze(1)

    @staticmethod
    def _to_nchw(image):
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError(f"expected NDHWC output, got {tuple(image.shape)}")
        return image[:, 0].movedim(-1, 1).contiguous()

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
