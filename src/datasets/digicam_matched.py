from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import to_absolute_path
from lensless.recon.rfft_convolve import RealFFTConvolve2D
from torch.utils.data import Dataset

from src.datasets.digicam import DigiCamRealDataset
from src.datasets.on_the_fly import _aligned_measurement
from src.digicam_protocol import build_digicam_mask_split


def build_row_slot_split(
    seed: int = 20260827,
    validation_size: int = 25,
) -> tuple[list[int], list[int]]:
    """Split the 250 rows available for every mask into train and validation."""
    candidates = np.arange(10, 250)
    shuffled = np.random.Generator(np.random.PCG64(seed)).permutation(candidates)
    validation = sorted(int(value) for value in shuffled[:validation_size])
    train = sorted(
        list(range(10)) + [int(value) for value in shuffled[validation_size:]]
    )
    return train, validation


def _source_index(mask_id: int, row_slot: int) -> int:
    # The source dataset is cycle-major: masks 15..99 repeat for every row slot.
    return int(row_slot) * 85 + int(mask_id) - 15


def _rows(role: str) -> list[dict[str, int]]:
    development_masks, _ = build_digicam_mask_split()
    train_slots, validation_slots = build_row_slot_split()
    if role == "train":
        slots = train_slots
    elif role == "inner_validation":
        slots = validation_slots
    else:
        raise ValueError("role must be train or inner_validation")
    return sorted(
        (
            {
                "source_index": _source_index(mask_id, row_slot),
                "mask_id": int(mask_id),
                "row_slot": int(row_slot),
            }
            for row_slot in slots
            for mask_id in development_masks
        ),
        key=lambda row: row["source_index"],
    )


class DigiCamMatchedDomainDataset(Dataset):
    """Real or matched-simulation rows for the same PSF-free training task."""

    def __init__(
        self,
        role: str,
        measurement_domain: str,
        repo_id: str,
        revision: str,
        split: str = "train",
        cache_dir: str | Path | None = None,
        source_dataset: Any | None = None,
        prepared_psf_bundle_path: str | Path | None = None,
        convolver_cache_size: int = 2,
    ) -> None:
        if measurement_domain not in {"real", "matched_sim"}:
            raise ValueError("measurement_domain must be real or matched_sim")

        if source_dataset is None:
            source_dataset = DigiCamRealDataset(
                repo_id=repo_id,
                revision=revision,
                split=split,
                cache_dir=cache_dir,
                indices=[],
            ).source_dataset

        self.rows = _rows(role)
        self.base = DigiCamRealDataset(
            repo_id=repo_id,
            revision=revision,
            split=split,
            cache_dir=cache_dir,
            source_dataset=source_dataset,
            indices=[row["source_index"] for row in self.rows],
            force_rgb=True,
            rotate_measurement=True,
            measurement_downsample=1.0,
            target_size=[200, 266],
            target_resize_mode="bilinear",
        )
        self.role = role
        self.measurement_domain = measurement_domain
        self.convolver_cache_size = int(convolver_cache_size)
        self.convolver_cache: OrderedDict[int, RealFFTConvolve2D] = OrderedDict()
        self.prepared_psfs: dict[int, np.ndarray] = {}

        if measurement_domain == "matched_sim":
            if prepared_psf_bundle_path is None:
                raise ValueError("matched_sim needs prepared_psf_bundle_path")
            path = Path(to_absolute_path(str(prepared_psf_bundle_path)))
            development_masks, _ = build_digicam_mask_split()
            with np.load(path, allow_pickle=False) as bundle:
                for mask_id in development_masks:
                    self.prepared_psfs[int(mask_id)] = np.ascontiguousarray(
                        bundle[f"mask_{mask_id}"]
                    )

    def __len__(self) -> int:
        return len(self.rows)

    def _convolver(self, mask_id: int) -> RealFFTConvolve2D:
        if mask_id in self.convolver_cache:
            convolver = self.convolver_cache.pop(mask_id)
            self.convolver_cache[mask_id] = convolver
            return convolver
        convolver = RealFFTConvolve2D(
            psf=torch.as_tensor(self.prepared_psfs[mask_id], dtype=torch.float32)
        )
        self.convolver_cache[mask_id] = convolver
        while len(self.convolver_cache) > self.convolver_cache_size:
            self.convolver_cache.popitem(last=False)
        return convolver

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample = self.base[index]
        mask_id = row["mask_id"]
        if self.measurement_domain == "matched_sim":
            sample["measurement"] = _aligned_measurement(
                sample["target"],
                self.prepared_psfs[mask_id],
                self._convolver(mask_id),
                [80, 100, 200, 266],
                quantize=True,
            )
        sample.update(
            {
                "source_index": row["source_index"],
                "row_slot": row["row_slot"],
                "protocol_role": self.role,
                "measurement_domain": self.measurement_domain,
            }
        )
        return sample


__all__ = ["DigiCamMatchedDomainDataset", "build_row_slot_split"]
