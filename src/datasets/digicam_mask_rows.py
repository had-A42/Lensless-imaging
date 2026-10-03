"""Explicit mask-by-row views of the upstream DigiCam training split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.datasets.digicam import DigiCamRealDataset

UPSTREAM_TRAIN_MASK_MIN = 15
UPSTREAM_TRAIN_MASK_MAX = 99
UPSTREAM_TRAIN_MASK_COUNT = 85


def digicam_train_source_index(mask_id: int, row_slot: int) -> int:
    mask_id = int(mask_id)
    row_slot = int(row_slot)
    if not UPSTREAM_TRAIN_MASK_MIN <= mask_id <= UPSTREAM_TRAIN_MASK_MAX:
        raise ValueError("development mask IDs must be within 15..99")
    if row_slot < 0:
        raise ValueError("row slots must be non-negative")
    return row_slot * UPSTREAM_TRAIN_MASK_COUNT + mask_id - UPSTREAM_TRAIN_MASK_MIN


class DigiCamMaskRowDataset(Dataset):
    """Select a deterministic Cartesian product of masks and row slots.

    The class deliberately permits only the upstream ``train`` split. It is
    intended for development protocols that must not accidentally open the
    official real test set.
    """

    def __init__(
        self,
        mask_ids: list[int],
        repo_id: str,
        revision: str,
        row_slots: list[int] | None = None,
        row_slot_start: int = 0,
        row_slot_count: int | None = None,
        split: str = "train",
        cache_dir: str | Path | None = None,
        source_dataset: Any | None = None,
        target_size: list[int] | None = None,
        rotate_measurement: bool = True,
        return_psf: bool = False,
        return_replay_psf: bool = False,
        prepared_psf_bundle_path: str | Path | None = None,
        simulator_config: Any | None = None,
    ) -> None:
        if split != "train":
            raise ValueError("DigiCamMaskRowDataset only permits split='train'")
        if row_slots is None:
            if row_slot_count is None or int(row_slot_count) <= 0:
                raise ValueError("row_slots or a positive row_slot_count is required")
            row_slots = range(
                int(row_slot_start),
                int(row_slot_start) + int(row_slot_count),
            )
        elif row_slot_count is not None or int(row_slot_start) != 0:
            raise ValueError(
                "explicit row_slots cannot be combined with row_slot_start/count"
            )
        self.mask_ids = tuple(int(value) for value in mask_ids)
        self.row_slots = tuple(int(value) for value in row_slots)
        if not self.mask_ids or not self.row_slots:
            raise ValueError("mask_ids and row_slots must be non-empty")
        if len(set(self.mask_ids)) != len(self.mask_ids):
            raise ValueError("mask_ids must be unique")
        if len(set(self.row_slots)) != len(self.row_slots):
            raise ValueError("row_slots must be unique")

        identities = [
            {
                "source_index": digicam_train_source_index(mask_id, row_slot),
                "mask_id": mask_id,
                "row_slot": row_slot,
            }
            for mask_id in self.mask_ids
            for row_slot in self.row_slots
        ]
        if source_dataset is None:
            source_dataset = DigiCamRealDataset(
                repo_id=repo_id,
                revision=revision,
                split=split,
                cache_dir=cache_dir,
                indices=[],
            ).source_dataset
        self.identities = tuple(identities)
        self.base = DigiCamRealDataset(
            repo_id=repo_id,
            revision=revision,
            split=split,
            cache_dir=cache_dir,
            source_dataset=source_dataset,
            indices=[row["source_index"] for row in identities],
            force_rgb=True,
            rotate_measurement=rotate_measurement,
            measurement_downsample=1.0,
            target_size=target_size,
            target_resize_mode="bilinear",
            return_psf=return_psf,
            return_replay_psf=return_replay_psf,
            prepared_psf_bundle_path=prepared_psf_bundle_path,
            simulator_config=simulator_config,
            expected_mask_count=len(self.mask_ids),
            expected_scenes_per_mask=len(self.row_slots),
        )

    def __len__(self) -> int:
        return len(self.identities)

    def __getitem__(self, index: int) -> dict[str, Any]:
        identity = self.identities[index]
        sample = self.base[index]
        if int(sample["mask_id"]) != identity["mask_id"]:
            raise ValueError(
                "upstream row identity mismatch: "
                f"expected mask {identity['mask_id']}, got {sample['mask_id']}"
            )
        sample.update(
            {
                "source_index": identity["source_index"],
                "row_slot": identity["row_slot"],
                "protocol_role": "explicit_development_grid",
            }
        )
        return sample


__all__ = [
    "DigiCamMaskRowDataset",
    "digicam_train_source_index",
]
