"""Deterministic mask splits shared by the real-data experiments."""

from typing import Iterable

import numpy as np

DIGICAM_DEVELOPMENT_SPLIT_SEED = 20260826
OFFICIAL_TEST_MASK_IDS = tuple(range(15))
OFFICIAL_TRAIN_MASK_IDS = tuple(range(15, 100))


def build_digicam_mask_split(
    mask_ids: Iterable[int] = OFFICIAL_TRAIN_MASK_IDS,
    *,
    seed: int = DIGICAM_DEVELOPMENT_SPLIT_SEED,
    calibration_mask_count: int = 68,
) -> tuple[list[int], list[int]]:
    """Return the deterministic mask-level calibration/gate partition"""

    normalized = sorted(int(mask_id) for mask_id in mask_ids)
    if len(normalized) != len(set(normalized)):
        raise ValueError("mask_ids must be unique")
    if not 0 < int(calibration_mask_count) < len(normalized):
        raise ValueError("calibration_mask_count must define two non-empty splits")
    permutation = np.random.default_rng(int(seed)).permutation(normalized).tolist()
    return (
        permutation[: int(calibration_mask_count)],
        permutation[int(calibration_mask_count) :],
    )
