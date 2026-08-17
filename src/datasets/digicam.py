from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

DEFAULT_DIGICAM_REPO = "bezzam/DigiCam-Mirflickr-MultiMask-1K"


class DigiCamRealDataset(Dataset):
    def __init__(
        self,
        repo_id: str = DEFAULT_DIGICAM_REPO,
        revision: str | None = None,
        split: str = "test",
        parquet_path: str | Path | None = None,
        indices: int | list[int] | None = None,
        index_start: int = 0,
        cache_dir: str | Path | None = None,
        source_dataset: Any | None = None,
        measurement_key: str = "lensless",
        target_key: str = "lensed",
        mask_key: str = "mask_label",
        force_rgb: bool = True,
        rotate_measurement: bool = True,
        measurement_downsample: float = 1.0,
        target_size: list[int] | None = None,
    ):
        if not repo_id:
            raise ValueError("repo_id must be a non-empty string")
        if not split:
            raise ValueError("split must be a non-empty string")
        if measurement_downsample <= 0:
            raise ValueError("measurement_downsample must be positive")
        if index_start < 0:
            raise ValueError("index_start must be non-negative")
        if target_size is not None:
            if len(target_size) != 2 or any(int(value) <= 0 for value in target_size):
                raise ValueError("target_size must contain positive [height, width]")
            target_size = tuple(int(value) for value in target_size)

        self.repo_id = repo_id
        self.revision = revision
        self.split = split
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.parquet_path = (
            str(Path(parquet_path).expanduser().resolve())
            if parquet_path is not None
            else None
        )
        self.measurement_key = measurement_key
        self.target_key = target_key
        self.mask_key = mask_key
        self.force_rgb = force_rgb
        self.rotate_measurement = rotate_measurement
        self.measurement_downsample = float(measurement_downsample)
        self.target_size = target_size
        self.index_start = int(index_start)

        if source_dataset is None:
            source_dataset = self._load_huggingface_dataset()
        if not hasattr(source_dataset, "__len__") or not hasattr(
            source_dataset, "__getitem__"
        ):
            raise TypeError("source_dataset must be an indexable dataset")
        self.source_dataset = source_dataset
        self.indices = self._normalize_indices(indices)

    def _load_huggingface_dataset(self):
        try:
            if self.parquet_path is not None:
                from datasets import Dataset
            else:
                from datasets import load_dataset
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Loading DigiCam from Hugging Face requires the 'datasets' "
                "package. Install it or pass a local source_dataset."
            ) from exc

        if self.parquet_path is not None:
            path = Path(self.parquet_path)
            if not path.is_file():
                raise FileNotFoundError(f"parquet file not found: {path}")
            return Dataset.from_parquet(str(path), cache_dir=self.cache_dir)

        return load_dataset(
            self.repo_id,
            revision=self.revision,
            split=self.split,
            cache_dir=self.cache_dir,
        )

    def _normalize_indices(self, indices: int | list[int] | None) -> tuple[int, ...]:
        if indices is None:
            normalized = tuple(range(self.index_start, len(self.source_dataset)))
        elif isinstance(indices, int):
            if indices < 0:
                raise ValueError("integer indices must be non-negative")
            normalized = tuple(range(self.index_start, self.index_start + indices))
        else:
            if self.index_start != 0:
                raise ValueError("index_start cannot be combined with explicit indices")
            normalized = tuple(int(index) for index in indices)

        source_length = len(self.source_dataset)
        if any(index < 0 or index >= source_length for index in normalized):
            raise IndexError(
                f"indices must be within [0, {source_length}), got {normalized}"
            )
        return normalized

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self.indices[index]
        record = self.source_dataset[source_index]
        if self.measurement_key not in record or self.target_key not in record:
            raise KeyError(
                "DigiCam record must contain "
                f"'{self.measurement_key}' and '{self.target_key}'"
            )

        measurement = self._image_to_chw_float(
            record[self.measurement_key], field=self.measurement_key
        )
        target = self._image_to_chw_float(
            record[self.target_key], field=self.target_key
        )

        if self.rotate_measurement:
            measurement = torch.rot90(measurement, k=2, dims=(-2, -1))
        if self.measurement_downsample != 1:
            output_size = tuple(
                max(1, int(size / self.measurement_downsample))
                for size in measurement.shape[-2:]
            )
            measurement = self._resize(measurement, output_size)
        if self.target_size is not None:
            target = self._resize(target, self.target_size)

        sample_id = self._metadata_value(
            record,
            keys=("sample_id", "id", "image_id"),
            default=f"{self.split}:{source_index}",
        )
        scene_id = self._metadata_value(
            record,
            keys=("scene_id", "image_id"),
            default=sample_id,
        )
        mask_id = self._metadata_value(
            record,
            keys=("mask_id", self.mask_key),
            default="unknown",
        )

        return {
            "measurement": measurement.contiguous(),
            "target": target.contiguous(),
            "sample_id": sample_id,
            "scene_id": scene_id,
            "mask_id": mask_id,
            "split": self.split,
        }

    def _image_to_chw_float(self, image: Any, field: str) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.detach().clone()
        else:
            array = np.array(image, copy=True)
            if np.issubdtype(array.dtype, np.integer):
                dtype_max = np.iinfo(array.dtype).max
                array = array.astype(np.float32) / dtype_max
            elif np.issubdtype(array.dtype, np.bool_):
                array = array.astype(np.float32)
            tensor = torch.from_numpy(array)

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            if tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)
            elif tensor.shape[0] not in (1, 3, 4):
                raise ValueError(
                    f"{field} must use HWC or CHW image layout, got "
                    f"{tuple(tensor.shape)}"
                )
        else:
            raise ValueError(
                f"{field} must be a 2D or 3D image, got {tensor.ndim} dimensions"
            )

        if tensor.shape[0] == 4:
            tensor = tensor[:3]
        if self.force_rgb and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.shape[0] not in (1, 3):
            raise ValueError(f"{field} must have one or three channels")

        if tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{field} contains non-finite values")
            if tensor.numel() and (tensor.min() < 0 or tensor.max() > 1):
                raise ValueError(f"floating-point {field} must be in [0, 1]")
        else:
            if tensor.dtype == torch.bool:
                tensor = tensor.to(dtype=torch.float32)
            else:
                dtype_max = torch.iinfo(tensor.dtype).max
                tensor = tensor.to(dtype=torch.float32) / dtype_max
        return tensor

    @staticmethod
    def _resize(image: torch.Tensor, size: list[int]) -> torch.Tensor:
        return F.interpolate(
            image.unsqueeze(0),
            size=tuple(size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    @staticmethod
    def _metadata_value(record: Any, keys: list[str], default: Any) -> Any:
        for key in keys:
            if key in record and record[key] is not None:
                value = record[key]
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    return value.item()
                return value
        return default
