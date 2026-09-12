from pathlib import Path

import numpy as np
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import center_crop, pil_to_tensor

from src.datasets.split_manifest import load_split_manifest


def celeba_filenames(root_dir, split, max_samples=None, split_seed=42):
    partition = {"train": "0", "validation": "1", "test": "2"}[split]
    path = Path(root_dir).expanduser() / "list_eval_partition.txt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run prepare_celeba.py first.")
    with path.open() as file:
        names = [
            name for name, code in (line.split() for line in file) if code == partition
        ]
    if not names or any(
        not name.endswith(".jpg") or not name[:-4].isdigit() for name in names
    ):
        raise ValueError(f"Expected canonical CelebA JPEG names in {path}")
    if max_samples is not None:
        if not 0 < max_samples <= len(names):
            raise ValueError(
                f"max_samples must be between 1 and {len(names)} for {split}"
            )
        indices = np.random.default_rng(split_seed).permutation(len(names))[
            :max_samples
        ]
        names = [names[index] for index in indices]
    return names


def celeba_manifest_filenames(root_dir, splits_path, split, split_seed=42):
    payload = load_split_manifest(splits_path)
    if payload["seed"] != int(split_seed):
        raise ValueError("CelebA manifest split_seed does not match the dataset config")
    if split not in payload["splits"]:
        raise ValueError("split must be train, validation or test")
    names = list(payload["splits"][split])
    if any(not name.endswith(".jpg") or not name[:-4].isdigit() for name in names):
        raise ValueError("CelebA manifest contains a non-canonical JPEG name")
    official = set(celeba_filenames(root_dir, split, None, split_seed))
    unknown = [name for name in names if name not in official]
    if unknown:
        raise ValueError(
            f"CelebA manifest ID is outside the official {split} partition: {unknown[0]}"
        )
    return names


class CelebASceneDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        max_samples=None,
        split_seed=42,
        image_size=32,
        superpixel_size=8,
        crop_size=178,
        splits_path=None,
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.split = split
        self.image_size = int(image_size)
        self.superpixel_size = int(superpixel_size)
        self.crop_size = int(crop_size)
        if min(self.image_size, self.superpixel_size, self.crop_size) < 1:
            raise ValueError(
                "image_size, superpixel_size and crop_size must be positive"
            )
        self.splits_path = (
            Path(splits_path).expanduser().resolve()
            if splits_path is not None
            else None
        )
        if self.splits_path is not None:
            self.filenames = celeba_manifest_filenames(
                root_dir,
                self.splits_path,
                split,
                split_seed,
            )
        else:
            self.filenames = celeba_filenames(root_dir, split, max_samples, split_seed)
        if self.splits_path is not None and max_samples is not None:
            max_samples = int(max_samples)
            if not 0 < max_samples <= len(self.filenames):
                raise ValueError("max_samples must be within the selected split")
            self.filenames = self.filenames[:max_samples]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        with Image.open(self.root_dir / "img_align_celeba" / name) as image:
            image = center_crop(image.convert("RGB"), [self.crop_size] * 2)
            target = pil_to_tensor(image).float().div(255)
        target = F.interpolate(
            target.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        target = target.repeat_interleave(self.superpixel_size, -2)
        target = target.repeat_interleave(self.superpixel_size, -1)
        scene_id = f"celeba_{Path(name).stem}"
        return {
            "target": target.contiguous(),
            "sample_id": scene_id,
            "scene_id": scene_id,
            "source_index": int(Path(name).stem) - 1,
            "split": self.split,
        }
