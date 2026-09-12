import re
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

from src.datasets.split_manifest import load_split_manifest


def mnist_scene_id(source_split, source_index):
    if source_split not in {"train", "test"}:
        raise ValueError("source_split must be train or test")
    return f"mnist_{source_split}_{int(source_index):05d}"


def stratified_mnist_split_indices(targets, split_seed, validation_per_class):
    validation_per_class = int(validation_per_class)
    if validation_per_class <= 0:
        raise ValueError("validation_per_class must be positive")

    generator = torch.Generator().manual_seed(int(split_seed))
    validation = []
    train = []
    for label in range(10):
        label_indices = torch.where(targets == label)[0]
        order = torch.randperm(len(label_indices), generator=generator)
        shuffled = label_indices[order].tolist()
        if validation_per_class >= len(shuffled):
            raise ValueError("validation split leaves no training examples")
        validation.extend(shuffled[:validation_per_class])
        train.extend(shuffled[validation_per_class:])
    return train, validation


class MNISTSceneDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        split_seed=42,
        validation_per_class=500,
        superpixel_size=8,
        channels=3,
        download=True,
        max_samples=None,
        splits_path=None,
    ):
        self.split = str(split)
        self.superpixel_size = int(superpixel_size)
        self.channels = int(channels)
        self.splits_path = (
            Path(splits_path).expanduser().resolve()
            if splits_path is not None
            else None
        )

        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        if self.superpixel_size <= 0:
            raise ValueError("superpixel_size must be positive")
        if self.channels not in {1, 3}:
            raise ValueError("channels must be 1 or 3")

        official_train = self.split != "test"
        self.dataset = MNIST(
            root=Path(root_dir).expanduser(),
            train=official_train,
            download=bool(download),
        )
        if self.splits_path is not None:
            self.indices = self._manifest_indices(
                split_seed=int(split_seed),
            )
        else:
            self.indices = self._split_indices(
                split_seed=int(split_seed),
                validation_per_class=int(validation_per_class),
            )
        if max_samples is not None:
            max_samples = int(max_samples)
            if not 0 < max_samples <= len(self.indices):
                raise ValueError("max_samples must be within the selected split")
            generator = torch.Generator().manual_seed(int(split_seed) + 104729)
            order = torch.randperm(len(self.indices), generator=generator)[:max_samples]
            self.indices = [self.indices[index] for index in order.tolist()]

    def _manifest_indices(self, split_seed):
        payload = load_split_manifest(self.splits_path)
        if payload["seed"] != split_seed:
            raise ValueError(
                "MNIST manifest split_seed does not match the dataset config"
            )
        namespace = "test" if self.split == "test" else "train"
        pattern = re.compile(rf"^mnist_{namespace}_(\d{{5}})$")
        indices = []
        for scene_id in payload["splits"][self.split]:
            match = pattern.fullmatch(scene_id)
            if match is None:
                raise ValueError(
                    f"MNIST manifest ID has the wrong source namespace: {scene_id}"
                )
            source_index = int(match.group(1))
            if source_index >= len(self.dataset):
                raise ValueError(
                    f"MNIST manifest source index is out of range: {scene_id}"
                )
            indices.append(source_index)
        return indices

    def _split_indices(self, split_seed, validation_per_class):
        if self.split == "test":
            return list(range(len(self.dataset)))
        train, validation = stratified_mnist_split_indices(
            self.dataset.targets,
            split_seed,
            validation_per_class,
        )
        return train if self.split == "train" else validation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = int(self.indices[index])
        image = self.dataset.data[source_index].float().div(255)
        image = image / image.amax().clamp_min(1e-8)
        image = F.pad(image, (2, 2, 2, 2))
        image = image.repeat_interleave(self.superpixel_size, 0)
        image = image.repeat_interleave(self.superpixel_size, 1)
        target = image.unsqueeze(0).repeat(self.channels, 1, 1)
        label = int(self.dataset.targets[source_index])
        source_split = "train" if self.split != "test" else "test"
        scene_id = mnist_scene_id(source_split, source_index)
        return {
            "target": target,
            "label": label,
            "sample_id": scene_id,
            "scene_id": scene_id,
            "source_index": source_index,
            "split": self.split,
        }
