import json

import pytest
import torch
from omegaconf import OmegaConf

from src.datasets.celeba import CelebASceneDataset
from src.datasets.mnist import MNISTSceneDataset
from src.datasets.split_manifest import load_split_manifest, write_split_manifest


def write_manifest(path, splits, seed=42):
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "splits": splits,
            }
        )
    )


def test_shared_manifest_rejects_cross_split_overlap(tmp_path):
    path = tmp_path / "splits.json"
    write_manifest(
        path,
        {"train": ["a"], "validation": ["a"], "test": ["b"]},
    )
    with pytest.raises(ValueError, match="multiple splits"):
        load_split_manifest(path)


def test_shared_manifest_rejects_metadata_outside_seed_and_splits(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "seed": 42,
                "dataset": "Example",
                "splits": {"train": ["a"], "validation": ["b"], "test": ["c"]},
            }
        )
    )
    with pytest.raises(ValueError, match="only seed and splits"):
        load_split_manifest(path)


def test_manifest_writer_requires_explicit_rewrite(tmp_path):
    path = tmp_path / "splits.json"
    first = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    second = {"train": ["d"], "validation": ["e"], "test": ["f"]}

    created = write_split_manifest({"seed": 42, "splits": first}, path, rewrite=False)
    assert created["created"]
    with pytest.raises(ValueError, match="Existing manifest differs"):
        write_split_manifest({"seed": 42, "splits": second}, path, rewrite=False)

    rewritten = write_split_manifest({"seed": 42, "splits": second}, path, rewrite=True)
    assert rewritten["rewritten"]
    assert load_split_manifest(path)["splits"] == second


def test_celeba_uses_manifest_order_with_deterministic_limit(tmp_path):
    (tmp_path / "list_eval_partition.txt").write_text(
        "000001.jpg 0\n000002.jpg 0\n000003.jpg 1\n000004.jpg 2\n"
    )
    manifest = tmp_path / "celeba.json"
    write_manifest(
        manifest,
        {
            "train": ["000002.jpg", "000001.jpg"],
            "validation": ["000003.jpg"],
            "test": ["000004.jpg"],
        },
        seed=42,
    )
    dataset = CelebASceneDataset(
        root_dir=tmp_path,
        splits_path=manifest,
        split="train",
        split_seed=42,
    )
    assert dataset.filenames == ["000002.jpg", "000001.jpg"]
    limited = CelebASceneDataset(
        root_dir=tmp_path,
        splits_path=manifest,
        split="train",
        split_seed=42,
        max_samples=1,
    )
    assert limited.filenames == ["000002.jpg"]


def test_mnist_uses_manifest_indices_and_namespaces(tmp_path, monkeypatch):
    class FakeMNIST:
        def __init__(self, root, train, download):
            del root, download
            count = 6 if train else 2
            self.data = torch.ones(count, 28, 28)
            self.targets = torch.tensor([0, 0, 1, 1, 2, 2][:count])

        def __len__(self):
            return len(self.data)

    monkeypatch.setattr("src.datasets.mnist.MNIST", FakeMNIST)
    manifest = tmp_path / "mnist.json"
    write_manifest(
        manifest,
        {
            "train": ["mnist_train_00004", "mnist_train_00000"],
            "validation": ["mnist_train_00002"],
            "test": ["mnist_test_00001"],
        },
        seed=42,
    )
    dataset = MNISTSceneDataset(
        root_dir=tmp_path,
        splits_path=manifest,
        split="train",
        split_seed=42,
        validation_per_class=1,
        download=False,
    )
    assert dataset.indices == [4, 0]
    limited = MNISTSceneDataset(
        root_dir=tmp_path,
        splits_path=manifest,
        split="train",
        split_seed=42,
        validation_per_class=1,
        max_samples=1,
        download=False,
    )
    repeated = MNISTSceneDataset(
        root_dir=tmp_path,
        splits_path=manifest,
        split="train",
        split_seed=42,
        validation_per_class=1,
        max_samples=1,
        download=False,
    )
    assert len(limited.indices) == 1
    assert limited.indices == repeated.indices


def test_dataset_configs_use_split_manifests():
    mirflickr = OmegaConf.load("src/configs/datasets/mirflickr_on_the_fly.yaml")
    celeba = OmegaConf.load("src/configs/datasets/celeba_on_the_fly.yaml")
    mnist = OmegaConf.load("src/configs/datasets/mnist_on_the_fly.yaml")
    assert mirflickr.train.splits_path == "manifests/mirflickr25k_splits.json"
    assert celeba.train.splits_path == "manifests/celeba_splits.json"
    assert mnist.train.splits_path == "manifests/mnist_splits.json"
    assert not mnist.train.download
