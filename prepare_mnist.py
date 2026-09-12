import json
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from torchvision.datasets import MNIST

from src.datasets.mnist import mnist_scene_id, stratified_mnist_split_indices
from src.datasets.split_manifest import split_manifest_summary, write_split_manifest


def _load_partition(root: Path, *, train: bool, download: bool):
    try:
        return MNIST(root=root, train=train, download=download)
    except RuntimeError as error:
        partition = "train" if train else "test"
        raise FileNotFoundError(
            f"MNIST {partition} partition is unavailable below {root}; "
            "run prepare_mnist.py or re-run it with download=true"
        ) from error


def _verify_partition(dataset, *, expected_count: int, name: str) -> int:
    if len(dataset) != expected_count:
        raise ValueError(
            f"Expected {expected_count} MNIST {name} samples, found {len(dataset)}"
        )
    if tuple(dataset.data.shape) != (expected_count, 28, 28):
        raise ValueError(
            f"Unexpected MNIST {name} image tensor shape: {tuple(dataset.data.shape)}"
        )
    if tuple(dataset.targets.shape) != (expected_count,):
        raise ValueError(
            f"Unexpected MNIST {name} label tensor shape: "
            f"{tuple(dataset.targets.shape)}"
        )
    return expected_count


def prepare(config) -> dict:
    root = Path(to_absolute_path(config.root_dir))
    root.mkdir(parents=True, exist_ok=True)
    datasets = {
        "train": _load_partition(root, train=True, download=bool(config.download)),
        "test": _load_partition(root, train=False, download=bool(config.download)),
    }

    verified_samples = sum(
        [
            _verify_partition(datasets["train"], expected_count=60_000, name="train"),
            _verify_partition(datasets["test"], expected_count=10_000, name="test"),
        ]
    )

    manifest_seed = int(config.manifest.seed)
    train_indices, validation_indices = stratified_mnist_split_indices(
        datasets["train"].targets,
        split_seed=manifest_seed,
        validation_per_class=int(config.manifest.validation_per_class),
    )
    manifest_payload = {
        "seed": manifest_seed,
        "splits": {
            "train": [mnist_scene_id("train", index) for index in train_indices],
            "validation": [
                mnist_scene_id("train", index) for index in validation_indices
            ],
            "test": [
                mnist_scene_id("test", index) for index in range(len(datasets["test"]))
            ],
        },
    }
    manifest_output = Path(to_absolute_path(config.manifest.output_path))
    manifest_write = write_split_manifest(
        manifest_payload,
        manifest_output,
        rewrite=bool(config.manifest.rewrite),
    )
    manifest_summary = split_manifest_summary(
        dataset="MNIST",
        root=root,
        output=manifest_output,
        payload=manifest_payload,
        write_result=manifest_write,
    )

    return {
        "status": "pass",
        "dataset": "MNIST",
        "root": str(root.resolve()),
        "verified_samples": verified_samples,
        "partition_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "manifest": manifest_summary,
    }


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="prepare_mnist",
)
def main(config) -> None:
    print(json.dumps(prepare(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
