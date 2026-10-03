"""Train one fixed MNIST classifier for reconstruction evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class IndexedMNIST(Dataset):
    def __init__(self, dataset: MNIST, scene_ids: list[str]) -> None:
        self.dataset = dataset
        self.indices = [int(scene_id.rsplit("_", 1)[1]) for scene_id in scene_ids]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        source = self.indices[index]
        image = self.dataset.data[source].float().div(255).unsqueeze(0)
        image = F.pad(image, (2, 2, 2, 2))
        return image, int(self.dataset.targets[source])


class FixedMNISTClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image))


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    correct = 0
    count = 0
    class_correct = [0] * 10
    class_count = [0] * 10
    with torch.no_grad():
        for image, label in loader:
            prediction = model(image.to(device)).argmax(1).cpu()
            match = prediction.eq(label)
            correct += int(match.sum())
            count += len(label)
            for digit in range(10):
                selected = label == digit
                class_correct[digit] += int(match[selected].sum())
                class_count[digit] += int(selected.sum())
    return {
        "accuracy": correct / count,
        "sample_count": count,
        "per_digit_accuracy": {
            str(digit): class_correct[digit] / class_count[digit]
            for digit in range(10)
        },
        "per_digit_count": {str(digit): class_count[digit] for digit in range(10)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    split_path = root / manifest["split_manifest"]
    if sha256(split_path) != manifest["split_manifest_sha256"]:
        raise ValueError("MNIST split manifest drift")
    if sha256(root / manifest["program"]) != manifest["program_sha256"]:
        raise ValueError("Classifier program drift")
    if manifest["data_partitions"] != ["train", "development"]:
        raise ValueError("Classifier must not use the official test partition")
    output = root / manifest["output"]
    if output.exists():
        raise FileExistsError(output)
    if not args.execute:
        print(json.dumps({"status": "ready", "output": str(output)}, indent=2))
        return
    output.mkdir(parents=True)
    save_json(output / "manifest_snapshot.json", manifest)
    seed = manifest["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.loads(split_path.read_text())
    source = MNIST(root / manifest["mnist_root"], train=True, download=False)
    train = IndexedMNIST(source, split["splits"]["train"])
    development = IndexedMNIST(source, split["splits"]["validation"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train,
        batch_size=manifest["batch_size"],
        shuffle=True,
        num_workers=manifest["num_workers"],
        generator=generator,
        persistent_workers=manifest["num_workers"] > 0,
    )
    development_loader = DataLoader(
        development,
        batch_size=manifest["batch_size"],
        shuffle=False,
        num_workers=manifest["num_workers"],
        persistent_workers=manifest["num_workers"] > 0,
    )
    model = FixedMNISTClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=manifest["learning_rate"])
    started = time.monotonic()
    rows = []
    for epoch in range(1, manifest["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(image), label)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite classifier loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(label)
            count += len(label)
        development_metrics = evaluate(model, development_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "development_accuracy": development_metrics["accuracy"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    checkpoint = output / "classifier_final.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": manifest["epochs"],
            "seed": seed,
            "architecture": "FixedMNISTClassifier",
            "split_manifest_sha256": manifest["split_manifest_sha256"],
        },
        checkpoint,
    )
    final_metrics = evaluate(model, development_loader, device)
    result = {
        "status": "complete",
        "endpoint_selection": "fixed final epoch",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "training_curve": rows,
        "development": final_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "device": device,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "official_test_accessed": False,
    }
    save_json(output / "result.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
