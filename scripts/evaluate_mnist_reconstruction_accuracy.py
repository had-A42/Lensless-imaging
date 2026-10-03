"""Evaluate fixed-classifier accuracy of one MNIST reconstruction endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


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


def classifier_input(image: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(image, 8, 8).mean(dim=1, keepdim=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("identifier")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    sys.path.insert(0, str(root))
    entry = next(item for item in manifest["entries"] if item["id"] == args.identifier)
    for field, hash_field in (
        ("config", "config_sha256"),
        ("checkpoint", "checkpoint_sha256"),
    ):
        if sha256(entry[field]) != entry[hash_field]:
            raise ValueError(f"{field} hash drift")
    classifier_path = root / manifest["classifier_checkpoint"]
    if sha256(classifier_path) != manifest["classifier_checkpoint_sha256"]:
        raise ValueError("Classifier checkpoint drift")
    output = root / entry["output"]
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    config = OmegaConf.load(entry["config"])
    config.model.checkpoint_path = None
    config.dataloader_builder.evaluation_only = True
    config.dataloader_builder.batch_size = 4
    config.dataloader_builder.validation_scenes_per_mask = 100
    config.dataloader_builder.num_workers = 4
    config.dataloader_builder.persistent_workers = False
    config.dataloader_builder.psf_cache.mode = "read_only"
    aware = entry["information_regime"] == "psf_aware"
    config.dataloader_builder.return_psf = aware
    from torch.utils.data import Subset
    from src.metrics.reconstruction import PooledPSNRMetric, PooledSSIMMetric
    from src.utils.init_utils import set_random_seed

    set_random_seed(entry["seed"])
    validation_scenes = instantiate(config.datasets.validation)
    id_to_position = {
        f"mnist_train_{int(source_index):05d}": position
        for position, source_index in enumerate(validation_scenes.indices)
    }
    selected_positions = [
        id_to_position[scene_id] for scene_id in manifest["scene_ids"]
    ]
    selected_scenes = Subset(validation_scenes, selected_positions)
    loaders, transforms = instantiate(
        config.dataloader_builder,
        datasets_config=config.datasets,
        simulator_config=config.simulator,
        validation_scenes=selected_scenes,
        _recursive_=False,
    )
    if transforms:
        raise ValueError("Unexpected transforms")
    state = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    if state["global_step"] != 50000 or state["sampler_step"] != 50000:
        raise ValueError("Reconstructor endpoint mismatch")
    model = instantiate(config.model).cuda().eval()
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    classifier_state = torch.load(
        str(classifier_path), map_location="cpu", weights_only=False
    )
    classifier = FixedMNISTClassifier().cuda().eval()
    classifier.load_state_dict(classifier_state["state_dict"], strict=True)
    psnr = PooledPSNRMetric(name="PSNR_32", pooling_factor=8, normalize_by_max=False)
    ssim = PooledSSIMMetric(name="SSIM_32", pooling_factor=8, normalize_by_max=False)
    rows = []
    with torch.no_grad():
        for batch in loaders["validation"]:
            measurement = batch["measurement"].cuda()
            target = batch["target"].cuda()
            kwargs = {"measurement": measurement}
            if aware:
                kwargs["psf"] = batch["psf"].cuda()
            prediction = model(**kwargs)["prediction"].float()
            predicted_label = classifier(classifier_input(prediction)).argmax(1).cpu()
            target_label = classifier(classifier_input(target)).argmax(1).cpu()
            label = batch["label"].long()
            psnr_values = psnr.per_image(prediction=prediction, target=target).cpu()
            ssim_values = ssim.per_image(prediction=prediction, target=target).cpu()
            for index in range(len(label)):
                rows.append(
                    {
                        "sample_index": len(rows),
                        "mask_id": str(batch["mask_id"][index]),
                        "scene_id": str(batch["scene_id"][index]),
                        "digit": int(label[index]),
                        "predicted_digit": int(predicted_label[index]),
                        "target_classifier_digit": int(target_label[index]),
                        "correct": int(predicted_label[index] == label[index]),
                        "target_classifier_correct": int(target_label[index] == label[index]),
                        "PSNR_32": float(psnr_values[index]),
                        "SSIM_32": float(ssim_values[index]),
                    }
                )
    frame = pd.DataFrame(rows)
    if len(frame) != 3200 or frame["mask_id"].nunique() != 32:
        raise ValueError("Development grid mismatch")
    if not np.isfinite(frame[["PSNR_32", "SSIM_32"]].to_numpy()).all():
        raise ValueError("Non-finite metrics")
    frame.to_csv(output / "per_image.csv", index=False)
    per_digit = (
        frame.groupby("digit")
        .agg(accuracy=("correct", "mean"), sample_count=("correct", "size"))
        .reset_index()
    )
    if set(per_digit["sample_count"]) != {320}:
        raise ValueError("Per-digit evaluation is not balanced")
    per_digit.to_csv(output / "per_digit.csv", index=False)
    per_mask = (
        frame.groupby("mask_id")
        .agg(
            accuracy=("correct", "mean"),
            target_classifier_accuracy=("target_classifier_correct", "mean"),
            PSNR_32=("PSNR_32", "mean"),
            SSIM_32=("SSIM_32", "mean"),
            sample_count=("correct", "size"),
        )
        .reset_index()
    )
    per_mask.to_csv(output / "per_mask.csv", index=False)
    summary = {
        "status": "complete",
        "id": entry["id"],
        "seed": entry["seed"],
        "information_regime": entry["information_regime"],
        "accuracy": float(frame["correct"].mean()),
        "target_classifier_accuracy": float(frame["target_classifier_correct"].mean()),
        "per_digit_accuracy": {
            str(int(row.digit)): float(row.accuracy)
            for row in per_digit.itertuples()
        },
        "sample_count": len(frame),
        "elapsed_seconds": time.monotonic() - started,
        "data_partition": "development",
        "official_test_accessed": False,
    }
    save_json(output / "summary.json", summary)
    save_json(
        output / "validation.json",
        {
            "status": "pass",
            "sample_count": 3200,
            "mask_count": 32,
            "unique_scene_count": 100,
            "scenes_per_digit": 10,
            "reconstruction_pairs_per_digit": 320,
            "classifier_checkpoint_sha256": manifest["classifier_checkpoint_sha256"],
            "official_test_accessed": False,
            "final_test_accessed": False,
        },
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
