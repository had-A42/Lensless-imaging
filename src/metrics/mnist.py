import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.metrics.base_metric import BaseMetric


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FixedMNISTClassifier(nn.Module):
    def __init__(self):
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

    def forward(self, image):
        return self.head(self.features(image))


class FixedMNISTClassifierAccuracyMetric(BaseMetric):
    def __init__(
        self,
        checkpoint_path,
        expected_sha256=None,
        pooling_factor=8,
        use_target=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        if expected_sha256 is not None:
            actual = _sha256(self.checkpoint_path)
            if actual != expected_sha256:
                raise ValueError("MNIST classifier checkpoint hash mismatch")
        self.pooling_factor = int(pooling_factor)
        self.use_target = bool(use_target)
        if self.pooling_factor <= 0:
            raise ValueError("pooling_factor must be positive")
        state = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.classifier = FixedMNISTClassifier().eval()
        self.classifier.load_state_dict(state["state_dict"], strict=True)

    def per_image(self, prediction, target, label, **batch):
        image = target if self.use_target else prediction
        if (
            image.shape[-2] % self.pooling_factor
            or image.shape[-1] % self.pooling_factor
        ):
            raise ValueError(
                "MNIST classifier input size must be divisible by pooling_factor"
            )
        image = F.avg_pool2d(image.float(), self.pooling_factor)
        image = image.mean(dim=1, keepdim=True)
        self.classifier = self.classifier.to(image.device)
        logits = self.classifier(image)
        return logits.argmax(dim=1).eq(label.to(image.device)).float().detach()

    def __call__(self, prediction, target, label, **batch):
        return self.per_image(prediction, target, label, **batch).mean()
