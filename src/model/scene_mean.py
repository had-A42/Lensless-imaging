import torch
from torch import nn
from tqdm import tqdm


class SceneMeanReconstructor(nn.Module):
    def __init__(self, scenes):
        super().__init__()
        mean = torch.zeros_like(scenes[0]["target"])
        for index in tqdm(range(len(scenes)), desc="Mean training image"):
            mean.add_(scenes[index]["target"])
        self.register_buffer("mean_image", mean.div_(len(scenes)))

    def forward(self, measurement, **batch):
        return {
            "prediction": self.mean_image.unsqueeze(0).expand(
                measurement.shape[0], -1, -1, -1
            )
        }
