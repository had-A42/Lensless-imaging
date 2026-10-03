import torch
from torch import nn
from tqdm import tqdm


class SceneMeanReconstructor(nn.Module):
    def __init__(self, scenes, output_normalization=None):
        super().__init__()
        if output_normalization not in {None, "min_max"}:
            raise ValueError("output_normalization must be None or min_max")
        self.output_normalization = output_normalization
        mean = torch.zeros_like(scenes[0]["target"])
        for index in tqdm(range(len(scenes)), desc="Mean training image"):
            mean.add_(scenes[index]["target"])
        self.register_buffer("mean_image", mean.div_(len(scenes)))

    def forward(self, measurement, **batch):
        prediction = self.mean_image.unsqueeze(0).expand(
            measurement.shape[0], -1, -1, -1
        )
        if self.output_normalization == "min_max":
            prediction = prediction - prediction.amin(dim=(1, 2, 3), keepdim=True)
            prediction = prediction / prediction.amax(
                dim=(1, 2, 3), keepdim=True
            ).clamp_min(1e-12)
        return {
            "prediction": prediction
        }
