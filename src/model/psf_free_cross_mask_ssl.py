import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PSFFreeCrossMaskSSL(nn.Module):
    def __init__(
        self,
        reconstructor: nn.Module,
        feature_channels: int = 32,
        projector_hidden_channels: int = 256,
        embedding_channels: int = 64,
        spatial_cells: int = 4,
        projector_seed: int = 52,
        output_crop: list[int] | None = None,
    ) -> None:
        super().__init__()
        dimensions = (
            feature_channels,
            projector_hidden_channels,
            embedding_channels,
            spatial_cells,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("feature and projector dimensions must be positive")

        self.reconstructor = reconstructor
        if output_crop is not None and tuple(output_crop) != tuple(
            getattr(self.reconstructor, "output_crop", ())
        ):
            raise ValueError("wrapper and reconstructor output crops must match")
        self.feature_channels = int(feature_channels)
        self.spatial_cells = int(spatial_cells)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(projector_seed))
            self.projector = nn.Sequential(
                nn.Linear(self.feature_channels, int(projector_hidden_channels)),
                nn.ReLU(),
                nn.Linear(int(projector_hidden_channels), int(embedding_channels)),
            )

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _reconstruct_view(self, measurement: Tensor) -> tuple[Tensor, Tensor]:
        output = self.reconstructor(measurement=measurement, return_features=True)
        prediction = output["prediction"]
        features = output["features"]
        if features.shape[1] != self.feature_channels:
            raise ValueError(
                f"reconstructor features must have {self.feature_channels} channels"
            )
        pooled = F.adaptive_avg_pool2d(
            features,
            output_size=(self.spatial_cells, self.spatial_cells),
        )
        cells = pooled.movedim(1, -1).flatten(1, 2)
        return prediction, self.projector(cells)

    def forward(
        self,
        measurement: Tensor | None = None,
        measurement_a: Tensor | None = None,
        measurement_b: Tensor | None = None,
        **batch: Tensor,
    ) -> dict[str, Tensor]:
        del batch
        if measurement is not None:
            if measurement_a is not None or measurement_b is not None:
                raise ValueError("pass either one measurement or a paired batch")
            return self.reconstructor(measurement=measurement)
        if measurement_a is None or measurement_b is None:
            raise ValueError("paired training needs measurement_a and measurement_b")

        prediction_a, embedding_a = self._reconstruct_view(measurement_a)
        prediction_b, embedding_b = self._reconstruct_view(measurement_b)
        return {
            "prediction_a": prediction_a,
            "prediction_b": prediction_b,
            "embedding_a": embedding_a,
            "embedding_b": embedding_b,
        }


__all__ = ["PSFFreeCrossMaskSSL"]
