from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PSFFreeCrossMaskJEPA(nn.Module):
    """Predict scene features across two lensless measurements."""

    def __init__(
        self,
        reconstructor: nn.Module,
        feature_channels: int = 32,
        embedding_channels: int = 64,
        predictor_hidden_channels: int = 256,
        predictor_heads: int = 4,
        predictor_layers: int = 2,
        spatial_cells: int = 4,
        target_fraction: float = 0.5,
        momentum: float = 0.996,
        projector_seed: int = 52,
        output_crop: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.reconstructor = reconstructor
        if output_crop is not None and tuple(output_crop) != tuple(
            getattr(self.reconstructor, "output_crop", ())
        ):
            raise ValueError("wrapper and reconstructor output crops must match")
        self.feature_channels = int(feature_channels)
        self.embedding_channels = int(embedding_channels)
        self.spatial_cells = int(spatial_cells)
        self.target_fraction = float(target_fraction)
        self.momentum = float(momentum)

        if self.feature_channels <= 0 or self.embedding_channels <= 0:
            raise ValueError("feature dimensions must be positive")
        if self.embedding_channels % int(predictor_heads):
            raise ValueError("embedding_channels must be divisible by predictor_heads")
        if self.spatial_cells < 2:
            raise ValueError("spatial_cells must be at least two")
        if not 0 < self.target_fraction < 1:
            raise ValueError("target_fraction must be between zero and one")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(projector_seed))
            self.projector = nn.Sequential(
                nn.Linear(
                    self.feature_channels,
                    self.embedding_channels,
                    bias=False,
                ),
                nn.LayerNorm(self.embedding_channels),
            )
            predictor_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_channels,
                nhead=int(predictor_heads),
                dim_feedforward=int(predictor_hidden_channels),
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.predictor = nn.TransformerEncoder(
                predictor_layer,
                num_layers=int(predictor_layers),
                enable_nested_tensor=False,
            )
            self.prediction_head = nn.Linear(
                self.embedding_channels,
                self.embedding_channels,
            )
            cell_count = self.spatial_cells**2
            self.position_embedding = nn.Parameter(
                torch.empty(1, cell_count, self.embedding_channels)
            )
            nn.init.normal_(self.position_embedding, std=0.02)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embedding_channels))

        self.target_reconstructor = deepcopy(self.reconstructor)
        self.target_projector = deepcopy(self.projector)
        self.target_reconstructor.requires_grad_(False)
        self.target_projector.requires_grad_(False)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_reconstructor.eval()
        self.target_projector.eval()
        return self

    @torch.no_grad()
    def _update_target(self) -> None:
        for online, target in (
            (self.reconstructor, self.target_reconstructor),
            (self.projector, self.target_projector),
        ):
            for online_parameter, target_parameter in zip(
                online.parameters(), target.parameters()
            ):
                target_parameter.lerp_(online_parameter, 1 - self.momentum)

    def _features(self, model: nn.Module, measurement: Tensor) -> Tensor:
        features = model(measurement=measurement, return_features=True)["features"]
        if features.shape[1] != self.feature_channels:
            raise ValueError(
                f"reconstructor features must have {self.feature_channels} channels"
            )
        features = F.adaptive_avg_pool2d(
            features,
            output_size=(self.spatial_cells, self.spatial_cells),
        )
        return features.flatten(2).transpose(1, 2)

    def _target_mask(self, tokens: Tensor) -> Tensor:
        batch_size, cell_count = tokens.shape[:2]
        target_count = max(1, round(cell_count * self.target_fraction))
        indices = (
            torch.rand(
                batch_size,
                cell_count,
                device=tokens.device,
            )
            .topk(target_count, dim=1)
            .indices
        )
        mask = torch.zeros(
            batch_size,
            cell_count,
            dtype=torch.bool,
            device=tokens.device,
        )
        return mask.scatter_(1, indices, True)

    def _predict(self, tokens: Tensor, target_mask: Tensor) -> Tensor:
        masked_tokens = torch.where(
            target_mask.unsqueeze(-1),
            self.mask_token.to(tokens),
            tokens,
        )
        predicted = self.predictor(masked_tokens + self.position_embedding.to(tokens))
        return self.prediction_head(predicted)

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

        if self.training:
            self._update_target()

        online_a = self.projector(self._features(self.reconstructor, measurement_a))
        online_b = self.projector(self._features(self.reconstructor, measurement_b))
        with torch.no_grad():
            target_a = self.target_projector(
                self._features(self.target_reconstructor, measurement_a)
            )
            target_b = self.target_projector(
                self._features(self.target_reconstructor, measurement_b)
            )

        target_mask_a = self._target_mask(online_a)
        target_mask_b = self._target_mask(online_b)
        return {
            "online_embedding_a": online_a,
            "online_embedding_b": online_b,
            "target_embedding_a": target_a,
            "target_embedding_b": target_b,
            "predicted_embedding_a": self._predict(online_b, target_mask_a),
            "predicted_embedding_b": self._predict(online_a, target_mask_b),
            "target_mask_a": target_mask_a,
            "target_mask_b": target_mask_b,
        }


__all__ = ["PSFFreeCrossMaskJEPA"]
