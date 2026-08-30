import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.loss.reconstruction import ReconstructionLoss


class CrossMaskJEPALoss(nn.Module):
    component_names = (
        "jepa_ab_loss",
        "jepa_ba_loss",
        "variance_loss",
        "covariance_loss",
        "online_embedding_std",
        "prediction_cosine",
        "target_embedding_std",
        "target_collapsed_fraction",
        "target_effective_rank",
        "scene_retrieval_accuracy",
        "scene_alignment_margin",
    )

    def __init__(
        self,
        collapse_threshold: float = 0.05,
        mse_weight: float = 1.0,
        ssim_weight: float = 0.0,
        lpips_weight: float = 1.0,
        lpips_net: str = "vgg",
        normalize_by_max: bool = True,
        variance_weight: float = 1.0,
        covariance_weight: float = 0.01,
        variance_epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        self.collapse_threshold = float(collapse_threshold)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.variance_epsilon = float(variance_epsilon)
        self.validation_loss = ReconstructionLoss(
            mse_weight=mse_weight,
            ssim_weight=ssim_weight,
            lpips_weight=lpips_weight,
            lpips_net=lpips_net,
            normalize_by_max=normalize_by_max,
        )

    def _representation_regularization(
        self,
        online_a: Tensor,
        online_b: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.autocast(device_type=online_a.device.type, enabled=False):
            online_a = online_a.float()
            online_b = online_b.float()
            batch_size, cell_count, embedding_dim = online_a.shape
            std_a = torch.sqrt(
                online_a.var(dim=0, correction=1) + self.variance_epsilon
            )
            std_b = torch.sqrt(
                online_b.var(dim=0, correction=1) + self.variance_epsilon
            )
            variance = 0.5 * (
                torch.relu(1 - std_a).mean() + torch.relu(1 - std_b).mean()
            )

            centered_a = online_a - online_a.mean(dim=0, keepdim=True)
            centered_b = online_b - online_b.mean(dim=0, keepdim=True)
            count = max(batch_size - 1, 1)
            covariance_a = torch.einsum("bcd,bce->cde", centered_a, centered_a) / count
            covariance_b = torch.einsum("bcd,bce->cde", centered_b, centered_b) / count
            covariance_a = covariance_a - torch.diag_embed(
                torch.diagonal(covariance_a, dim1=-2, dim2=-1)
            )
            covariance_b = covariance_b - torch.diag_embed(
                torch.diagonal(covariance_b, dim1=-2, dim2=-1)
            )
            covariance = (
                0.5
                * (covariance_a.square().sum() + covariance_b.square().sum())
                / (cell_count * embedding_dim)
            )
            embedding_std = 0.5 * (std_a.mean() + std_b.mean())
            return variance, covariance, embedding_std

    @staticmethod
    def _masked_cosine(
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction = F.normalize(prediction.float(), dim=-1)
            target = F.normalize(target.detach().float(), dim=-1)
            distance = 2 - 2 * (prediction * target).sum(dim=-1)
            weights = mask.float()
            return (distance * weights).sum() / weights.sum().clamp_min(1)

    def _diagnostics(
        self,
        online_a: Tensor,
        online_b: Tensor,
        target_a: Tensor,
        target_b: Tensor,
        loss: Tensor,
    ) -> dict[str, Tensor]:
        with torch.autocast(device_type=target_a.device.type, enabled=False):
            return self._diagnostics_float(
                online_a,
                online_b,
                target_a,
                target_b,
                loss,
            )

    def _diagnostics_float(
        self,
        online_a: Tensor,
        online_b: Tensor,
        target_a: Tensor,
        target_b: Tensor,
        loss: Tensor,
    ) -> dict[str, Tensor]:
        targets = F.normalize(
            torch.cat((target_a.float(), target_b.float()), dim=0),
            dim=-1,
        )
        target_std = targets.std(dim=0, correction=1)
        collapsed = (target_std < self.collapse_threshold).float().mean()

        scene_targets = 0.5 * (
            F.normalize(target_a.float(), dim=-1)
            + F.normalize(target_b.float(), dim=-1)
        )
        scene_targets = scene_targets.flatten(1)
        centered = scene_targets - scene_targets.mean(dim=0, keepdim=True)
        gram = centered @ centered.T / max(scene_targets.shape[0] - 1, 1)
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
        total_variance = eigenvalues.sum()
        probabilities = eigenvalues / total_variance.clamp_min(1e-12)
        nonzero_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        )
        effective_rank = torch.where(total_variance > 0, nonzero_rank, total_variance)

        online_a = F.normalize(online_a.float().flatten(1), dim=1)
        online_b = F.normalize(online_b.float().flatten(1), dim=1)
        target_a = F.normalize(target_a.float().flatten(1), dim=1)
        target_b = F.normalize(target_b.float().flatten(1), dim=1)
        similarities_ab = online_a @ target_b.T
        similarities_ba = online_b @ target_a.T
        expected = torch.arange(online_a.shape[0], device=online_a.device)
        retrieval = 0.5 * (
            (similarities_ab.argmax(dim=1) == expected).float().mean()
            + (similarities_ba.argmax(dim=1) == expected).float().mean()
        )
        if online_a.shape[0] > 1:
            diagonal = 0.5 * (similarities_ab.diagonal() + similarities_ba.diagonal())
            eye = torch.eye(
                online_a.shape[0],
                device=online_a.device,
                dtype=torch.bool,
            )
            negative = 0.5 * (
                similarities_ab.masked_fill(eye, float("-inf")).amax(dim=1)
                + similarities_ba.masked_fill(eye, float("-inf")).amax(dim=1)
            )
            margin = (diagonal - negative).mean()
        else:
            margin = loss.new_zeros(())

        return {
            "prediction_cosine": (1 - 0.5 * loss).detach(),
            "target_embedding_std": target_std.mean().detach(),
            "target_collapsed_fraction": collapsed.detach(),
            "target_effective_rank": effective_rank.detach(),
            "scene_retrieval_accuracy": retrieval.detach(),
            "scene_alignment_margin": margin.detach(),
        }

    def forward(
        self,
        prediction: Tensor | None = None,
        target: Tensor | None = None,
        predicted_embedding_a: Tensor | None = None,
        predicted_embedding_b: Tensor | None = None,
        online_embedding_a: Tensor | None = None,
        online_embedding_b: Tensor | None = None,
        target_embedding_a: Tensor | None = None,
        target_embedding_b: Tensor | None = None,
        target_mask_a: Tensor | None = None,
        target_mask_b: Tensor | None = None,
        **batch,
    ) -> dict[str, Tensor]:
        del batch
        if prediction is not None or target is not None:
            if prediction is None or target is None:
                raise ValueError("validation needs prediction and target")
            result = self.validation_loss(prediction=prediction, target=target)
            zero = result["loss"].new_zeros(())
            result.update({name: zero for name in self.component_names})
            return result

        values = (
            predicted_embedding_a,
            predicted_embedding_b,
            online_embedding_a,
            online_embedding_b,
            target_embedding_a,
            target_embedding_b,
            target_mask_a,
            target_mask_b,
        )
        if any(value is None for value in values):
            raise ValueError("JEPA training needs both online and target views")

        loss_ba = self._masked_cosine(
            predicted_embedding_a,
            target_embedding_a,
            target_mask_a,
        )
        loss_ab = self._masked_cosine(
            predicted_embedding_b,
            target_embedding_b,
            target_mask_b,
        )
        variance, covariance, embedding_std = self._representation_regularization(
            online_embedding_a,
            online_embedding_b,
        )
        jepa_loss = 0.5 * (loss_ab + loss_ba)
        loss = (
            jepa_loss
            + self.variance_weight * variance
            + self.covariance_weight * covariance
        )
        return {
            "loss": loss,
            "jepa_ab_loss": loss_ab,
            "jepa_ba_loss": loss_ba,
            "variance_loss": variance,
            "covariance_loss": covariance,
            "online_embedding_std": embedding_std.detach(),
            **self._diagnostics(
                online_embedding_a,
                online_embedding_b,
                target_embedding_a,
                target_embedding_b,
                jepa_loss,
            ),
            "mse_loss": loss.new_zeros(()),
            "lpips_loss": loss.new_zeros(()),
        }


__all__ = ["CrossMaskJEPALoss"]
