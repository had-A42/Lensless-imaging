from torch.nn import functional as F

from src.metrics.base_metric import BaseMetric


class BlockResidualRMSE(BaseMetric):

    def __init__(self, block_size=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = int(block_size)

    def per_image(self, prediction, target, **batch):
        error = prediction.float() - target.float()
        block = self.block_size
        if block < 1 or error.shape[-2] % block or error.shape[-1] % block:
            raise ValueError(
                "Image dimensions must be divisible by a positive block_size"
            )
        coarse = F.avg_pool2d(error, block)
        detail = error - coarse.repeat_interleave(block, -2).repeat_interleave(
            block, -1
        )
        return detail.square().flatten(1).mean(1).sqrt().detach()

    def __call__(self, prediction, target, **batch):
        return self.per_image(prediction, target, **batch).mean()


class PeriodicResidualRMSE(BaseMetric):

    def __init__(self, period=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = int(period)

    def per_image(self, prediction, target, **batch):
        error = prediction.float() - target.float()
        n, c, h, w = error.shape
        period = self.period
        if period < 1 or h % period or w % period:
            raise ValueError("Image dimensions must be divisible by a positive period")
        phases = error.reshape(n, c, h // period, period, w // period, period)
        phases = phases.mean(dim=(2, 4))
        phases = phases - phases.mean(dim=(-2, -1), keepdim=True)
        return phases.square().flatten(1).mean(1).sqrt().detach()

    def __call__(self, prediction, target, **batch):
        return self.per_image(prediction, target, **batch).mean()
