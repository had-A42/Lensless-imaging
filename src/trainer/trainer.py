import math

import torch
from torchvision.utils import make_grid

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        paired_batch = "measurement_a" in batch and "measurement_b" in batch
        metric_funcs = [] if paired_batch else self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        with self.autocast_context():
            outputs = self.model(**batch)
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        loss = batch["loss"]
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise TypeError("loss must be a scalar torch.Tensor")
        if not torch.isfinite(loss.detach()).item():
            stage = "training" if self.is_train else "evaluation"
            raise FloatingPointError(
                f"Non-finite total loss during {stage} "
                f"at global_step={getattr(self, 'global_step', 0)}, "
                f"sampler_step={getattr(self, 'sampler_step', 0)}: "
                f"{loss.detach().item()}"
            )

        if self.is_train:
            grad_scaler = getattr(self, "grad_scaler", None)
            if grad_scaler is None:
                loss.backward()  # sum of all losses is always called loss
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(self.optimizer)
            self._clip_grad_norm()
            grad_norm = self._get_grad_norm()
            if not math.isfinite(grad_norm):
                raise FloatingPointError(
                    "Non-finite gradient norm after backward and before "
                    f"optimizer.step at global_step={getattr(self, 'global_step', 0)}, "
                    f"sampler_step={getattr(self, 'sampler_step', 0)}: {grad_norm}"
                )
            batch["grad_norm"] = grad_norm
            if grad_scaler is None:
                self.optimizer.step()
            else:
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        batch_size = batch[self.device_tensor_names(batch)[0]].shape[0]

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item(), n=batch_size)

        metric_values = {}
        for met in metric_funcs:
            value = met(**batch)
            metrics.update(met.name, value, n=batch_size)
            metric_values[met.name] = float(value.detach().cpu())
        batch["metric_values"] = metric_values
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example
        # the method is called only every self.log_step steps
        if not self.config.writer.get("log_images", False):
            return
        if mode == "train":
            image_log_step = self.config.writer.get("image_log_step")
            if image_log_step is not None and batch_idx % int(image_log_step) != 0:
                return

        if any(key not in batch for key in ("measurement", "prediction", "target")):
            return

        # logging scheme might be different for different partitions
        # in train mode the method is called only every self.log_step steps
        count = min(
            int(self.config.writer.get("max_images", 4)),
            batch["measurement"].shape[0],
        )
        measurements = []
        reconstructions = []
        for index in range(count):
            measurements.append(
                batch["measurement"][index].detach().float().cpu().clamp(0, 1)
            )
            reconstructions.extend(
                [
                    batch["prediction"][index].detach().float().cpu().clamp(0, 1),
                    batch["target"][index].detach().float().cpu().clamp(0, 1),
                ]
            )

        self.writer.add_image(
            "measurement",
            make_grid(measurements, nrow=count, padding=2),
        )

        self.writer.add_image(
            "prediction_target",
            make_grid(reconstructions, nrow=2, padding=2),
        )
