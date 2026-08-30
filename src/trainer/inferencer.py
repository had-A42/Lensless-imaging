import json
import time

import pandas as pd
import torch
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
        logger=None,
        writer=None,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        self.logger = logger
        self.writer = writer
        self.output_type = self.config.inferencer.get("output_type", "classification")
        self.per_image_rows = {}
        self.saved_example_labels = {}
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def _from_pretrained(self, pretrained_path):
        super()._from_pretrained(pretrained_path)

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part, output_offset=0):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        if getattr(self, "output_type", "classification") == "reconstruction":
            return self._process_reconstruction_batch(
                batch=batch,
                metrics=metrics,
                part=part,
                output_offset=output_offset,
            )

        if metrics is not None:
            batch_size = batch[self.cfg_trainer.device_tensors[0]].shape[0]
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch), n=batch_size)

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk

        batch_size = batch["logits"].shape[0]
        for i in range(batch_size):
            # clone because of
            # https://github.com/pytorch/pytorch/issues/1995
            logits = batch["logits"][i].clone()
            label = batch["labels"][i].clone()
            pred_label = logits.argmax(dim=-1)

            output_id = output_offset + i

            output = {
                "pred_label": pred_label,
                "label": label,
            }

            if self.save_path is not None:
                # you can use safetensors or other lib here
                torch.save(output, self.save_path / part / f"output_{output_id}.pth")

        return batch

    @staticmethod
    def _item(values, index):
        if isinstance(values, torch.Tensor):
            value = values[index]
            return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
        return values[index]

    def _process_reconstruction_batch(self, batch, metrics, part, output_offset):
        batch_size = batch["prediction"].shape[0]
        metric_values = {}
        if metrics is not None:
            for metric in self.metrics["inference"]:
                if hasattr(metric, "per_image"):
                    values = metric.per_image(**batch)
                else:
                    values = metric(**batch).repeat(batch_size)
                values = values.detach().cpu().flatten()
                if values.numel() != batch_size:
                    raise ValueError(
                        f"{metric.name} returned {values.numel()} values for "
                        f"a batch of {batch_size}"
                    )
                metrics.update(metric.name, values.mean(), n=batch_size)
                metric_values[metric.name] = values

        rows = self.per_image_rows.setdefault(part, [])
        for index in range(batch_size):
            row = {"sample_index": output_offset + index}
            for key in (
                "sample_id",
                "scene_id",
                "mask_id",
                "split",
                "label",
            ):
                if key in batch:
                    row[key] = self._item(batch[key], index)
            for name, values in metric_values.items():
                row[name] = float(values[index])
            rows.append(row)

        self._save_reconstruction_examples(part, batch, output_offset)
        return batch

    def _save_reconstruction_examples(self, part, batch, output_offset):
        example_indices = set(self.config.inferencer.get("example_indices", []))
        example_labels = set(self.config.inferencer.get("example_labels", []))
        saved_labels = self.saved_example_labels.setdefault(part, set())
        if not example_indices and not example_labels:
            return

        for index in range(batch["prediction"].shape[0]):
            output_id = output_offset + index
            label = self._item(batch["label"], index) if "label" in batch else None
            save_by_label = label in example_labels and label not in saved_labels
            names = []
            if output_id in example_indices:
                names.append(f"example_{output_id:04d}")
            if save_by_label:
                names.append(f"label_{label}_example_{output_id:04d}")
                saved_labels.add(label)
            if not names:
                continue
            output = {
                "measurement": batch["measurement"][index].detach().cpu(),
                "prediction": batch["prediction"][index].detach().cpu(),
                "target": batch["target"][index].detach().cpu(),
            }
            for key in (
                "sample_id",
                "scene_id",
                "mask_id",
                "split",
                "label",
            ):
                if key in batch:
                    output[key] = self._item(batch[key], index)
            for name in names:
                torch.save(output, self.save_path / part / f"{name}.pth")

                if self.writer is not None and self.config.writer.get(
                    "log_images", False
                ):
                    self.writer.set_step(0, part)
                    self.writer.add_image(
                        f"{name}_measurement",
                        output["measurement"].clamp(0, 1),
                    )
                    self.writer.add_image(
                        f"{name}_prediction_target",
                        make_grid(
                            [
                                output["prediction"].clamp(0, 1),
                                output["target"].clamp(0, 1),
                            ],
                            nrow=2,
                        ),
                    )

    def _save_reconstruction_results(self, part, elapsed_seconds):
        rows = self.per_image_rows.get(part, [])
        if not rows:
            return {}

        per_image = pd.DataFrame(rows)
        metric_names = [metric.name for metric in self.metrics["inference"]]
        per_mask = per_image.groupby("mask_id", sort=True).agg(
            sample_count=("sample_index", "count"),
            **{
                f"{name}_{stat}": (name, stat)
                for name in metric_names
                for stat in ("mean", "std")
            },
        )
        per_mask = per_mask.reset_index()

        sample_weighted = {name: float(per_image[name].mean()) for name in metric_names}
        mask_balanced = {
            name: float(per_mask[f"{name}_mean"].mean()) for name in metric_names
        }
        mask_std = {
            name: (float(per_mask[f"{name}_mean"].std()) if len(per_mask) > 1 else None)
            for name in metric_names
        }
        per_class = None
        class_macro = None
        if "label" in per_image:
            per_class = per_image.groupby("label", sort=True).agg(
                sample_count=("sample_index", "count"),
                **{
                    f"{name}_{stat}": (name, stat)
                    for name in metric_names
                    for stat in ("mean", "std")
                },
            )
            per_class = per_class.reset_index()
            class_macro = {
                name: float(per_class[f"{name}_mean"].mean()) for name in metric_names
            }
        model_load_seconds = float(getattr(self.model, "load_seconds", 0.0))
        inference_seconds = max(elapsed_seconds - model_load_seconds, 1e-12)
        summary = {
            "sample_count": len(per_image),
            "mask_count": len(per_mask),
            "samples_per_mask": sorted(per_mask["sample_count"].unique().tolist()),
            "sample_weighted": sample_weighted,
            "mask_balanced": mask_balanced,
            "mask_std": mask_std,
            "end_to_end_seconds": elapsed_seconds,
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
            "samples_per_second": len(per_image) / inference_seconds,
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated()
                if str(self.device).startswith("cuda")
                else 0
            ),
        }
        if per_class is not None:
            summary.update(
                {
                    "class_count": len(per_class),
                    "samples_per_class": sorted(
                        per_class["sample_count"].unique().tolist()
                    ),
                    "class_macro": class_macro,
                }
            )
        if self.config.get("provenance") is not None:
            summary["provenance"] = OmegaConf.to_container(
                self.config.provenance,
                resolve=True,
            )
            self._check_expected_counts(summary)

        output_dir = self.save_path / part
        per_image.to_csv(output_dir / "per_image.csv", index=False)
        per_mask.to_csv(output_dir / "per_mask.csv", index=False)
        if per_class is not None:
            per_class.to_csv(output_dir / "per_class.csv", index=False)
        with (output_dir / "summary.json").open("w") as file:
            json.dump(summary, file, indent=2)

        if self.writer is not None:
            self.writer.set_step(0, part)
            self.writer.add_scalars(sample_weighted)
            self.writer.add_scalars(
                {
                    f"{name}_mask_balanced": value
                    for name, value in mask_balanced.items()
                }
            )
            if class_macro is not None:
                self.writer.add_scalars(
                    {
                        f"{name}_class_macro": value
                        for name, value in class_macro.items()
                    }
                )
            self.writer.add_scalar("end_to_end_seconds", elapsed_seconds)
            self.writer.add_scalar("model_load_seconds", model_load_seconds)
            self.writer.add_scalar("inference_seconds", inference_seconds)
            self.writer.add_scalar("samples_per_second", summary["samples_per_second"])
            self.writer.add_scalar("peak_vram_bytes", summary["peak_vram_bytes"])
            self.writer.add_table("per_image", per_image)
            self.writer.add_table("per_mask", per_mask)
            if per_class is not None:
                self.writer.add_table("per_class", per_class)

        return summary

    def _check_expected_counts(self, summary):
        expected = self.config.provenance
        checks = {
            "expected_samples": summary["sample_count"],
            "expected_masks": summary["mask_count"],
        }
        for name, actual in checks.items():
            value = expected.get(name)
            if value is not None and int(value) != actual:
                raise ValueError(f"{name} is {value}, but inference produced {actual}")

        scenes_per_mask = expected.get("expected_scenes_per_mask")
        if scenes_per_mask is not None and summary["samples_per_mask"] != [
            int(scenes_per_mask)
        ]:
            raise ValueError(
                "expected_scenes_per_mask is "
                f"{scenes_per_mask}, but inference produced "
                f"{summary['samples_per_mask']}"
            )

        class_checks = {
            "expected_classes": summary.get("class_count"),
            "expected_samples_per_class": summary.get("samples_per_class"),
        }
        for name, actual in class_checks.items():
            value = expected.get(name)
            if value is None:
                continue
            expected_value = [int(value)] if name.endswith("per_class") else int(value)
            if actual != expected_value:
                raise ValueError(f"{name} is {value}, but inference produced {actual}")

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            if str(self.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            start_time = time.perf_counter()
            output_offset = 0
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                    output_offset=output_offset,
                )
                output_offset += batch[self.cfg_trainer.device_tensors[0]].shape[0]

        if getattr(self, "output_type", "classification") == "reconstruction":
            return self._save_reconstruction_results(
                part,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        if self.evaluation_metrics is None:
            return {}
        return self.evaluation_metrics.result()
