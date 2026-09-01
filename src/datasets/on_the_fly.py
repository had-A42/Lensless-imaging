from collections import OrderedDict
from functools import partial

import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from lensless.recon.rfft_convolve import RealFFTConvolve2D
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.digicam_synth.mask_protocol import (
    DEFAULT_MASK_SEED,
    get_mask_records,
    mask_seed,
)
from src.digicam_synth.pipeline import (
    forward_single_sample,
    generate_random_pattern,
    pattern_to_psf,
)
from src.digicam_synth.psf_cache import PSFCache

DEFAULT_VALIDATION_SEED = 52


def _validate_operator_prompt_mode(mode):
    mode = str(mode).lower()
    if mode not in {"none", "true", "shuffled"}:
        raise ValueError("operator_prompt_mode must be none, true or shuffled")
    return mode


def _prompt_mask_record(mask_records, mask_index, mode):
    if mode == "none":
        return None
    if mode == "true":
        return mask_records[mask_index]
    if len(mask_records) < 2:
        raise ValueError("shuffled operator prompts need at least two masks")
    return mask_records[(mask_index + 1) % len(mask_records)]


def build_digicam_mask(config, seed, create_simulator=True):
    rng = np.random.default_rng(seed)
    pattern = generate_random_pattern(config, rng)
    return pattern_to_psf(
        config,
        pattern,
        rng,
        create_simulator=create_simulator,
    )


def _sample_seed(run_seed, step, slot):
    state = np.random.SeedSequence([run_seed, step, slot, 91]).generate_state(1)
    return int(state[0])


def _to_chw(image, size):
    image = torch.as_tensor(np.asarray(image), dtype=torch.float32)
    if image.ndim == 2:
        image = image.unsqueeze(-1)
    if image.ndim != 3:
        raise ValueError("image must have HW, HWC or CHW shape")
    if image.shape[-1] in (1, 3):
        image = image.permute(2, 0, 1)
    elif image.shape[0] not in (1, 3):
        raise ValueError("image must have one or three channels")
    image = image.contiguous()
    if size is not None and tuple(image.shape[-2:]) != tuple(size):
        image = F.interpolate(
            image.unsqueeze(0),
            size=tuple(size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
    return image.clamp(0, 1)


def _aligned_measurement(target, psf, convolver, roi, quantize):
    psf = torch.as_tensor(np.asarray(psf), dtype=torch.float32)
    if psf.ndim != 4 or psf.shape[0] != 1:
        raise ValueError("psf must have DHWC shape with depth 1")

    top, left, height, width = (int(value) for value in roi)
    if target.shape[-2:] != (height, width):
        raise ValueError("target size must match the configured ROI")
    if (
        top < 0
        or left < 0
        or top + height > psf.shape[1]
        or left + width > psf.shape[2]
    ):
        raise ValueError("ROI exceeds the full sensor canvas")

    canvas = torch.zeros_like(psf)
    canvas[:, top : top + height, left : left + width] = target.movedim(0, -1)
    measurement = convolver.convolve(canvas).clamp_min(0)
    measurement = measurement / measurement.amax().clamp_min(1e-8)
    if quantize:
        measurement = torch.round(measurement * 255) / 255
    return measurement.squeeze(0).movedim(-1, 0).contiguous()


def _prepare_convolution_psf(psf):
    psf = torch.as_tensor(np.asarray(psf), dtype=torch.float32)
    if psf.ndim != 4 or psf.shape[0] != 1:
        raise ValueError("psf must have DHWC shape with depth 1")

    psf = torch.flip(psf, dims=(-3, -2))
    return psf / psf.norm().clamp_min(1e-12)


class DigiCamOnTheFlyDataset:
    def __init__(
        self,
        scenes,
        simulator_config,
        measurement_size=(64, 64),
        target_size=(64, 64),
        simulation_mode="far_field",
        roi=None,
        finite_cache_size=8,
        psf_cache=None,
        mask_factory=None,
    ):
        self.scenes = scenes
        self.simulator_config = simulator_config
        self.measurement_size = (
            tuple(int(value) for value in measurement_size)
            if measurement_size is not None
            else None
        )
        self.target_size = (
            tuple(int(value) for value in target_size)
            if target_size is not None
            else None
        )
        self.simulation_mode = str(simulation_mode)
        self.roi = tuple(int(value) for value in roi) if roi is not None else None
        self.finite_cache_size = int(finite_cache_size)
        self.mask_factory = mask_factory or partial(
            build_digicam_mask,
            create_simulator=self.simulation_mode == "far_field",
        )
        self.mask_cache = OrderedDict()
        self.current_mask_seed = None
        self.current_mask = None
        self.convolver_cache = OrderedDict()

        for name, size in (
            ("measurement_size", self.measurement_size),
            ("target_size", self.target_size),
        ):
            if size is not None and (len(size) != 2 or min(size) <= 0):
                raise ValueError(f"{name} must be null or [height, width]")
        if self.simulation_mode not in {"far_field", "roi_convolution"}:
            raise ValueError("simulation_mode must be far_field or roi_convolution")
        if self.simulation_mode == "roi_convolution":
            if self.roi is None or len(self.roi) != 4:
                raise ValueError("roi_convolution needs [top, left, height, width]")
            if self.target_size != self.roi[-2:]:
                raise ValueError("target_size must match ROI height and width")
        if self.finite_cache_size < 0:
            raise ValueError("finite_cache_size must be non-negative")
        self.psf_cache = PSFCache(psf_cache, self.simulator_config)
        if self.psf_cache.mode != "off" and self.simulation_mode != "roi_convolution":
            raise ValueError("PSF disk cache currently supports roi_convolution only")

    def __len__(self):
        return len(self.scenes)

    def _get_mask(self, request):
        seed = int(request["mask_seed"])
        if seed == self.current_mask_seed:
            return self.current_mask

        use_cache = request["mode"] == "finite" and self.finite_cache_size > 0
        if use_cache and seed in self.mask_cache:
            value = self.mask_cache.pop(seed)
            self.mask_cache[seed] = value
            self.current_mask_seed = seed
            self.current_mask = value
            return value

        disk_value = self.psf_cache.load(seed, request["mode"])
        if disk_value is None:
            value = self.mask_factory(self.simulator_config, seed)
            self.psf_cache.save(seed, request["mode"], value[0])
        else:
            value = disk_value, None, {}
        self.current_mask_seed = seed
        self.current_mask = value
        if use_cache:
            self.mask_cache[seed] = value
            while len(self.mask_cache) > self.finite_cache_size:
                self.mask_cache.popitem(last=False)
        return value

    def warmup_psf_cache(self, mask_records):
        for record in mask_records:
            self._get_mask(
                {
                    "mask_seed": int(record["mask_seed"]),
                    "mode": "finite",
                }
            )
        self.mask_cache.clear()
        self.current_mask_seed = None
        self.current_mask = None

    def _get_convolver(self, seed, psf):
        if seed in self.convolver_cache:
            convolver = self.convolver_cache.pop(seed)
            self.convolver_cache[seed] = convolver
            return convolver

        convolver = RealFFTConvolve2D(psf=_prepare_convolution_psf(psf))
        self.convolver_cache[seed] = convolver
        while len(self.convolver_cache) > 2:
            self.convolver_cache.popitem(last=False)
        return convolver

    def __getitem__(self, request):
        if not isinstance(request, dict):
            raise TypeError("on-the-fly dataset expects a sampler request")
        if request.get("paired", False):
            return self._paired_item(request)

        scene = self.scenes[int(request["scene_index"])]
        psf, simulator, _ = self._get_mask(request)
        if self.simulation_mode == "roi_convolution":
            target = _to_chw(scene["target"], self.target_size)
            convolver = self._get_convolver(int(request["mask_seed"]), psf)
            measurement = _aligned_measurement(
                target,
                psf,
                convolver,
                self.roi,
                quantize=bool(self.simulator_config.optics.quantize),
            )
            if self.measurement_size is not None:
                measurement = _to_chw(measurement, self.measurement_size)
        else:
            result = forward_single_sample(
                scene["target"],
                psf=psf,
                simulator=simulator,
                config=self.simulator_config,
                seed=int(request["sample_seed"]),
            )
            measurement = _to_chw(result["measurement"], self.measurement_size)
            target = _to_chw(result["ground_truth"], self.target_size)

        sample = {
            "measurement": measurement,
            "target": target,
            "sample_id": f'{scene["scene_id"]}__{request["mask_id"]}',
            "scene_id": scene["scene_id"],
            "source_index": scene["source_index"],
            "mask_id": request["mask_id"],
            "mask_seed": int(request["mask_seed"]),
            "sample_seed": int(request["sample_seed"]),
            "step": int(request["step"]),
            "split": scene["split"],
            "mode": request["mode"],
        }
        if "label" in scene:
            sample["label"] = scene["label"]
        if request.get("return_psf", False):
            prompt_psf = psf
            prompt_mask_seed = request.get("prompt_mask_seed")
            if prompt_mask_seed is not None and int(prompt_mask_seed) != int(
                request["mask_seed"]
            ):
                prompt_psf, _, _ = self._get_mask(
                    {
                        "mask_seed": int(prompt_mask_seed),
                        "mode": request["mode"],
                    }
                )
            sample["psf"] = (
                _prepare_convolution_psf(prompt_psf)
                .squeeze(0)
                .movedim(-1, 0)
                .contiguous()
            )
            sample["prompt_mask_id"] = request.get("prompt_mask_id", request["mask_id"])
        return sample

    def _paired_item(self, request):
        views = {}
        for view in ("a", "b"):
            view_request = {
                "scene_index": request["scene_index"],
                "mask_id": request[f"mask_id_{view}"],
                "mask_seed": request[f"mask_seed_{view}"],
                "sample_seed": request[f"sample_seed_{view}"],
                "step": request["step"],
                "mode": request["mode"],
                "return_psf": True,
            }
            views[view] = self.__getitem__(view_request)

        if views["a"]["scene_id"] != views["b"]["scene_id"]:
            raise RuntimeError("paired views must use the same scene")
        sample = {
            "measurement_a": views["a"]["measurement"],
            "measurement_b": views["b"]["measurement"],
            "psf_a": views["a"]["psf"],
            "psf_b": views["b"]["psf"],
            "scene_id": views["a"]["scene_id"],
            "source_index": views["a"]["source_index"],
            "mask_id_a": views["a"]["mask_id"],
            "mask_id_b": views["b"]["mask_id"],
            "mask_seed_a": views["a"]["mask_seed"],
            "mask_seed_b": views["b"]["mask_seed"],
            "sample_seed_a": views["a"]["sample_seed"],
            "sample_seed_b": views["b"]["sample_seed"],
            "step": views["a"]["step"],
            "split": views["a"]["split"],
            "mode": views["a"]["mode"],
        }
        if request.get("return_target", False):
            sample["target"] = views["a"]["target"]
        return sample


class DigiCamMaskBatchSampler:
    def __init__(
        self,
        scene_count,
        batch_size,
        steps,
        run_seed,
        mode,
        mask_records=None,
        infinite_base_seed=DEFAULT_MASK_SEED,
        return_psf=False,
        operator_prompt_mode="none",
        rank=0,
        world_size=1,
    ):
        self.scene_count = int(scene_count)
        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.run_seed = int(run_seed)
        self.mode = str(mode)
        self.mask_records = list(mask_records or [])
        self.infinite_base_seed = int(infinite_base_seed)
        self.operator_prompt_mode = _validate_operator_prompt_mode(operator_prompt_mode)
        self.return_psf = bool(return_psf or self.operator_prompt_mode != "none")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_step = 0

        if self.mode not in {"finite", "infinite"}:
            raise ValueError("mode must be finite or infinite")
        if self.scene_count < self.batch_size or self.batch_size <= 0:
            raise ValueError("scene_count must be at least batch_size")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.mode == "finite" and not self.mask_records:
            raise ValueError("finite mode needs mask records")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be inside world_size")

    def __len__(self):
        return self.steps - self.start_step

    def set_start_step(self, step):
        step = int(step)
        if step < 0 or step > self.steps:
            raise ValueError(f"start step must be within [0, {self.steps}]")
        self.start_step = step

    def _finite_mask(self, position):
        count = len(self.mask_records)
        cycle, offset = divmod(position, count)
        rng = np.random.default_rng(np.random.SeedSequence([self.run_seed, cycle, 17]))
        return self.mask_records[int(rng.permutation(count)[offset])]

    def _infinite_mask(self, position):
        return {
            "mask_id": f"infinite_{position:08d}",
            "mask_seed": mask_seed(self.infinite_base_seed, "infinite", position),
        }

    def __iter__(self):
        for step in range(self.start_step, self.steps):
            position = step * self.world_size + self.rank
            if self.mode == "finite":
                mask = self._finite_mask(position)
                mask_index = next(
                    index
                    for index, record in enumerate(self.mask_records)
                    if record["mask_seed"] == mask["mask_seed"]
                )
                prompt_mask = _prompt_mask_record(
                    self.mask_records,
                    mask_index,
                    self.operator_prompt_mode,
                )
            else:
                mask = self._infinite_mask(position)
                if self.operator_prompt_mode != "none":
                    raise ValueError("operator prompts currently require finite masks")
                prompt_mask = None

            scene_rng = np.random.default_rng(
                np.random.SeedSequence([self.run_seed, position, 43])
            )
            scene_indices = scene_rng.choice(
                self.scene_count, size=self.batch_size, replace=False
            )
            requests = [
                {
                    "scene_index": int(scene_index),
                    "mask_id": mask["mask_id"],
                    "mask_seed": int(mask["mask_seed"]),
                    "sample_seed": _sample_seed(self.run_seed, position, slot),
                    "step": position,
                    "mode": self.mode,
                    "return_psf": self.return_psf,
                }
                for slot, scene_index in enumerate(scene_indices)
            ]
            if prompt_mask is not None:
                for request in requests:
                    request["prompt_mask_id"] = prompt_mask["mask_id"]
                    request["prompt_mask_seed"] = int(prompt_mask["mask_seed"])
            yield requests


class DigiCamPairedMaskBatchSampler:
    def __init__(
        self,
        scene_count,
        batch_size,
        steps,
        run_seed,
        mask_records,
        return_target=False,
        rank=0,
        world_size=1,
    ):
        self.scene_count = int(scene_count)
        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.run_seed = int(run_seed)
        self.mask_records = list(mask_records)
        self.return_target = bool(return_target)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_step = 0

        if self.scene_count < self.batch_size or self.batch_size <= 0:
            raise ValueError("scene_count must be at least batch_size")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if len(self.mask_records) < 2:
            raise ValueError("paired training needs at least two masks")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be inside world_size")

    def __len__(self):
        return self.steps - self.start_step

    def set_start_step(self, step):
        step = int(step)
        if step < 0 or step > self.steps:
            raise ValueError(f"start step must be within [0, {self.steps}]")
        self.start_step = step

    def __iter__(self):
        for step in range(self.start_step, self.steps):
            position = step * self.world_size + self.rank
            mask_rng = np.random.default_rng(
                np.random.SeedSequence([self.run_seed, position, 71])
            )
            mask_indices = mask_rng.choice(
                len(self.mask_records), size=2, replace=False
            )
            mask_a, mask_b = (self.mask_records[int(index)] for index in mask_indices)

            scene_rng = np.random.default_rng(
                np.random.SeedSequence([self.run_seed, position, 43])
            )
            scene_indices = scene_rng.choice(
                self.scene_count, size=self.batch_size, replace=False
            )
            yield [
                {
                    "paired": True,
                    "return_target": self.return_target,
                    "scene_index": int(scene_index),
                    "mask_id_a": mask_a["mask_id"],
                    "mask_seed_a": int(mask_a["mask_seed"]),
                    "mask_id_b": mask_b["mask_id"],
                    "mask_seed_b": int(mask_b["mask_seed"]),
                    "sample_seed_a": _sample_seed(self.run_seed, position, 2 * slot),
                    "sample_seed_b": _sample_seed(
                        self.run_seed, position, 2 * slot + 1
                    ),
                    "step": position,
                    "mode": "finite",
                }
                for slot, scene_index in enumerate(scene_indices)
            ]


class DigiCamCrossMaskGateBatchSampler:

    def __init__(
        self,
        scene_count,
        batch_size,
        mask_records,
        run_seed,
        scenes_per_mask=32,
        scene_selector_salt=59,
    ):
        self.scene_count = int(scene_count)
        self.batch_size = int(batch_size)
        self.mask_records = list(mask_records)
        self.run_seed = int(run_seed)
        self.scenes_per_mask = int(scenes_per_mask)
        self.scene_selector_salt = int(scene_selector_salt)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.mask_records) < 2:
            raise ValueError("cross-mask gate needs at least two masks")
        if not 0 < self.scenes_per_mask <= self.scene_count:
            raise ValueError("scenes_per_mask must be within the scene split")

    def __len__(self):
        batches_per_mask = (
            self.scenes_per_mask + self.batch_size - 1
        ) // self.batch_size
        return len(self.mask_records) * batches_per_mask

    def __iter__(self):
        rng = np.random.default_rng(
            np.random.SeedSequence([self.run_seed, self.scene_selector_salt])
        )
        scene_indices = rng.permutation(self.scene_count)[: self.scenes_per_mask]
        step = 0
        for mask_index, source_mask in enumerate(self.mask_records):
            probe_mask = self.mask_records[(mask_index + 1) % len(self.mask_records)]
            for start in range(0, self.scenes_per_mask, self.batch_size):
                batch_scene_indices = scene_indices[start : start + self.batch_size]
                yield [
                    {
                        "paired": True,
                        "return_target": True,
                        "scene_index": int(scene_index),
                        "mask_id_a": source_mask["mask_id"],
                        "mask_seed_a": int(source_mask["mask_seed"]),
                        "mask_id_b": probe_mask["mask_id"],
                        "mask_seed_b": int(probe_mask["mask_seed"]),
                        "sample_seed_a": _sample_seed(
                            self.run_seed, mask_index, int(scene_index)
                        ),
                        "sample_seed_b": _sample_seed(
                            self.run_seed,
                            mask_index + len(self.mask_records),
                            int(scene_index),
                        ),
                        "step": step,
                        "mode": "finite",
                    }
                    for scene_index in batch_scene_indices
                ]
                step += 1


class DigiCamValidationBatchSampler:
    def __init__(
        self,
        scene_count,
        batch_size,
        mask_records,
        run_seed,
        scenes_per_mask=4,
        return_psf=False,
        operator_prompt_mode="none",
        scene_offset=0,
        scene_selector_salt=59,
    ):
        self.scene_count = int(scene_count)
        self.batch_size = int(batch_size)
        self.mask_records = list(mask_records)
        self.run_seed = int(run_seed)
        self.scenes_per_mask = int(scenes_per_mask)
        self.operator_prompt_mode = _validate_operator_prompt_mode(operator_prompt_mode)
        self.return_psf = bool(return_psf or self.operator_prompt_mode != "none")
        self.scene_offset = int(scene_offset)
        self.scene_selector_salt = int(scene_selector_salt)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.mask_records:
            raise ValueError("validation needs mask records")
        if not 0 < self.scenes_per_mask <= self.scene_count:
            raise ValueError("scenes_per_mask must be within the validation split")
        if self.scene_offset < 0 or (
            self.scene_offset + self.scenes_per_mask > self.scene_count
        ):
            raise ValueError("scene_offset and scenes_per_mask exceed the scene split")

    def __len__(self):
        batches_per_mask = (
            self.scenes_per_mask + self.batch_size - 1
        ) // self.batch_size
        return len(self.mask_records) * batches_per_mask

    def __iter__(self):
        rng = np.random.default_rng(
            np.random.SeedSequence([self.run_seed, self.scene_selector_salt])
        )
        scene_indices = rng.permutation(self.scene_count)[
            self.scene_offset : self.scene_offset + self.scenes_per_mask
        ]
        step = 0

        for mask_index, mask in enumerate(self.mask_records):
            prompt_mask = _prompt_mask_record(
                self.mask_records,
                mask_index,
                self.operator_prompt_mode,
            )
            for start in range(0, self.scenes_per_mask, self.batch_size):
                batch_scene_indices = scene_indices[start : start + self.batch_size]
                requests = [
                    {
                        "scene_index": int(scene_index),
                        "mask_id": mask["mask_id"],
                        "mask_seed": int(mask["mask_seed"]),
                        "sample_seed": _sample_seed(self.run_seed, int(scene_index), 0),
                        "step": step,
                        "mode": "finite",
                        "return_psf": self.return_psf,
                    }
                    for scene_index in batch_scene_indices
                ]
                if prompt_mask is not None:
                    for request in requests:
                        request["prompt_mask_id"] = prompt_mask["mask_id"]
                        request["prompt_mask_seed"] = int(prompt_mask["mask_seed"])
                yield requests
                step += 1


def _scene_dataset(config):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    if config.get("root_dir") is not None:
        config.root_dir = to_absolute_path(config.root_dir)
    if config.get("splits_path") is not None:
        config.splits_path = to_absolute_path(config.splits_path)
    return instantiate(config)


def build_cross_mask_gate_dataloader(
    datasets_config,
    simulator_config,
    validation_mask_count=32,
    scenes_per_mask=32,
    scene_selector_salt=59,
    batch_size=4,
    evaluation_mask_seed=DEFAULT_MASK_SEED,
    validation_seed=DEFAULT_VALIDATION_SEED,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=True,
    measurement_size=None,
    target_size=(200, 266),
    roi=(80, 100, 200, 266),
    finite_cache_size=2,
    psf_cache=None,
    validation_scenes=None,
    mask_factory=None,
):
    if validation_scenes is None:
        validation_scenes = _scene_dataset(datasets_config.validation)
    mask_records = get_mask_records(
        int(evaluation_mask_seed), "validation", int(validation_mask_count)
    )
    psf_cache = dict(psf_cache or {})
    if psf_cache.get("root_dir") is not None:
        psf_cache["root_dir"] = to_absolute_path(psf_cache["root_dir"])
    dataset = DigiCamOnTheFlyDataset(
        validation_scenes,
        simulator_config,
        measurement_size=measurement_size,
        target_size=target_size,
        simulation_mode="roi_convolution",
        roi=roi,
        finite_cache_size=finite_cache_size,
        psf_cache=psf_cache,
        mask_factory=mask_factory,
    )
    if dataset.psf_cache.warmup:
        dataset.warmup_psf_cache(mask_records)
    sampler = DigiCamCrossMaskGateBatchSampler(
        scene_count=len(validation_scenes),
        batch_size=batch_size,
        mask_records=mask_records,
        run_seed=validation_seed,
        scenes_per_mask=scenes_per_mask,
        scene_selector_salt=scene_selector_salt,
    )
    loader_args = {"num_workers": int(num_workers), "pin_memory": bool(pin_memory)}
    if int(num_workers) > 0:
        if prefetch_factor is not None:
            loader_args["prefetch_factor"] = int(prefetch_factor)
        loader_args["persistent_workers"] = bool(persistent_workers)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        generator=torch.Generator().manual_seed(int(validation_seed)),
        **loader_args,
    )
    return loader, mask_records


def build_failure_prediction_dataloaders(
    datasets_config,
    simulator_config,
    validation_mask_count=32,
    calibration_mask_count=16,
    reference_mask_count=16,
    scenes_per_mask=16,
    batch_size=1,
    evaluation_mask_seed=DEFAULT_MASK_SEED,
    reconstruction_mask_seed=DEFAULT_MASK_SEED,
    validation_seed=DEFAULT_VALIDATION_SEED,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=True,
    measurement_size=None,
    target_size=(200, 266),
    roi=(80, 100, 200, 266),
    finite_cache_size=2,
    psf_cache=None,
    validation_scenes=None,
    mask_factory=None,
):
    validation_mask_count = int(validation_mask_count)
    calibration_mask_count = int(calibration_mask_count)
    if not 0 < calibration_mask_count < validation_mask_count:
        raise ValueError(
            "calibration_mask_count must be between zero and validation_mask_count"
        )
    if validation_scenes is None:
        validation_scenes = _scene_dataset(datasets_config.validation)

    validation_records = get_mask_records(
        int(evaluation_mask_seed),
        "validation",
        validation_mask_count,
    )
    reference_records = get_mask_records(
        int(reconstruction_mask_seed),
        "train",
        int(reference_mask_count),
    )
    psf_cache = dict(psf_cache or {})
    if psf_cache.get("root_dir") is not None:
        psf_cache["root_dir"] = to_absolute_path(psf_cache["root_dir"])
    dataset = DigiCamOnTheFlyDataset(
        validation_scenes,
        simulator_config,
        measurement_size=measurement_size,
        target_size=target_size,
        simulation_mode="roi_convolution",
        roi=roi,
        finite_cache_size=finite_cache_size,
        psf_cache=psf_cache,
        mask_factory=mask_factory,
    )
    if dataset.psf_cache.warmup:
        dataset.warmup_psf_cache(reference_records + validation_records)

    loader_args = {"num_workers": int(num_workers), "pin_memory": bool(pin_memory)}
    if int(num_workers) > 0:
        if prefetch_factor is not None:
            loader_args["prefetch_factor"] = int(prefetch_factor)
        loader_args["persistent_workers"] = bool(persistent_workers)

    mask_splits = {
        "reference": reference_records,
        "calibration": validation_records[:calibration_mask_count],
        "test": validation_records[calibration_mask_count:],
    }
    dataloaders = {}
    for split_index, (name, records) in enumerate(mask_splits.items()):
        sampler = DigiCamValidationBatchSampler(
            scene_count=len(validation_scenes),
            batch_size=batch_size,
            mask_records=records,
            run_seed=validation_seed,
            scenes_per_mask=scenes_per_mask,
            scene_offset=split_index * int(scenes_per_mask),
        )
        dataloaders[name] = DataLoader(
            dataset,
            batch_sampler=sampler,
            generator=torch.Generator().manual_seed(int(validation_seed) + split_index),
            **loader_args,
        )
    return dataloaders, {}


def build_on_the_fly_dataloaders(
    datasets_config,
    simulator_config,
    train_mode=None,
    finite_mask_count=None,
    train_steps=None,
    validation_mask_count=None,
    validation_scenes_per_mask=4,
    validation_steps=None,
    batch_size=4,
    run_seed=1,
    base_mask_seed=DEFAULT_MASK_SEED,
    train_mask_seed=None,
    evaluation_mask_seed=None,
    validation_seed=DEFAULT_VALIDATION_SEED,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=True,
    measurement_size=(64, 64),
    target_size=(64, 64),
    simulation_mode="far_field",
    roi=None,
    finite_cache_size=8,
    psf_cache=None,
    train_scenes=None,
    validation_scenes=None,
    mask_factory=None,
    evaluation_only=False,
    paired_train=False,
    paired_scene_count=None,
    cross_validation_steps=0,
    return_psf=False,
    operator_prompt_mode="none",
):
    if train_mask_seed is None:
        train_mask_seed = base_mask_seed
    if evaluation_mask_seed is None:
        evaluation_mask_seed = base_mask_seed

    if validation_mask_count is None:
        validation_mask_count = 32 if validation_steps is None else validation_steps
    elif validation_steps is not None and int(validation_steps) != int(
        validation_mask_count
    ):
        raise ValueError("validation_steps and validation_mask_count must match")

    if validation_scenes is None:
        validation_scenes = _scene_dataset(datasets_config.validation)

    validation_records = get_mask_records(
        evaluation_mask_seed,
        "validation",
        int(validation_mask_count),
    )

    psf_cache = dict(psf_cache or {})
    if psf_cache.get("root_dir") is not None:
        psf_cache["root_dir"] = to_absolute_path(psf_cache["root_dir"])

    validation_dataset = DigiCamOnTheFlyDataset(
        validation_scenes,
        simulator_config,
        measurement_size=measurement_size,
        target_size=target_size,
        simulation_mode=simulation_mode,
        roi=roi,
        finite_cache_size=finite_cache_size,
        psf_cache=psf_cache,
        mask_factory=mask_factory,
    )

    if validation_dataset.psf_cache.warmup:
        validation_dataset.warmup_psf_cache(validation_records)
    validation_sampler = DigiCamValidationBatchSampler(
        scene_count=len(validation_scenes),
        batch_size=batch_size,
        scenes_per_mask=validation_scenes_per_mask,
        run_seed=validation_seed,
        mask_records=validation_records,
        return_psf=return_psf,
        operator_prompt_mode=operator_prompt_mode,
    )

    loader_args = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        if prefetch_factor is not None:
            loader_args["prefetch_factor"] = int(prefetch_factor)
        loader_args["persistent_workers"] = bool(persistent_workers)
    validation_generator = torch.Generator().manual_seed(int(validation_seed))
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        generator=validation_generator,
        **loader_args,
    )
    dataloaders = {"validation": validation_loader}
    if int(cross_validation_steps) > 0:
        cross_validation_sampler = DigiCamPairedMaskBatchSampler(
            scene_count=len(validation_scenes),
            batch_size=batch_size,
            steps=int(cross_validation_steps),
            run_seed=validation_seed,
            mask_records=validation_records,
        )
        dataloaders["cross_validation"] = DataLoader(
            validation_dataset,
            batch_sampler=cross_validation_sampler,
            generator=torch.Generator().manual_seed(int(validation_seed)),
            **loader_args,
        )
    if evaluation_only:
        return dataloaders, {}

    if train_mode is None or train_steps is None:
        raise ValueError("training needs train_mode and train_steps")
    if train_mode == "finite" and finite_mask_count is None:
        raise ValueError("finite training needs finite_mask_count")
    if train_scenes is None:
        train_scenes = _scene_dataset(datasets_config.train)

    train_records = None
    if train_mode == "finite":
        train_records = get_mask_records(
            train_mask_seed,
            "train",
            int(finite_mask_count),
        )
    train_dataset = DigiCamOnTheFlyDataset(
        train_scenes,
        simulator_config,
        measurement_size=measurement_size,
        target_size=target_size,
        simulation_mode=simulation_mode,
        roi=roi,
        finite_cache_size=finite_cache_size,
        psf_cache=psf_cache,
        mask_factory=mask_factory,
    )
    if train_dataset.psf_cache.warmup and train_records is not None:
        train_dataset.warmup_psf_cache(train_records)
    if paired_train:
        if train_mode != "finite":
            raise ValueError("paired training currently requires finite masks")
        if simulation_mode != "roi_convolution":
            raise ValueError("paired training currently requires roi_convolution")
        paired_scene_count = (
            len(train_scenes) if paired_scene_count is None else int(paired_scene_count)
        )
        if not 0 < paired_scene_count <= len(train_scenes):
            raise ValueError("paired_scene_count must be within the train split")
        train_sampler = DigiCamPairedMaskBatchSampler(
            scene_count=paired_scene_count,
            batch_size=batch_size,
            steps=train_steps,
            run_seed=run_seed,
            mask_records=train_records,
        )
    else:
        train_sampler = DigiCamMaskBatchSampler(
            scene_count=len(train_scenes),
            batch_size=batch_size,
            steps=train_steps,
            run_seed=run_seed,
            mode=train_mode,
            mask_records=train_records,
            infinite_base_seed=train_mask_seed,
            return_psf=return_psf,
            operator_prompt_mode=operator_prompt_mode,
        )
    train_generator = torch.Generator().manual_seed(int(run_seed))
    dataloaders["train"] = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        generator=train_generator,
        **loader_args,
    )
    return dataloaders, {}
