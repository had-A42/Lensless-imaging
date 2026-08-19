from collections import OrderedDict
from functools import partial

import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
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
        self.current_convolver_seed = None
        self.current_convolver = None

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
        if seed == self.current_convolver_seed:
            return self.current_convolver

        try:
            from lensless.recon.rfft_convolve import RealFFTConvolve2D
        except ImportError as error:
            raise ImportError("aligned simulation requires LenslessPiCam") from error

        self.current_convolver_seed = seed
        self.current_convolver = RealFFTConvolve2D(
            psf=torch.as_tensor(np.asarray(psf), dtype=torch.float32)
        )
        return self.current_convolver

    def __getitem__(self, request):
        if not isinstance(request, dict):
            raise TypeError("on-the-fly dataset expects a sampler request")

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

        return {
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
            else:
                mask = self._infinite_mask(position)

            scene_rng = np.random.default_rng(
                np.random.SeedSequence([self.run_seed, position, 43])
            )
            scene_indices = scene_rng.choice(
                self.scene_count, size=self.batch_size, replace=False
            )
            yield [
                {
                    "scene_index": int(scene_index),
                    "mask_id": mask["mask_id"],
                    "mask_seed": int(mask["mask_seed"]),
                    "sample_seed": _sample_seed(self.run_seed, position, slot),
                    "step": position,
                    "mode": self.mode,
                }
                for slot, scene_index in enumerate(scene_indices)
            ]


class DigiCamValidationBatchSampler:
    def __init__(
        self,
        scene_count,
        batch_size,
        mask_records,
        run_seed,
        scenes_per_mask=4,
    ):
        self.scene_count = int(scene_count)
        self.batch_size = int(batch_size)
        self.mask_records = list(mask_records)
        self.run_seed = int(run_seed)
        self.scenes_per_mask = int(scenes_per_mask)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.mask_records:
            raise ValueError("validation needs mask records")
        if not 0 < self.scenes_per_mask <= self.scene_count:
            raise ValueError("scenes_per_mask must be within the validation split")

    def __len__(self):
        batches_per_mask = (
            self.scenes_per_mask + self.batch_size - 1
        ) // self.batch_size
        return len(self.mask_records) * batches_per_mask

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.run_seed, 59]))
        scene_indices = rng.permutation(self.scene_count)[: self.scenes_per_mask]
        step = 0

        for mask in self.mask_records:
            for start in range(0, self.scenes_per_mask, self.batch_size):
                batch_scene_indices = scene_indices[start : start + self.batch_size]
                yield [
                    {
                        "scene_index": int(scene_index),
                        "mask_id": mask["mask_id"],
                        "mask_seed": int(mask["mask_seed"]),
                        "sample_seed": _sample_seed(self.run_seed, int(scene_index), 0),
                        "step": step,
                        "mode": "finite",
                    }
                    for scene_index in batch_scene_indices
                ]
                step += 1


def _scene_dataset(config):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config.root_dir = to_absolute_path(config.root_dir)
    config.splits_path = to_absolute_path(config.splits_path)
    return instantiate(config)


def build_on_the_fly_dataloaders(
    datasets_config,
    simulator_config,
    train_mode,
    finite_mask_count,
    train_steps,
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

    if train_scenes is None:
        train_scenes = _scene_dataset(datasets_config.train)
    if validation_scenes is None:
        validation_scenes = _scene_dataset(datasets_config.validation)

    train_records = None
    if train_mode == "finite":
        train_records = get_mask_records(
            train_mask_seed,
            "train",
            int(finite_mask_count),
        )
    validation_records = get_mask_records(
        evaluation_mask_seed,
        "validation",
        int(validation_mask_count),
    )

    psf_cache = dict(psf_cache or {})
    if psf_cache.get("root_dir") is not None:
        psf_cache["root_dir"] = to_absolute_path(psf_cache["root_dir"])

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

    if train_dataset.psf_cache.warmup:
        if train_records is not None:
            train_dataset.warmup_psf_cache(train_records)
        validation_dataset.warmup_psf_cache(validation_records)

    train_sampler = DigiCamMaskBatchSampler(
        scene_count=len(train_scenes),
        batch_size=batch_size,
        steps=train_steps,
        run_seed=run_seed,
        mode=train_mode,
        mask_records=train_records,
        infinite_base_seed=train_mask_seed,
    )
    validation_sampler = DigiCamValidationBatchSampler(
        scene_count=len(validation_scenes),
        batch_size=batch_size,
        scenes_per_mask=validation_scenes_per_mask,
        run_seed=validation_seed,
        mask_records=validation_records,
    )

    loader_args = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        if prefetch_factor is not None:
            loader_args["prefetch_factor"] = int(prefetch_factor)
        loader_args["persistent_workers"] = bool(persistent_workers)
    train_generator = torch.Generator().manual_seed(int(run_seed))
    validation_generator = torch.Generator().manual_seed(int(validation_seed))
    return {
        "train": DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            generator=train_generator,
            **loader_args,
        ),
        "validation": DataLoader(
            validation_dataset,
            batch_sampler=validation_sampler,
            generator=validation_generator,
            **loader_args,
        ),
    }, {}
