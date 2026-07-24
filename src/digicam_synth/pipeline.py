import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from tqdm.auto import tqdm

from lensless.hardware.sensor import VirtualSensor
from lensless.hardware.slm import get_intensity_psf, get_programmable_mask
from lensless.utils.image import rgb2gray
from lensless.utils.io import get_dtype, load_image, save_image

from waveprop.devices import SLMParam, slm_dict
from waveprop.simulation import FarFieldSimulator
from waveprop.pytorch_util import RealFFTConvolve2D


def _resize_like(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    in_height, in_width = image.shape[:2]
    out_height, out_width = target_shape
    y_idx = (np.linspace(0, in_height - 1, out_height)).round().astype(np.int64)
    x_idx = (np.linspace(0, in_width - 1, out_width)).round().astype(np.int64)
    return image[y_idx][:, x_idx]


def convert_to_target_type(type_name: str, tensor: np.ndarray) -> np.ndarray:
    if type_name == "float32":
        return tensor.astype(np.float32)
    if type_name == "uint8":
        return np.round(tensor * 255.0).clip(0, 255).astype(np.uint8)
    raise ValueError(f"Unknown type: {type_name}")


def generate_random_pattern(config, rng: np.random.Generator) -> np.ndarray:
    if config.pattern.dist_type == "uniform":
        pattern = rng.uniform(
            config.pattern.min_val,
            config.pattern.max_val,
            size=config.pattern.shape,
        ).astype(np.float32)
    elif config.pattern.dist_type == "bernoulli":
        pattern = (rng.random(config.pattern.shape) < config.pattern.bernoulli_p).astype(
            np.float32
        )
        pattern = pattern * (config.pattern.max_val - config.pattern.min_val) + config.pattern.min_val
    elif config.pattern.dist_type == "gaussian":
        pattern = rng.normal(
            loc=0.5,
            scale=config.pattern.gaussian_std,
            size=config.pattern.shape,
        ).astype(np.float32)
        pattern = np.clip(pattern, config.pattern.min_val, config.pattern.max_val)
    else:
        raise ValueError(f"unknown dist type: {config.pattern.dist_type}")
    return pattern.astype(np.float32)


def pattern_to_psf(
    config,
    pattern: np.ndarray,
    rng: np.random.Generator,
    filepath: Path | None = None,
) -> tuple[np.ndarray, FarFieldSimulator, dict[str, Any]]:
    torch_device = config.torch_device
    dtype = get_dtype(config.dtype, config.use_torch)

    rotate_angle = None
    if config.optics.use_rotate:
        rotate_angle = float(rng.uniform(config.optics.rotate_min, config.optics.rotate_max))

    sensor = VirtualSensor.from_name(
        config.optics.sensor,
        downsample=config.optics.downsample if config.optics.downsample > 1 else None,
    )

    slm_param = slm_dict[config.optics.slm]
    color_filter = None
    if SLMParam.COLOR_FILTER in slm_param.keys():
        color_filter = slm_param[SLMParam.COLOR_FILTER]
        if config.use_torch:
            color_filter = torch.from_numpy(color_filter.copy()).to(
                device=torch_device, dtype=dtype
            )
        else:
            color_filter = color_filter.astype(dtype)

    if config.optics.slm == "adafruit":
        pattern = pattern.reshape((-1, pattern.shape[-1]), order="F")

    if config.use_torch:
        pattern = torch.from_numpy(pattern)

    mask = get_programmable_mask(
        vals=pattern,
        sensor=sensor,
        slm_param=slm_param,
        rotate=rotate_angle,
        flipud=config.optics.flipud,
        color_filter=color_filter,
        deadspace=config.optics.deadspace,
    )

    vertical_shift = 0
    if config.optics.use_vertical_shift:
        vertical_shift = int(
            rng.uniform(config.optics.vertical_shift_min, config.optics.vertical_shift_max)
        )
        if config.use_torch:
            mask = torch.roll(mask, vertical_shift // config.optics.downsample, dims=1)
        else:
            mask = np.roll(mask, vertical_shift // config.optics.downsample, axis=1)

    horizontal_shift = 0
    if config.optics.use_horizontal_shift:
        horizontal_shift = int(
            rng.uniform(config.optics.horizontal_shift_min, config.optics.horizontal_shift_max)
        )
        if config.use_torch:
            mask = torch.roll(mask, horizontal_shift // config.optics.downsample, dims=2)
        else:
            mask = np.roll(mask, horizontal_shift // config.optics.downsample, axis=2)

    scene2mask = float(rng.uniform(config.optics.scene2mask_min, config.optics.scene2mask_max))
    mask2sensor = float(rng.uniform(config.optics.mask2sensor_min, config.optics.mask2sensor_max))

    psf = get_intensity_psf(
        mask=mask,
        sensor=sensor,
        waveprop=config.optics.use_waveprop,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
    )

    simulator = FarFieldSimulator(
        psf=psf,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        sensor=config.optics.sensor,
        snr_db=config.optics.snr_db,
        object_height=config.optics.object_height,
        device_conv=torch_device,
        is_torch=config.use_torch,
    )


    if config.grayscale and len(psf.shape) == 3:
        psf = rgb2gray(psf.transpose(1, 2, 0)).squeeze()  # (H,W)
        psf = psf[np.newaxis, ...]  # (1,H,W)
    else:
        psf = psf
    
    if config.use_torch:
        psf_np = psf.cpu().detach().numpy()
        mask_np = mask.cpu().detach().numpy()
    else:
        psf_np = psf.copy()
        mask_np = mask.copy()

    psf_np = np.transpose(psf_np, (1, 2, 0))
    mask_np = np.transpose(mask_np, (1, 2, 0))
        
    if filepath is not None:
        # save_image(psf_np, str(filepath))
        np.save(filepath, psf_np[np.newaxis, ...])

    psf_metadata = {
        "pattern_shape": list(np.asarray(pattern).shape),
        "mask_shape": list(np.asarray(mask_np).shape),
        "psf_shape": list(np.asarray(psf_np).shape),
        "rotate_angle": rotate_angle,
        "vertical_shift": vertical_shift,
        "horizontal_shift": horizontal_shift,
        "scene2mask": scene2mask,
        "mask2sensor": mask2sensor,
        "sensor": config.optics.sensor,
        "slm": config.optics.slm,
        "grayscale": bool(config.grayscale),
    }
    assert len(psf.shape) == 3 and (psf.shape[0] == 3 or psf.shape[0] == 1)
    return psf, simulator, psf_metadata


def synthesize_measurement(
    config,
    rng: np.random.Generator,
    save_dir: Path,
    psf_dir: Path,
    psf_id: int,
    image_fps: list[Path],
) -> list[dict[str, Any]]:
    pattern = generate_random_pattern(config, rng)

    psf_filepath = psf_dir / f"psf_{psf_id}.npy"
    psf, simulator, psf_metadata = pattern_to_psf(config, pattern, rng, psf_filepath)

    with open(psf_dir / f"psf_{psf_id}.metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "psf_id": psf_id,
                "psf_file": psf_filepath.name,
                "pattern_min": float(np.min(pattern)),
                "pattern_max": float(np.max(pattern)),
                "pattern_mean": float(np.mean(pattern)),
                **psf_metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    saved_samples = []

    for image_fp in tqdm(image_fps, "Measuring", leave=False):
        image = load_image(str(image_fp))  # [H, W, 3] или [H, W, 1]
        original_shape = list(image.shape)
        resized = False

        if image.shape[:2] != psf.shape[1:3]:
            image = _resize_like(image, psf.shape[1:3])
            resized = True

        ground_truth = image.copy()

        image = np.transpose(image, (2, 0, 1)) # [3, H, W]  или [1, H, W]
        if config.use_torch:
            image = torch.from_numpy(image)

        if image.shape[0] == 1 and psf.shape[0] == 3:
            image = np.repeat(image, 3, axis=-1)

        if image.shape[0] != psf.shape[0]:
            raise ValueError(f"Image channels ({image.shape[0]}) != PSF channels ({psf.shape[0]})")
    
        # conv = RealFFTConvolve2D(psf, device="cpu")
        # lensless = conv(image)

        lensless, ground_truth = simulator.propagate(image, return_object_plane=True)

        if config.use_torch:
            ground_truth = ground_truth.cpu().detach().numpy()
        ground_truth = np.transpose(ground_truth, (1, 2, 0))

        bn = image_fp.stem

        sample_dir = save_dir / (bn + f"_{psf_id=}")
        sample_dir.mkdir(parents=True, exist_ok=True)

        # original_fp = sample_dir / "origin.png"
        # save_image(ground_truth, str(original_fp))

        # lensless_fp = sample_dir / "lensless.png"
        if config.use_torch:
            lensless = torch.permute(lensless, (1, 2, 0)).cpu().detach().numpy()
        else:
            lensless = np.transpose(lensless, (1, 2, 0))
            
        # save_image(lensless, str(lensless_fp))
        np.savez_compressed(sample_dir / "sample", lensless=lensless.astype(np.uint8), ground_truth=ground_truth.astype(np.uint8))

        metadata = {
            "sample_name": bn,
            "source_file": str(image_fp),
            "psf_id": psf_id,
            "psf_file": str(psf_filepath),
            "sample_dir": str(sample_dir),
            "original_image_shape": original_shape,
            # "processed_image_shape": list(np.asarray(image).shape),
            # "lensless_shape": list(np.asarray(lensless).shape),
            "measurement_shape": list(np.asarray(lensless).shape),
            "resized_to_match_psf": resized,
            "grayscale": bool(config.grayscale),
            "pattern_stats": {
                "min": float(np.min(pattern)),
                "max": float(np.max(pattern)),
                "mean": float(np.mean(pattern)),
            },
            "psf_metadata": psf_metadata,
        }

        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        saved_samples.append(metadata)

    return saved_samples


def save_sample(
    gt: np.ndarray,
    pattern: np.ndarray,
    psf: np.ndarray,
    measurement: np.ndarray,
    metadata: dict[str, Any],
    out_dir: str | Path,
    sample_id: int,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sdir = out_dir / f"sample_{sample_id:06d}"
    sdir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        sdir / "sample.npz",
        gt=np.asarray(gt),
        pattern=np.asarray(pattern),
        psf=np.asarray(psf),
        measurement=np.asarray(measurement),
    )

    with open(sdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return sdir