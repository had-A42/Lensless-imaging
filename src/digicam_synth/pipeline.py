from __future__ import annotations

import hashlib
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None


_RGB_WEIGHTS = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_hwc(value: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 2:
        array = array[..., None]
    elif array.ndim != 3:
        raise ValueError(f"{name} must be HW, HWC or CHW; received shape {array.shape}")

    if array.shape[-1] not in (1, 3):
        if array.shape[0] in (1, 3):
            array = np.moveaxis(array, 0, -1)
        else:
            raise ValueError(
                f"{name} must have 1 or 3 channels; received shape {array.shape}"
            )

    if array.shape[-1] not in (1, 3):
        raise ValueError(
            f"{name} must have 1 or 3 channels; received shape {array.shape}"
        )
    return np.ascontiguousarray(array)


def _rgb_to_gray_hwc(image: np.ndarray) -> np.ndarray:
    if image.shape[-1] == 1:
        return image
    return np.sum(image.astype(np.float32) * _RGB_WEIGHTS, axis=-1, keepdims=True)


def _prepare_scene(
    image: Any,
    *,
    channels: int,
    max_val: float,
    grayscale: bool,
) -> np.ndarray:
    scene = _as_hwc(image, name="image")
    source_dtype = scene.dtype
    scene = scene.astype(np.float32)

    if np.issubdtype(source_dtype, np.integer):
        source_max = float(np.iinfo(source_dtype).max)
        if source_max > 0:
            scene *= max_val / source_max
    elif scene.size and float(np.nanmax(scene)) <= 1.0 + 1e-6:
        scene *= max_val

    if not np.isfinite(scene).all():
        raise ValueError("image contains NaN or infinity")
    scene = np.clip(scene, 0.0, max_val)

    if grayscale or channels == 1:
        scene = _rgb_to_gray_hwc(scene)
    elif scene.shape[-1] == 1 and channels == 3:
        scene = np.repeat(scene, 3, axis=-1)

    if scene.shape[-1] != channels:
        raise ValueError(
            f"Image channels ({scene.shape[-1]}) do not match PSF channels ({channels})"
        )
    return np.ascontiguousarray(scene, dtype=np.float32)


def convert_to_target_type(
    type_name: str,
    tensor: Any,
    *,
    source_max: float = 1.0,
) -> np.ndarray:
    if source_max <= 0:
        raise ValueError("source_max must be positive")
    array = _to_numpy(tensor).astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Cannot serialize an array containing NaN or infinity")
    normalized = np.clip(array / float(source_max), 0.0, 1.0)
    if type_name == "float32":
        return normalized.astype(np.float32)
    if type_name == "uint8":
        return np.rint(normalized * 255.0).astype(np.uint8)
    raise ValueError(f"Unknown output dtype: {type_name}")


def generate_random_pattern(config: Any, rng: np.random.Generator) -> np.ndarray:
    pattern_config = config.pattern
    shape = tuple(int(value) for value in pattern_config.shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(
            f"pattern.shape must contain two positive dimensions, got {shape}"
        )

    min_val = float(pattern_config.min_val)
    max_val = float(pattern_config.max_val)
    if not min_val < max_val:
        raise ValueError("pattern.min_val must be smaller than pattern.max_val")

    if pattern_config.dist_type == "uniform":
        pattern = rng.uniform(min_val, max_val, size=shape)
    elif pattern_config.dist_type == "bernoulli":
        probability = float(pattern_config.bernoulli_p)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("pattern.bernoulli_p must be in [0, 1]")
        pattern = (rng.random(shape) < probability).astype(np.float32)
        pattern = pattern * (max_val - min_val) + min_val
    elif pattern_config.dist_type == "gaussian":
        std = float(pattern_config.gaussian_std)
        if std < 0:
            raise ValueError("pattern.gaussian_std must be non-negative")
        mean = (min_val + max_val) / 2.0
        pattern = rng.normal(loc=mean, scale=std, size=shape)
        pattern = np.clip(pattern, min_val, max_val)
    else:
        raise ValueError(f"Unknown pattern distribution: {pattern_config.dist_type}")
    return np.asarray(pattern, dtype=np.float32)


def _load_optics_backend() -> dict[str, Any]:
    def import_backend() -> dict[str, Any]:
        from lensless.hardware.sensor import VirtualSensor
        from lensless.hardware.slm import get_intensity_psf, get_programmable_mask
        from lensless.utils.io import get_dtype, load_image
        from lensless.utils.simulation import FarFieldSimulator
        from waveprop.devices import SLMParam, slm_dict

        return {
            "VirtualSensor": VirtualSensor,
            "get_intensity_psf": get_intensity_psf,
            "get_programmable_mask": get_programmable_mask,
            "get_dtype": get_dtype,
            "load_image": load_image,
            "FarFieldSimulator": FarFieldSimulator,
            "SLMParam": SLMParam,
            "slm_dict": slm_dict,
        }

    needs_turtle_stub = False
    try:
        import turtle
    except ModuleNotFoundError as error:
        if error.name != "_tkinter":
            raise
        needs_turtle_stub = True

    previous_turtle = None
    if needs_turtle_stub:
        previous_turtle = sys.modules.pop("turtle", None)
        turtle_stub = ModuleType("turtle")
        turtle_stub.pu = None
        sys.modules["turtle"] = turtle_stub
    try:
        slm_module = sys.modules.get("lensless.hardware.slm")
        if slm_module is not None and not getattr(
            slm_module, "waveprop_available", True
        ):
            importlib.reload(slm_module)
        return import_backend()
    except ImportError as error:
        raise RuntimeError(
            "Synthetic optics requires LenslessPiCam, waveprop and their dependencies. "
            "Install requirements.txt in the project environment."
        ) from error
    finally:
        if needs_turtle_stub:
            sys.modules.pop("turtle", None)
            if previous_turtle is not None:
                sys.modules["turtle"] = previous_turtle


def _sample_integer(rng: np.random.Generator, minimum: int, maximum: int) -> int:
    if maximum < minimum:
        raise ValueError(f"Invalid integer range [{minimum}, {maximum}]")
    return int(rng.integers(minimum, maximum + 1))


def _chw_to_dhwc(value: Any) -> Any:
    if len(value.shape) != 3 or value.shape[0] not in (1, 3):
        raise ValueError(
            f"Expected CHW array with 1 or 3 channels, got {tuple(value.shape)}"
        )
    if torch is not None and isinstance(value, torch.Tensor):
        return value.movedim(0, -1).unsqueeze(0)
    return np.moveaxis(np.asarray(value), 0, -1)[None, ...]


def _grayscale_chw(value: Any) -> Any:
    if value.shape[0] == 1:
        return value
    if value.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 PSF channels, got {value.shape[0]}")
    if torch is not None and isinstance(value, torch.Tensor):
        weights = torch.as_tensor(_RGB_WEIGHTS, dtype=value.dtype, device=value.device)
        return torch.sum(value * weights[:, None, None], dim=0, keepdim=True)
    return np.sum(
        np.asarray(value) * _RGB_WEIGHTS[:, None, None], axis=0, keepdims=True
    )


def pattern_to_psf(
    config: Any,
    pattern: np.ndarray,
    rng: np.random.Generator,
    filepath: Path | None = None,
    create_simulator: bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
    backend = _load_optics_backend()
    use_torch = bool(config.use_torch)
    if use_torch and torch is None:
        raise RuntimeError("config.use_torch=true but PyTorch is not installed")

    torch_device = str(config.torch_device)
    dtype = backend["get_dtype"](config.dtype, use_torch)
    pattern_original_shape = list(pattern.shape)

    rotate_angle = None
    if config.optics.use_rotate:
        rotate_angle = float(
            rng.uniform(config.optics.rotate_min, config.optics.rotate_max)
        )

    sensor = backend["VirtualSensor"].from_name(
        config.optics.sensor,
        downsample=config.optics.downsample if config.optics.downsample > 1 else None,
    )
    slm_param = backend["slm_dict"][config.optics.slm]

    color_filter = None
    if backend["SLMParam"].COLOR_FILTER in slm_param:
        color_filter = slm_param[backend["SLMParam"].COLOR_FILTER]
        if use_torch:
            color_filter = torch.from_numpy(color_filter.copy()).to(
                device=torch_device, dtype=dtype
            )
        else:
            color_filter = color_filter.astype(dtype)

    programmed_pattern = np.asarray(pattern, dtype=np.float32)
    if config.optics.slm == "adafruit":
        programmed_pattern = programmed_pattern.reshape(
            (-1, programmed_pattern.shape[-1]), order="F"
        )
    if use_torch:
        programmed_pattern = torch.from_numpy(programmed_pattern).to(
            device=torch_device, dtype=dtype
        )
    else:
        programmed_pattern = programmed_pattern.astype(dtype)

    mask = backend["get_programmable_mask"](
        vals=programmed_pattern,
        sensor=sensor,
        slm_param=slm_param,
        rotate=rotate_angle,
        flipud=config.optics.flipud,
        color_filter=color_filter,
        deadspace=config.optics.deadspace,
    )

    vertical_shift = 0
    if config.optics.use_vertical_shift:
        vertical_shift = _sample_integer(
            rng,
            int(config.optics.vertical_shift_min),
            int(config.optics.vertical_shift_max),
        )
    horizontal_shift = 0
    if config.optics.use_horizontal_shift:
        horizontal_shift = _sample_integer(
            rng,
            int(config.optics.horizontal_shift_min),
            int(config.optics.horizontal_shift_max),
        )

    vertical_shift_px = vertical_shift // int(config.optics.downsample)
    horizontal_shift_px = horizontal_shift // int(config.optics.downsample)
    if vertical_shift_px:
        mask = (
            torch.roll(mask, vertical_shift_px, dims=1)
            if use_torch
            else np.roll(mask, vertical_shift_px, axis=1)
        )
    if horizontal_shift_px:
        mask = (
            torch.roll(mask, horizontal_shift_px, dims=2)
            if use_torch
            else np.roll(mask, horizontal_shift_px, axis=2)
        )

    scene2mask = float(
        rng.uniform(config.optics.scene2mask_min, config.optics.scene2mask_max)
    )
    mask2sensor = float(
        rng.uniform(config.optics.mask2sensor_min, config.optics.mask2sensor_max)
    )
    psf_chw = backend["get_intensity_psf"](
        mask=mask,
        sensor=sensor,
        waveprop=config.optics.use_waveprop,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
    )
    if config.grayscale:
        # (H,W)
        psf_chw = _grayscale_chw(psf_chw)  # (1,H,W)
    psf = _chw_to_dhwc(psf_chw)
    psf_np = _to_numpy(psf).astype(np.float32)
    if not np.isfinite(psf_np).all():
        raise ValueError("Generated PSF contains NaN or infinity")
    if psf_np.size == 0 or float(psf_np.max()) <= 0.0:
        raise ValueError(
            "Generated PSF has no positive energy. Check pattern size, sensor "
            "downsample and deadspace settings."
        )

    max_val = float(_config_get(config.optics, "max_val", 255))
    quantize = bool(_config_get(config.optics, "quantize", True))
    return_float = bool(_config_get(config.optics, "return_float", True))
    simulator = None
    if create_simulator:
        simulator = backend["FarFieldSimulator"](
            psf=psf,
            scene2mask=scene2mask,
            mask2sensor=mask2sensor,
            sensor=config.optics.sensor,
            snr_db=_config_get(config.optics, "snr_db", None),
            object_height=config.optics.object_height,
            max_val=max_val,
            quantize=quantize,
            return_float=return_float,
            device_conv=torch_device,
            is_torch=use_torch,
        )

    mask_hwc = np.moveaxis(_to_numpy(mask), 0, -1).astype(np.float32)
    if filepath is not None:
        filepath = Path(filepath)
        if filepath.suffix != ".npy":
            raise ValueError(f"PSF filepath must end in .npy, got {filepath}")
        # save_image(psf_np, str(filepath))
        np.save(filepath, psf_np)

    psf_metadata = {
        "array_contract": "DHWC",
        "pattern_sha256": hashlib.sha256(
            np.ascontiguousarray(pattern).tobytes()
        ).hexdigest(),
        "pattern_shape": pattern_original_shape,
        "programmed_pattern_shape": list(programmed_pattern.shape),
        "mask_shape": list(mask_hwc.shape),
        "psf_shape": list(psf_np.shape),
        "psf_dtype": str(psf_np.dtype),
        "psf_min": float(psf_np.min()),
        "psf_max": float(psf_np.max()),
        "psf_sum": float(psf_np.sum(dtype=np.float64)),
        "rotate_angle_deg": rotate_angle,
        "vertical_shift_sensor_px": vertical_shift,
        "horizontal_shift_sensor_px": horizontal_shift,
        "vertical_shift_psf_px": vertical_shift_px,
        "horizontal_shift_psf_px": horizontal_shift_px,
        "scene2mask_m": scene2mask,
        "mask2sensor_m": mask2sensor,
        "sensor": str(config.optics.sensor),
        "slm": str(config.optics.slm),
        "downsample": int(config.optics.downsample),
        "grayscale": bool(config.grayscale),
    }
    return psf, simulator, psf_metadata


@contextmanager
def _deterministic_backend_seed(seed: int) -> Iterator[None]:
    numpy_state = np.random.get_state()
    np.random.seed(seed)
    if torch is None:
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
        return

    cuda_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        np.random.set_state(numpy_state)


def _sample_object_height(value: Any, rng: np.random.Generator) -> float:
    is_range = not isinstance(value, (str, bytes)) and hasattr(value, "__len__")
    if is_range:
        if len(value) != 2:
            raise ValueError(
                "optics.object_height range must contain exactly two values"
            )
        low, high = float(value[0]), float(value[1])
        if high < low:
            raise ValueError("optics.object_height upper bound must be >= lower bound")
        return float(rng.uniform(low, high))
    return float(value)


def forward_single_sample(
    image: Any,
    *,
    psf: Any,
    simulator: Any,
    config: Any,
    seed: int,
) -> dict[str, Any]:
    psf_array = _to_numpy(psf)
    if (
        psf_array.ndim != 4
        or psf_array.shape[0] != 1
        or psf_array.shape[-1] not in (1, 3)
    ):
        raise ValueError(f"PSF must have shape (1, H, W, C), got {psf_array.shape}")

    max_val = float(_config_get(config.optics, "max_val", 255))
    output_dtype = str(_config_get(config, "output_dtype", "uint8"))
    scene = _prepare_scene(
        image,
        channels=int(psf_array.shape[-1]),
        max_val=max_val,
        grayscale=bool(config.grayscale),
    )

    scene_backend: Any = scene
    if bool(config.use_torch):
        if torch is None:
            raise RuntimeError("config.use_torch=true but PyTorch is not installed")
        dtype = torch.float32 if str(config.dtype) == "float32" else torch.float64
        scene_backend = torch.from_numpy(scene).to(
            device=str(config.torch_device), dtype=dtype
        )

    sample_rng = np.random.default_rng(seed)
    object_height = _sample_object_height(config.optics.object_height, sample_rng)
    previous_object_height = simulator.object_height
    simulator.object_height = object_height
    try:
        with _deterministic_backend_seed(seed):
            measurement, target = simulator.propagate_image(
                scene_backend, return_object_plane=True
            )
    finally:
        simulator.object_height = previous_object_height

    measurement_hwc = _as_hwc(measurement, name="measurement")
    target_hwc = _as_hwc(target, name="target")
    measurement_out = convert_to_target_type(
        output_dtype, measurement_hwc, source_max=max_val
    )
    target_out = convert_to_target_type(output_dtype, target_hwc, source_max=max_val)

    return {
        "measurement": measurement_out,
        "ground_truth": target_out,
        "metadata": {
            "seed": int(seed),
            "object_height_m": object_height,
            "array_contract": "HWC",
            "output_dtype": output_dtype,
            "output_range": [0.0, 1.0] if output_dtype == "float32" else [0, 255],
            "measurement_shape": list(measurement_out.shape),
            "ground_truth_shape": list(target_out.shape),
        },
    }
