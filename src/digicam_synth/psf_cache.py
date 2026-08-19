import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def _json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value)


class PSFCache:
    def __init__(self, config, simulator_config):
        config = dict(config or {})
        self.mode = str(config.get("mode", "off"))
        self.request_modes = set(config.get("request_modes", ["finite"]))
        self.warmup = bool(config.get("warmup", False))

        if self.mode not in {"off", "read_only", "read_write"}:
            raise ValueError("psf_cache.mode must be off, read_only or read_write")
        if not self.request_modes <= {"finite", "infinite"}:
            raise ValueError("psf_cache.request_modes contains an unknown mode")
        if self.warmup and self.mode == "off":
            raise ValueError("psf_cache.warmup needs an enabled cache")

        resolved = OmegaConf.to_container(simulator_config, resolve=True)
        self.config_hash = _sha256(_json_bytes(resolved))
        root_dir = Path(config.get("root_dir", "data/psf_cache"))
        self.cache_dir = root_dir / self.config_hash

    def applies_to(self, request_mode):
        return self.mode != "off" and request_mode in self.request_modes

    def path(self, seed):
        return self.cache_dir / f"{int(seed)}.npz"

    def load(self, seed, request_mode):
        if not self.applies_to(request_mode):
            return None

        path = self.path(seed)
        if not path.exists():
            if self.mode == "read_only":
                raise FileNotFoundError(f"PSF is missing from read-only cache: {path}")
            return None

        try:
            with np.load(path, allow_pickle=False) as artifact:
                psf = np.ascontiguousarray(artifact["psf"])
                metadata = json.loads(artifact["metadata"].tobytes().decode("utf-8"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid PSF cache artifact: {path}") from error

        expected = {
            "config_hash": self.config_hash,
            "mask_seed": int(seed),
            "psf_sha256": _sha256(psf.tobytes()),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"PSF cache metadata mismatch for {path}: {key}")
        if psf.ndim != 4 or psf.shape[0] != 1:
            raise RuntimeError(f"Cached PSF must have DHWC shape: {path}")
        if not np.isfinite(psf).all() or float(psf.max()) <= 0:
            raise RuntimeError(f"Cached PSF has invalid values: {path}")
        return psf

    def save(self, seed, request_mode, psf):
        if not self.applies_to(request_mode) or self.mode != "read_write":
            return

        psf = _numpy(psf)
        metadata = {
            "config_hash": self.config_hash,
            "mask_seed": int(seed),
            "psf_sha256": _sha256(psf.tobytes()),
        }
        metadata_bytes = np.frombuffer(_json_bytes(metadata), dtype=np.uint8)
        path = self.path(seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez(temporary, psf=psf, metadata=metadata_bytes)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
