"""Read legacy tensor weights without executing arbitrary checkpoint pickle objects."""

import hashlib
import json
import os
import pickle
import pickletools
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def metadata_literals(data):
    """Inspect pickle instructions; never call pickle.load or any GLOBAL/REDUCE."""
    ops = list(pickletools.genops(data))
    values = {}
    boundary = None
    for index, (opcode, argument, position) in enumerate(ops):
        if opcode.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE"):
            if argument == "optimizer" and boundary is None:
                boundary = position
            if argument in ("epoch", "global_step", "sampler_step", "T_max"):
                j = index + 1
                while j < len(ops) and ops[j][0].name in (
                    "BINPUT",
                    "LONG_BINPUT",
                    "MEMOIZE",
                    "PUT",
                ):
                    j += 1
                if j < len(ops) and ops[j][0].name in (
                    "BININT",
                    "BININT1",
                    "BININT2",
                    "INT",
                    "LONG1",
                    "LONG4",
                ):
                    values.setdefault(argument, set()).add(int(ops[j][1]))
    if boundary is None or any(len(v) != 1 for v in values.values()):
        raise ValueError("Unrecognised or ambiguous legacy checkpoint metadata")
    return boundary, {key: next(iter(value)) for key, value in values.items()}


def load_checkpoint(path, cache_dir=None):
    """Use weights_only=True, extracting only a validated tensor prefix for legacy files.

    Legacy config/RNG objects are never deserialised. Their tensor-only prefix
    must decode using PyTorch's restricted loader and have exactly three keys.
    The archived YAML sidecar supplies configuration, checked separately by SS01.
    """
    path = Path(path)
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except pickle.UnpicklingError:
        pass
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith("/data.pkl")
        )
        if archive.getinfo(metadata_name).file_size > 8_000_000:
            raise ValueError("Checkpoint metadata exceeds the limit")
        data = archive.read(metadata_name)
        boundary, values = metadata_literals(data)
        if not {"epoch", "global_step", "sampler_step", "T_max"} <= values.keys():
            raise ValueError("Missing legacy endpoint metadata")
        # These project checkpoints store arch, epoch, state_dict, then optimizer.
        # SETITEMS closes the outer dictionary; STOP ends the restricted payload.
        prefix = data[:boundary] + b"u."
        root = (
            Path(cache_dir)
            if cache_dir
            else Path(__file__).resolve().parents[2] / "outputs/ss01_safe_weights"
        )
        root.mkdir(parents=True, exist_ok=True)
        source_hash = digest(path)
        target = root / f"{source_hash}.pth"
        proof = target.with_suffix(".json")
        valid_cache = False
        if target.exists() and proof.exists():
            recorded = json.loads(proof.read_text())
            valid_cache = recorded.get("source_sha256") == source_hash and recorded.get(
                "tensor_archive_sha256"
            ) == digest(target)
        if not valid_cache:
            with tempfile.NamedTemporaryFile(
                dir=root, suffix=".pth", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                with zipfile.ZipFile(
                    temporary_path, "w", compression=zipfile.ZIP_STORED
                ) as clean:
                    for member in archive.infolist():
                        if member.filename == metadata_name:
                            clean.writestr(member.filename, prefix)
                        else:
                            with archive.open(member) as src, clean.open(
                                member.filename, "w"
                            ) as dst:
                                shutil.copyfileobj(src, dst, length=1024 * 1024)
                decoded = torch.load(
                    str(temporary_path),
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
                if set(decoded) != {"arch", "epoch", "state_dict"}:
                    raise ValueError("Unexpected legacy tensor-prefix structure")
                if decoded["epoch"] != values["epoch"] or not isinstance(
                    decoded["state_dict"], dict
                ):
                    raise ValueError("Inconsistent legacy tensor metadata")
                if not all(
                    isinstance(v, torch.Tensor) for v in decoded["state_dict"].values()
                ):
                    raise ValueError("Non-tensor entry in the model state")
                del decoded
                os.replace(temporary_path, target)
                proof.write_text(
                    json.dumps(
                        {
                            "source_sha256": source_hash,
                            "tensor_archive_sha256": digest(target),
                            "method": "static prefix extraction plus weights_only=True",
                        },
                        indent=2,
                    )
                    + "\n"
                )
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
    state = torch.load(str(target), map_location="cpu", weights_only=True, mmap=True)
    state.update({key: values[key] for key in ("epoch", "global_step", "sampler_step")})
    state["lr_scheduler"] = {"T_max": values["T_max"]}
    state["config"] = OmegaConf.load(path.parent / "config.yaml")
    state["legacy_metadata_source"] = "static integer fields plus archived config.yaml"
    return state


def encode_rng(state):
    state = dict(state)
    rng = state["numpy"]
    state["numpy"] = (rng[0], rng[1].tolist(), int(rng[2]), int(rng[3]), float(rng[4]))
    return state


def decode_rng(state):
    state = dict(state)
    rng = state["numpy"]
    state["numpy"] = (
        rng[0],
        np.asarray(rng[1], dtype=np.uint32),
        int(rng[2]),
        int(rng[3]),
        float(rng[4]),
    )
    return state


def safe_trainer_class():
    from src.trainer import Trainer

    class SafeTrainer(Trainer):
        def _save_checkpoint(
            self, epoch, save_best=False, only_best=False, filename=None
        ):
            if save_best:
                raise ValueError(
                    "SS01 uses final endpoints, never best-checkpoint selection"
                )
            state = {
                "arch": type(self.model).__name__,
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "monitor_best": self.mnt_best,
                "not_improved_count": self.not_improved_count,
                "global_step": self.global_step,
                "sampler_step": self.sampler_step,
                "rng_state": encode_rng(self._rng_state()),
                "dataloader_rng_state": self._dataloader_rng_state(),
                "grad_scaler": (
                    self.grad_scaler.state_dict()
                    if self.grad_scaler is not None
                    else None
                ),
                "config": OmegaConf.to_container(self.config, resolve=True),
                "safe_checkpoint_schema": 1,
            }
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self.checkpoint_dir / (filename or f"checkpoint-epoch{epoch}.pth")
            temporary = path.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(path)
            self.logger.info(f"Saving safe checkpoint: {path}")
            self._prune_periodic_checkpoints()
            return str(path)

        def _resume_checkpoint(self, path):
            from .common import require_checkpoint_config

            state = torch.load(
                str(path), map_location="cpu", weights_only=True, mmap=True
            )
            if state.get("safe_checkpoint_schema") != 1:
                raise ValueError(
                    "Resume requires a checkpoint produced by this safe runner"
                )
            require_checkpoint_config(
                state, OmegaConf.to_container(self.config, resolve=True)
            )
            self.model.load_state_dict(state["state_dict"], strict=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            self.start_epoch = int(state["epoch"]) + 1
            self.global_step = int(state["global_step"])
            self.sampler_step = int(state["sampler_step"])
            self.mnt_best = state["monitor_best"]
            self.not_improved_count = state["not_improved_count"]
            if self.grad_scaler is not None and state["grad_scaler"] is not None:
                self.grad_scaler.load_state_dict(state["grad_scaler"])
            self._set_sampler_step(self.sampler_step)
            self._restore_dataloader_rng_state(state["dataloader_rng_state"])
            self._restore_rng_state(decode_rng(state["rng_state"]))
            self.logger.info(f"Resumed safe checkpoint at step {self.global_step}")

    return SafeTrainer
