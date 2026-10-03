"""Protocol invariants shared by preparation, execution and aggregation."""

import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
SEEDS = (42, 52, 62)
ARMS = ("scratch", "gopro")
METRICS = ("PSNR", "SSIM", "LPIPS")
GOPRO_SHA256 = "12c6f5c8053cdede1e37d442ebf0824817007c32214c1b51b29e50636db39354"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extend_split(original, images, train_count=16384, extension_seed=20260907):
    """Append unused eligible scenes; preserve all old IDs and list ordering."""
    from src.datasets.mirflickr import (
        is_external_real_test_scene,
        mirflickr_source_index,
        validate_mirflickr_splits,
    )

    validate_mirflickr_splits(original)
    old = original["splits"]
    if train_count <= len(old["train"]):
        raise ValueError("The expanded training set must be strictly larger")
    used = {mirflickr_source_index(p) for part in old.values() for p in part}
    if not used <= set(images):
        raise ValueError("Some original images are absent from the source inventory")
    eligible = sorted(
        i for i in images if i not in used and not is_external_real_test_scene(i)
    )
    count = train_count - len(old["train"])
    if count > len(eligible):
        raise ValueError(
            "Not enough eligible unused images; holdouts cannot be reassigned"
        )
    selected = np.random.default_rng(extension_seed).permutation(eligible)[:count]
    extended = copy.deepcopy(original)
    extended["splits"]["train"].extend(str(images[int(i)]) for i in selected)
    validate_mirflickr_splits(extended)
    assert extended["splits"]["validation"] == old["validation"]
    assert extended["splits"]["test"] == old["test"]
    assert extended["splits"]["train"][: len(old["train"])] == old["train"]
    return extended


def load_reference(arm, seed, gopro_path):
    cfg = OmegaConf.load(HERE / "references" / f"{arm}-seed{seed}.yaml")
    return resolve_reference(cfg, gopro_path)


def resolve_reference(config, gopro_path):
    cfg = OmegaConf.create(
        OmegaConf.to_container(config, resolve=False)
        if OmegaConf.is_config(config)
        else config
    )
    # Replace the recorded environment dependency explicitly before resolving.
    pretrained = cfg.initialization.name == "xrestormer-gopro-pretrained"
    if cfg.initialization.name not in (
        "xrestormer-gopro-pretrained",
        "xrestormer-scratch",
    ):
        raise ValueError("Unexpected initialisation in the archived configuration")
    cfg.initialization.checkpoint_path = str(gopro_path) if pretrained else None
    cfg.model.checkpoint_path = cfg.initialization.checkpoint_path
    resolved = OmegaConf.to_container(cfg, resolve=True)
    # Older wrappers accepted the hash as a constructor argument. The current
    # wrapper does not; this runner checks the same hash before every load.
    old_hash = resolved["model"].pop("checkpoint_sha256", None)
    if old_hash not in (None, GOPRO_SHA256):
        raise ValueError("Unexpected historical initialisation hash")
    resolved["model"]["_target_"] = "src.model.psf_free_xrestormer.PSFFreeXRestormer"
    for key, value in {
        "operator_prompt": "none",
        "operator_code_dim": 16,
        "operator_hidden_dim": 128,
        "output_normalization": "positive_max",
    }.items():
        resolved["model"].setdefault(key, value)
    return resolved


def make_training_config(reference, name, package, manifest, image_root):
    cfg = copy.deepcopy(reference)
    for part in ("train", "validation"):
        cfg["datasets"][part]["splits_path"] = str(manifest)
        cfg["datasets"][part]["root_dir"] = str(image_root)
    cfg["dataloader_builder"]["psf_cache"]["root_dir"] = str(package / "psf_cache")
    cfg["trainer"].update(
        save_dir=str(package / "training"),
        monitor="off",
        early_stop=-1,
        save_period=1,
        keep_last_checkpoints=2,
        save_initial_checkpoint=True,
        from_pretrained=None,
        resume_from=None,
        override=False,
        skip_oom=False,
    )
    cfg["writer"].update(
        run_name=name,
        mode="offline",
        group="scene-scaling-ss01",
        log_checkpoints=False,
        log_best_checkpoint_at_end=False,
        save_code=False,
        log_git_state=False,
        log_images=False,
    )
    cfg["writer"].pop("run_id", None)
    return cfg


def scientific_signature(config):
    """Ignore storage/checkpoint-selection settings, never optimisation or simulation."""
    cfg = copy.deepcopy(config)
    ds = cfg["datasets"]
    for value in ds.values():
        value.pop("root_dir", None)
        value.pop("splits_path", None)
    loader = cfg["dataloader_builder"]
    loader.get("psf_cache", {}).pop("root_dir", None)
    trainer = cfg["trainer"]
    for key in (
        "save_dir",
        "monitor",
        "early_stop",
        "save_period",
        "keep_last_checkpoints",
        "save_initial_checkpoint",
        "override",
        "from_pretrained",
        "resume_from",
    ):
        trainer.pop(key, None)
    return {
        key: cfg[key]
        for key in (
            "model",
            "datasets",
            "dataloader_builder",
            "simulator",
            "optimizer",
            "lr_scheduler",
            "loss_function",
            "metrics",
            "trainer",
        )
    }


def require_checkpoint_config(state, reference):
    """Validate the saved training recipe, not just a filename and step count."""
    actual = resolve_reference(state["config"], reference["model"]["checkpoint_path"])
    if scientific_signature(actual) != scientific_signature(reference):
        raise ValueError(
            "Checkpoint training recipe differs from the archived reference"
        )


def require_config_parity(new, control):
    if scientific_signature(new) != scientific_signature(control):
        raise ValueError("Scientific configuration differs from its historical control")
    tr, dl = new["trainer"], new["dataloader_builder"]
    assert tr["seed"] in SEEDS and tr["n_epochs"] == 5 and tr["epoch_len"] == 10000
    assert (
        tr["total_steps"] == dl["train_steps"] == new["lr_scheduler"]["T_max"] == 50000
    )
    assert dl["batch_size"] == 1 and dl["finite_mask_count"] == 100
    assert dl["validation_mask_count"] == dl["validation_scenes_per_mask"] == 32
    assert dl["evaluation_mask_seed"] == 42 and dl["validation_seed"] == 52
    assert tr["amp"] == {"enabled": True, "dtype": "bfloat16"}
    assert (
        new["trainer"]["from_pretrained"] is None
        and new["trainer"]["resume_from"] is None
    )


def sampler_audit(config, small_count, large_count):
    from src.datasets.on_the_fly import DigiCamMaskBatchSampler
    from src.digicam_synth.mask_protocol import get_mask_records

    dl = config["dataloader_builder"]
    records = get_mask_records(dl["train_mask_seed"], "train", 100)
    samplers = [
        DigiCamMaskBatchSampler(
            scene_count=n,
            batch_size=1,
            steps=50000,
            run_seed=config["trainer"]["seed"],
            mode="finite",
            mask_records=records,
        )
        for n in (small_count, large_count)
    ]
    hashes = [hashlib.sha256(), hashlib.sha256()]
    mask_hash = hashlib.sha256()
    unique = [set(), set()]
    count = 0
    for small, large in zip(*samplers):
        assert small[0]["mask_id"] == large[0]["mask_id"]
        assert small[0]["mask_seed"] == large[0]["mask_seed"]
        for i, batch in enumerate((small, large)):
            hashes[i].update(json.dumps(batch, sort_keys=True).encode())
            unique[i].add(batch[0]["scene_index"])
        mask_hash.update(str(small[0]["mask_seed"]).encode() + b"\n")
        count += 1
    assert count == 50000
    return {
        "steps": count,
        "same_mask_sequence": True,
        "mask_sequence_sha256": mask_hash.hexdigest(),
        "small_requests_sha256": hashes[0].hexdigest(),
        "large_requests_sha256": hashes[1].hexdigest(),
        "unique_scenes_selected": {
            str(n): len(s) for n, s in zip((small_count, large_count), unique)
        },
    }


def evaluation_grid(manifest, scenes=128):
    from src.datasets.mirflickr import mirflickr_source_index
    from src.datasets.on_the_fly import DigiCamValidationBatchSampler
    from src.digicam_synth.mask_protocol import get_mask_records

    paths = manifest["splits"]["validation"]
    sampler = DigiCamValidationBatchSampler(
        scene_count=len(paths),
        batch_size=1,
        mask_records=get_mask_records(42, "validation", 32),
        run_seed=52,
        scenes_per_mask=scenes,
    )
    return [
        {
            "sample_index": i,
            "mask_id": request[0]["mask_id"],
            "scene_id": f"mirflickr_{mirflickr_source_index(paths[request[0]['scene_index']]):05d}",
        }
        for i, request in enumerate(sampler)
    ]


def source_hashes():
    paths = [ROOT / "train.py", ROOT / "inference.py"]
    paths += sorted((ROOT / "src").rglob("*.py"))
    paths += sorted(HERE.glob("*.py"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def verify_package(package):
    plan = read_json(Path(package) / "plan.json")
    for name, expected in plan["source_hashes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(
                f"Source changed after preparation: {name}; prepare an isolated package again"
            )
    for name, expected in plan["file_hashes"].items():
        if sha256(Path(package) / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    return plan


def checkpoint_state(path, seed):
    from .safe_checkpoint import load_checkpoint

    state = load_checkpoint(path)
    if state.get("global_step") != 50000 or state.get("epoch") != 5:
        raise ValueError("Checkpoint is not the final standalone 50k endpoint")
    cfg = state.get("config")
    if (
        cfg is None
        or int(cfg["trainer"]["total_steps"]) != 50000
        or int(cfg["trainer"]["seed"]) != seed
    ):
        raise ValueError("Checkpoint training horizon or seed differs")
    if state.get("lr_scheduler", {}).get("T_max") != 50000:
        raise ValueError("A checkpoint from a longer schedule is not a 50k control")
    if sum(v.numel() for v in state["state_dict"].values()) != 26041208:
        raise ValueError("Unexpected X-Restormer state size")
    return state


def balanced_metrics(rows, grid):
    if len(rows) != len(grid):
        raise ValueError(f"Expected {len(grid)} evaluation rows, got {len(rows)}")
    expected = {(r["mask_id"], r["scene_id"]) for r in grid}
    actual = [(r["mask_id"], r["scene_id"]) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Evaluation scene/mask identities differ from the frozen grid")
    groups = {}
    for row in rows:
        groups.setdefault(row["mask_id"], []).append(row)
    values = {}
    for metric in METRICS:
        data = [
            sum(float(r[metric]) for r in group) / len(group)
            for group in groups.values()
        ]
        if not all(math.isfinite(v) for v in data):
            raise ValueError(f"Non-finite {metric}")
        values[metric] = sum(data) / len(data)
    return values
