"""Create immutable SS01 manifests, six training configs and evaluation jobs."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from .common import (
    GOPRO_SHA256,
    HERE,
    ROOT,
    SEEDS,
    checkpoint_state,
    evaluation_grid,
    extend_split,
    load_reference,
    make_training_config,
    read_json,
    require_checkpoint_config,
    require_config_parity,
    sampler_audit,
    save_json,
    sha256,
    source_hashes,
)


def find_control(control, explicit=None, reference=None):
    candidates = [explicit] if explicit else control["checkpoint_candidates"]
    rejected = []
    for name in candidates:
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            continue
        try:
            state = checkpoint_state(path, control["seed"])
            if reference is not None:
                require_checkpoint_config(state, reference)
            del state
            digest = sha256(path)
            expected = control.get("expected_checkpoint_sha256")
            if expected and expected != digest:
                raise ValueError(
                    "Checkpoint hash differs from the archived final endpoint"
                )
            return {
                "status": "available",
                "path": str(path.resolve()),
                "sha256": digest,
                "global_step": 50000,
                "rejected_candidates": rejected,
            }
        except (ValueError, KeyError, RuntimeError) as error:
            rejected.append({"path": str(path), "reason": str(error)})
    return {
        "status": "missing_final_checkpoint",
        "path": None,
        "rejected_candidates": rejected,
    }


def prepare(
    output,
    image_root,
    original_manifest,
    gopro_checkpoint,
    checkpoint_map=None,
    seeds=SEEDS,
    gpus=(3, 4, 5),
):
    from src.datasets.mirflickr import (
        discover_mirflickr_images,
        validate_mirflickr_splits,
    )

    seeds, gpus = tuple(seeds), tuple(gpus)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        raise ValueError("Choose distinct seeds from 42, 52 and 62")
    if not gpus or len(set(gpus)) != len(gpus) or not set(gpus) <= {3, 4, 5}:
        raise ValueError("Choose distinct GPUs from 3, 4 and 5")
    output, image_root = Path(output).resolve(), Path(image_root).resolve()
    if output.exists():
        raise FileExistsError(
            "Use a new package directory; existing packages are immutable"
        )
    original = read_json(original_manifest)
    if {k: len(v) for k, v in original["splits"].items()} != {
        "train": 4096,
        "validation": 128,
        "test": 256,
    }:
        raise ValueError("SS01 requires the existing 4096/128/256 split")
    images = discover_mirflickr_images(image_root)
    if len(images) != 25000:
        raise ValueError("Expected the complete 25,000-image MIRFLICKR inventory")
    validate_mirflickr_splits(original, root_dir=image_root)
    relative = {i: p.relative_to(image_root).as_posix() for i, p in images.items()}
    expanded = extend_split(original, relative)
    validate_mirflickr_splits(expanded, root_dir=image_root)
    output.mkdir(parents=True)
    small = output / "manifests/scenes4096.json"
    large = output / "manifests/scenes16384.json"
    save_json(small, original)
    save_json(large, expanded)
    original_hash = sha256(original_manifest)
    save_json(
        output / "split_validation.json",
        {
            "original_manifest": str(Path(original_manifest).resolve()),
            "original_sha256": original_hash,
            "extension_seed": 20260907,
            "old_train_is_prefix": True,
            "validation_and_test_lists_unchanged": True,
            "external_real_test_excluded": True,
            "counts": {
                "small_train": 4096,
                "large_train": 16384,
                "validation": 128,
                "test": 256,
            },
            "new_test_evaluation": False,
        },
    )
    save_json(output / "grid128.json", evaluation_grid(original, 128))
    save_json(output / "grid32.json", evaluation_grid(original, 32))
    gopro_checkpoint = Path(gopro_checkpoint).resolve()
    gopro_available = gopro_checkpoint.is_file()
    if gopro_available and sha256(gopro_checkpoint) != GOPRO_SHA256:
        raise ValueError(
            "The supplied GoPro weights do not match the archived initialisation"
        )
    overrides = read_json(checkpoint_map) if checkpoint_map else {}
    controls = [c for c in read_json(HERE / "controls.json") if c["seed"] in seeds]
    runs, jobs, readiness, schedules = [], [], {}, {}
    for control in controls:
        arm, seed = control["arm"], control["seed"]
        ref = load_reference(arm, seed, gopro_checkpoint)
        name = f"ss01-xrest-{arm}-scenes16384-masks100-50k-seed{seed}"
        reference_name = f"ss01-control-{arm}-scenes4096-masks100-50k-seed{seed}"
        cfg = make_training_config(ref, name, output, large, image_root)
        old_cfg = make_training_config(ref, reference_name, output, small, image_root)
        require_config_parity(cfg, ref)
        require_config_parity(cfg, old_cfg)
        config_file = output / "configs" / f"{name}.yaml"
        control_file = output / "configs" / f"{reference_name}.yaml"
        config_file.parent.mkdir(exist_ok=True)
        OmegaConf.save(OmegaConf.create(cfg), config_file, resolve=True)
        OmegaConf.save(OmegaConf.create(old_cfg), control_file, resolve=True)
        if seed not in schedules:
            schedules[seed] = sampler_audit(cfg, 4096, 16384)
        found = find_control(control, overrides.get(control["id"]), ref)
        readiness[control["id"]] = found
        run = {
            "id": name,
            "arm": arm,
            "seed": seed,
            "config": str(config_file.relative_to(output)),
            "control_config": str(control_file.relative_to(output)),
            "control_id": control["id"],
            "control": {**control, "archive": found},
            "new_checkpoint": str(output / "training" / name / "checkpoint-epoch5.pth"),
            "steps": 50000,
            "training_scenes": 16384,
        }
        runs.append(run)
        for n, config_path, checkpoint in (
            (4096, control_file, found["path"]),
            (16384, config_file, run["new_checkpoint"]),
        ):
            jobs.append(
                {
                    "id": f"{arm}-seed{seed}-scenes{n}",
                    "run_id": name,
                    "arm": arm,
                    "seed": seed,
                    "training_scenes": n,
                    "config": str(config_path.relative_to(output)),
                    "checkpoint": checkpoint,
                    "output": f"evaluations/{arm}-seed{seed}-scenes{n}",
                    "precision": "bf16",
                    "steps": 50000,
                    "validation_scenes": 128,
                    "mask_count": 32,
                    "expected_pairs": 4096,
                }
            )
    save_json(output / "sampler_validation.json", schedules)
    # Match arms on one GPU per seed; no existing queues or GPU 0–2 are touched.
    queue = {}
    for i, seed in enumerate(seeds):
        queue.setdefault(str(gpus[i % len(gpus)]), []).extend(
            r["id"] for r in runs if r["seed"] == seed
        )
    plan = {
        "experiment": "SS01",
        "schema_version": 1,
        "status": "prepared_not_started",
        "training_seeds": list(seeds),
        "statistical_scope": (
            "single-seed pilot" if len(seeds) == 1 else "multiseed comparison"
        ),
        "new_trainings": len(runs),
        "total_new_optimizer_steps": sum(r["steps"] for r in runs),
        "gopro": {
            "path": str(gopro_checkpoint),
            "sha256": GOPRO_SHA256,
            "available": gopro_available,
        },
        "image_root": str(image_root),
        "runs": runs,
        "evaluations": jobs,
        "queue": queue,
        "source_hashes": source_hashes(),
        "file_hashes": {
            str(p.relative_to(output)): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
        "endpoint": "final 50000 steps; no validation selection",
        "primary_grid": "grid128.json",
        "legacy_grid": "grid32.json",
        "test_access": "no synthetic or official real test evaluation",
    }
    save_json(output / "plan.json", plan)
    save_json(
        output / "readiness.json",
        {
            "configs_and_splits": "passed",
            "gopro_available": gopro_available,
            "controls": readiness,
            "cuda_preflight": "not_run",
            "training_started": False,
        },
    )
    if sha256(original_manifest) != original_hash:
        raise RuntimeError("Original manifest changed during preparation")
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="outputs/scene_scaling_20260907")
    p.add_argument("--image-root", default="data/raw/mirflickr25k/extracted")
    p.add_argument("--original-manifest", default="manifests/mirflickr25k_splits.json")
    p.add_argument(
        "--gopro-checkpoint", default="model_weights/xrestormer/net_g_latest.pth"
    )
    p.add_argument(
        "--checkpoint-map",
        help="Optional JSON: control ID -> recovered final checkpoint",
    )
    p.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    p.add_argument("--gpus", nargs="+", type=int, choices=(3, 4, 5), default=[3, 4, 5])
    args = p.parse_args()
    plan = prepare(
        args.output,
        args.image_root,
        args.original_manifest,
        args.gopro_checkpoint,
        args.checkpoint_map,
        args.seeds,
        args.gpus,
    )
    print(
        f"Prepared {len(plan['runs'])} new trainings and {len(plan['evaluations'])} evaluations."
    )
    for run in plan["runs"]:
        print(run["control_id"], run["control"]["archive"]["status"])
    print(
        "GPU execution has not started. See readiness.json and the experiment protocol."
    )


if __name__ == "__main__":
    main()
