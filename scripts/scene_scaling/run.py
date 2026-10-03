"""Run gated SS01 jobs explicitly; preparation never starts a GPU process."""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from .common import (
    GOPRO_SHA256,
    METRICS,
    ROOT,
    balanced_metrics,
    checkpoint_state,
    read_csv,
    read_json,
    require_checkpoint_config,
    require_config_parity,
    save_json,
    sha256,
    verify_package,
)


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    header = f"{value.dtype}:{list(value.shape)}:".encode()
    return hashlib.sha256(
        header + value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes()
    ).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode() + tensor_hash(value).encode())
    return h.hexdigest()


def load_config(package, relative):
    return OmegaConf.load(Path(package) / relative)


def config_digest(package, relative):
    return sha256(Path(package) / relative)


def require_control_result(package, plan, run):
    job = next(
        j
        for j in plan["evaluations"]
        if j["run_id"] == run["id"] and j["training_scenes"] == 4096
    )
    result = validated_evaluation(package, plan, job)
    if result is None or not result.get("legacy_parity_pass"):
        raise ValueError(
            "Evaluate the existing 4096-scene control first; legacy-grid parity is required"
        )
    return result


def validate_initialisation(config):
    path = config.model.get("checkpoint_path")
    if path is not None and sha256(path) != GOPRO_SHA256:
        raise ValueError("Unexpected external pretraining weights")


def preflight(package, plan, run, device="cuda"):
    import torch
    from hydra.utils import instantiate

    from src.datasets.data_utils import get_dataloaders
    from src.utils.init_utils import set_random_seed

    control_result = (
        require_control_result(package, plan, run) if device == "cuda" else None
    )
    cfg = load_config(package, run["config"])
    control = load_config(package, run["control_config"])
    require_config_parity(
        OmegaConf.to_container(cfg, resolve=True),
        OmegaConf.to_container(control, resolve=True),
    )
    validate_initialisation(cfg)
    # Dataset size must not change initial model tensors, in either initialisation arm.
    hashes = []
    for c in (control, cfg):
        set_random_seed(c.trainer.seed)
        loaders, transforms = get_dataloaders(c, device)
        assert not transforms and len(loaders["train"].dataset.scenes) in (4096, 16384)
        model = instantiate(c.model)
        hashes.append(state_hash(model))
        del model, loaders
    assert hashes[0] == hashes[1], "Initial tensors differ between scene-count arms"
    set_random_seed(cfg.trainer.seed)
    local = copy.deepcopy(cfg)
    # A technical probe runs in-process. Scientific training retains the archived worker count.
    local.dataloader_builder.num_workers = 0
    loaders, _ = get_dataloaders(local, device)
    model = instantiate(cfg.model).to(device).train()
    assert state_hash(model) == hashes[1]
    criterion = instantiate(cfg.loss_function).to(device)
    it = iter(loaders["train"])
    batches = []
    for _ in range(2):
        batch = next(it)
        assert tuple(batch["measurement"].shape) == (1, 3, 380, 507)
        assert tuple(batch["target"].shape) == (1, 3, 200, 266)
        model.zero_grad(set_to_none=True)
        started = time.monotonic()
        with torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
        ):
            prediction = model(measurement=batch["measurement"].to(device))[
                "prediction"
            ]
            loss = criterion(prediction=prediction, target=batch["target"].to(device))[
                "loss"
            ]
        loss.backward()
        grad = math.sqrt(
            sum(
                float(p.grad.detach().float().square().sum())
                for p in model.parameters()
                if p.grad is not None
            )
        )
        if not torch.isfinite(loss) or not math.isfinite(grad) or grad == 0:
            raise ValueError(
                "Preflight loss/gradient is non-finite or the gradient is zero"
            )
        batches.append(
            {
                "scene_id": batch["scene_id"][0],
                "mask_id": batch["mask_id"][0],
                "measurement_sha256": tensor_hash(batch["measurement"]),
                "target_sha256": tensor_hash(batch["target"]),
                "loss": float(loss.detach()),
                "grad_norm": grad,
                "forward_backward_seconds": time.monotonic() - started,
            }
        )
    assert state_hash(model) == hashes[1], "A no-update preflight changed the weights"
    proof = {
        "status": "passed",
        "device": device,
        "initial_state_sha256": hashes[1],
        "initial_tensors_equal_control": True,
        "optimizer_updates": 0,
        "config_sha256": config_digest(package, run["config"]),
        "control_checkpoint_sha256": (
            control_result["checkpoint_sha256"] if control_result else None
        ),
        "plan_sha256": sha256(Path(package) / "plan.json"),
        "batches": batches,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
    }
    save_json(Path(package) / "preflight" / f"{run['id']}-{device}.json", proof)
    print(json.dumps(proof, indent=2))


def train_job(package, plan, run, resume=False):
    import torch
    from hydra.utils import instantiate

    from src.datasets.data_utils import get_dataloaders
    from src.utils.init_utils import set_random_seed, setup_saving_and_logging

    from .safe_checkpoint import safe_trainer_class

    control_result = require_control_result(package, plan, run)
    proof = read_json(Path(package) / "preflight" / f"{run['id']}-cuda.json")
    assert proof["status"] == "passed" and proof["device"] == "cuda"
    assert proof["plan_sha256"] == sha256(Path(package) / "plan.json")
    assert proof["config_sha256"] == config_digest(package, run["config"])
    assert proof["control_checkpoint_sha256"] == control_result["checkpoint_sha256"]
    cfg = load_config(package, run["config"])
    validate_initialisation(cfg)
    out = Path(cfg.trainer.save_dir) / cfg.writer.run_name
    final = Path(run["new_checkpoint"])
    if final.exists():
        raise FileExistsError(
            "A final checkpoint already exists; evaluate it instead of retraining"
        )
    if resume:
        candidates = sorted(
            out.glob("checkpoint-epoch*.pth"),
            key=lambda p: int(p.stem.split("epoch")[1]),
        )
        if not candidates:
            raise ValueError("No completed-epoch checkpoint is available for resume")
        saved = torch.load(
            str(candidates[-1]), map_location="cpu", weights_only=True, mmap=True
        )
        if (
            not 0 < saved["global_step"] < 50000
            or saved["lr_scheduler"]["T_max"] != 50000
        ):
            raise ValueError("Invalid resume budget")
        if (
            not {"rng_state", "dataloader_rng_state", "optimizer", "sampler_step"}
            <= saved.keys()
        ):
            raise ValueError("Resume state is incomplete")
        if saved["sampler_step"] != saved["global_step"]:
            raise ValueError("Skipped training examples make this resume unsuitable")
        if saved["global_step"] != saved["epoch"] * cfg.trainer.epoch_len:
            raise ValueError("Resume checkpoint is not a completed-epoch state")
        require_checkpoint_config(saved, OmegaConf.to_container(cfg, resolve=True))
        if str(saved["config"]["datasets"]["train"]["splits_path"]) != str(
            cfg.datasets.train.splits_path
        ):
            raise ValueError(
                "Resume checkpoint belongs to a different training manifest"
            )
        del saved
        cfg.trainer.resume_from = candidates[-1].name
    set_random_seed(cfg.trainer.seed)
    logger = setup_saving_and_logging(cfg)
    writer = instantiate(cfg.writer, logger, OmegaConf.to_container(cfg, resolve=True))
    started = time.monotonic()
    try:
        loaders, transforms = get_dataloaders(cfg, "cuda")
        model = instantiate(cfg.model).cuda()
        if state_hash(model) != proof["initial_state_sha256"]:
            raise ValueError(
                "Actual initial tensors differ from preflight; no training updates performed"
            )
        loss = instantiate(cfg.loss_function).cuda()
        metrics = instantiate(cfg.metrics)
        optimizer = instantiate(
            cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters())
        )
        scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)
        trainer = safe_trainer_class()(
            model=model,
            criterion=loss,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=cfg,
            device="cuda",
            dataloaders=loaders,
            epoch_len=cfg.trainer.epoch_len,
            logger=logger,
            writer=writer,
            batch_transforms=transforms,
            skip_oom=False,
        )
        trainer.train()
    finally:
        writer.finish()
    state = checkpoint_state(final, run["seed"])
    assert state["sampler_step"] == 50000
    save_json(
        out / "completion.json",
        {
            "status": "trained",
            "global_step": 50000,
            "examples_presented": 50000,
            "checkpoint_sha256": sha256(final),
            "plan_sha256": sha256(Path(package) / "plan.json"),
            "elapsed_seconds_this_invocation": time.monotonic() - started,
            "resumed": resume,
        },
    )


def validated_evaluation(package, plan, job):
    marker = Path(package) / job["output"] / "completion.json"
    if not marker.exists():
        return None
    value = read_json(marker)
    if value["status"] != "evaluated" or value["plan_sha256"] != sha256(
        Path(package) / "plan.json"
    ):
        raise ValueError(
            "Evaluation proof does not match the current immutable package"
        )
    if not job["checkpoint"] or sha256(job["checkpoint"]) != value["checkpoint_sha256"]:
        raise ValueError("Evaluated checkpoint was changed or removed")
    csv_path = marker.parent / "validation/per_image.csv"
    if sha256(csv_path) != value["per_image_sha256"]:
        raise ValueError("Per-image evaluation results changed")
    return value


def evaluate(package, plan, job):
    import torch
    from hydra.utils import instantiate

    from src.datasets.data_utils import get_dataloaders
    from src.trainer import Inferencer
    from src.utils.init_utils import set_random_seed

    if validated_evaluation(package, plan, job):
        print(f"Already evaluated and verified: {job['id']}")
        return
    if not job["checkpoint"]:
        raise FileNotFoundError(
            f"Missing final checkpoint for {job['id']}; recover it and prepare a new package"
        )
    run = next(r for r in plan["runs"] if r["id"] == job["run_id"])
    digest = sha256(job["checkpoint"])
    if job["training_scenes"] == 4096 and digest != run["control"]["archive"]["sha256"]:
        raise ValueError("Historical checkpoint changed after preparation")
    state = checkpoint_state(job["checkpoint"], job["seed"])
    cfg = load_config(package, job["config"])
    require_checkpoint_config(state, OmegaConf.to_container(cfg, resolve=True))
    if job["training_scenes"] == 16384:
        stored_manifest = str(state["config"]["datasets"]["train"]["splits_path"])
        if stored_manifest != str(cfg.datasets.train.splits_path):
            raise ValueError("The new checkpoint was trained with a different manifest")
    cfg.model.checkpoint_path = None
    cfg.initialization.checkpoint_path = None
    cfg.dataloader_builder.evaluation_only = True
    cfg.dataloader_builder.validation_scenes_per_mask = 128
    # Keep batch 1 and BF16 model inference, matching historical trainer validation.
    cfg.inferencer = {
        "seed": job["seed"],
        "device": "cuda",
        "device_tensors": ["measurement", "target"],
        "output_type": "reconstruction",
        "example_indices": [0, 1, 2, 3],
        "from_pretrained": job["checkpoint"],
        "skip_model_load": True,
    }
    cfg.metrics.device = "cuda"
    out = Path(package) / job["output"]
    out.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, out / "resolved_config.yaml", resolve=True)
    set_random_seed(job["seed"])
    loaders, transforms = get_dataloaders(cfg, "cuda")
    model = instantiate(cfg.model).cuda()
    model.load_state_dict(state["state_dict"], strict=True)
    del state

    class BF16Model(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, **batch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.network(**batch)

    class AuditedInferencer(Inferencer):
        def _process_reconstruction_batch(self, batch, metrics, part, output_offset):
            result = super()._process_reconstruction_batch(
                batch, metrics, part, output_offset
            )
            n = len(batch["measurement"])
            for i, row in enumerate(self.per_image_rows[part][-n:]):
                row["measurement_sha256"] = tensor_hash(batch["measurement"][i])
                row["target_sha256"] = tensor_hash(batch["target"][i])
            return result

    torch.cuda.reset_peak_memory_stats()
    evaluator = AuditedInferencer(
        model=BF16Model(model),
        config=cfg,
        device="cuda",
        dataloaders=loaders,
        save_path=out,
        metrics=instantiate(cfg.metrics),
        batch_transforms=transforms,
        skip_model_load=True,
    )
    started = time.monotonic()
    evaluator.run_inference()
    rows = read_csv(out / "validation/per_image.csv")
    primary = balanced_metrics(rows, read_json(Path(package) / "grid128.json"))
    legacy_grid = read_json(Path(package) / "grid32.json")
    legacy_keys = {(r["mask_id"], r["scene_id"]) for r in legacy_grid}
    legacy = balanced_metrics(
        [r for r in rows if (r["mask_id"], r["scene_id"]) in legacy_keys], legacy_grid
    )
    result = {
        "status": "evaluated",
        "id": job["id"],
        "precision": "BF16 model / FP32 metrics",
        "primary128": primary,
        "legacy32": legacy,
        "checkpoint_sha256": digest,
        "per_image_sha256": sha256(out / "validation/per_image.csv"),
        "plan_sha256": sha256(Path(package) / "plan.json"),
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "legacy_parity_pass": None,
    }
    if job["training_scenes"] == 4096:
        deltas = {m: legacy[m] - run["control"]["legacy32_metrics"][m] for m in METRICS}
        result["legacy_parity_deltas"] = deltas
        result["legacy_parity_pass"] = all(
            abs(v) <= (0.01 if m == "PSNR" else 0.001) for m, v in deltas.items()
        )
        if not result["legacy_parity_pass"]:
            save_json(out / "parity_failure.json", result)
            raise ValueError(
                "Historical control parity failed; dependent training is blocked"
            )
    save_json(out / "completion.json", result)
    print(json.dumps(result, indent=2))


def assert_gpu_idle(gpu):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    memory, utilisation = [int(x.strip()) for x in output.strip().split(",")]
    # Utilisation can briefly retain the previous child's last sample after exit.
    if memory <= 512 and utilisation > 10:
        time.sleep(2)
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        memory, utilisation = [int(x.strip()) for x in output.strip().split(",")]
    if memory > 512 or utilisation > 10:
        raise RuntimeError(
            f"GPU {gpu} is occupied ({memory} MiB, {utilisation}%); no process was stopped"
        )


def worker(package, plan, gpu, resume):
    package = Path(package)
    queue = plan["queue"].get(str(gpu))
    if queue is None:
        raise ValueError("This package assigns work only to GPU 3, 4 and 5")
    lock_dir = package / "locks"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f"gpu{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in queue:
            run = next(r for r in plan["runs"] if r["id"] == name)
            old = next(
                j
                for j in plan["evaluations"]
                if j["run_id"] == name and j["training_scenes"] == 4096
            )
            new = next(
                j
                for j in plan["evaluations"]
                if j["run_id"] == name and j["training_scenes"] == 16384
            )
            stages = [("evaluate", "--evaluation", old["id"])]
            if not Path(run["new_checkpoint"]).exists():
                stages += [("preflight", "--run", name), ("train", "--run", name)]
            stages += [("evaluate", "--evaluation", new["id"])]
            for stage in stages:
                assert_gpu_idle(gpu)
                command = [
                    sys.executable,
                    "-m",
                    "scripts.scene_scaling.run",
                    stage[0],
                    "--package",
                    str(package),
                    "--gpu",
                    str(gpu),
                    *stage[1:],
                ]
                if (
                    stage[0] == "train"
                    and resume
                    and (package / "training" / name).exists()
                ):
                    command.append("--resume")
                save_json(
                    package / f"worker_gpu{gpu}.json",
                    {"status": "running", "run": name, "stage": stage[0]},
                )
                print(" ".join(command), flush=True)
                try:
                    subprocess.run(command, cwd=ROOT, check=True)
                    if stage[0] == "evaluate":
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "scripts.scene_scaling.summarize",
                                "--package",
                                str(package),
                            ],
                            cwd=ROOT,
                            check=True,
                        )
                except subprocess.CalledProcessError as error:
                    save_json(
                        package / f"worker_gpu{gpu}.json",
                        {
                            "status": "failed",
                            "run": name,
                            "stage": stage[0],
                            "exit_code": error.returncode,
                        },
                    )
                    raise
        save_json(
            package / f"worker_gpu{gpu}.json", {"status": "complete", "runs": queue}
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "train", "evaluate", "worker"))
    p.add_argument("--package", default="outputs/scene_scaling_20260907")
    p.add_argument("--run")
    p.add_argument("--evaluation")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument(
        "--resume", action="store_true", help="Explicitly resume from a completed epoch"
    )
    p.add_argument(
        "--cpu-probe",
        action="store_true",
        help="Technical preflight only; never qualifies a CUDA training run",
    )
    args = p.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError(f"Run from the isolated project root: {ROOT}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("HF_HOME", str(ROOT / "data/huggingface"))
    os.environ["WANDB_MODE"] = "offline"
    package = Path(args.package).resolve()
    plan = verify_package(package)
    if args.command == "worker":
        worker(package, plan, args.gpu, args.resume)
        return
    import torch

    torch.set_num_threads(4)
    if args.cpu_probe and args.command != "preflight":
        raise ValueError("CPU probes cannot train or supply scientific evaluation")
    if not args.cpu_probe:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA GPU with BF16 support is required")
    if args.command == "evaluate":
        job = next(j for j in plan["evaluations"] if j["id"] == args.evaluation)
        evaluate(package, plan, job)
    else:
        run = next(r for r in plan["runs"] if r["id"] == args.run)
        if args.command == "preflight":
            preflight(package, plan, run, "cpu" if args.cpu_probe else "cuda")
        else:
            train_job(package, plan, run, args.resume)


if __name__ == "__main__":
    main()
