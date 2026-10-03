"""Train a fresh MNIST repeat beside its frozen, historical source tree.

Copy this file and safe_checkpoint.py into a directory containing the archived
src/, reference_config.yaml and SOURCE.json. No existing run is overwritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    memory = subprocess.check_output(
        ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    if int(memory.strip()) > 512:
        raise RuntimeError("Selected GPU is occupied")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("WANDB_MODE", "offline")

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from safe_checkpoint import safe_trainer_class, digest
    from src.datasets.data_utils import get_dataloaders
    from src.utils.init_utils import set_random_seed, setup_saving_and_logging

    torch.set_num_threads(4)
    cfg = OmegaConf.load(ROOT / "reference_config.yaml")
    source = json.loads((ROOT / "SOURCE.json").read_text())
    cfg.trainer.seed = args.seed
    cfg.dataloader_builder.run_seed = args.seed
    cfg.writer.run_name = f"mnist-{source['mode']}-10k-seed{args.seed}" + ("-smoke" if args.smoke else "")
    cfg.writer.mode = "offline"
    cfg.writer.log_git_state = False
    cfg.writer.log_best_checkpoint_at_end = False
    cfg.writer.log_checkpoints = False
    cfg.trainer.save_dir = str(ROOT / "saved")
    cfg.trainer.monitor = "off"
    cfg.trainer.override = False
    cfg.dataloader_builder.psf_cache.root_dir = str(ROOT / "psf_cache")
    for part in ("train", "validation"):
        cfg.datasets[part].root_dir = "/home/hadhad/project/Lensless-imaging/data/raw/mnist"
        cfg.datasets[part].download = False
    assert cfg.model.checkpoint_path is None
    assert cfg.trainer.resume_from is None and cfg.trainer.from_pretrained is None
    assert cfg.trainer.total_steps == cfg.lr_scheduler.T_max == 10000
    assert cfg.trainer.n_epochs == 4 and cfg.trainer.epoch_len == 2500
    assert cfg.dataloader_builder.train_mask_seed == 42 and cfg.dataloader_builder.evaluation_mask_seed == 52
    assert cfg.dataloader_builder.batch_size == 4 and not cfg.trainer.amp.enabled
    assert not cfg.trainer.skip_oom
    if args.smoke:
        cfg.dataloader_builder.validation_mask_count = 2
        cfg.dataloader_builder.validation_scenes_per_mask = 4

    set_random_seed(args.seed)
    logger = setup_saving_and_logging(cfg)
    writer = instantiate(cfg.writer, logger, OmegaConf.to_container(cfg, resolve=True))
    loaders, transforms = get_dataloaders(cfg, "cuda")
    model = instantiate(cfg.model).cuda()
    criterion = instantiate(cfg.loss_function).cuda()
    metrics = instantiate(cfg.metrics)
    optimizer = instantiate(cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)
    out = Path(cfg.trainer.save_dir) / cfg.writer.run_name
    write_json(out / "provenance.json", {
        **source, "seed": args.seed, "gpu": args.gpu, "technical_smoke": args.smoke,
        "pytorch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "changes_from_reference": ["training and sampler seed", "run identity and output paths", "offline logging", "final endpoint with safe checkpoint serialization", "reuse installed MNIST without download"],
    })

    class Recorder(safe_trainer_class()):
        def _train_epoch(self, epoch):
            result = super()._train_epoch(epoch)
            with (out / "history.jsonl").open("a") as f:
                f.write(json.dumps({"epoch": epoch, "step": self.global_step, **result}, allow_nan=False) + "\n")
            self.final_metrics = result
            return result

        def process_batch(self, batch, metrics):
            result = super().process_batch(batch, metrics)
            if self.is_train and (self.global_step == 0 or (self.global_step + 1) % 50 == 0):
                row = {"step": self.global_step + 1, "loss": float(result["loss"].detach()),
                       "grad_norm": float(result["grad_norm"]), "lr": scheduler.get_last_lr()[0]}
                with (out / "trace.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
            return result

        def _log_batch(self, batch_idx, batch, mode="train"):
            super()._log_batch(batch_idx, batch, mode)
            if mode == "validation":
                panel = {k: batch[k].detach().cpu() for k in ("measurement", "target", "prediction")}
                panel.update({k: list(batch[k]) for k in ("scene_id", "mask_id")})
                panel.update({"step": self.global_step, "seed": args.seed, "selection": "last validation batch"})
                torch.save(panel, out / f"examples-step{self.global_step}.pth")

    trainer = Recorder(model=model, criterion=criterion, metrics=metrics, optimizer=optimizer,
                       lr_scheduler=scheduler, config=cfg, device="cuda", dataloaders=loaders,
                       epoch_len=2500, logger=logger, writer=writer, batch_transforms=transforms, skip_oom=False)
    started = time.monotonic()
    try:
        if args.smoke:
            trainer.epoch_len = 2
            trainer._train_epoch(1)
            trainer._save_checkpoint(1)
        else:
            trainer.train()
        expected = 2 if args.smoke else 10000
        epoch = 1 if args.smoke else 4
        assert trainer.global_step == trainer.sampler_step == expected
        final = out / f"checkpoint-epoch{epoch}.pth"
        state = torch.load(str(final), map_location="cpu", weights_only=True, mmap=True)
        assert state["global_step"] == expected and state["lr_scheduler"]["T_max"] == 10000
        assert all(torch.equal(v.detach().cpu(), state["state_dict"][k]) for k, v in model.state_dict().items())
        write_json(out / "complete.json", {
            "status": "complete", "technical_smoke": args.smoke, "mode": source["mode"], "seed": args.seed,
            "steps": expected, "sampler_steps": trainer.sampler_step, "schedule_horizon": scheduler.T_max,
            "checkpoint": str(final), "checkpoint_sha256": digest(final),
            "elapsed_seconds": time.monotonic() - started, "metrics": trainer.final_metrics,
        })
    finally:
        writer.finish()


if __name__ == "__main__":
    main()
