#!/usr/bin/env python3
"""PSFF-REFINE-01: matched residual refinement and optional continuation control."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from compatibility_gradient_pilot import file_hash, peak, save_json, tensor_hash, two_way_interval


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(width, width, 3, padding=1))

    def forward(self, value):
        return value + .1 * self.body(value)


class Refinement(nn.Module):
    """Context coordinates are separate from image coordinates; global attention links them."""
    def __init__(self, context):
        super().__init__()
        if context not in {"self_context", "measurement_context"}:
            raise ValueError(context)
        self.context = context
        widths = (32, 64, 96, 128)
        self.head = nn.Conv2d(3, widths[0], 3, padding=1)
        self.enc = nn.ModuleList(nn.Sequential(ResidualBlock(w), ResidualBlock(w)) for w in widths)
        self.down = nn.ModuleList(nn.Conv2d(a, b, 3, stride=2, padding=1) for a,b in zip(widths[:-1], widths[1:]))
        self.up = nn.ModuleList(nn.Conv2d(a, b, 1) for a,b in zip(widths[:0:-1], widths[-2::-1]))
        self.dec = nn.ModuleList(nn.Sequential(nn.Conv2d(2*w, w, 3, padding=1), nn.SiLU(),
                                              ResidualBlock(w), ResidualBlock(w)) for w in widths[-2::-1])
        sensor, previous = [], 3
        for width in (16, 32, 64, 128):
            sensor.extend([nn.Conv2d(previous, width, 3, stride=2, padding=1), nn.GroupNorm(8,width), nn.SiLU()])
            previous = width
        self.sensor = nn.Sequential(*sensor)
        self.query_position = nn.Parameter(torch.zeros(1, 128, 25, 34))
        self.context_position = nn.Parameter(torch.zeros(1, 96, 128))
        self.query_norm = nn.LayerNorm(128)
        self.context_norm = nn.LayerNorm(128)
        self.attention = nn.MultiheadAttention(128, 4, dropout=0., batch_first=True)
        self.tail = nn.Conv2d(32, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, base, measurement):
        if tuple(base.shape[1:]) != (3,200,266) or tuple(measurement.shape[1:]) != (3,380,507):
            raise ValueError("Expected full sensor and fixed ROI")
        if self.context == "self_context":
            context = F.pad(base, (100,141,80,100))
        else:
            context = measurement
        tokens = F.adaptive_avg_pool2d(self.sensor(context), (8,12)).flatten(2).transpose(1,2)
        tokens = self.context_norm(tokens + self.context_position)
        value = self.head(base)
        skips = []
        for i, encoder in enumerate(self.enc):
            value = encoder(value)
            if i < len(self.down):
                skips.append(value)
                value = self.down[i](value)
        position = F.interpolate(self.query_position, size=value.shape[-2:], mode="bilinear", align_corners=False)
        query = self.query_norm((value + position).flatten(2).transpose(1,2))
        attended, _ = self.attention(query, tokens, tokens, need_weights=False)
        value = value + attended.transpose(1,2).reshape_as(value)
        for up, decoder, skip in zip(self.up, self.dec, reversed(skips)):
            value = up(F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False))
            value = decoder(torch.cat((value, skip), 1))
        return base + .1 * torch.tanh(self.tail(value)).float()


def evaluate_summary(rows, names, masks, scenes):
    lookup = {(r["mask_order"],r["scene_order"],r["arm"]):r for r in rows}
    if len(lookup) != len(names)*masks*scenes:
        raise ValueError("Incomplete/duplicate evaluation grid")
    metrics = ("PSNR", "SSIM", "LPIPS")
    result = {"status":"complete", "metrics":{}, "contrasts":{}, "confirmation_diagnostics":{}}
    for arm in names:
        result["metrics"][arm] = {k:float(np.mean([lookup[m,s,arm][k] for m in range(masks) for s in range(scenes)])) for k in metrics}
    pairs = [(arm,"baseline") for arm in names if arm != "baseline"]
    if "measurement_context" in names:
        pairs += [("measurement_context","self_context"),("measurement_context","wrong_context")]
    for arm, reference in pairs:
        label = f"{arm}_minus_{reference}"
        result["contrasts"][label] = {}
        result["confirmation_diagnostics"][label] = {}
        for metric in metrics:
            grid = np.array([[lookup[m,s,arm][metric]-lookup[m,s,reference][metric] for s in range(scenes)] for m in range(masks)])
            result["contrasts"][label][metric] = two_way_interval(grid)
            result["confirmation_diagnostics"][label][metric] = two_way_interval(grid[masks//2:,scenes//2:])
    def signal(label, threshold):
        if label not in result["contrasts"]:
            return False
        c = result["contrasts"][label]
        return bool(c["PSNR"]["mean"] >= threshold and c["PSNR"]["lo"] > 0
                    and c["SSIM"]["mean"] >= -.002 and c["LPIPS"]["mean"] <= .005)
    result["h1_signal"] = any(signal(f"{arm}_minus_baseline", .10) for arm in ("self_context","measurement_context"))
    result["h2_signal"] = signal("measurement_context_minus_self_context", .05)
    reference = {"PSNR":14.923134836368263,"SSIM":.3443276314283139,"LPIPS":.5560499958810396}
    result["baseline_parity_delta"] = {m:result["metrics"]["baseline"][m]-v for m,v in reference.items()}
    result["baseline_parity_pass"] = (masks == 32 and scenes == 32 and all(abs(result["baseline_parity_delta"][m])<tol for m,tol in {"PSNR":.01,"SSIM":.001,"LPIPS":.001}.items()))
    return result


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    base = torch.rand(1,3,200,266)
    measurement = torch.rand(1,3,380,507)
    a = Refinement("self_context")
    b = copy.deepcopy(a)
    b.context = "measurement_context"
    assert torch.equal(a(base,measurement),base)
    assert torch.equal(b(base,measurement),base)
    # Activate head to test actual information paths, beyond zero initialization.
    nn.init.normal_(a.tail.weight,std=.01)
    b.load_state_dict(a.state_dict())
    assert torch.equal(a(base,measurement), a(base,torch.zeros_like(measurement)))
    assert not torch.allclose(b(base,measurement), b(base,torch.zeros_like(measurement)))
    x = b(base,measurement)
    (x-(base*.9)).square().mean().backward()
    assert b.sensor[0].weight.grad is not None and b.sensor[0].weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    print("PASS: exact baseline initialization, isolated self-context, active measurement path, finite gradients, equal capacity",flush=True)


def run(args):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Expose exactly one CUDA device")
    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/"status.json",{"status":"starting"})
    started=time.monotonic()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from src.datasets.data_utils import get_dataloaders
    from src.loss.reconstruction import ReconstructionLoss
    from src.metrics.reconstruction import PSNRMetric, SSIMMetric, LPIPSMetric

    base_run=Path("saved/scale100k-xrest-gopro-finite-100-seed42-508a878-r3")
    config_path=base_run/"config.yaml"
    checkpoint_path=base_run/"checkpoint-epoch10.pth"
    config=OmegaConf.load(config_path)
    config.initialization.checkpoint_path=None
    config.model.checkpoint_path=None
    config.trainer.total_steps=args.steps
    config.trainer.seed=args.seed
    builder=config.dataloader_builder
    builder.batch_size=args.batch_size; builder.num_workers=args.workers
    builder.validation_mask_count=args.val_masks; builder.validation_scenes_per_mask=args.val_scenes
    builder.train_mask_seed=42; builder.evaluation_mask_seed=42; builder.psf_cache.warmup=False
    resolved=OmegaConf.to_container(config,resolve=True)
    if resolved["protocol"]["version"]!="corrected-v1" or resolved["dataloader_builder"]["simulation_mode"]!="roi_convolution":
        raise ValueError("Wrong simulation protocol")
    (output/"source_config.yaml").write_text(config_path.read_text())
    OmegaConf.save(OmegaConf.create(resolved),output/"resolved_config.yaml")
    provenance={"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                "checkpoint_sha256":file_hash(checkpoint_path),"source_config_sha256":file_hash(config_path),
                "script_sha256":file_hash(Path(__file__)),"utility_sha256":file_hash(Path(__file__).with_name("compatibility_gradient_pilot.py")),
                "torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),
                "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
                "examples_per_arm":args.steps*args.batch_size,"loss":"MSE + LPIPS-VGG, peak normalized",
                "source_hashes":{str(p):file_hash(ROOT/p) for p in (Path("src/datasets/on_the_fly.py"),Path("src/digicam_synth/pipeline.py"),Path("src/model/psf_free_xrestormer.py"),Path("src/loss/reconstruction.py"),Path("src/metrics/reconstruction.py"),Path("manifests/mirflickr25k_splits.json"))}}
    loaders,transforms=get_dataloaders(config,"cuda")
    if transforms:
        raise ValueError("Unexpected batch transforms")
    base=instantiate(config.model).cuda().eval().requires_grad_(False)
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    base.load_state_dict(checkpoint["state_dict"],strict=True)
    del checkpoint
    if args.mode=="refine":
        self_context=Refinement("self_context").cuda()
        measurement_context=copy.deepcopy(self_context)
        measurement_context.context="measurement_context"
        models={"self_context":self_context,"measurement_context":measurement_context}
        learning_rate=1e-4
    else:
        models={"continued":copy.deepcopy(base).requires_grad_(True)}
        learning_rate=1e-5
    optimizers={name:torch.optim.Adam(model.parameters(),lr=learning_rate) for name,model in models.items()}
    schedulers={name:torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.steps,eta_min=1e-6) for name,optimizer in optimizers.items()}
    provenance["trainable_parameters"]={name:sum(p.numel() for p in model.parameters()) for name,model in models.items()}
    provenance["learning_rate"]=learning_rate
    save_json(output/"provenance.json",provenance)
    writer=None
    try:
        import wandb
        writer=wandb.init(project="lensless-imaging",entity="had-2005-hse-university",mode="offline",name=output.name,
                          group="psff-refine-01",dir=str(output),config=provenance)
    except ImportError:
        pass
    criterion=ReconstructionLoss(mse_weight=1.,lpips_weight=1.,lpips_net="vgg",normalize_by_max=True).cuda()
    # The optional learned perceptual net is built by the first loss call.
    for model in models.values(): model.train()
    save_json(output/"status.json",{"status":"training","mode":args.mode})
    with (output/"training.jsonl").open("w",buffering=1) as stream:
        for step,batch in enumerate(loaders["train"],start=1):
            y,target=batch["measurement"].cuda(),batch["target"].cuda()
            if tuple(y.shape[1:])!=(3,380,507) or tuple(target.shape[1:])!=(3,200,266):
                raise ValueError("Wrong batch shapes")
            if args.mode=="refine":
                with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                    b=peak(torch.cat([base(measurement=item[None])["prediction"].float() for item in y]))
            values={}
            for name,model in models.items():
                optimizer=optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                if args.mode=="refine":
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        prediction=model(b,y)
                    losses=criterion(prediction.float(),target.float())
                    loss=losses["loss"]
                    if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
                    loss.backward()
                    values.update({f"{name}/{k}":float(v.detach()) for k,v in losses.items()})
                else:
                    # Same effective batch/exposure, bounded activation memory.
                    accumulated={}
                    for i in range(len(y)):
                        with torch.autocast("cuda",dtype=torch.bfloat16):
                            prediction=model(measurement=y[i:i+1])["prediction"].float()
                        losses=criterion(prediction,target[i:i+1])
                        if not torch.isfinite(losses["loss"]): raise RuntimeError("Nonfinite loss")
                        (losses["loss"]/len(y)).backward()
                        for key,value in losses.items(): accumulated[key]=accumulated.get(key,0.)+float(value.detach())/len(y)
                    values.update({f"{name}/{key}":value for key,value in accumulated.items()})
                nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step(); schedulers[name].step()
            if step==1 or step%50==0 or step==args.steps:
                record={"step":step,"seconds":time.monotonic()-started,**values}
                stream.write(json.dumps(record)+"\n"); print(json.dumps(record),flush=True)
                if writer: writer.log(record,step=step)
    for name,model in models.items():
        torch.save({"state_dict":model.state_dict(),"provenance":provenance,"arm":name,"steps":args.steps},output/f"{name}_final.pth")
        model.eval().requires_grad_(False)
    del optimizers,schedulers,criterion
    save_json(output/"status.json",{"status":"evaluating"})
    metrics={"PSNR":PSNRMetric(normalize_by_max=True),"SSIM":SSIMMetric(normalize_by_max=True),
             "LPIPS":LPIPSMetric(net_type="vgg",device="cuda",normalize_by_max=True)}
    rows,mask_order,scene_order=[],{},{}
    qualitative=[]
    with (output/"per_sample.csv").open("w",newline="") as stream:
        csv_writer=None
        for index,batch in enumerate(loaders["validation"],start=1):
            y,target=batch["measurement"].cuda(),batch["target"].cuda()
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                raw_base=torch.cat([base(measurement=item[None])["prediction"].float() for item in y])
                b=peak(raw_base)
                predictions={"baseline":b}
                for name,model in models.items():
                    if args.mode=="refine": predictions[name]=model(b,y).float()
                    else: predictions[name]=torch.cat([model(measurement=item[None])["prediction"].float() for item in y])
                if args.mode=="refine":
                    if len(y)<2 or len(set(batch["mask_id"]))!=1: raise ValueError("Wrong context control needs >=2 scenes of one mask")
                    predictions["wrong_context"]=models["measurement_context"](b,y.roll(1,0)).float()
            metadata=[]
            for i,sample_id in enumerate(batch["sample_id"]):
                mid,sid=batch["mask_id"][i],batch["scene_id"][i]
                mask_order.setdefault(mid,len(mask_order)); scene_order.setdefault(sid,len(scene_order))
                meta={"sample_id":sample_id,"mask_id":mid,"scene_id":sid,"source_index":int(batch["source_index"][i]),
                      "mask_order":mask_order[mid],"scene_order":scene_order[sid],"measurement_sha256":tensor_hash(y[i]),
                      "target_sha256":tensor_hash(target[i]),"base_sha256":tensor_hash(raw_base[i])}
                metadata.append(meta)
                if mask_order[mid] in (0,16) and scene_order[sid]<4:
                    qualitative.append({"metadata":meta,"target":target[i].cpu(),
                                        **{name:value[i].cpu() for name,value in predictions.items()}})
            for name,prediction in predictions.items():
                with torch.no_grad(): values={key:metric.per_image(prediction,target).cpu().tolist() for key,metric in metrics.items()}
                for i,meta in enumerate(metadata):
                    row={**meta,"arm":name,**{key:value[i] for key,value in values.items()}}
                    if not all(np.isfinite(row[key]) for key in values): raise RuntimeError("Nonfinite evaluation")
                    if csv_writer is None:
                        csv_writer=csv.DictWriter(stream,fieldnames=list(row)); csv_writer.writeheader()
                    csv_writer.writerow(row); rows.append(row)
            stream.flush()
            if index%16==0 or index==1:
                print(json.dumps({"evaluation_batch":index,"seconds":time.monotonic()-started}),flush=True)
    result=evaluate_summary(rows,list(predictions),args.val_masks,args.val_scenes)
    result.update({"seconds":time.monotonic()-started,"peak_vram_bytes":torch.cuda.max_memory_allocated(),
                   "technical_smoke":args.steps!=3000 or args.val_masks!=32 or args.val_scenes!=32,
                   "scope":"Development grid, one frozen base checkpoint; no closed test opened."})
    if not result["technical_smoke"] and not result["baseline_parity_pass"]:
        result.update(status="requires_baseline_audit",h1_signal=False,h2_signal=False)
    torch.save(qualitative,output/"qualitative_fixed.pt")
    save_json(output/"summary.json",result)
    save_json(output/"status.json",{"status":result["status"],"h1_signal":result["h1_signal"],"h2_signal":result["h2_signal"]})
    if writer:
        writer.summary.update(result); writer.finish()
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--mode",choices=("refine","continue"),default="refine")
    parser.add_argument("--steps",type=int,default=3000)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--val-masks",type=int,default=32)
    parser.add_argument("--val-scenes",type=int,default=32)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.self_test: self_test(); return
    if args.output is None or args.steps<=0 or args.batch_size<2 or not (2<=args.val_masks<=32 and 2<=args.val_scenes<=32):
        parser.error("Provide output, positive steps, batch>=2, and masks/scenes in [2,32]")
    try:
        run(args)
    except Exception as error:
        # Do not alter another run if the requested output path already existed.
        if not isinstance(error,FileExistsError) and args.output.exists():
            save_json(args.output/"failure.json",{"status":"failed","type":type(error).__name__,"message":str(error)})
        raise


if __name__=="__main__":
    main()
