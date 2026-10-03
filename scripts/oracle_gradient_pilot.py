#!/usr/bin/env python3
"""PSFF-ORACLE-GRAD-01. True-PSF results are oracle controls, never PSF-free claims."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))

import numpy as np
import torch
from compatibility_gradient_pilot import ETAS,file_hash,peak,rms_unit,save_json,tensor_hash,two_way_interval


def direction(base,measurement,reprojector):
    # No clean target enters this function.
    candidate=base.detach().clone().requires_grad_(True)
    residual=(reprojector(candidate)-measurement).square().flatten(1).mean(1)
    gradient,=torch.autograd.grad(residual.sum(),candidate)
    if not torch.isfinite(gradient).all(): raise RuntimeError("Nonfinite physics gradient")
    return rms_unit(-gradient.detach())


def summarize(rows,masks,scenes):
    lookup={(r["mask_order"],r["scene_order"],r["arm"],r["eta"]):r for r in rows}
    if len(lookup)!=masks*scenes*3*len(ETAS): raise ValueError("Incomplete grid")
    def grid(arm,eta,metric,part):
        rr=range(masks//2) if part=="calibration" else range(masks//2,masks)
        cc=range(scenes//2) if part=="calibration" else range(scenes//2,scenes)
        return np.array([[lookup[m,s,arm,eta][metric]-lookup[m,s,"true_psf",0.][metric] for s in cc] for m in rr])
    result={"status":"complete","selected_eta":{},"calibration":{},"confirmation":{},"signal":{}}
    for arm in ("true_psf","mean_train_psf"):
        table=[]
        for eta in ETAS:
            d={k:float(grid(arm,eta,k,"calibration").mean()) for k in ("PSNR","SSIM","LPIPS")}
            table.append({"eta":eta,**d,"eligible":d["SSIM"]>=-.002 and d["LPIPS"]<=.005})
        chosen=max((r for r in table if r["eligible"]),key=lambda r:(r["PSNR"],-r["eta"]))["eta"]
        result["selected_eta"][arm]=chosen; result["calibration"][arm]=table
        c={k:two_way_interval(grid(arm,chosen,k,"confirmation")) for k in ("PSNR","SSIM","LPIPS")}
        result["confirmation"][arm]=c
        result["signal"][arm]=bool(c["PSNR"]["mean"]>=.10 and c["PSNR"]["lo"]>0 and c["SSIM"]["mean"]>=-.002 and c["LPIPS"]["mean"]<=.005)
    chosen=result["selected_eta"]["true_psf"]
    result["confirmation"]["wrong_psf"]={k:two_way_interval(grid("wrong_psf",chosen,k,"confirmation")) for k in ("PSNR","SSIM","LPIPS")}
    result["true_minus_wrong"]={k:two_way_interval(grid("true_psf",chosen,k,"confirmation")-grid("wrong_psf",chosen,k,"confirmation")) for k in ("PSNR","SSIM","LPIPS")}
    baseline={k:float(np.mean([r[k] for r in rows if r["arm"]=="true_psf" and r["eta"]==0.])) for k in ("PSNR","SSIM","LPIPS")}
    result["baseline_full_grid"]=baseline
    ref={"PSNR":14.923134836368263,"SSIM":.3443276314283139,"LPIPS":.5560499958810396}
    result["baseline_parity_delta"]={k:baseline[k]-v for k,v in ref.items()}
    result["baseline_parity_pass"]=(masks==32 and scenes==32 and all(abs(result["baseline_parity_delta"][k])<tol for k,tol in {"PSNR":.01,"SSIM":.001,"LPIPS":.001}.items()))
    return result


def self_test():
    from src.loss.cross_mask_physics import _ROIReprojector
    torch.manual_seed(9); torch.set_num_threads(2)
    psf=torch.rand(2,3,24,32); psf=psf/psf.flatten(1).norm(dim=1)[:,None,None,None]
    op=_ROIReprojector(psf,[2,3,16,20])
    target=torch.rand(2,3,16,20)
    measurement=op(target).detach()
    base=(target+.05*torch.randn_like(target)).clamp(0,1)
    before=(op(base)-measurement).square().mean()
    after=(op(base+.0001*direction(base,measurement,op))-measurement).square().mean()
    assert after<before,(before,after)
    quantized=torch.round(measurement*255)/255
    assert (quantized-measurement).abs().max()<=.5/255+1e-6
    print("PASS: physical gradient descent and quantization bound",flush=True)


def run(args):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from src.datasets.data_utils import get_dataloaders
    from src.datasets.on_the_fly import _prepare_convolution_psf
    from src.digicam_synth.mask_protocol import get_mask_records
    from src.loss.cross_mask_physics import _ROIReprojector
    from src.metrics.reconstruction import PSNRMetric,SSIMMetric,LPIPSMetric
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("Expose one CUDA device")
    args.output.mkdir(parents=True,exist_ok=False)
    output=args.output; started=time.monotonic()
    save_json(output/"status.json",{"status":"starting"})
    torch.manual_seed(42); torch.set_num_threads(4)
    config_path=Path("saved/scale100k-xrest-gopro-finite-100-seed42-508a878-r3/config.yaml")
    checkpoint_path=config_path.parent/"checkpoint-epoch10.pth"
    config=OmegaConf.load(config_path)
    config.initialization.checkpoint_path=None; config.model.checkpoint_path=None
    builder=config.dataloader_builder
    builder.evaluation_only=True; builder.return_psf=True
    builder.batch_size=4; builder.num_workers=2
    builder.validation_mask_count=args.val_masks; builder.validation_scenes_per_mask=args.val_scenes
    builder.train_mask_seed=42; builder.evaluation_mask_seed=42; builder.psf_cache.warmup=False
    resolved=OmegaConf.to_container(config,resolve=True)
    if resolved["protocol"]["version"]!="corrected-v1": raise ValueError("Wrong simulator")
    OmegaConf.save(OmegaConf.create(resolved),output/"resolved_config.yaml")
    provenance={"checkpoint_sha256":file_hash(checkpoint_path),"source_config_sha256":file_hash(config_path),
                "script_sha256":file_hash(Path(__file__)),"utility_sha256":file_hash(Path(__file__).with_name("compatibility_gradient_pilot.py")),
                "physics_sha256":file_hash(ROOT/"src/loss/cross_mask_physics.py"),
                "dataset_sha256":file_hash(ROOT/"src/datasets/on_the_fly.py"),
                "torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),
                "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"etas":ETAS,
                "scope":"True/wrong PSF are oracle diagnostics. Mean PSF is training-derived."}
    loaders,transforms=get_dataloaders(config,"cuda")
    if transforms: raise ValueError("Unexpected transforms")
    loader=loaders["validation"]; dataset=loader.dataset
    def load_psf(record):
        raw,_,_=dataset._get_mask({"mask_seed":int(record["mask_seed"]),"mode":"finite"})
        return _prepare_convolution_psf(raw).squeeze(0).movedim(-1,0).contiguous()
    training_records=get_mask_records(42,"train",100)
    mean_psf=None
    for record in training_records:
        psf=load_psf(record)
        mean_psf=psf.clone() if mean_psf is None else mean_psf+psf
    mean_psf=mean_psf/mean_psf.norm().clamp_min(1e-12)
    provenance["mean_psf_sha256"]=tensor_hash(mean_psf)
    provenance["train_mask_seeds"]=[int(r["mask_seed"]) for r in training_records]
    save_json(output/"provenance.json",provenance)
    wandb_run=None
    try:
        import wandb
        wandb_run=wandb.init(project="lensless-imaging",entity="had-2005-hse-university",mode="offline",
                             name=output.name,group="psff-oracle-grad-01",dir=str(output),config=provenance)
    except ImportError:
        pass
    records=get_mask_records(42,"validation",args.val_masks)
    order_by_id={r["mask_id"]:i for i,r in enumerate(records)}
    base=instantiate(config.model).cuda().eval().requires_grad_(False)
    state=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    base.load_state_dict(state["state_dict"],strict=True); del state
    metrics={"PSNR":PSNRMetric(normalize_by_max=True),"SSIM":SSIMMetric(normalize_by_max=True),"LPIPS":LPIPSMetric(net_type="vgg",device="cuda",normalize_by_max=True)}
    rows=[]; scene_order={}; worst_error=0.
    save_json(output/"status.json",{"status":"evaluating"})
    with (output/"per_sample.csv").open("w",newline="") as stream:
        writer=None
        for index,batch in enumerate(loader,start=1):
            y,target=batch["measurement"].cuda(),batch["target"].cuda()
            psf=batch["psf"].cuda()
            if len(set(batch["mask_id"]))!=1: raise ValueError("Expected one mask per batch")
            mask_order=order_by_id[batch["mask_id"][0]]
            wrong_record=records[(mask_order+1)%len(records)]
            wrong=load_psf(wrong_record).cuda()[None].expand_as(psf).contiguous()
            mean=mean_psf.cuda()[None].expand_as(psf).contiguous()
            operators={"true_psf":_ROIReprojector(psf,[80,100,200,266]),
                       "mean_train_psf":_ROIReprojector(mean,[80,100,200,266]),
                       "wrong_psf":_ROIReprojector(wrong,[80,100,200,266])}
            with torch.no_grad():
                error=float((operators["true_psf"](target)-y).abs().max())
                worst_error=max(worst_error,error)
                if error>.5/255+1e-5: raise RuntimeError(f"Forward parity failure: {error}")
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                raw_base=torch.cat([base(measurement=item[None])["prediction"].float() for item in y])
            b=peak(raw_base)
            directions={name:direction(b,y,operator) for name,operator in operators.items()}
            metadata=[]
            for i,sample_id in enumerate(batch["sample_id"]):
                sid=batch["scene_id"][i]; scene_order.setdefault(sid,len(scene_order))
                metadata.append({"sample_id":sample_id,"mask_id":batch["mask_id"][i],"scene_id":sid,"source_index":int(batch["source_index"][i]),
                                 "mask_order":mask_order,"scene_order":scene_order[sid],
                                 "measurement_sha256":tensor_hash(y[i]),"target_sha256":tensor_hash(target[i]),"base_sha256":tensor_hash(raw_base[i]),
                                 "true_psf_sha256":tensor_hash(psf[i]),"wrong_psf_sha256":tensor_hash(wrong[i]),"wrong_mask_id":wrong_record["mask_id"],"forward_error_max":error})
            cached=None
            for arm,d in directions.items():
                for eta in ETAS:
                    if eta==0 and cached is not None: values=cached
                    else:
                        with torch.no_grad():
                            candidate=b+eta*d
                            values={name:metric.per_image(candidate,target).cpu().tolist() for name,metric in metrics.items()}
                            values["true_reprojection_mse"]=(operators["true_psf"](candidate)-y).square().flatten(1).mean(1).cpu().tolist()
                        if eta==0: cached=values
                    for i,meta in enumerate(metadata):
                        row={**meta,"arm":arm,"eta":eta,**{k:v[i] for k,v in values.items()}}
                        if not all(np.isfinite(row[k]) for k in values): raise RuntimeError("Nonfinite metric")
                        if writer is None: writer=csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row); rows.append(row)
            stream.flush()
            if index==1 or index%16==0: print(json.dumps({"evaluation_batch":index,"seconds":time.monotonic()-started}),flush=True)
    result=summarize(rows,args.val_masks,args.val_scenes)
    result.update(seconds=time.monotonic()-started,peak_vram_bytes=torch.cuda.max_memory_allocated(),
                  max_forward_quantization_error=worst_error,technical_smoke=args.val_masks!=32 or args.val_scenes!=32)
    if not result["technical_smoke"] and not result["baseline_parity_pass"]:
        result["status"]="requires_baseline_audit"; result["signal"]={k:False for k in result["signal"]}
    save_json(output/"summary.json",result); save_json(output/"status.json",{"status":result["status"],"signal":result["signal"]})
    if wandb_run:
        wandb_run.summary.update(result); wandb_run.finish()
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--self-test",action="store_true")
    p.add_argument("--output",type=Path); p.add_argument("--val-masks",type=int,default=32); p.add_argument("--val-scenes",type=int,default=32)
    args=p.parse_args()
    if args.self_test: self_test(); return
    if args.output is None or not (2<=args.val_masks<=32 and 2<=args.val_scenes<=32): p.error("Output and grid sizes 2..32 required")
    run(args)


if __name__=="__main__":main()
