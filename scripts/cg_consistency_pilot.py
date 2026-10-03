#!/usr/bin/env python3
"""PSFF-CG-01: regularized linear correction; true PSF is an oracle control."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
import numpy as np
import torch
from torch.nn import functional as F
from compatibility_gradient_pilot import file_hash,peak,save_json,tensor_hash,two_way_interval

LAMBDAS=(1e-6,1e-5,1e-4,1e-3,1e-2)
PARAMETERS=((0.,0.),)+tuple((lam,a) for lam in LAMBDAS for a in (.25,1.))


class LinearROI:
    def __init__(self,psf,roi=(80,100,200,266)):
        from lensless.recon.rfft_convolve import RealFFTConvolve2D
        self.roi=tuple(roi); self.shape=psf.shape[-2:]
        self.conv=RealFFTConvolve2D(psf=psf.float().movedim(1,-1).unsqueeze(1),dtype=torch.float32)
        self.spectrum=self.conv._H.abs().square()
        self.scale=self.spectrum.flatten(1).amax(1)

    def canvas(self,x):
        top,left,height,width=self.roi; h,w=self.shape
        return F.pad(x,(left,w-left-width,top,h-top-height)).movedim(1,-1).unsqueeze(1)

    def roi_crop(self,x):
        top,left,height,width=self.roi
        return x[:,0,top:top+height,left:left+width,:].movedim(-1,1).contiguous()

    def forward(self,x):
        return self.conv.convolve(self.canvas(x)).squeeze(1).movedim(-1,1).contiguous()

    def adjoint(self,y):
        return self.roi_crop(self.conv.deconvolve(y.movedim(1,-1).unsqueeze(1)))

    def precondition(self,r,lam):
        padded=self.conv._pad(self.canvas(r))
        transformed=torch.fft.rfft2(padded,dim=(-3,-2))
        filtered=transformed/(self.spectrum+lam.view(-1,1,1,1,1))
        x=torch.fft.irfft2(filtered,s=self.conv._padded_shape[-3:-1],dim=(-3,-2))
        return self.roi_crop(self.conv._crop(x))


def dot(a,b): return (a*b).flatten(1).sum(1)
def times(a,x): return a[:,None,None,None]*x


@torch.no_grad()
def solve(operator,rhs,relative,iterations=20):
    lam=relative*operator.scale
    normal=lambda x:operator.adjoint(operator.forward(x))+times(lam,x)
    x=torch.zeros_like(rhs); r=rhs.clone()
    z=operator.precondition(r,lam); p=z.clone(); rho=dot(r,z)
    for _ in range(iterations):
        ap=normal(p); denominator=dot(p,ap).clamp_min(1e-30)
        step=rho/denominator
        x=x+times(step,p); r=r-times(step,ap)
        z=operator.precondition(r,lam); next_rho=dot(r,z)
        p=z+times(next_rho/rho.clamp_min(1e-30),p); rho=next_rho
    residual=(normal(x)-rhs).flatten(1).norm(dim=1)/rhs.flatten(1).norm(dim=1).clamp_min(1e-20)
    if not torch.isfinite(x).all() or not torch.isfinite(residual).all(): raise RuntimeError("Nonfinite PCG")
    return x,residual


def self_test():
    torch.manual_seed(71); torch.set_num_threads(2)
    for h,w,roi in ((24,32,(2,3,16,20)),(380,507,(80,100,200,266))):
        psf=torch.rand(1,3,h,w); psf=psf/psf.norm()
        op=LinearROI(psf,roi)
        x=torch.randn(1,3,roi[2],roi[3]); y=torch.randn(1,3,h,w)
        left=dot(op.forward(x),y); right=dot(x,op.adjoint(y))
        error=float((left-right).abs()/torch.maximum(left.abs(),right.abs()).clamp_min(1e-8))
        assert error<1e-3,(h,w,error)
        if h==24:
            truth=torch.rand_like(x); b=truth+.05*torch.randn_like(x)
            residual=op.forward(truth-b); rhs=op.adjoint(residual)
            delta,normal_error=solve(op,rhs,.01)
            before=dot(residual,residual)
            after=dot(op.forward(delta)-residual,op.forward(delta)-residual)+.01*op.scale*dot(delta,delta)
            assert (after<before).all() and ((b+delta-truth).square().mean()<(b-truth).square().mean())
            assert normal_error.max()<.05,normal_error
    print("PASS: full sensor adjoint, positive regularization, PCG objective decrease",flush=True)


def summarize(rows,masks,scenes):
    lookup={(r['mask_order'],r['scene_order'],r['arm'],r['relative_lambda'],r['alpha']):r for r in rows}
    if len(lookup)!=masks*scenes*2*len(PARAMETERS): raise ValueError('Incomplete grid')
    def grid(arm,param,metric,part):
        rr=range(masks//2) if part=='calibration' else range(masks//2,masks)
        cc=range(scenes//2) if part=='calibration' else range(scenes//2,scenes)
        return np.array([[lookup[(m,s,arm)+tuple(param)][metric]-lookup[m,s,'true_psf',0.,0.][metric] for s in cc] for m in rr])
    result={'status':'complete','selected':{},'calibration':{},'confirmation':{},'signal':{}}
    for arm in ('true_psf','mean_train_psf'):
        table=[]
        for param in PARAMETERS:
            d={k:float(grid(arm,param,k,'calibration').mean()) for k in ('PSNR','SSIM','LPIPS')}
            table.append({'relative_lambda':param[0],'alpha':param[1],**d,'eligible':d['SSIM']>=-.002 and d['LPIPS']<=.005})
        chosen=max((r for r in table if r['eligible']),key=lambda r:(r['PSNR'],-r['alpha']))
        param=(chosen['relative_lambda'],chosen['alpha'])
        result['selected'][arm]={'relative_lambda':param[0],'alpha':param[1]}; result['calibration'][arm]=table
        c={k:two_way_interval(grid(arm,param,k,'confirmation')) for k in ('PSNR','SSIM','LPIPS')}
        result['confirmation'][arm]=c
        result['signal'][arm]=bool(c['PSNR']['mean']>=.1 and c['PSNR']['lo']>0 and c['SSIM']['mean']>=-.002 and c['LPIPS']['mean']<=.005)
    result['baseline_full_grid']={k:float(np.mean([r[k] for r in rows if r['arm']=='true_psf' and r['alpha']==0])) for k in ('PSNR','SSIM','LPIPS')}
    reference={'PSNR':14.923134836368263,'SSIM':.3443276314283139,'LPIPS':.5560499958810396}
    result['baseline_parity_delta']={k:result['baseline_full_grid'][k]-v for k,v in reference.items()}
    result['baseline_parity_pass']=masks==32 and scenes==32 and all(abs(result['baseline_parity_delta'][k])<tol for k,tol in {'PSNR':.01,'SSIM':.001,'LPIPS':.001}.items())
    return result


def run(args):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from src.datasets.data_utils import get_dataloaders
    from src.datasets.on_the_fly import _prepare_convolution_psf
    from src.digicam_synth.mask_protocol import get_mask_records
    from src.metrics.reconstruction import PSNRMetric,SSIMMetric,LPIPSMetric
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Expose one CUDA device')
    args.output.mkdir(parents=True,exist_ok=False); out=args.output; started=time.monotonic()
    torch.manual_seed(42); torch.set_num_threads(4)
    config_path=Path('saved/scale100k-xrest-gopro-finite-100-seed42-508a878-r3/config.yaml')
    checkpoint_path=config_path.parent/'checkpoint-epoch10.pth'
    config=OmegaConf.load(config_path); config.initialization.checkpoint_path=None; config.model.checkpoint_path=None
    d=config.dataloader_builder; d.evaluation_only=True; d.return_psf=True; d.batch_size=4; d.num_workers=2
    d.validation_mask_count=args.val_masks; d.validation_scenes_per_mask=args.val_scenes
    d.train_mask_seed=42; d.evaluation_mask_seed=42; d.psf_cache.warmup=False
    resolved=OmegaConf.to_container(config,resolve=True)
    if resolved['protocol']['version']!='corrected-v1':raise ValueError('Wrong simulator')
    OmegaConf.save(OmegaConf.create(resolved),out/'resolved_config.yaml')
    loaders,transforms=get_dataloaders(config,'cuda')
    if transforms:raise ValueError('Unexpected transforms')
    loader=loaders['validation']; mean=None
    for record in get_mask_records(42,'train',100):
        raw,_,_=loader.dataset._get_mask({'mask_seed':int(record['mask_seed']),'mode':'finite'})
        psf=_prepare_convolution_psf(raw).squeeze(0).movedim(-1,0).contiguous()
        mean=psf.clone() if mean is None else mean+psf
    mean=mean/mean.norm()
    provenance={'script_sha256':file_hash(Path(__file__)),'checkpoint_sha256':file_hash(checkpoint_path),
                'source_config_sha256':file_hash(config_path),'mean_psf_sha256':tensor_hash(mean),
                'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'parameters':PARAMETERS,'cg_iterations':20,'scope':'true PSF oracle vs fixed mean training PSF; development only'}
    save_json(out/'provenance.json',provenance)
    import wandb
    log=wandb.init(project='lensless-imaging',entity='had-2005-hse-university',mode='offline',name=out.name,group='psff-cg-01',dir=str(out),config=provenance)
    base=instantiate(config.model).cuda().eval().requires_grad_(False)
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=False); base.load_state_dict(state['state_dict'],strict=True); del state
    metrics={'PSNR':PSNRMetric(normalize_by_max=True),'SSIM':SSIMMetric(normalize_by_max=True),'LPIPS':LPIPSMetric(net_type='vgg',device='cuda',normalize_by_max=True)}
    rows=[]; mask_order={}; scene_order={}; worst=0.
    save_json(out/'status.json',{'status':'evaluating'})
    with (out/'per_sample.csv').open('w',newline='') as stream:
        writer=None
        for index,batch in enumerate(loader,start=1):
            y,target=batch['measurement'].cuda(),batch['target'].cuda(); psf=batch['psf'].cuda()
            operators={'true_psf':LinearROI(psf),'mean_train_psf':LinearROI(mean.cuda()[None].expand_as(psf).contiguous())}
            with torch.no_grad():
                error=float((peak(operators['true_psf'].forward(target).clamp_min(0))-y).abs().max()); worst=max(worst,error)
                if error>.5/255+1e-5:raise RuntimeError('Forward parity failure')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                raw=torch.cat([base(measurement=item[None])['prediction'].float() for item in y])
            b=peak(raw); meta=[]
            for i,sample_id in enumerate(batch['sample_id']):
                mid,sid=batch['mask_id'][i],batch['scene_id'][i]; mask_order.setdefault(mid,len(mask_order)); scene_order.setdefault(sid,len(scene_order))
                meta.append({'sample_id':sample_id,'mask_id':mid,'scene_id':sid,'mask_order':mask_order[mid],'scene_order':scene_order[sid],
                             'measurement_sha256':tensor_hash(y[i]),'target_sha256':tensor_hash(target[i]),'base_sha256':tensor_hash(raw[i]),'true_psf_sha256':tensor_hash(psf[i])})
            for arm,op in operators.items():
                with torch.no_grad():
                    ab=op.forward(b); gain=(ab*y).flatten(1).sum(1)/y.square().flatten(1).sum(1).clamp_min(1e-12)
                    gain=gain.clamp_min(1e-8); rhs=op.adjoint(times(gain,y)-ab)
                previous=None; delta=torch.zeros_like(b); residual=torch.zeros(len(y),device='cuda')
                for relative,alpha in PARAMETERS:
                    if relative and relative!=previous:
                        delta,residual=solve(op,rhs,relative); previous=relative
                    with torch.no_grad():values={k:m.per_image(b+alpha*delta,target).cpu().tolist() for k,m in metrics.items()}
                    for i,metadata in enumerate(meta):
                        row={**metadata,'arm':arm,'relative_lambda':relative,'alpha':alpha,'gain':float(gain[i]),'normal_residual':float(residual[i]),**{k:v[i] for k,v in values.items()}}
                        if not all(np.isfinite(row[k]) for k in values):raise RuntimeError('Nonfinite metric')
                        if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row); rows.append(row)
            stream.flush()
            if index==1 or index%16==0:print(json.dumps({'evaluation_batch':index,'seconds':time.monotonic()-started}),flush=True)
    result=summarize(rows,args.val_masks,args.val_scenes)
    result.update(seconds=time.monotonic()-started,peak_vram_bytes=torch.cuda.max_memory_allocated(),technical_smoke=args.val_masks!=32 or args.val_scenes!=32,max_forward_quantization_error=worst)
    if not result['technical_smoke'] and not result['baseline_parity_pass']:
        result['status']='requires_baseline_audit'; result['signal']={k:False for k in result['signal']}
    save_json(out/'summary.json',result); save_json(out/'status.json',{'status':result['status'],'signal':result['signal']})
    log.summary.update(result); log.finish(); print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--val-masks',type=int,default=32);p.add_argument('--val-scenes',type=int,default=32);a=p.parse_args()
    if a.self_test:self_test()
    elif a.output is None or not(2<=a.val_masks<=32 and 2<=a.val_scenes<=32):p.error('Output and grid sizes2..32 required')
    else:run(a)
