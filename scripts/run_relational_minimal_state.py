#!/usr/bin/env python3
"""Finite old-fit-only state ablation, no simulation or confirmation access."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from src.probes.candidate_utility_oracle import atomic_json,sha256
from src.probes.relational_dynamics_pilot import normalization,score
from src.probes.relational_minimal_state import project,REPRESENTATIONS
from scripts.evaluate_relational_autorun import H24Dynamics,paired
SOURCE=ROOT/'results/relational_autorun'
OUT=ROOT/'results/relational_minimal_state'

def train(rep,rt):
    directory=OUT/rep;directory.mkdir()
    with np.load(SOURCE/'models/dataset.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    schema=json.loads((SOURCE/'models/schema.json').read_text())
    d['x'],indices=project(d['x'],schema,rep)
    meta=json.loads((SOURCE/'models/samples.json').read_text())
    fit=d['fit'];dev=~fit;weights=d['weights']
    st=normalization(d['x'],d['actions'],d['delta'],fit,weights)
    np.savez_compressed(directory/'normalization.npz',**st)
    atomic_json(directory/'projection.json',dict(indices=indices,shape=list(d['x'].shape)))
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda';x=torch.tensor((d['x']-st['x_mean'])/st['x_std'],device=device);actions=torch.tensor((d['actions']-st['action_mean'])/st['action_std'],device=device)
    y=torch.tensor(d['delta']/st['delta_scale'],device=device);binary=torch.tensor(np.maximum(d['binary'],0),dtype=torch.float32,device=device);valid=torch.tensor(d['binary']>=0,device=device)
    fw=weights[fit,None];pos=(fw*(d['binary'][fit]==1)).sum(0);neg=(fw*(d['binary'][fit]==0)).sum(0)
    pw=torch.tensor(np.clip(neg/np.maximum(pos,1e-12),1,20),device=device);wt=torch.tensor(weights/weights[fit].mean(),device=device)
    ev={};predictions={}
    def metrics(delta,prob):
        return dict(old_fit=score(delta,prob,d,meta,fit),new_development=score(delta,prob,d,meta,dev),new_pairs=paired(delta,d,meta,dev))
    ev['unchanged']=metrics(np.zeros_like(d['delta']),np.where(d['current']>=0,d['current'],np.nan))
    for seed in (11,23,47):
        for family in ('no_action','action'):
            torch.manual_seed(seed);model=H24Dynamics(x.shape[-1]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            rng=np.random.default_rng(seed);a=torch.zeros_like(actions) if family=='no_action' else actions;name=f'{family}_seed{seed}';losses=[]
            for epoch in range(80):
                order=rng.permutation(np.flatnonzero(fit));model.train();loss_sum=0
                for start in range(0,len(order),128):
                    ii=torch.tensor(order[start:start+128],device=device);out=model(x[ii],a[ii])
                    cont=F.smooth_l1_loss(out[:,:12],y[ii],reduction='none').mean(1)
                    bce=F.binary_cross_entropy_with_logits(out[:,12:],binary[ii],pos_weight=pw,reduction='none')
                    aux=(bce*valid[ii]).sum(1)/valid[ii].sum(1).clamp_min(1);loss=((cont+.2*aux)*wt[ii]).mean();assert torch.isfinite(loss)
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();loss_sum+=float(loss.detach())
                losses.append(loss_sum);rt.update(phase='training_old_data_only',model=name,epoch=epoch+1);atomic_json(OUT/'runtime.json',rt)
            model.eval()
            with torch.no_grad():out=torch.cat([model(x[i:i+128],a[i:i+128]).cpu() for i in range(0,len(meta),128)])
            delta=out[:,:12].numpy()*st['delta_scale'];prob=torch.sigmoid(out[:,12:]).numpy();assert np.isfinite(delta).all()
            predictions[name]=(delta,prob);ev[name]=metrics(delta,prob);ev[name]['fit_loss']=losses
            np.savez_compressed(directory/f'{name}_predictions.npz',delta=delta,probability=prob)
            torch.save(dict(state_dict=model.cpu().state_dict(),feature_dimension=int(x.shape[-1]),horizon=24),directory/f'{name}.pt')
            rt['completed_fits']+=1;atomic_json(OUT/'runtime.json',rt);print(json.dumps(dict(stage='models',completed_fits=rt['completed_fits'])),flush=True)
    for family in ('no_action','action'):
        delta=np.mean([predictions[f'{family}_seed{s}'][0] for s in (11,23,47)],axis=0);prob=np.mean([predictions[f'{family}_seed{s}'][1] for s in (11,23,47)],axis=0)
        name=family+'_ensemble';ev[name]=metrics(delta,prob);np.savez_compressed(directory/f'{name}_predictions.npz',delta=delta,probability=prob)

    atomic_json(directory/'evaluation.json',dict(models=ev,new_data_used_for_fitting=False,confirmation_opened=False))

def main():
    cfg=json.loads((ROOT/'configs/relational_minimal_state_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    OUT.mkdir(exist_ok=True)
    if (OUT/'runtime.json').exists():raise RuntimeError('Existing attempt retained; no automatic retries')
    rt=dict(status='RUNNING',started_epoch=time.time(),completed_fits=0,maximum_fits=24,completed_representations=[])
    atomic_json(OUT/'runtime.json',rt)
    try:
        for rep in REPRESENTATIONS:
            rt['representation']=rep;atomic_json(OUT/'runtime.json',rt)
            train(rep,rt)
            rt['completed_representations'].append(rep);atomic_json(OUT/'runtime.json',rt)
        from scripts.report_relational_minimal_state import main as report
        report()
        rt.update(status='COMPLETE',phase='verified_and_reported',finished_epoch=time.time(),report='docs/relational_minimal_state_audit.md')
        atomic_json(OUT/'runtime.json',rt)
    except Exception:
        rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'runtime.json',rt);raise
if __name__=='__main__':main()
