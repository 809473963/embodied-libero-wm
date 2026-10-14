#!/usr/bin/env python3
"""Bounded natural-input perception pilot followed by frozen dynamics substitution."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,json,time,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.probes.candidate_utility_oracle import atomic_json,sha256
from src.probes.relational_dynamics_pilot import relation,score
from src.probes.relational_minimal_state import project
from scripts.evaluate_relational_autorun import H24Dynamics,paired
OUT=ROOT/'results/relational_natural_state'
SRC=ROOT/'results/relational_autorun'

class Perception(nn.Module):
    def __init__(self,visual):
        super().__init__();self.visual=visual
        if visual:
            self.cnn=nn.Sequential(nn.Conv2d(6,16,5,2,2),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,32,3,2,1),nn.ReLU(),nn.Flatten(),nn.Linear(32*8*8,64),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(83 if visual else 19,128),nn.ReLU(),nn.Linear(128,18))
    def forward(self,image,proprio_action):
        x=torch.cat([self.cnn(image),proprio_action],1) if self.visual else proprio_action
        return self.head(x)

def build():
    old=json.loads((ROOT/'results/relational_event_h24/manifest.json').read_text())['records']
    new=json.loads((SRC/'branches/manifest.json').read_text())['records']
    images=[];initials={};initial_ids=[];obs=[];targets=[];fit=[];roots=[];dedup={};mapping={};sources={}
    initial_by_root={};root_fits={}
    for isfit,records in [(True,old),(False,new)]:
        for r in records:
            root=r['root_id'] if isfit else r['root'];path=r['dense']['arrays'] if isfit else r['path'];expected=r['dense']['sha256'] if isfit else r['sha256']
            assert sha256(ROOT/path)==expected,path;sources[path]=expected
            with np.load(ROOT/path,allow_pickle=False) as f:a={k:f[k] for k in ['rgb','proprio','natural_step','pose','grip_rot','grip_pos','grip_velocity','actions']}
            steps=a['natural_step'];assert len(set(steps.tolist()))==len(steps);assert steps[0]==0
            assert a['rgb'].dtype==np.uint8 and a['rgb'].shape[1:]==(128,128,3)
            # Deterministic 2x2 box downsampling, no label-dependent transforms.
            small=a['rgb'].astype(np.uint16).reshape(-1,64,2,64,2,3).sum((2,4));small=(small//4).astype(np.uint8)
            if root not in initials:initials[root]=small[0];initial_by_root[root]=len(initial_by_root);root_fits[root]=isfit
            else:np.testing.assert_array_equal(initials[root],small[0]);assert root_fits[root]==isfit
            rel=relation(a);local={}
            for j,t in enumerate(steps):
                pa=np.r_[a['proprio'][j],np.zeros(7) if t==0 else a['actions'][t-1]].astype(np.float32)
                y=np.r_[rel[t],a['grip_velocity'][t]].astype(np.float32)
                # Equal branches' shared prefix is one supervised example. Distinct labels stay distinct.
                key=(root,int(t),hashlib.sha256(small[j].tobytes()+pa.tobytes()+y.tobytes()).hexdigest())
                if key not in dedup:
                    dedup[key]=len(images);images.append(small[j]);obs.append(pa);targets.append(y);fit.append(isfit);roots.append(root);initial_ids.append(initial_by_root[root])
                local[int(t)]=dedup[key]
            mapping[(root,r['anchor'],r['variant'])]=[local[t] for t in range(r['anchor']-10,r['anchor']+1)]
    meta=json.loads((SRC/'models/samples.json').read_text())
    history=np.array([mapping[(m['root'],m['anchor'],m['variant'])] for m in meta])
    for root,anchor in {(m['root'],m['anchor']) for m in meta}:
        ix=[i for i,m in enumerate(meta) if (m['root'],m['anchor'])==(root,anchor)]
        for i in ix[1:]:np.testing.assert_array_equal(history[ix[0]],history[i])
    fit=np.array(fit);roots=np.array(roots);assert not set(roots[fit])&set(roots[~fit])
    weights=np.array([1/(roots==r).sum() for r in roots],np.float32)
    d=dict(images=np.array(images),initial_images=np.stack(list(initials.values())),initial_ids=np.array(initial_ids),obs=np.array(obs),target=np.array(targets),fit=fit,weights=weights,history=history)
    np.savez_compressed(OUT/'dataset.npz',**d)
    atomic_json(OUT/'inventory.json',dict(sources=sources,roots=roots.tolist(),fit_frames=int(fit.sum()),development_frames=int((~fit).sum()),fit_scenes=len(set(roots[fit])),development_scenes=len(set(roots[~fit])),identity_contract='fixed physical bowl slots; no learned language grounding',frame_grain='root, control_step, input_and_target_hash',confirmation_opened=False))
    return d,roots

def stats(d):
    w=d['weights'][d['fit']];w=w/w.sum();out={}
    for key in ('obs','target'):
        x=d[key][d['fit']];mean=(x*w[:,None]).sum(0);std=np.sqrt(((x-mean)**2*w[:,None]).sum(0)).clip(.001)
        out[key+'_mean']=mean;out[key+'_std']=std
    return out

def evaluate(name,pred,d,roots):
    dev=~d['fit'];err=np.linalg.norm((pred[:,:12]-d['target'][:,:12]).reshape(-1,4,3),axis=2)*1000
    scene={r:err[(roots==r)&dev].mean(0).tolist() for r in sorted(set(roots[dev]))}
    with np.load(SRC/'models/dataset.npz',allow_pickle=False) as f:dyn={k:f[k] for k in f.files}
    schema=json.loads((SRC/'models/schema.json').read_text());meta=json.loads((SRC/'models/samples.json').read_text())
    x,_=project(dyn['x'],schema,'geometry_history');x[...,:18]=pred[d['history']]
    with np.load(ROOT/'results/relational_minimal_state/geometry_history/normalization.npz',allow_pickle=False) as f:st={k:f[k] for k in f.files}
    a=torch.tensor((dyn['actions']-st['action_mean'])/st['action_std'],device='cuda');xt=torch.tensor((x-st['x_mean'])/st['x_std'],device='cuda')
    deltas=[];probs=[]
    for seed in (11,23,47):
        model=H24Dynamics(26).cuda();checkpoint=torch.load(ROOT/f'results/relational_minimal_state/geometry_history/action_seed{seed}.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state_dict']);model.eval()
        with torch.no_grad():out=torch.cat([model(xt[i:i+128],a[i:i+128]).cpu() for i in range(0,len(x),128)])
        deltas.append(out[:,:12].numpy()*st['delta_scale']);probs.append(torch.sigmoid(out[:,12:]).numpy())
    delta=np.mean(deltas,0);prob=np.mean(probs,0);mask=~dyn['fit']
    # Absolute future relation includes current-state perception error; delta-only is insufficient.
    true_start=dyn['x'][:,-1,104:116];estimated_start=pred[d['history'][:,-1],:12]
    endpoint_delta=delta+estimated_start-true_start
    np.savez_compressed(OUT/f'{name}_bridge.npz',delta=delta,probability=prob,endpoint_delta=endpoint_delta)
    return dict(perception_per_scene_mm=scene,perception_scene_macro_mm=np.mean(list(scene.values()),0).tolist(),
        delta=score(delta,prob,dyn,meta,mask),absolute_endpoint=score(endpoint_delta,prob,dyn,meta,mask),pairs=paired(delta,dyn,meta,mask))

def main():
    cfg=json.loads((ROOT/'configs/relational_natural_state_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    OUT.mkdir(exist_ok=True)
    if (OUT/'runtime.json').exists():raise RuntimeError('Prior attempt retained; no automatic restart')
    rt=dict(status='RUNNING',phase='data_alignment',started_epoch=time.time(),completed_fits=0,maximum_fits=6);atomic_json(OUT/'runtime.json',rt)
    try:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        d,roots=build();st=stats(d);np.savez_compressed(OUT/'normalization.npz',**st)
        pa=torch.tensor((d['obs']-st['obs_mean'])/st['obs_std'],device='cuda');y=torch.tensor((d['target']-st['target_mean'])/st['target_std'],device='cuda')
        weights=torch.tensor(d['weights']/d['weights'][d['fit']].mean(),device='cuda');preds={};metrics={}
        def batch(ii):
            current=d['images'][ii];initial=d['initial_images'][d['initial_ids'][ii]]
            return torch.tensor(np.concatenate([initial,current],axis=-1).transpose(0,3,1,2).copy(),dtype=torch.float32,device='cuda')/255
        for family in ('proprio','rgb_proprio'):
            for seed in (11,23,47):
                torch.manual_seed(seed);model=Perception(family=='rgb_proprio').cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);name=f'{family}_seed{seed}';losses=[]
                for epoch in range(40):
                    model.train();order=rng.permutation(np.flatnonzero(d['fit']));loss_sum=0
                    for start in range(0,len(order),128):
                        ii=order[start:start+128];out=model(batch(ii) if model.visual else None,pa[ii]);loss=(F.smooth_l1_loss(out,y[ii],reduction='none').mean(1)*weights[ii]).mean();assert torch.isfinite(loss)
                        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();loss_sum+=float(loss.detach())
                    losses.append(loss_sum);rt.update(phase='fit_old_scenes_only',model=name,epoch=epoch+1);atomic_json(OUT/'runtime.json',rt)
                model.eval();outputs=[]
                with torch.no_grad():
                    for start in range(0,len(pa),128):
                        ii=np.arange(start,min(start+128,len(pa)));outputs.append(model(batch(ii) if model.visual else None,pa[ii]).cpu().numpy())
                pred=np.concatenate(outputs)*st['target_std']+st['target_mean'];assert np.isfinite(pred).all();preds[name]=pred
                np.savez_compressed(OUT/f'{name}_prediction.npz',state=pred)
                torch.save(dict(state_dict=model.cpu().state_dict(),visual=family=='rgb_proprio'),OUT/f'{name}.pt')
                metrics[name]=evaluate(name,pred,d,roots);metrics[name]['fit_loss']=losses
                rt['completed_fits']+=1;atomic_json(OUT/'runtime.json',rt);print(name,flush=True)
            name=family+'_ensemble';pred=np.mean([preds[f'{family}_seed{s}'] for s in (11,23,47)],0)
            np.savez_compressed(OUT/f'{name}_prediction.npz',state=pred);metrics[name]=evaluate(name,pred,d,roots)
        atomic_json(OUT/'evaluation.json',dict(models=metrics,confirmation_opened=False,new_data_used_for_fitting=False))
        from scripts.report_relational_natural_state import main as report
        report()
        rt.update(status='COMPLETE',phase='verified_and_reported',finished_epoch=time.time(),report='docs/relational_natural_state_audit.md');atomic_json(OUT/'runtime.json',rt)
    except Exception:
        rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'runtime.json',rt);raise
if __name__=='__main__':main()
