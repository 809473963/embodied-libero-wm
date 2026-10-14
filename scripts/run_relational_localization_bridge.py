#!/usr/bin/env python3
"""Frozen CV recipe refit on all old scenes; single matched development bridge."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.train_relational_localization_cv import Model,normalization
from scripts import run_relational_natural_state as bridge
from src.probes.candidate_utility_oracle import atomic_json,sha256
OUT=ROOT/'results/relational_localization_bridge'

def main():
 cfg=json.loads((ROOT/'configs/relational_localization_bridge_protocol.json').read_text())
 for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
 OUT.mkdir(exist_ok=True)
 if (OUT/'runtime.json').exists():raise RuntimeError('Existing attempt retained')
 rt=dict(status='RUNNING',completed_fits=0,maximum_fits=6,phase='validate_old_fit_alignment',started_epoch=time.time());atomic_json(OUT/'runtime.json',rt)
 try:
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  with np.load(ROOT/'results/relational_localization_cv/dataset.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
  with np.load(ROOT/'results/relational_natural_state/dataset.npz',allow_pickle=False) as f:n={k:f[k] for k in f.files}
  meta=json.loads((ROOT/'results/relational_localization_cv/metadata.json').read_text())['rows'];roots=np.array([m['root'] for m in meta]);nr=np.array(json.loads((ROOT/'results/relational_natural_state/inventory.json').read_text())['roots'])
  np.testing.assert_array_equal(roots,nr[n['fit']])
  for key in ['images','obs','target']:np.testing.assert_array_equal(d[key],n[key][n['fit']])
  np.testing.assert_array_equal(d['initial_images'][d['initial_ids']],n['initial_images'][n['initial_ids'][n['fit']]])
  assert not set(roots)&set(nr[~n['fit']]);w=np.array([1/(roots==r).sum() for r in roots],np.float32);fit=np.ones(len(roots),bool);st=normalization(d,fit,w)
  np.savez_compressed(OUT/'normalization.npz',**st)
  atomic_json(OUT/'alignment.json',dict(status='PASS',old_frames=len(roots),old_scenes=len(set(roots)),development_frames=int((~n['fit']).sum()),development_scenes=len(set(nr[~n['fit']])),old_arrays_exact=True))
  obs=torch.tensor((d['obs']-st['obs_mean'])/st['obs_std'],device='cuda');no=torch.tensor((n['obs']-st['obs_mean'])/st['obs_std'],device='cuda');y=torch.tensor((d['target']-st['target_mean'])/st['target_std'],device='cuda');pt=torch.tensor((d['points']-st['points_mean'])/st['points_std'],device='cuda');mask=torch.tensor(d['masks'],device='cuda');wt=torch.tensor(w/w.mean(),device='cuda')
  fg=(d['masks']*w[:,None,None,None]).sum((0,2,3))/(w.sum()*64);pos=torch.tensor(np.clip((1-fg)/np.maximum(fg,1e-6),1,20),device='cuda')[:,None,None]
  def batch(data,ii):
   im=np.concatenate([data['initial_images'][data['initial_ids'][ii]],data['images'][ii]],-1)
   return torch.tensor(im.transpose(0,3,1,2).copy(),device='cuda',dtype=torch.float32)/255
  predictions={};metrics={};bridge.OUT=OUT
  for seed in [11,23,47]:
   for family in ['direct','localized']:
    name=f'{family}_seed{seed}';torch.manual_seed(seed);model=Model().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);losses=[]
    for epoch in range(40):
     model.train();order=rng.permutation(len(roots));loss_sum=0
     for start in range(0,len(order),128):
      ii=order[start:start+128];state,point,m=model(batch(d,ii),obs[ii]);loss=F.smooth_l1_loss(state,y[ii],reduction='none').mean(1)
      if family=='localized':loss=loss+.25*F.smooth_l1_loss(point,pt[ii],reduction='none').mean(1)+.1*F.binary_cross_entropy_with_logits(m,mask[ii],pos_weight=pos,reduction='none').mean((1,2,3))
      loss=(loss*wt[ii]).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();loss_sum+=float(loss.detach())
     losses.append(loss_sum);rt.update(phase='fit_old_only',model=name,epoch=epoch+1);atomic_json(OUT/'runtime.json',rt)
    model.eval();pieces=[]
    with torch.no_grad():
     for start in range(0,len(nr),128):
      ii=np.arange(start,min(start+128,len(nr)));pieces.append(model(batch(n,ii),no[ii])[0].cpu().numpy())
    pred=np.concatenate(pieces)*st['target_std']+st['target_mean'];assert np.isfinite(pred).all();predictions[name]=pred
    np.savez_compressed(OUT/f'{name}_prediction.npz',state=pred);torch.save(dict(state_dict=model.cpu().state_dict(),family=family,seed=seed,loss=losses),OUT/f'{name}.pt')
    metrics[name]=bridge.evaluate(name,pred,n,nr);rt['completed_fits']+=1;atomic_json(OUT/'runtime.json',rt);print(name,flush=True)
  for family in ['direct','localized']:
   name=family+'_ensemble';pred=np.mean([predictions[f'{family}_seed{s}'] for s in [11,23,47]],0);np.savez_compressed(OUT/f'{name}_prediction.npz',state=pred);metrics[name]=bridge.evaluate(name,pred,n,nr)
  atomic_json(OUT/'evaluation.json',dict(models=metrics,new_development_used_for_fitting=False,confirmation_opened=False))
  from scripts.report_relational_localization_bridge import main as report
  report();rt.update(status='COMPLETE',phase='verified_and_reported',finished_epoch=time.time(),report='docs/relational_localization_bridge_audit.md');atomic_json(OUT/'runtime.json',rt)
 except Exception:
  rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'runtime.json',rt);raise
if __name__=='__main__':main()
