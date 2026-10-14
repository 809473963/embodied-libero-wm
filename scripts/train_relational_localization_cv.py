#!/usr/bin/env python3
"""Matched-architecture localization auxiliary-loss ablation in old scenes only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.probes.candidate_utility_oracle import atomic_json,sha256
OUT=ROOT/'results/relational_localization_cv'
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=nn.Sequential(nn.Conv2d(6,16,5,2,2),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,32,3,2,1),nn.ReLU());self.pool=nn.Sequential(nn.Flatten(),nn.Linear(2048,64),nn.ReLU());self.state=nn.Sequential(nn.Linear(83,128),nn.ReLU(),nn.Linear(128,18));self.points=nn.Linear(64,9);self.mask=nn.Conv2d(32,3,1)
 def forward(self,image,obs):
  h=self.encoder(image);z=self.pool(h);return self.state(torch.cat([z,obs],1)),self.points(z),self.mask(h)
def normalization(d,fit,w):
 w=w[fit]/w[fit].sum();out={}
 for k in ['obs','target','points']:
  x=d[k][fit];mean=(x*w[:,None]).sum(0);std=np.sqrt(((x-mean)**2*w[:,None]).sum(0)).clip(.001);out[k+'_mean']=mean;out[k+'_std']=std
 return out
def main():
 cfg=json.loads((ROOT/'configs/relational_localization_cv_protocol.json').read_text())
 for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
 collection=json.loads((OUT/'collection_runtime.json').read_text());assert collection['status']=='COMPLETE';assert sha256(OUT/'dataset.npz')==collection['dataset_sha256']
 if (OUT/'training_runtime.json').exists():raise RuntimeError('Existing training attempt retained')
 rt=dict(status='RUNNING',completed_fits=0,maximum_fits=18,started_epoch=time.time());atomic_json(OUT/'training_runtime.json',rt)
 try:
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  with np.load(OUT/'dataset.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
  meta=json.loads((OUT/'metadata.json').read_text())['rows'];roots=np.array([m['root'] for m in meta]);w=np.array([1/(roots==r).sum() for r in roots],np.float32)
  allpred={f'{family}_seed{s}':np.full(d['target'].shape,np.nan,np.float32) for family in ['direct','localized'] for s in [11,23,47]}
  for fold in range(3):
   fit=d['fold']!=fold;test=~fit;assert not set(roots[fit])&set(roots[test]);st=normalization(d,fit,w);np.savez_compressed(OUT/f'fold{fold}_normalization.npz',**st)
   obs=torch.tensor((d['obs']-st['obs_mean'])/st['obs_std'],device='cuda');y=torch.tensor((d['target']-st['target_mean'])/st['target_std'],device='cuda');pt=torch.tensor((d['points']-st['points_mean'])/st['points_std'],device='cuda');mask=torch.tensor(d['masks'],device='cuda');wt=torch.tensor(w/w[fit].mean(),device='cuda')
   fg=(d['masks'][fit]*w[fit,None,None,None]).sum((0,2,3))/(w[fit].sum()*64);pos=torch.tensor(np.clip((1-fg)/np.maximum(fg,1e-6),1,20),device='cuda')[:,None,None]
   def batch(ii):
    im=np.concatenate([d['initial_images'][d['initial_ids'][ii]],d['images'][ii]],-1)
    return torch.tensor(im.transpose(0,3,1,2).copy(),device='cuda',dtype=torch.float32)/255
   for seed in [11,23,47]:
    for family in ['direct','localized']:
     torch.manual_seed(seed);model=Model().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);losses=[]
     for epoch in range(40):
      model.train();order=rng.permutation(np.flatnonzero(fit));loss_sum=0
      for start in range(0,len(order),128):
       ii=order[start:start+128];state,point,m=model(batch(ii),obs[ii]);loss=F.smooth_l1_loss(state,y[ii],reduction='none').mean(1)
       if family=='localized':loss=loss+.25*F.smooth_l1_loss(point,pt[ii],reduction='none').mean(1)+.1*F.binary_cross_entropy_with_logits(m,mask[ii],pos_weight=pos,reduction='none').mean((1,2,3))
       loss=(loss*wt[ii]).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();loss_sum+=float(loss.detach())
      losses.append(loss_sum);rt.update(fold=fold,seed=seed,family=family,epoch=epoch+1);atomic_json(OUT/'training_runtime.json',rt)
     model.eval();ix=np.flatnonzero(test);pred=[];pts=[];ms=[]
     with torch.no_grad():
      for start in range(0,len(ix),128):
       ii=ix[start:start+128];a,b,c=model(batch(ii),obs[ii]);pred.append(a.cpu().numpy());pts.append(b.cpu().numpy());ms.append(torch.sigmoid(c).cpu().numpy())
     name=f'{family}_seed{seed}';allpred[name][ix]=np.concatenate(pred)*st['target_std']+st['target_mean']
     np.savez_compressed(OUT/f'fold{fold}_{name}_prediction.npz',indices=ix,state=allpred[name][ix],points=np.concatenate(pts)*st['points_std']+st['points_mean'],mask=np.concatenate(ms))
     torch.save(dict(state_dict=model.cpu().state_dict(),fold=fold,seed=seed,family=family,loss=losses),OUT/f'fold{fold}_{name}.pt')
     rt['completed_fits']+=1;atomic_json(OUT/'training_runtime.json',rt);print(f'{rt["completed_fits"]}/18 {name} fold{fold}',flush=True)
  for name,p in allpred.items():assert np.isfinite(p).all();np.savez_compressed(OUT/f'oof_{name}.npz',state=p)
  for family in ['direct','localized']:np.savez_compressed(OUT/f'oof_{family}_ensemble.npz',state=np.mean([allpred[f'{family}_seed{s}'] for s in [11,23,47]],0))
  rt.update(status='COMPLETE',finished_epoch=time.time());atomic_json(OUT/'training_runtime.json',rt)
 except Exception:
  rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'training_runtime.json',rt);raise
if __name__=='__main__':main()
