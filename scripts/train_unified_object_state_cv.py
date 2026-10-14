#!/usr/bin/env python3
"""Old-scene matched shared-node versus independent-edge perception experiment."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.probes.candidate_utility_oracle import atomic_json,sha256
OUT=ROOT/'results/unified_object_state_cv'
EDGES=[(i,j) for i in range(9) for j in range(i+1,9)]
GROUPS={
 'object': [k for k,(i,j) in enumerate(EDGES) if j!=8 and i<5],
 'robot': [k for k,(i,j) in enumerate(EDGES) if j==8],
 'context': [k for k,(i,j) in enumerate(EDGES) if i>=5 and j<8]}
KEY=[k for k,(i,j) in enumerate(EDGES) if i<5 and (j<5 or j==8)]
class Model(nn.Module):
 def __init__(self,family):
  super().__init__();self.family=family
  self.encoder=nn.Sequential(nn.Conv2d(6,16,5,2,2),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,32,3,2,1),nn.ReLU(),nn.Flatten(),nn.Linear(2048,64),nn.ReLU())
  self.hidden=nn.Sequential(nn.Linear(83,128),nn.ReLU())
  self.head=nn.Linear(128,24 if family=='shared' else 108)
 def forward(self,image,obs,grip,center):
  raw=self.head(self.hidden(torch.cat([self.encoder(image),obs],1))).reshape(-1,8 if self.family=='shared' else 36,3)
  if self.family=='shared':
   nodes=torch.cat([center[None]+.05*raw,grip[:,None]],1)
   edges=torch.stack([nodes[:,i]-nodes[:,j] for i,j in EDGES],1)
  else:
   edges=center[None]+.05*raw
   nodes=torch.cat([torch.stack([edges[:,EDGES.index((i,8))]+grip for i in range(8)],1),grip[:,None]],1)
  return nodes,edges

def edge_loss(pred,target):
 per=F.smooth_l1_loss(pred/.05,target/.05,reduction='none').mean(2)
 return sum(per[:,v].mean(1) for v in GROUPS.values())/3

def scene_mean(x,roots):
 return float(np.mean([np.mean(x[roots==r]) for r in sorted(set(roots))]))

def evaluate(nodes,edges,truth,relations,roots):
 ne=np.linalg.norm(nodes-truth,axis=2)*1000;ee=np.linalg.norm(edges-relations,axis=2)*1000
 closures=np.stack([np.linalg.norm(edges[:,EDGES.index((i,j))]+edges[:,EDGES.index((j,k))]-edges[:,EDGES.index((i,k))],axis=1)*1000 for i in range(9) for j in range(i+1,9) for k in range(j+1,9)],1)
 return dict(node_mm=[scene_mean(ne[:,i],roots) for i in range(9)],edge_mm=[scene_mean(ee[:,i],roots) for i in range(36)],groups_mm={g:scene_mean(ee[:,ix].mean(1),roots) for g,ix in GROUPS.items()},key_mean_mm=scene_mean(ee[:,KEY].mean(1),roots),worst_key_mm=scene_mean(ee[:,KEY].max(1),roots),triangle_closure_mm=scene_mean(closures.mean(1),roots),per_scene={r:dict(key_mean_mm=float(ee[roots==r][:,KEY].mean()),worst_key_mm=float(ee[roots==r][:,KEY].max(1).mean())) for r in sorted(set(roots))})

def main():
 cfg=json.loads((ROOT/'configs/unified_object_state_cv_protocol.json').read_text())
 for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
 OUT.mkdir(exist_ok=True)
 if (OUT/'runtime.json').exists():raise RuntimeError('Existing attempt retained')
 rt=dict(status='RUNNING',completed_fits=0,maximum_fits=18,started_epoch=time.time(),confirmation_opened=False);atomic_json(OUT/'runtime.json',rt)
 try:
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  with np.load(ROOT/'results/relational_localization_cv/dataset.npz',allow_pickle=False) as f:d={k:f[k] for k in ['images','initial_images','initial_ids','obs','fold']}
  with np.load(ROOT/'results/unified_object_state_audit_v3/node_supervision.npz',allow_pickle=False) as f:n={k:f[k] for k in ['positions','relations','roots','fold']}
  np.testing.assert_array_equal(d['fold'],n['fold']);roots=n['roots'];assert len(roots)==4843 and len(set(roots))==13
  w=np.array([1/(roots==r).sum() for r in roots],np.float32)
  # Both families are supervised only through the same physical edge loss.
  truth=torch.tensor(n['relations'],device='cuda',dtype=torch.float32);grip=torch.tensor(d['obs'][:,:3],device='cuda')
  predictions={'mean_position':(np.full(n['positions'].shape,np.nan,np.float32),np.full(n['relations'].shape,np.nan,np.float32))};params={}
  def images(ii):
   im=np.concatenate([d['initial_images'][d['initial_ids'][ii]],d['images'][ii]],-1)
   return torch.tensor(im.transpose(0,3,1,2).copy(),device='cuda',dtype=torch.float32)/255
  for fold in range(3):
   fit=d['fold']!=fold;test=~fit;assert not set(roots[fit])&set(roots[test]);weights=w[fit]/w[fit].sum()
   om=(d['obs'][fit]*weights[:,None]).sum(0);std=np.sqrt(((d['obs'][fit]-om)**2*weights[:,None]).sum(0)).clip(.001)
   centers={family:(n['positions'][fit,:8] if family=='shared' else n['relations'][fit]).astype(np.float64) for family in ['shared','independent']}
   centers={family:(v*weights[:,None,None]).sum(0).astype(np.float32) for family,v in centers.items()}
   np.savez_compressed(OUT/f'fold{fold}_normalization.npz',obs_mean=om,obs_std=std,**centers)
   baseline=np.concatenate([np.broadcast_to(centers['shared'],(int(test.sum()),8,3)),d['obs'][test,None,:3]],1)
   predictions['mean_position'][0][test]=baseline
   predictions['mean_position'][1][test]=np.stack([baseline[:,i]-baseline[:,j] for i,j in EDGES],1)
   obs=torch.tensor((d['obs']-om)/std,device='cuda');wt=torch.tensor(w/w[fit].mean(),device='cuda')
   for seed in [11,23,47]:
    for family in ['shared','independent']:
     torch.manual_seed(seed);model=Model(family).cuda();params[family]=sum(p.numel() for p in model.parameters());center=torch.tensor(centers[family],device='cuda')
     opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);losses=[]
     for epoch in range(40):
      model.train();order=rng.permutation(np.flatnonzero(fit));loss_sum=0
      for start in range(0,len(order),128):
       ii=order[start:start+128];_,pred=model(images(ii),obs[ii],grip[ii],center);loss=(edge_loss(pred,truth[ii])*wt[ii]).mean();assert torch.isfinite(loss)
       opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();loss_sum+=float(loss.detach())
      losses.append(loss_sum);rt.update(fold=fold,seed=seed,family=family,epoch=epoch+1);atomic_json(OUT/'runtime.json',rt)
     model.eval();ix=np.flatnonzero(test);ns=[];es=[]
     with torch.no_grad():
      for start in range(0,len(ix),128):
       ii=ix[start:start+128];a,b=model(images(ii),obs[ii],grip[ii],center);ns.append(a.cpu().numpy());es.append(b.cpu().numpy())
     name=f'{family}_seed{seed}'
     if name not in predictions:predictions[name]=(np.full(n['positions'].shape,np.nan,np.float32),np.full(n['relations'].shape,np.nan,np.float32))
     predictions[name][0][ix]=np.concatenate(ns);predictions[name][1][ix]=np.concatenate(es)
     np.savez_compressed(OUT/f'fold{fold}_{name}_prediction.npz',indices=ix,nodes=np.concatenate(ns),edges=np.concatenate(es))
     torch.save(dict(state_dict=model.cpu().state_dict(),fold=fold,seed=seed,family=family,loss=losses),OUT/f'fold{fold}_{name}.pt')
     rt['completed_fits']+=1;atomic_json(OUT/'runtime.json',rt);print(f'{rt["completed_fits"]}/18 {name} fold{fold}',flush=True)
  for family in ['shared','independent']:
   predictions[f'{family}_ensemble']=tuple(np.mean([predictions[f'{family}_seed{s}'][k] for s in [11,23,47]],0) for k in [0,1])
  metrics={}
  for name,(nodes,edges) in predictions.items():
   assert np.isfinite(nodes).all() and np.isfinite(edges).all();np.savez_compressed(OUT/f'oof_{name}.npz',nodes=nodes,edges=edges)
   metrics[name]=evaluate(nodes,edges,n['positions'],n['relations'],roots)
  wins=[]
  for seed in [11,23,47]:
   a=metrics[f'shared_seed{seed}'];b=metrics[f'independent_seed{seed}']
   wins.append(a['key_mean_mm']<=.9*b['key_mean_mm'] and a['worst_key_mm']<=.9*b['worst_key_mm'] and all(a['edge_mm'][k]<=1.1*b['edge_mm'][k] for k in KEY))
  result=dict(metrics=metrics,parameters=params,edges=EDGES,groups=GROUPS,key_edges=KEY,seed_gate=wins,shared_prioritized=sum(wins)>=2,independent_node_definition='edge(object,grip) + legal measured grip; other predicted edges need not agree',confirmation_opened=False)
  atomic_json(OUT/'evaluation.json',result)
  lines=['# 统一对象位置三折匹配实验','', '仅旧13场景4,843帧；3折×3 seeds×2结构，每次40 epochs。开发/确认数据未开启。','', '| 结构 | 关键边均值 mm | 每帧最坏关键边 mm | 三角闭合 mm |','|---|---:|---:|---:|']
  for name in ['mean_position','independent_ensemble','shared_ensemble']:
   m=metrics[name];lines.append(f"| {name} | {m['key_mean_mm']:.3f} | {m['worst_key_mm']:.3f} | {m['triangle_closure_mm']:.6f} |")
  lines+=['',f'逐seed预声明工程线：{wins}；是否优先共享结构：{sum(wins)>=2}。完整逐节点/36边/分组/场景结果见evaluation.json。','',f'参数数目：{params}。公共视觉与融合层一致，输出头不同；并非完全等参数实验。','', '共享结构闭合低是构造性质，不能替代误差改善。独立结构的节点位置由其对象—夹爪边恢复；不对其他边作事后投影。物理槽位并非语言身份；本轮不学习姿态、接触、速度或事件，也不证明闭环收益。','', '只使用最终epoch，不选择最佳checkpoint；当前数据属于反复使用的内部开发数据，不能称为独立泛化确认。']
  report=ROOT/'docs/unified_object_state_cv_audit.md';report.write_text('\n'.join(lines)+'\n')
  rt.update(status='COMPLETE',phase='reported',finished_epoch=time.time());atomic_json(OUT/'runtime.json',rt)
 except Exception:
  rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'runtime.json',rt);raise
if __name__=='__main__':main()
