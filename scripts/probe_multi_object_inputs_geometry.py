#!/usr/bin/env python3
"""Frozen-weight sensitivities and depth-free geometry diagnostics on old OOF scenes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.diagnose_multi_object_perception import inputs,read_npz,OUT,CV,DATA,CFG,NAMES,macro,stats,project,rays_to_plane,save
from scripts.train_unified_object_state_cv import Model,EDGES,evaluate
from src.probes.candidate_utility_oracle import atomic_json,sha256

def sensitivity():
 import torch
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 d,n,meta=inputs();roots=n['roots'];variants=json.loads(CFG.read_text())['sensitivity']+['shifted_bowl2_mask_control']
 seeds=[11,23,47];rng=np.random.default_rng(20261009);shuffle=np.arange(len(roots))
 # Sort roots: shuffle must not depend on Python hash randomization.
 for r in sorted(set(roots)):
  ix=np.flatnonzero(roots==r);shuffle[ix]=rng.permutation(ix)
 pred={};verification=[];runtime=dict(status='RUNNING',completed_models=0,maximum_models=18,training=False);atomic_json(OUT/'sensitivity_runtime.json',runtime)
 try:
  for fold in range(3):
   ix=np.flatnonzero(d['fold']==fold);norm=read_npz(CV/f'fold{fold}_normalization.npz')
   for family in ['shared','independent']:
    center=torch.tensor(norm[family],device='cuda');obs=torch.tensor((d['obs']-norm['obs_mean'])/norm['obs_std'],device='cuda');grip=torch.tensor(d['obs'][:,:3],device='cuda')
    for seed in seeds:
     model=Model(family).cuda();ckpt=torch.load(CV/f'fold{fold}_{family}_seed{seed}.pt',map_location='cpu',weights_only=False);assert ckpt['fold']==fold and ckpt['seed']==seed and ckpt['family']==family
     model.load_state_dict(ckpt['state_dict']);model.eval()
     for variant in variants:
      name=f'{family}_seed{seed}_{variant}'
      if name not in pred:pred[name]=[np.zeros(n['positions'].shape,np.float32),np.zeros(n['relations'].shape,np.float32)]
      with torch.no_grad():
       for start in range(0,len(ix),128):
        ii=ix[start:start+128];ini=d['initial_images'][d['initial_ids'][ii]].copy();cur=d['images'][ii].copy()
        if variant=='zero_current':cur[:]=0
        elif variant=='zero_initial':ini[:]=0
        elif variant=='shuffle_current_within_scene':cur=d['images'][shuffle[ii]]
        elif variant in ['mask_bowl1','mask_bowl2','shifted_bowl2_mask_control']:
         obj=0 if variant=='mask_bowl1' else 1
         mask=np.repeat(np.repeat(d['masks'][ii,obj]>0,8,1),8,2)
         if variant=='shifted_bowl2_mask_control':mask=np.roll(mask,32,axis=2)
         cur[mask]=0
        im=np.concatenate([ini,cur],-1).transpose(0,3,1,2).copy();obsbatch=obs[ii] if variant!='mean_proprio_action' else torch.zeros_like(obs[ii])
        a,b=model(torch.tensor(im,device='cuda',dtype=torch.float32)/255,obsbatch,grip[ii],center)
        pred[name][0][ii]=a.cpu().numpy();pred[name][1][ii]=b.cpu().numpy()
     expected=read_npz(CV/f'fold{fold}_{family}_seed{seed}_prediction.npz');actual=pred[f'{family}_seed{seed}_original']
     for k,key in enumerate(['nodes','edges']):np.testing.assert_allclose(actual[k][ix],expected[key],atol=2e-7,rtol=0)
     verification.append(dict(fold=fold,family=family,seed=seed,max_position_difference_m=float(abs(actual[0][ix]-expected['nodes']).max())))
     runtime['completed_models']+=1;atomic_json(OUT/'sensitivity_runtime.json',runtime);print(runtime['completed_models'],flush=True)
  metrics={};arrays={}
  for family in ['shared','independent']:
   base=np.mean([pred[f'{family}_seed{s}_original'][0] for s in seeds],0)
   for variant in variants:
    nodes,edges=[np.mean([pred[f'{family}_seed{s}_{variant}'][k] for s in seeds],0) for k in [0,1]]
    name=f'{family}_{variant}';m=evaluate(nodes,edges,n['positions'],n['relations'],roots)
    m['prediction_shift_mm']=macro(np.linalg.norm(nodes-base,axis=2)*1000,roots).tolist()
    m['per_seed']={str(s):evaluate(*pred[f'{family}_seed{s}_{variant}'],n['positions'],n['relations'],roots) for s in seeds}
    metrics[name]=m;arrays[name]=nodes
  np.savez_compressed(OUT/'sensitivity_predictions.npz',**arrays,shuffle=shuffle)
  save('input_sensitivity.json',dict(training=False,models=metrics,original_forward_checks=verification,caveat='Test-time perturbation, not retrained modality ablation; zero input is distribution shift. Masks are privileged coarse visible-region interventions, not deployable input. Proprio intervention keeps separately measured grip fixed. Sensitivity alone does not establish correct object localization.'))
  runtime.update(status='COMPLETE');atomic_json(OUT/'sensitivity_runtime.json',runtime)
 except Exception:
  import traceback
  runtime.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'sensitivity_runtime.json',runtime);raise

def geometry():
 d,n,meta=inputs();roots=n['roots'];truth=n['positions'];cam=json.loads((ROOT/'results/camera_projection_contract/camera_inventory.json').read_text())['camera_reference'];M=np.array(cam['world_to_raw_pixel']);uv,_=project(truth[:,:5],M)
 heights=np.zeros((len(roots),5));priors={}
 for fold in range(3):
  fit=d['fold']!=fold;test=~fit
  z=np.median([np.median(truth[roots==r,:5,2],axis=0) for r in sorted(set(roots[fit]))],axis=0)
  heights[test]=z;priors[str(fold)]=z.tolist()
 results={};arrays={};rng=np.random.default_rng(20261009)
 for px in [0,1,2,4]:
  repeats=[]
  for repeat in range(1 if px==0 else 5):
   imagepoint=uv+rng.normal(0,px,uv.shape);xyz,valid=rays_to_plane(imagepoint,heights,M)
   assert valid.all(),'Invalid ray; must retain failures instead of dropping'
   err=np.linalg.norm(xyz-truth[:,:5],axis=2)*1000
   repeats.append(dict(errors=stats(err,roots),scenes={r:err[roots==r].mean(0).tolist() for r in sorted(set(roots))}))
   arrays[f'plane_oracle_uv_noise{px}_repeat{repeat}']=xyz
  results[str(px)]=repeats
 predicted_uv=np.zeros((len(roots),3,2))
 for fold in range(3):
  ps=[]
  for seed in [11,23,47]:
   q=read_npz(DATA/f'fold{fold}_localized_seed{seed}_prediction.npz');np.testing.assert_array_equal(q['indices'],np.flatnonzero(d['fold']==fold));ps.append(q['points'][:,:6].reshape(-1,3,2))
  predicted_uv[d['fold']==fold]=np.mean(ps,0)
 natural,valid=rays_to_plane(predicted_uv,heights[:,:3],M);assert valid.all();arrays['plane_existing_oof_uv']=natural
 natural_error=stats(np.linalg.norm(natural-truth[:,:3],axis=2)*1000,roots)
 # Existing learned UV+depth path: predicted, never GT, depth.
 learned=np.zeros((len(roots),3,3))
 for fold in range(3):
  points=np.mean([read_npz(DATA/f'fold{fold}_localized_seed{s}_prediction.npz')['points'] for s in [11,23,47]],0)
  raw=points[:,:6].reshape(-1,3,2).copy();raw[:,:,1]=127-raw[:,:,1];dep=points[:,6:9]
  hp=np.concatenate([raw*dep[:,:,None],dep[:,:,None],np.ones(dep.shape+(1,))],-1)@np.linalg.inv(M).T
  learned[d['fold']==fold]=hp[:,:,:3]
 arrays['existing_oof_uv_predicted_depth']=learned
 # Conditional near-height groups are diagnostic strata, never a deployment routing rule.
 clean=arrays['plane_oracle_uv_noise0_repeat0'];err=np.linalg.norm(clean-truth[:,:5],axis=2)*1000
 conditional={NAMES[o]:{label:stats(err[mask,o],roots[mask]) for label,mask in [('height_prior_within_5mm',abs(heights[:,o]-truth[:,o,2])<=.005),('height_prior_over_5mm',abs(heights[:,o]-truth[:,o,2])>.005)]} for o in range(5)}
 np.savez_compressed(OUT/'geometry_predictions.npz',height_priors=heights,predicted_uv=predicted_uv,**arrays)
 save('geometry_feasibility.json',dict(method='Ray intersection with fit-only object-origin height prior; not a universal 3D reconstruction',inputs_A='GT projected body-origin UV + calibrated camera + fit-scene median height; no heldout GT depth or Z supplied to solver',inputs_B='same + iid Gaussian 128px coordinate noise, seeds fixed; not natural detector noise',inputs_C='existing old-fold localized UV ensemble + same fit-only height prior; uses no GT keypoints or test depth',priors_by_fold=priors,noise_results=results,conditional_height_strata=conditional,natural_existing_oof_uv=natural_error,existing_oof_uv_predicted_depth=stats(np.linalg.norm(learned-truth[:,:3],axis=2)*1000,roots),invalid_natural_rays=int((~valid).sum()),scope='C covers bowls and plate only; no masks/keypoints for remaining five objects. No model retraining; previously trained auxiliary head is not optimized for this geometric objective.',observability='One view plus unknown depth leaves a ray. Different support heights violate single-plane assumption. Fixed initial/current camera is not a calibrated stereo pair. Mesh exists but identifiable natural CAD keypoints are not established; bowl yaw symmetry is separate from depth ambiguity.',oracle=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['sensitivity','geometry']);a=p.parse_args()
 target=OUT/('sensitivity_runtime.json' if a.stage=='sensitivity' else 'geometry_feasibility.json')
 if target.exists():raise RuntimeError('Existing attempt retained: '+str(target))
 globals()[a.stage]();print(a.stage,'complete')
