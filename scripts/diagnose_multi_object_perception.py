#!/usr/bin/env python3
"""Old-scene forensic analysis. Never reads new-development or confirmation datasets."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,argparse,subprocess,time,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.candidate_utility_oracle import atomic_json,sha256
from scripts.train_unified_object_state_cv import Model,EDGES,KEY,evaluate
OUT=ROOT/'results/multi_object_perception_diagnosis';CV=ROOT/'results/unified_object_state_cv';DATA=ROOT/'results/relational_localization_cv'
CFG=ROOT/'configs/multi_object_perception_diagnosis_protocol.json'
NAMES=['bowl1','bowl2','plate','cookies','ramekin','table','cabinet','stove','grip']

def read_npz(path):
 with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def inputs():
 d=read_npz(DATA/'dataset.npz');n=read_npz(ROOT/'results/unified_object_state_audit_v3/node_supervision.npz');meta=json.loads((DATA/'metadata.json').read_text())
 assert len(n['roots'])==4843 and len(set(n['roots']))==13
 np.testing.assert_array_equal(d['fold'],n['fold']);assert set(n['roots'])==set(meta['folds'])
 return d,n,meta
def macro(x,roots):return np.stack([np.mean(x[roots==r],axis=0) for r in sorted(set(roots))]).mean(0)
def stats(x,roots):
 if len(x)==0:return None
 return dict(frames=len(x),scenes=len(set(roots)),scene_macro_mean=np.asarray(macro(x,roots)).tolist(),scene_macro_median=np.mean([np.median(x[roots==r],axis=0) for r in sorted(set(roots))],axis=0).tolist(),scene_macro_p90=np.mean([np.percentile(x[roots==r],90,axis=0) for r in sorted(set(roots))],axis=0).tolist(),scene_macro_p95=np.mean([np.percentile(x[roots==r],95,axis=0) for r in sorted(set(roots))],axis=0).tolist(),pooled_quantiles_descriptive=np.percentile(x,[50,90,95],axis=0).tolist())
def project(x,M):
 z=np.concatenate([x,np.ones(x.shape[:-1]+(1,))],-1)@M.T
 uv=z[...,:2]/z[...,2,None];uv[...,1]=127-uv[...,1]
 return uv,z[...,2]
def rays_to_plane(uv,zprior,M):
 raw=uv.copy();raw[...,1]=127-raw[...,1];inv=np.linalg.inv(M)
 c=(inv@np.array([0.,0.,0.,1.]))[:3]
 directions=np.concatenate([raw,np.ones(raw.shape[:-1]+(1,)),np.zeros(raw.shape[:-1]+(1,))],-1)@inv.T;v=directions[...,:3]
 den=v[...,2];valid=np.abs(den)>1e-9
 t=np.divide(np.asarray(zprior)-c[2],den,out=np.zeros_like(den),where=valid)
 valid&=t>0
 out=c+v*t[...,None]
 return out,valid

def permute_edges(edges,perm):
 lookup={e:k for k,e in enumerate(EDGES)};out=np.empty_like(edges)
 for k,(i,j) in enumerate(EDGES):
  a,b=perm[i],perm[j];out[:,k]=edges[:,lookup[tuple(sorted((a,b)))]]*(1 if a<b else -1)
 return out

def save(name,x):
 path=OUT/name
 if path.exists():raise RuntimeError('Refusing overwrite '+str(path))
 atomic_json(path,x)

def baseline():
 d,n,meta=inputs();checks={};hashes={};normalization=[]
 for config in ['configs/unified_object_state_cv_protocol.json','configs/unified_object_state_audit_v3_protocol.json']:
  c=json.loads((ROOT/config).read_text())
  for p,h in c['frozen_files'].items():assert sha256(ROOT/p)==h,p;hashes[p]=h
 w=np.array([1/(n['roots']==r).sum() for r in n['roots']],np.float32)
 for f in range(3):
  fit=d['fold']!=f;test=~fit;assert not set(n['roots'][fit])&set(n['roots'][test]);norm=read_npz(CV/f'fold{f}_normalization.npz');weights=w[fit]/w[fit].sum()
  mean=(d['obs'][fit]*weights[:,None]).sum(0);std=np.sqrt(((d['obs'][fit]-mean)**2*weights[:,None]).sum(0)).clip(.001)
  np.testing.assert_array_equal(norm['obs_mean'],mean);np.testing.assert_array_equal(norm['obs_std'],std)
  for family,v in [('shared',n['positions'][fit,:8]),('independent',n['relations'][fit])]:np.testing.assert_array_equal(norm[family],(v.astype(float)*weights[:,None,None]).sum(0).astype(np.float32))
  normalization.append(dict(fold=f,fit_scenes=sorted(set(n['roots'][fit])),test_scenes=sorted(set(n['roots'][test]))))
 original=json.loads((CV/'evaluation.json').read_text())
 for name in original['metrics']:
  p=read_npz(CV/f'oof_{name}.npz');m=evaluate(p['nodes'],p['edges'],n['positions'],n['relations'],n['roots']);assert json.dumps(m,sort_keys=True)==json.dumps(original['metrics'][name],sort_keys=True);checks[name]=m
  hashes[str((CV/f'oof_{name}.npz').relative_to(ROOT))]=sha256(CV/f'oof_{name}.npz')
  if '_seed' in name:
   seen=np.zeros(4843,int)
   for fold in range(3):
    q=read_npz(CV/f'fold{fold}_{name}_prediction.npz');ix=q['indices'];np.testing.assert_array_equal(ix,np.flatnonzero(d['fold']==fold));np.testing.assert_array_equal(p['nodes'][ix],q['nodes']);np.testing.assert_array_equal(p['edges'][ix],q['edges']);seen[ix]+=1
    weight=CV/f'fold{fold}_{name}.pt';assert weight.exists();hashes[str(weight.relative_to(ROOT))]=sha256(weight)
   assert (seen==1).all()
 for family in ['shared','independent']:
  for field in ['nodes','edges']:
   np.testing.assert_array_equal(read_npz(CV/f'oof_{family}_ensemble.npz')[field],np.mean([read_npz(CV/f'oof_{family}_seed{s}.npz')[field] for s in [11,23,47]],0))
 for r in sorted(set(n['roots'])):
  ii=np.flatnonzero(n['roots']==r);assert len(set(d['initial_ids'][ii]))==1
  first=ii[[meta['rows'][j]['step']==0 for j in ii]][0];np.testing.assert_array_equal(d['initial_images'][d['initial_ids'][first]],d['images'][first])
 camera=ROOT/'results/camera_projection_contract/camera_inventory.json';hashes[str(camera.relative_to(ROOT))]=sha256(camera)
 save('baseline_manifest.json',dict(status='PASS',git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),sources=hashes,folds=normalization,metrics=checks,data_shapes={k:list(v.shape) for k,v in d.items()},statistical_unit='scene macro; frames are correlated; seeds share scenes',input_audit='Model.forward receives images, normalized legal obs, measured grip and fit-only centers. Paths/scene IDs/GT pose/masks only index data or form supervision; no direct network input.',history='initial RGB plus current RGB; previous action, no future images; 128 RGB downsampled by 2x2 integer box average',limitations='Historical architecture selection used these same old scenes; this is not independent confirmation.',source_files_unchanged=True))

def diagnose():
 d,n,meta=inputs();roots=n['roots'];truth=n['positions'];times=np.array([r['step'] for r in meta['rows']]);analysis={};identity={};visibility={}
 M=np.array(json.loads((ROOT/'results/camera_projection_contract/camera_inventory.json').read_text())['camera_reference']['world_to_raw_pixel']);uv,depth=project(truth,M)
 np.testing.assert_allclose(uv[:,:3].reshape(-1,6),d['points'][:,:6],atol=1e-4,rtol=0)
 area=d['masks'].sum((2,3))*256
 for name in json.loads((CV/'evaluation.json').read_text())['metrics']:
  p=read_npz(CV/f'oof_{name}.npz');delta=(p['nodes']-truth)*1000;err=np.linalg.norm(delta,axis=2)
  strata={}
  for obj in range(5):
   bins={}
   for label,mask in [('t0',times==0),('t1_31',(times>=1)&(times<32)),('t32_63',(times>=32)&(times<64)),('t64_95',(times>=64)&(times<96)),('t96plus',times>=96),('previous_open',d['obs'][:,-1]<-.5),('previous_close',d['obs'][:,-1]>.5),('moving_gt_speed_gt_10mm_s',np.linalg.norm(n['object_pose_velocity'][:,obj,7:10],axis=1)>.01),('both_pads',n['pad_contact'][:,:,obj].all(1)),('external_contact',n['non_gripper_contact'][:,obj].astype(bool))]:
    bins[label]=stats(err[mask,obj],roots[mask])
   strata[NAMES[obj]]=bins
  coverage={}
  for r in sorted(set(roots)):
   mask=roots==r;fold=int(d['fold'][mask][0]);fit=d['fold']!=fold;center=read_npz(CV/f'fold{fold}_normalization.npz')['shared'];v=truth[mask,1];vp=p['nodes'][mask,1]
   outside=(v<truth[fit,1].min(0))|(v>truth[fit,1].max(0));distance=np.maximum(truth[fit,1].min(0)-v,0)+np.minimum(truth[fit,1].max(0)-v,0)
   # Exact nearest training coordinate is a descriptive GT-space coverage diagnostic, never input.
   nearest=[]
   for batch in np.array_split(v,max(1,len(v)//64)):nearest.extend(np.sqrt(((batch[:,None]-truth[fit,1])**2).sum(2)).min(1).tolist())
   toward=center[1]-v;res=vp-v;cos=np.sum(toward*res,1)/(np.linalg.norm(toward,axis=1)*np.linalg.norm(res,axis=1)+1e-12)
   coverage[r]=dict(fold=fold,frames=int(mask.sum()),node_error_mm=err[mask].mean(0).tolist(),bowl2_signed_xyz_mm=delta[mask,1].mean(0).tolist(),bowl2_truth_min=v.min(0).tolist(),bowl2_truth_max=v.max(0).tolist(),bowl2_pred_min=vp.min(0).tolist(),bowl2_pred_max=vp.max(0).tolist(),bowl2_fit_mean=center[1].tolist(),bowl2_outside_fit_bbox_fraction=outside.any(1).mean().item(),bowl2_bbox_distance_mm=float(np.linalg.norm(distance,axis=1).mean()*1000),bowl2_nearest_fit_position_mm=float(np.mean(nearest)*1000),bowl2_prediction_to_fit_mean_mm=float(np.linalg.norm(vp-center[1],axis=1).mean()*1000),bowl2_truth_to_fit_mean_mm=float(np.linalg.norm(v-center[1],axis=1).mean()*1000),bowl2_residual_toward_fit_mean_cosine=float(cos.mean()))
  analysis[name]=dict(euclidean_mm=stats(err,roots),axis_mae_mm=stats(abs(delta),roots),signed_bias_mm=macro(delta,roots).tolist(),xy_mm=stats(np.linalg.norm(delta[:,:,:2],axis=2),roots),height_abs_mm=stats(abs(delta[:,:,2]),roots),strata=strata,scenes=coverage,folds={str(f):stats(err[d['fold']==f],roots[d['fold']==f]) for f in range(3)})
  if name not in ['shared_ensemble','independent_ensemble']:continue
  perm=[1,0,2,3,4,5,6,7,8];pn=p['nodes'][:,perm];pe=permute_edges(p['edges'],perm);swerr=np.linalg.norm(pn-truth,axis=2)
  use=swerr[:,:2].sum(1)<(err[:,:2]/1000).sum(1);on=p['nodes'].copy();oe=p['edges'].copy();on[use]=pn[use];oe[use]=pe[use]
  sceneuse=np.zeros(len(roots),bool)
  for r in set(roots):
   mask=roots==r
   if swerr[mask,:2].sum()<(err[mask,:2]/1000).sum():sceneuse[mask]=True
  sn=p['nodes'].copy();se=p['edges'].copy();sn[sceneuse]=pn[sceneuse];se[sceneuse]=pe[sceneuse]
  identity[name]=dict(oracle=True,criterion='min summed Euclidean bowl1+bowl2 error; two objects means identity or swap is exact optimal assignment; not optimized for key-edge metric',original=evaluate(p['nodes'],p['edges'],truth,n['relations'],roots),always_swap=evaluate(pn,pe,truth,n['relations'],roots),per_frame_best=evaluate(on,oe,truth,n['relations'],roots),per_scene_fixed_best=evaluate(sn,se,truth,n['relations'],roots),scene_macro_swap_fraction=float(macro(use,roots)),scenes={r:dict(frames=int((roots==r).sum()),swap_fraction=float(use[roots==r].mean()),bowl2_original_mm=float(err[roots==r,1].mean()),bowl2_oracle_mm=float(np.linalg.norm(on[roots==r,1]-truth[roots==r,1],axis=1).mean()*1000)) for r in sorted(set(roots))})
  vis={}
  for obj in range(3):
   fov=(depth[:,obj]>0)&(uv[:,obj,0]>=0)&(uv[:,obj,0]<128)&(uv[:,obj,1]>=0)&(uv[:,obj,1]<128)
   vis[NAMES[obj]]=dict(visible_pixels=stats(area[:,obj],roots),body_origin_in_frame_fraction=float(macro(fov,roots)),strata={label:stats(err[mask,obj],roots[mask]) for label,mask in [('zero_visible',area[:,obj]<1),('1_63_pixels',(area[:,obj]>=1)&(area[:,obj]<64)),('64_255_pixels',(area[:,obj]>=64)&(area[:,obj]<256)),('256plus',area[:,obj]>=256)]},scenes={r:dict(mean_visible_pixels=float(area[roots==r,obj].mean()),zero_visible_fraction=float((area[roots==r,obj]<1).mean()),mean_error_mm=float(err[roots==r,obj].mean())) for r in sorted(set(roots))})
  visibility[name]=vis
 save('object_error_analysis.json',dict(statistics='scene macro means and mean of within-scene quantiles; pooled quantiles descriptive only',models=analysis,paired_shared_minus_independent_mm={r:(np.array(analysis['shared_ensemble']['scenes'][r]['node_error_mm'])-analysis['independent_ensemble']['scenes'][r]['node_error_mm']).tolist() for r in sorted(set(roots))}))
 save('identity_diagnosis.json',identity)
 save('visibility_diagnosis.json',dict(models=visibility,known='first 3 object visible geom masks are from exact RGB-aligned native128 instance rendering; area reconstructed from fractional pooled masks',unknown='no amodal masks: cannot identify occlusion fraction or separate occluders; remaining objects have no masks and are not labelled invisible',projection='body origin in-frame is not full-object visibility or visible centroid',appearance='Both bowls instantiate the same akita_black_bowl mesh/texture; instance name is not an appearance cue. Position/history can distinguish instances, but ambiguous occlusion remains.',source='scripts/collect_relational_localization.py'))
 np.savez_compressed(OUT/'diagnostic_frames.npz',times=times,uv=uv,depth=depth,visible_pixels=area)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['baseline','diagnose']);args=parser.parse_args();OUT.mkdir(exist_ok=True)
 globals()[args.stage]()
 print(args.stage,'complete')
