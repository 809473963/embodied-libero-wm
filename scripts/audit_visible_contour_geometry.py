#!/usr/bin/env python3
"""Known-shape, visible-mask geometry diagnostic; no learned detector or hidden GT pose."""
import sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scripts.diagnose_multi_object_perception import inputs,read_npz,project,stats,macro
from scripts.train_unified_object_state_cv import evaluate
from scripts.collect_visible_correspondence import OUT,NAMES
from src.probes.candidate_utility_oracle import atomic_json,sha256

def bounds(mask):
 y,x=np.nonzero(mask)
 if len(x)<4:return None
 return np.array([x.min()-.5,y.min()-.5,x.max()+.5,y.max()+.5])
def perturb(mask,variant):
 x=mask.astype(np.uint8)
 if variant=='erode1':return cv2.erode(x,np.ones((3,3),np.uint8)).astype(bool)
 if variant=='dilate1':return cv2.dilate(x,np.ones((3,3),np.uint8)).astype(bool)
 if variant=='res64':return np.repeat(np.repeat(x.reshape(64,2,64,2).mean((1,3))>=.5,2,0),2,1)
 return mask

def mesh_projection(vertices,R,camera_translation,K,camera_rotation):
 camverts=vertices@R.T@camera_rotation.T+camera_translation
 h=camverts@K.T;uv=h[:,:2]/h[:,2,None];uv[:,1]=127-uv[:,1]
 return uv,camverts[:,2]

def fit_shape(mask,vertices,R,K,camera_pose,mode='bbox'):
 box=bounds(mask)
 if box is None:return None
 contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
 contour=np.concatenate([x[:,0,:] for x in contours],0).astype(float)
 contour=contour[np.linspace(0,len(contour)-1,min(48,len(contour))).astype(int)]
 camR=camera_pose[:3,:3].T;u,v=(box[:2]+box[2:])/2;ray=np.linalg.inv(K)@np.array([u,127-v,1.])
 def residual(t):
  uv,z=mesh_projection(vertices,R,t,K,camR)
  if np.min(z)<=.02:return np.full(4+(len(contour) if mode=='contour' else 0),1e4)
  b=np.r_[uv.min(0),uv.max(0)];r=(b-box)
  if mode=='contour':
   hull=cv2.convexHull(uv.astype(np.float32)).reshape(-1,2)
   dist=np.array([cv2.pointPolygonTest(hull,(float(p[0]),float(p[1])),True) for p in contour])
   r=np.r_[r,dist/np.sqrt(len(dist))*2]
  return r
 candidates=[]
 for depth in [.4,.8,1.2]:
  initial=ray*depth
  fit=least_squares(residual,initial,bounds=([-2.,-2.,.15],[2.,2.,2.5]),max_nfev=60,loss='soft_l1',f_scale=1,diff_step=1e-4)
  candidates.append(fit)
 # Choose by observable residual only. Neither translation nor orientation GT is consulted.
 fit=min(candidates,key=lambda x:x.cost);world=camera_pose[:3,:3]@fit.x+camera_pose[:3,3]
 uv,z=mesh_projection(vertices,R,fit.x,K,camR)
 return dict(xyz=world,residual_rms=float(np.sqrt(np.mean(residual(fit.x)**2))),solver_success=bool(fit.success),nfev=int(fit.nfev),projected_bbox=np.r_[uv.min(0),uv.max(0)],positive_depth=bool((z>0).all()))

def main():
 cfg=json.loads((ROOT/'configs/visible_contour_geometry_protocol.json').read_text())
 for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
 if (OUT/'geometry_runtime.json').exists():raise RuntimeError('Existing attempt retained')
 rt=dict(status='RUNNING',completed=0,started_epoch=time.time(),training=False);atomic_json(OUT/'geometry_runtime.json',rt)
 try:
  d,n,meta=inputs();m=read_npz(OUT/'visible_masks.npz');c=read_npz(OUT/'geometry_cases.npz');shapes=read_npz(OUT/'known_mesh_hulls.npz');ix=c['indices'];roots=n['roots'];truth=n['positions'];np.testing.assert_array_equal(m['roots'],roots)
  cam=json.loads((OUT/'collection_manifest.json').read_text())['camera'];K=np.array(cam['intrinsic']);cp=np.array(cam['corrected_camera_pose_world']);M=np.array(cam['world_to_raw_pixel'])
  # Prior orientations use fit scenes only, equal scene weights; no test pose enters shape fitting.
  Rs=np.zeros((3,5,3,3));prior={}
  for fold in range(3):
   fit=d['fold']!=fold;w=np.array([1/(roots[fit]==r).sum() for r in roots[fit]])
   for obj in range(5):
    q=n['object_pose_velocity'][fit,obj][:,[4,5,6,3]];Rs[fold,obj]=Rotation.from_quat(q).mean(weights=w).as_matrix()
   prior[str(fold)]=Rs[fold].tolist()
  masks=np.unpackbits(m['packed_masks'][ix],axis=-1).astype(bool);variants=['bbox_native','contour_native','contour_erode1','contour_dilate1','contour_res64'];results={};arrays={};cache={}
  oldshared=read_npz(ROOT/'results/unified_object_state_cv/oof_shared_ensemble.npz');oldind=read_npz(ROOT/'results/unified_object_state_cv/oof_independent_ensemble.npz');oldmean=read_npz(ROOT/'results/unified_object_state_cv/oof_mean_position.npz')
  for variant in variants:
   pred=np.full((len(ix),5,3),np.nan);rms=np.full((len(ix),5),np.nan);success=np.zeros((len(ix),5),bool);boxes=np.full((len(ix),5,4),np.nan)
   for j,i in enumerate(ix):
    fold=int(d['fold'][i]);mode='bbox' if variant.startswith('bbox') else 'contour';change=variant.split('_')[1]
    for obj in range(5):
     mask=perturb(masks[j,obj],change);key=(fold,obj,mode,mask.tobytes())
     if key not in cache:cache[key]=fit_shape(mask,shapes[f'object{obj}'],Rs[fold,obj],K,cp,mode)
     sol=cache[key]
     if sol is not None:
      pred[j,obj]=sol['xyz'];rms[j,obj]=sol['residual_rms'];success[j,obj]=sol['solver_success'] and sol['positive_depth'];boxes[j,obj]=sol['projected_bbox']
    rt['completed']+=1
    if j%20==0:rt.update(variant=variant);atomic_json(OUT/'geometry_runtime.json',rt)
   valid=np.isfinite(pred).all(2);err=np.linalg.norm(pred-truth[ix,:5],axis=2)*1000
   complete=valid.all(1);fallback=oldshared['nodes'][ix].copy()
   for obj in range(5):fallback[valid[:,obj],obj]=pred[valid[:,obj],obj]
   edges=np.stack([fallback[:,a]-fallback[:,b] for a,b in __import__('scripts.train_unified_object_state_cv',fromlist=['EDGES']).EDGES],1)
   metric=evaluate(fallback,edges,truth[ix],n['relations'][ix],roots[ix])
   # Low residual does not imply geometrically correct depth under partial masks.
   results[variant]=dict(valid_counts=valid.sum(0).tolist(),total_frames=len(ix),solver_converged_counts=success.sum(0).tolist(),complete_frames=int(complete.sum()),per_object_valid_only_mm={NAMES[o]:stats(err[valid[:,o],o],roots[ix][valid[:,o]]) for o in range(5)},all_rows_with_original_model_fallback=metric,complete_case_metrics=evaluate(fallback[complete],edges[complete],truth[ix][complete],n['relations'][ix][complete],roots[ix][complete]) if complete.any() else None,scenes={r:dict(frames=int((roots[ix]==r).sum()),object_valid_counts=valid[roots[ix]==r].sum(0).tolist(),object_error_valid_only_mm=[float(err[(roots[ix]==r)&valid[:,o],o].mean()) if ((roots[ix]==r)&valid[:,o]).any() else None for o in range(5)]) for r in sorted(set(roots))},residual_rms_valid_only=[float(rms[valid[:,o],o].mean()) if valid[:,o].any() else None for o in range(5)])
   arrays[variant]=pred;arrays[variant+'_bbox']=boxes;arrays[variant+'_valid']=valid
  np.savez_compressed(OUT/'geometry_predictions.npz',indices=ix,orientation_priors=Rs,**arrays)
  baselines={name:evaluate(p['nodes'][ix],p['edges'][ix],truth[ix],n['relations'][ix],roots[ix]) for name,p in [('shared',oldshared),('independent',oldind),('mean',oldmean)]}
  atomic_json(OUT/'geometry_evaluation.json',dict(variants=results,baselines_same104frames=baselines,orientation_priors=prior,shape_prior='Known metric CAD convex hull; upright/typical orientation estimated only from fold-fit quaternions',oracle_input='Visible instance segmentation including physical identity. No GT test position/depth/rotation is supplied to fitting. Current image detector not trained.',limitations='Convex projected outline, fixed fit orientation, partial visible masks may distort bbox/contour. These finite fitters are conditional oracle diagnostics, not best possible use of silhouette. No GT-assisted initialization or result selection.',failed_observation_policy='Less than4 mask pixels -> missing. Report valid counts; full population summary explicitly falls back to old shared predictions, not hidden case dropping.',cache_entries=len(cache)))
  uv,_=project(truth[:,:5],M);cent=m['visible_centroid'];offset=np.linalg.norm(cent-uv,axis=2);visibility={}
  for obj in range(5):
   valid=m['area'][:,obj]>0
   visibility[NAMES[obj]]=dict(area_pixels=stats(m['area'][:,obj].astype(float),roots),zero_area_scene_macro=float(macro(~valid,roots)),centroid_to_body_origin_pixels=stats(offset[valid,obj],roots[valid]),scenes={r:dict(frames=int((roots==r).sum()),zero_fraction=float((~valid[roots==r]).mean()),mean_visible_area=float(m['area'][roots==r,obj].mean())) for r in sorted(set(roots))})
  atomic_json(OUT/'visibility_evaluation.json',dict(objects=visibility,label_reproducibility='All4843 RGB frames exact; first3 pooled masks reproduce old labels; 13 repeated segmentations bitwise identical. Not evidence of human annotation repeatability or learned segmenter performance.',mask_available_for_all5=True,amodal_occlusion_fraction='unknown; projected CAD bounds are not full-scene occlusion-free raster area'))
  rt.update(status='COMPLETE',finished_epoch=time.time());atomic_json(OUT/'geometry_runtime.json',rt)
 except Exception:
  rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'geometry_runtime.json',rt);raise
if __name__=='__main__':main()
