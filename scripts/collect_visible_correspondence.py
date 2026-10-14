#!/usr/bin/env python3
"""Exact native128 visible masks for five objects, old 4843 rows only."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.libero'))
import numpy as np
from scipy.spatial import ConvexHull
from src.probes.candidate_utility_oracle import atomic_json,sha256
OUT=ROOT/'results/visible_correspondence_audit'
NAMES=['akita_black_bowl_1','akita_black_bowl_2','plate_1','cookies_1','glazed_rim_porcelain_ramekin_1']

def local_mesh(sim,ids,bodyid):
 """Compile fixed model geometry in object body frame, never a pose estimate."""
 points=[];R=sim.data.body_xmat[bodyid].reshape(3,3);origin=sim.data.body_xpos[bodyid]
 for gid in ids:
  if int(sim.model.geom_type[gid])!=7 or int(sim.model.geom_group[gid])!=1:continue
  mid=int(sim.model.geom_dataid[gid]);adr=int(sim.model.mesh_vertadr[mid]);num=int(sim.model.mesh_vertnum[mid]);v=np.asarray(sim.model.mesh_vert[adr:adr+num],float)
  world=v@sim.data.geom_xmat[gid].reshape(3,3).T+sim.data.geom_xpos[gid]
  points.append((world-origin)@R)
 if not points:raise RuntimeError('No visual mesh for body')
 pts=np.concatenate(points);hull=ConvexHull(pts);return pts[hull.vertices]

def main():
 cfg=json.loads((ROOT/'configs/visible_correspondence_protocol.json').read_text())
 for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
 OUT.mkdir(exist_ok=True)
 if (OUT/'collection_runtime.json').exists():raise RuntimeError('Existing collection retained')
 rt=dict(status='RUNNING',frames=0,maximum_frames=4843,env_step_calls=0,training=False,started_epoch=time.time());atomic_json(OUT/'collection_runtime.json',rt);env=None
 try:
  import mujoco
  from scripts.evaluate_c0_all import make_env
  from scripts.inventory_visual_target_masks import render_rgb,render_segmentation,model_geom_ids
  from scripts.audit_camera_projection_contract import camera_record
  meta=json.loads((ROOT/'results/relational_localization_cv/metadata.json').read_text());rows=meta['rows'];assert len(rows)==4843
  with np.load(ROOT/'results/relational_localization_cv/dataset.npz',allow_pickle=False) as f:oldimages=f['images'];oldmasks=f['masks'];fold=f['fold']
  recs=json.loads((ROOT/'results/relational_event_h24/manifest.json').read_text())['records'];records={r['dense']['arrays']:r for r in recs}
  contracts=json.loads((ROOT/'configs/task_grounding_contract.json').read_text())['tasks'];reference=json.loads((ROOT/'results/camera_projection_contract/camera_inventory.json').read_text())['camera_reference']
  roots=np.array([r['root'] for r in rows]);cases=np.concatenate([np.flatnonzero(roots==r)[np.linspace(0,(roots==r).sum()-1,8).astype(int)] for r in sorted(set(roots))]);case_set=set(cases.tolist());case_images={}
  packed=np.zeros((len(rows),5,128,16),np.uint8);areas=np.zeros((len(rows),5),int);bboxes=np.full((len(rows),5,4),np.nan);centroids=np.full((len(rows),5,2),np.nan)
  current=None;oldpath=None;a=None;sources={};shapes={};geometry=[];repeats=[]
  for i,row in enumerate(rows):
   path=row['source'];t=row['step'];rec=records[path];assert rec['dense']['geometry']['entities'][:5]==NAMES
   if path!=oldpath:
    assert sha256(ROOT/path)==meta['sources'][path];sources[path]=meta['sources'][path]
    with np.load(ROOT/path,allow_pickle=False) as f:a={k:f[k] for k in ['qpos','qvel','rgb','natural_step','pose']}
    oldpath=path
   if row['root']!=current:
    if env:env.close()
    current=row['root'];task=int(current[4:6]);env=make_env(ROOT/contracts[task]['bddl_path']);env.seed(30000+100*task+int(current[-2:])-30);env.env.reset()
    cam=camera_record(env.env,'agentview');np.testing.assert_allclose(cam['world_to_raw_pixel'],reference['world_to_raw_pixel'],atol=1e-10,rtol=0)
    ids=[sorted(model_geom_ids(env.sim,env.env.get_object(name))) for name in NAMES];assert all(ids)
    bodies=[env.sim.model.body_name2id(b) for b in rec['dense']['geometry']['bodies'][:5]]
    for k in range(5):
     # Include mesh geoms only; invisible collision primitives do not define silhouette shape prior.
     vertices=local_mesh(env.sim,ids[k],bodies[k])
     if k not in shapes:shapes[k]=vertices
     else:
      # Mesh hull ordering may differ numerically: compare min/max as calibration check.
      np.testing.assert_allclose(vertices.min(0),shapes[k].min(0),atol=1e-6,rtol=0);np.testing.assert_allclose(vertices.max(0),shapes[k].max(0),atol=1e-6,rtol=0)
    geometry.append(dict(root=current,entities=NAMES,geometry_ids=ids))
   s=env.sim;s.data.qpos[:]=a['qpos'][t];s.data.qvel[:]=a['qvel'][t]
   if t>0:
    assert int(s.model.opt.integrator)==0;mujoco.mj_integratePos(s.model._model,s.data.qpos,s.data.qvel,-s.model.opt.timestep)
   s.forward();j=int(np.flatnonzero(a['natural_step']==t)[0]);rgb=render_rgb(env.env)
   if not np.array_equal(rgb,a['rgb'][j]):
    np.savez_compressed(OUT/'rgb_mismatch.npz',rendered=rgb,stored=a['rgb'][j]);raise AssertionError(f'RGB mismatch row{i}')
   small=(rgb.astype(np.uint16).reshape(64,2,64,2,3).sum((1,3))//4).astype(np.uint8);np.testing.assert_array_equal(small,oldimages[i])
   seg=render_segmentation(env.env);masks=np.array([(seg[...,0]==5)&np.isin(seg[...,1],g) for g in ids])
   np.testing.assert_array_equal(masks[:3].reshape(3,8,16,8,16).mean((2,4)).astype(np.float32),oldmasks[i])
   if i==0 or rows[i-1]['root']!=current:
    seg2=render_segmentation(env.env);np.testing.assert_array_equal(seg,seg2);repeats.append(i)
   packed[i]=np.packbits(masks,axis=-1);areas[i]=masks.sum((1,2))
   for k in range(5):
    y,x=np.nonzero(masks[k])
    if len(x):bboxes[i,k]=[x.min()-.5,y.min()-.5,x.max()+.5,y.max()+.5];centroids[i,k]=[x.mean(),y.mean()]
   if i in case_set:case_images[i]=rgb.copy()
   if i%100==0:rt.update(frames=i+1,root=current);atomic_json(OUT/'collection_runtime.json',rt)
  env.close();env=None
  np.savez_compressed(OUT/'visible_masks.npz',packed_masks=packed,area=areas,bbox=bboxes,visible_centroid=centroids,roots=roots,fold=fold)
  np.savez_compressed(OUT/'geometry_cases.npz',indices=cases,rgb=np.stack([case_images[i] for i in cases]))
  np.savez_compressed(OUT/'known_mesh_hulls.npz',**{f'object{k}':v for k,v in shapes.items()})
  atomic_json(OUT/'collection_manifest.json',dict(frames=len(rows),scenes=13,objects=NAMES,source_sha256=sources,scene_geometry=geometry,camera=reference,repeat_rows=repeats,exact_rgb_frames=len(rows),first_three_masks_match_old=True,statistical_unit='scene; repeated frames are not independent scenes',label_semantics='Native visible instance raster, not amodal mask or body-center annotation. Repeated renderer agreement is label reproducibility, not human annotation or natural recoverability.',shape_prior='Fixed model mesh hull in body frame, includes known metric size. Reset object pose cancels in coordinate conversion, never used as test pose for fitting.',new_development_opened=False,confirmation_opened=False,dataset_sha256=sha256(OUT/'visible_masks.npz')))
  rt.update(status='COMPLETE',frames=len(rows),finished_epoch=time.time());atomic_json(OUT/'collection_runtime.json',rt)
 except Exception:
  rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'collection_runtime.json',rt);raise
 finally:
  if env:env.close()
if __name__=='__main__':main()
