#!/usr/bin/env python3
"""Old-scene localization supervision, strictly matched to saved RGB."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys,json,time,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.libero'))
import numpy as np
from src.probes.candidate_utility_oracle import atomic_json,sha256
from src.probes.relational_dynamics_pilot import relation
from src.probes.camera_projection_contract import official_float_projection
OUT=ROOT/'results/relational_localization_cv'

def main():
 cfg=json.loads((ROOT/'configs/relational_localization_cv_protocol.json').read_text())
 for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
 OUT.mkdir(exist_ok=True)
 if (OUT/'collection_runtime.json').exists():raise RuntimeError('Existing collection retained')
 rt=dict(status='RUNNING',phase='exact_rgb_and_labels',frames=0,maximum_frames=8208,started_epoch=time.time());atomic_json(OUT/'collection_runtime.json',rt)
 env=None
 try:
  import mujoco
  from scripts.evaluate_c0_all import make_env
  from scripts.inventory_visual_target_masks import render_rgb,render_segmentation,model_geom_ids
  from scripts.audit_camera_projection_contract import camera_record
  records=json.loads((ROOT/'results/relational_event_h24/manifest.json').read_text())['records']
  contracts=json.loads((ROOT/'configs/task_grounding_contract.json').read_text())['tasks']
  reference=json.loads((ROOT/'results/camera_projection_contract/camera_inventory.json').read_text())['camera_reference']
  cache={};initials={};images=[];initial_ids=[];obs=[];target=[];points=[];masks=[];meta=[];active=None;sources={}
  for rec in records:
   root=rec['root_id'];path=ROOT/rec['dense']['arrays'];assert sha256(path)==rec['dense']['sha256'];sources[str(path.relative_to(ROOT))]=rec['dense']['sha256']
   with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in ['qpos','qvel','rgb','natural_step','proprio','pose','grip_rot','grip_pos','grip_velocity','actions']}
   small=(a['rgb'].astype(np.uint16).reshape(-1,64,2,64,2,3).sum((2,4))//4).astype(np.uint8)
   if root not in initials:initials[root]=small[0]
   else:np.testing.assert_array_equal(initials[root],small[0])
   if active!=root:
    if env:env.close()
    task=int(root[4:6]);env=make_env(ROOT/contracts[task]['bddl_path']);env.seed(30000+100*task+int(root[-2:])-30);env.env.reset();active=root
    cam=camera_record(env.env,'agentview');np.testing.assert_allclose(cam['world_to_raw_pixel'],reference['world_to_raw_pixel'],atol=1e-10,rtol=0)
    geom_ids=[list(model_geom_ids(env.sim,env.env.get_object(n))) for n in ['akita_black_bowl_1','akita_black_bowl_2','plate_1']]
    assert all(geom_ids)
   rel=relation(a)
   for j,t in enumerate(a['natural_step']):
    pa=np.r_[a['proprio'][j],np.zeros(7) if t==0 else a['actions'][t-1]].astype(np.float32);y=np.r_[rel[t],a['grip_velocity'][t]].astype(np.float32)
    key=(root,int(t),hashlib.sha256(small[j].tobytes()+pa.tobytes()+y.tobytes()).hexdigest())
    if key in cache:continue
    s=env.sim;s.data.qpos[:]=a['qpos'][t];s.data.qvel[:]=a['qvel'][t]
    if t>0:
     assert int(s.model.opt.integrator)==0
     mujoco.mj_integratePos(s.model._model,s.data.qpos,s.data.qvel,-s.model.opt.timestep)
    s.forward();rgb=render_rgb(env.env)
    if not np.array_equal(rgb,a['rgb'][j]):
     np.savez_compressed(OUT/'mismatch.npz',stored=a['rgb'][j],reconstructed=rgb)
     raise AssertionError(f'RGB mismatch {root} t={t} source={path.name}; labels not admitted')
    seg=render_segmentation(env.env)
    mm=np.array([(seg[...,0]==5)&np.isin(seg[...,1],ids) for ids in geom_ids],np.float32).reshape(3,8,16,8,16).mean((2,4))
    uv,depth=official_float_projection(a['pose'][t,:3,:3],np.array(cam['world_to_raw_pixel']));uv[:,1]=127-uv[:,1];assert (depth>0).all()
    cache[key]=len(images);images.append(small[j]);initial_ids.append(list(initials).index(root));obs.append(pa);target.append(y);points.append(np.r_[uv.flatten(),depth]);masks.append(mm);meta.append(dict(root=root,step=int(t),source=str(path.relative_to(ROOT))))
    rt.update(frames=len(images),root=root,step=int(t));assert len(images)<=rt['maximum_frames']
    if len(images)%100==0:atomic_json(OUT/'collection_runtime.json',rt)
  roots=sorted(initials);folds={r:i%3 for i,r in enumerate(roots)}
  np.savez_compressed(OUT/'dataset.npz',images=np.array(images),initial_images=np.stack(list(initials.values())),initial_ids=np.array(initial_ids),obs=np.array(obs),target=np.array(target),points=np.array(points,np.float32),masks=np.array(masks),fold=np.array([folds[m['root']] for m in meta]))
  atomic_json(OUT/'metadata.json',dict(rows=meta,folds=folds,sources=sources,exact_rgb_frames=len(images),confirmation_opened=False,new_development_opened=False))
  rt.update(status='COMPLETE',phase='all_labels_rgb_verified',finished_epoch=time.time(),dataset_sha256=sha256(OUT/'dataset.npz'));atomic_json(OUT/'collection_runtime.json',rt)
  print(json.dumps(rt),flush=True)
 except Exception:
  rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'collection_runtime.json',rt);raise
 finally:
  if env:env.close()
if __name__=='__main__':main()
