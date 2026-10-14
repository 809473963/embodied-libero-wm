#!/usr/bin/env python3
"""Conditional numerical PnP oracle with known mesh correspondences, no GT depth input."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from scripts.diagnose_multi_object_perception import inputs,OUT,project,stats,save
from src.probes.candidate_utility_oracle import sha256

def solve(vertices,raw_uv,K,camera_pose):
 ok,r,t=cv2.solvePnP(vertices,raw_uv,K,None,flags=cv2.SOLVEPNP_EPNP)
 if not ok:return None
 ok,r,t=cv2.solvePnP(vertices,raw_uv,K,None,r,t,True,flags=cv2.SOLVEPNP_ITERATIVE)
 if not ok or not np.isfinite(t).all():return None
 xyz=(camera_pose@np.r_[t.ravel(),1])[:3]
 return xyz,bool(t[2,0]>0)

def main():
 if (OUT/'pnp_oracle.json').exists():raise RuntimeError('Existing PnP attempt retained')
 d,n,meta=inputs();cfg=json.loads((ROOT/'configs/multi_object_pnp_oracle_protocol.json').read_text());roots=n['roots'];ix=np.concatenate([np.flatnonzero(roots==r)[np.linspace(0,(roots==r).sum()-1,8).astype(int)] for r in sorted(set(roots))])
 mesh=ROOT/'third_party/LIBERO/libero/libero/assets/stable_scanned_objects/akita_black_bowl/visual/akita_black_bowl_vis.msh'
 with mesh.open('rb') as f:header=np.fromfile(f,np.int32,4);verts=np.fromfile(f,np.float32,3*int(header[0])).reshape(-1,3).astype(float)*.7
 # Fixed geometry-only sampling, independent of scene and performance.
 vertices=verts[::100];chosen=[int(vertices[:,0].argmin())]
 for _ in range(7):
  dist=np.linalg.norm(vertices[:,None]-vertices[chosen][None],axis=2).min(1);chosen.append(int(dist.argmax()))
 points=np.ascontiguousarray(vertices[chosen]);assert np.linalg.matrix_rank(points-points.mean(0))==3
 cam=json.loads((ROOT/'results/camera_projection_contract/camera_inventory.json').read_text())['camera_reference'];M=np.array(cam['world_to_raw_pixel']);K=np.array(cam['intrinsic']);cp=np.array(cam['corrected_camera_pose_world']);np.testing.assert_allclose(np.linalg.inv(cp)[:3],np.linalg.inv(K)@M[:3],atol=1e-12)
 world=[]
 for i in ix:
  row=[]
  for obj in [0,1]:
   pose=n['object_pose_velocity'][i,obj];rot=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix();row.append(points@rot.T+pose[:3])
  world.append(row)
 uv,depth=project(np.array(world),M);uv[...,1]=127-uv[...,1];rng=np.random.default_rng(cfg['seed']);outputs={};arrays={}
 for noise in cfg['noise_pixels_128']:
  runs=[]
  for repeat in range(1 if noise==0 else cfg['repetitions']):
   obs=uv+rng.normal(0,noise,uv.shape);pred=np.zeros((len(ix),2,3));failures=[];behind=0
   for j in range(len(ix)):
    for obj in [0,1]:
     sol=solve(points,np.ascontiguousarray(obs[j,obj]),K,cp)
     if sol is None:failures.append([int(ix[j]),obj]);pred[j,obj]=np.nan
     else:pred[j,obj]=sol[0];behind+=int(not sol[1])
   # All failures retained; no finite-only mean silently reported.
   err=np.linalg.norm(pred-n['positions'][ix,:2],axis=2)*1000
   runs.append(dict(errors=stats(err,roots[ix]) if not failures else None,solver_failures=failures,behind_camera_count=behind))
   arrays[f'noise{noise}_repeat{repeat}']=pred
  outputs[str(noise)]=runs
 np.savez_compressed(OUT/'pnp_oracle_predictions.npz',indices=ix,mesh_keypoints=points,raw_uv=uv,**arrays)
 save('pnp_oracle.json',dict(config=cfg,mesh_sha256=sha256(mesh),mesh_scale=.7,source_vertices=len(verts),keypoints=points.tolist(),frames=len(ix),scenes=13,results=outputs,frame_assumption='mesh geom has no local pos/quat in asset object body; known shape prior, eight GT-labelled vertices may be occluded or rotationally ambiguous',interpretation='Conditional solver precision/sensitivity only, not evidence that natural RGB supplies the vertex identities. No heldout world depth/pose sent to solver.'))
 print('PnP complete')
if __name__=='__main__':main()
