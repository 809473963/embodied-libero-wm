import numpy as np
from scripts.diagnose_multi_object_perception import project,rays_to_plane,permute_edges,EDGES,macro

def test_plane_solver_uses_height_prior_not_true_depth():
 M=np.array([[100.,0,64,0],[0,100,64,0],[0,0,1,0],[0,0,0,1]])
 x=np.array([[.2,.1,2.],[.4,-.2,3.]])
 uv,_=project(x,M);y,valid=rays_to_plane(uv,np.array([2.,2.]),M)
 assert valid.all();np.testing.assert_allclose(y[0],x[0]);np.testing.assert_allclose(y[1],x[1]*2/3)
 assert np.linalg.norm(y[1]-x[1])>1

def test_swap_relabels_edges_with_sign_and_keeps_inconsistency():
 xyz=np.random.default_rng(2).normal(size=(5,9,3));p=[1,0,2,3,4,5,6,7,8]
 e=np.stack([xyz[:,i]-xyz[:,j] for i,j in EDGES],1)
 np.testing.assert_allclose(permute_edges(e,p),np.stack([xyz[:,p[i]]-xyz[:,p[j]] for i,j in EDGES],1))
 np.testing.assert_allclose(permute_edges(permute_edges(e,p),p),e)

def test_scene_weighting_not_frame_weighting():
 assert macro(np.array([0.,12.,12.,12.]),np.array(['a','b','b','b']))==6

def test_pnp_has_no_truth_pose_or_depth_initialization():
 import cv2
 from scripts.probe_multi_object_pnp_oracle import solve
 points=np.array([[x,y,z] for x in [-.04,.04] for y in [-.04,.04] for z in [0,.05]],np.float64)
 K=np.array([[150.,0,64],[0,150.,64],[0,0,1.]])
 uv,_=cv2.projectPoints(points,np.array([.1,.2,.05]),np.array([.1,.05,.7]),K,None)
 pred,front=solve(points,uv.reshape(-1,2),K,np.eye(4))
 assert front;np.testing.assert_allclose(pred,[.1,.05,.7],atol=1e-7)

def test_geometry_natural_api_cannot_use_evaluation_depth():
 import inspect
 assert list(inspect.signature(rays_to_plane).parameters)==['uv','zprior','M']
 M=np.array([[100.,0,64,0],[0,100,64,0],[0,0,1,0],[0,0,0,1]])
 # Two GT points share the same ray: solver cannot know which depth is correct.
 uv,_=project(np.array([[.1,.2,1.],[.2,.4,2.]]),M)
 pred,valid=rays_to_plane(uv,np.array([1.5,1.5]),M)
 assert valid.all();np.testing.assert_allclose(pred[0],pred[1])
