import numpy as np
from src.probes.relational_graph_audit_v2 import to_world,closure,reconcile,macro_error

def test_closure_requires_rotating_local_vectors():
 r=np.array([[[0.,-1,0],[1,0,0],[0,0,1]]]);a=np.array([[.2,.1,.3]]);b=np.array([[-.1,.4,.2]]);g=np.array([[.1,.1,.1]])
 local=lambda x:np.einsum('nji,nj->ni',r,x)
 x=np.stack([a,local(a-g),b,local(b-g)],axis=1)
 assert np.linalg.norm(closure(x))>.1
 np.testing.assert_allclose(closure(to_world(x,r)),0,atol=1e-12)

def test_projection_closes_graph_and_cannot_increase_squared_error_to_consistent_truth():
 rng=np.random.default_rng(5);truth=reconcile(rng.normal(size=(9,4,3)));x=truth+rng.normal(size=truth.shape);fixed=reconcile(x)
 np.testing.assert_allclose(closure(fixed),0,atol=1e-12)
 assert np.square(fixed-truth).sum()<=np.square(x-truth).sum()
 np.testing.assert_allclose(reconcile(fixed),fixed,atol=1e-12)

def test_scene_macro_does_not_overweight_repeated_anchors():
 pred=np.array([[.001,0,0],[.001,0,0],[.009,0,0]])
 assert macro_error(pred,np.zeros_like(pred),np.array(['a','a','b']))['scene_macro_mm']==5.


def test_float32_inputs_are_promoted_before_strict_closure():
 x=np.random.default_rng(6).normal(size=(22,4,3)).astype(np.float32)
 fixed=reconcile(x)
 assert fixed.dtype==np.float64
 np.testing.assert_allclose(closure(fixed),0,atol=1e-10)
