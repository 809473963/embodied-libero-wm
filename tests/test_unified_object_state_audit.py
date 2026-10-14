import numpy as np
from scripts.audit_unified_object_state import angle_degrees

def test_rotation_comparison_uses_relative_angle_and_handles_roundoff():
 a=np.eye(3)[None];b=np.array([[[0,-1.,0],[1.,0,0],[0,0,1.]]]);np.testing.assert_allclose(angle_degrees(a,b),90)
 np.testing.assert_allclose(angle_degrees(a,a*(1+1e-15)),0)

def test_complete_graph_edge_orientation_and_triangle_closure():
 xyz=np.random.default_rng(5).normal(size=(4,9,3));edges=[(i,j) for i in range(9) for j in range(i+1,9)];assert len(edges)==36
 rel=np.stack([xyz[:,i]-xyz[:,j] for i,j in edges],1)
 for i,j,k in [(0,1,8),(1,4,6),(2,5,8)]:np.testing.assert_allclose(rel[:,edges.index((i,j))]+rel[:,edges.index((j,k))],rel[:,edges.index((i,k))],atol=1e-12)
