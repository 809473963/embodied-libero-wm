import numpy as np
import torch
from scripts.train_unified_object_state_cv import Model,EDGES,GROUPS,edge_loss,evaluate,scene_mean

def test_shared_nodes_use_measured_robot_and_edges_preserve_geometry():
 torch.manual_seed(3);m=Model('shared');im=torch.zeros(2,6,64,64);obs=torch.zeros(2,19);grip=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
 nodes,edges=m(im,obs,grip,torch.zeros(8,3))
 torch.testing.assert_close(nodes[:,8],grip)
 # Same image/proprio creates same object estimates; changing measured robot changes only its incident edges.
 for k,(i,j) in enumerate(EDGES):
  torch.testing.assert_close(edges[1,k]-edges[0,k],-(grip[1]-grip[0]) if j==8 else torch.zeros(3))
 loss=edge_loss(edges,torch.zeros_like(edges)).mean();loss.backward();assert m.head.weight.grad.abs().sum()>0

def test_metrics_use_scene_macro_and_detect_inconsistency():
 assert scene_mean(np.array([0.,10.,10.,10.]),np.array(['a','b','b','b']))==5
 assert sorted(sum(GROUPS.values(),[]))==list(range(36))
 xyz=np.random.default_rng(0).normal(size=(4,9,3));rel=np.stack([xyz[:,i]-xyz[:,j] for i,j in EDGES],1)
 good=evaluate(xyz,rel,xyz,rel,np.array(['a','a','b','b']));assert good['key_mean_mm']==0 and good['triangle_closure_mm']<1e-10
 bad=rel.copy();bad[:,0,0]+=.01
 assert evaluate(xyz,bad,xyz,rel,np.array(['a','a','b','b']))['triangle_closure_mm']>0
