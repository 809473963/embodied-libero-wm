import numpy as np
import torch
from scripts.train_relational_localization_cv import Model,normalization

def test_fold_stats_ignore_held_out_labels():
 rng=np.random.default_rng(2);d={k:rng.normal(size=(8,n)).astype(np.float32) for k,n in [('obs',19),('target',18),('points',9)]};fit=np.array([1,1,1,1,0,0,0,0],bool);w=np.ones(8)
 before=normalization(d,fit,w)
 for v in d.values():v[~fit]+=999
 for k,v in normalization(d,fit,w).items():np.testing.assert_array_equal(v,before[k])

def test_auxiliary_heads_and_relation_head_share_encoder():
 torch.manual_seed(11);m=Model();a,b,c=m(torch.randn(2,6,64,64),torch.randn(2,19));assert a.shape==(2,18) and b.shape==(2,9) and c.shape==(2,3,8,8)
 (b.square().mean()+c.square().mean()).backward();assert m.encoder[0].weight.grad.abs().sum()>0;assert m.state[0].weight.grad is None
