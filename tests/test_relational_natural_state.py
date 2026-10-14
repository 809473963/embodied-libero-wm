import numpy as np
import torch
from scripts.run_relational_natural_state import Perception,stats

def test_normalization_does_not_observe_development():
    d=dict(obs=np.arange(76,dtype=np.float32).reshape(4,19),target=np.arange(72,dtype=np.float32).reshape(4,18),fit=np.array([True,True,False,False]),weights=np.ones(4))
    a=stats(d)
    d['obs'][2:]+=10000;d['target'][2:]-=20000
    for k,v in stats(d).items():np.testing.assert_array_equal(v,a[k])

def test_natural_shapes_and_proprio_control_ignores_rgb():
    torch.manual_seed(1);p=torch.randn(2,19);im=torch.randn(2,6,64,64)
    m=Perception(False).eval();torch.testing.assert_close(m(im,p),m(None,p))
    assert Perception(True)(im,p).shape==(2,18)
