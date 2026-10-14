import numpy as np
from src.probes.relational_new_scene import fingerprint,near,distance,probe_actions

def pose():
    a=np.zeros((8,13));a[:,3]=1;return a

def test_quaternion_sign_not_new_scene():
    a=pose();b=a.copy();b[:,3:7]*=-1
    assert fingerprint(a)==fingerprint(b) and near(a,b)

def test_position_and_rotation_both_matter():
    a=pose();b=a.copy();b[0,0]=.002
    assert not near(a,b) and fingerprint(a)!=fingerprint(b)
    b=a.copy();b[0,3:7]=[np.cos(.1),0,0,np.sin(.1)]
    assert not near(a,b) and distance(a,b)['max_orientation_deg']>1

def test_same_geometry_different_velocity_not_novel_layout():
    a=pose();b=a.copy();b[:,7:]=5
    assert fingerprint(a)==fingerprint(b)

def test_fixed_probe_is_bounded_without_policy_or_future():
    a=probe_actions();assert a.shape==(24,7) and np.isfinite(a).all() and (abs(a)<=1).all()
    assert (a[:8,6]==-1).all() and not a[:8,:6].any()
    np.testing.assert_array_equal(a,probe_actions())
