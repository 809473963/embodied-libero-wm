import numpy as np
from src.probes.relational_autorun import select_anchors
P=dict(window_intervals=10,local_drift_m=.005,rise_m=.005,linear_speed_m_s=.02,angular_speed_rad_s=.5,placement_drift_m=.005)

def example():
    n=301;a=dict(step=np.arange(n),pose=np.zeros((n,8,13)),grip_pos=np.zeros((n,3)),grip_rot=np.tile(np.eye(3),(n,1,1)),pad_contact=np.zeros((n,2,8),bool),gripper_contact=np.zeros((n,8),bool),non_gripper_contact=np.zeros((n,8),bool),on=np.zeros((n,2),bool))
    a['pose'][:,:,3]=1;a['pad_contact'][:,:,0]=True;a['gripper_contact'][:,0]=True
    a['pose'][:32,0,0]=.1;a['pose'][32:,0,0]=.01;a['grip_pos'][:]=a['pose'][:,0,:3]
    return a

def test_held_near_far_are_separate_causal_strata():
    a=example();r=select_anchors(a,P)
    assert r['held_far']==16 and r['held_near']==32
    a['pose'][33:,0,0]=100
    assert select_anchors(a,P)['held_near']==32

def test_missing_release_not_filled_using_success_or_future():
    a=example();a['success']=np.ones(301,bool)
    assert select_anchors(a,P)['released_on_history'] is None
    a['success'][:]=False
    assert select_anchors(a,P)['released_on_history'] is None
