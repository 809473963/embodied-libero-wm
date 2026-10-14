import numpy as np
from src.probes.relational_event_sampling import select,candidates
from src.probes.relational_event_labels import labels
P=dict(window_intervals=10,local_drift_m=.005,rise_m=.005,linear_speed_m_s=.02,angular_speed_rad_s=.5,placement_drift_m=.005)

def recording(n=280):
    a=dict(step=np.arange(n),pose=np.zeros((n,8,13)),grip_pos=np.ones((n,3)),grip_rot=np.tile(np.eye(3),(n,1,1)),pad_contact=np.zeros((n,2,8),bool),gripper_contact=np.zeros((n,8),bool),non_gripper_contact=np.zeros((n,8),bool),on=np.zeros((n,2),bool))
    a['pose'][:,:,3]=1
    return a

def test_selection_uses_first_current_match_not_future():
    a=recording();a['grip_pos'][24:]=0
    assert select(a,P)['near_without_contact']==24
    a['grip_pos'][25:]=100
    assert select(a,P)['near_without_contact']==24
    assert select(a,P)['detached_attachment'] is None

def test_strata_missing_not_replaced_and_history_required():
    a=recording();a['on'][:,0]=True;a['non_gripper_contact'][:,0]=True
    assert select(a,P)['released_on_with_history'] is None
    a['pad_contact'][30:60,:,0]=True;a['gripper_contact'][30:60,0]=True
    a['non_gripper_contact'][30:60,0]=False
    assert select(a,P)['bilateral_not_detached']==32
    assert select(a,P)['detached_attachment']==40
    assert select(a,P)['released_on_with_history']==64

def test_h24_plans_only_use_root_h8_and_known_tail():
    a0=np.arange(56,dtype=np.float32).reshape(8,7)/100
    plans=candidates(a0);assert plans.shape==(6,24,7)
    np.testing.assert_array_equal(plans[0,:8],a0)
    assert not plans[0,8:,:6].any() and (plans[2,:,6]==-1).all()
    assert (plans[4,:,6]==1).all() and not plans[4,8:,:6].any()
    assert (plans[5,8:,6]==-1).all() and (abs(plans)<=1).all()

def test_release_can_be_observed_in_h24_not_h8():
    a=recording();a['grip_pos'][:]=0;a['on'][:,0]=True;a['non_gripper_contact'][:,0]=True
    a['gripper_contact'][:65,0]=True
    v=labels(a,P)['values']
    assert v[72,3]==0 and v[88,3]==1
