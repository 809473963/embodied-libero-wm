import numpy as np
from src.probes.relational_event_labels import labels, future_targets

CFG = dict(window_intervals=10, local_drift_m=.005, rise_m=.005,
           linear_speed_m_s=.02, angular_speed_rad_s=.5, placement_drift_m=.005)


def recording(n=45):
    pose=np.zeros((n,8,13));pose[:,:,3]=1
    return dict(pose=pose, step=np.arange(n), grip_pos=np.zeros((n,3)),
                grip_rot=np.tile(np.eye(3),(n,1,1)),
                pad_contact=np.zeros((n,2,8),bool), gripper_contact=np.zeros((n,8),bool),
                non_gripper_contact=np.zeros((n,8),bool), on=np.zeros((n,2),bool))


def held():
    a=recording();a['pad_contact'][:,:,0]=True;a['gripper_contact'][:,0]=True
    a['non_gripper_contact'][0,0]=True
    a['pose'][1:,0,2]=.02;a['grip_pos'][1:,2]=.02
    return a


def test_past_only_and_censoring():
    a=held();full=labels(a,CFG)['values']
    for n in (1,10,11,20,35):
        truncated={k:v[:n] for k,v in a.items()}
        np.testing.assert_array_equal(labels(truncated,CFG)['values'],full[:n])
    assert (full[:10]==-1).all()
    targets=future_targets(full,[8,20]);assert (targets[-8:,0]==-1).all()
    np.testing.assert_array_equal(targets[:-20,1],full[20:])


def test_gap_motion_and_lift_reference():
    a=held();assert labels(a,CFG)['values'][11,2]==1
    a['non_gripper_contact'][:]=False
    assert labels(a,CFG)['values'][11,2]==-1
    a['pad_contact'][5,1,0]=False
    assert labels(a,CFG)['values'][10,0]==0
    a=held();a['grip_pos'][:,0]=np.arange(45)*.002
    assert (labels(a,CFG)['values'][10:,0]==0).all()


def test_on_held_or_rotating_is_not_quiet_placement():
    a=held();a['on'][:,0]=True;a['non_gripper_contact'][:,0]=True
    assert (labels(a,CFG)['values'][10:,3]==0).all()
    a['gripper_contact'][:]=False;a['pose'][:,0,12]=1
    assert (labels(a,CFG)['values'][10:,3]==0).all()


def test_release_requires_detached_history_not_touch():
    a=held();a['pad_contact'][20:]=False;a['gripper_contact'][20:]=False
    a['on'][20:,0]=True;a['non_gripper_contact'][20:,0]=True
    out=labels(a,CFG);assert out['values'][30,4]==1
    assert out['onset'][30,3]==1
    a['pad_contact'][:]=False
    out=labels(a,CFG);assert out['values'][30,3]==1 and out['values'][30,4]==-1


def test_translation_and_yaw_equivariance():
    a=held();expected=labels(a,CFG)['values']
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    a['pose'][:,:,:3]=a['pose'][:,:,:3]@R.T+[2,3,4]
    a['grip_pos']=a['grip_pos']@R.T+[2,3,4]
    a['grip_rot']=np.einsum('ij,tjk->tik',R,a['grip_rot'])
    np.testing.assert_array_equal(labels(a,CFG)['values'],expected)


def test_wrong_object_has_separate_evidence():
    a=held();assert labels(a,CFG,0)['values'][11,0]==1
    assert labels(a,CFG,1)['values'][11,0]==0


def test_missing_control_steps_rejected():
    import pytest
    a=held();a['step'][5]=6
    with pytest.raises(AssertionError):labels(a,CFG)
