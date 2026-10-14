import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('dense_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_dense_relation_preflight.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def recording(n=25):
    pose=np.zeros((n,8,13));pose[:,:,3]=1
    return {'pose':pose,'grip_pos':np.zeros((n,3)), 'grip_rot':np.tile(np.eye(3),(n,1,1)),
            'pad_contact':np.zeros((n,2,8),bool),'gripper_contact':np.zeros((n,8),bool),
            'non_gripper_contact':np.zeros((n,8),bool),'on':np.zeros((n,2),bool)}


def test_intervals_count_elapsed_steps_not_samples():
    assert m.intervals([0,1,1,0,1])==[[1,2],[4,4]]
    assert m.qualifying_windows(np.ones(10,bool),np.zeros((10,3)),10,.005)==[]
    assert m.qualifying_windows(np.ones(11,bool),np.zeros((11,3)),10,.005)==[0]


def test_single_contact_gap_breaks_hold_window():
    x=recording(11);x['pad_contact'][:,:,0]=True;x['pad_contact'][5,1,0]=False
    assert not m.derive(x)['hold_candidate_window_starts']


def test_hand_motion_does_not_imply_lifted_object():
    x=recording();x['pad_contact'][:,:,0]=True;x['grip_pos'][:,2]=np.arange(25)*.002
    assert not m.derive(x)['hold_candidate_window_starts']
    assert not m.derive(x)['lift_hold_candidate_window_starts']


def test_lift_hold_requires_prior_support_and_detachment():
    x=recording();x['pad_contact'][:,:,0]=True
    x['pose'][1:,0,2]=.02;x['grip_pos'][1:,2]=.02
    assert not m.derive(x)['lift_hold_candidate_window_starts']
    x['non_gripper_contact'][0,0]=True
    assert m.derive(x)['lift_hold_candidate_window_starts']
    x['non_gripper_contact'][:,0]=True
    assert not m.derive(x)['lift_hold_candidate_window_starts']


def test_on_while_held_is_not_release_placement():
    x=recording();x['on'][:,0]=True;x['non_gripper_contact'][:,0]=True;x['gripper_contact'][:,0]=True
    assert not m.derive(x)['released_placement_candidate_window_starts']
    x['gripper_contact'][1:,0]=False
    assert m.derive(x)['released_placement_candidate_window_starts']
    x['pose'][:,0,7]=.03
    assert not m.derive(x)['released_placement_candidate_window_starts']
