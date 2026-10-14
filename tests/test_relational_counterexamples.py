import numpy as np
import pytest
from src.probes.relational_counterexamples import candidates,validate,NAMES

def test_fixed_recipes_preserve_order_and_nonmodified_dimensions():
    a=np.arange(56,dtype=np.float32).reshape(8,7)/60
    b=candidates(a);assert b.shape==(9,8,7) and len(NAMES)==9
    np.testing.assert_array_equal(b[0],a);np.testing.assert_array_equal(b[1,:,:6],a[:,:6]);assert (b[1,:,6]==-1).all() and (b[2,:,6]==1).all()
    for axis in range(3):
        for i,sign in ((3+2*axis,1),(4+2*axis,-1)):
            keep=[k for k in range(7) if k!=axis]
            np.testing.assert_array_equal(b[i][:,keep],a[:,keep])
            np.testing.assert_allclose(b[i,:,axis],np.clip(a[:,axis]+sign*.2,-1,1))
    np.testing.assert_array_equal(a,np.arange(56,dtype=np.float32).reshape(8,7)/60)

def test_clip_and_invalid_actions():
    b=candidates(np.ones((8,7),np.float32));assert (abs(b)<=1).all()
    with pytest.raises(AssertionError):candidates(np.zeros((7,7)))
    with pytest.raises(AssertionError):candidates(np.full((8,7),np.nan))

def recording(anchor=64):
    n=anchor+9;steps=np.r_[0,np.arange(anchor-10,anchor+9)]
    a=dict(step=np.arange(n),actions=np.zeros((n-1,7)),contact_offsets=np.zeros(n+1,int),contact_geom_pairs=np.empty((0,2),int),contact_wrenches=np.empty((0,6)),contact_frames=np.empty((0,9)),contact_distances=np.empty(0),pad_contact=np.zeros((n,2,8),bool),gripper_contact=np.zeros((n,8),bool),non_gripper_contact=np.zeros((n,8),bool),contacts=np.zeros((n,8,8),bool),grasp=np.zeros(n,bool),natural_step=steps,rgb=np.zeros((len(steps),128,128,3),np.uint8),proprio=np.zeros((len(steps),12)))
    g=dict(geom_entity_owner={},gripper_ids=[1,2],left_pad_ids=[1],right_pad_ids=[2])
    return a,g

def test_rgb_timestamps_and_length_are_checked():
    a,g=recording();validate(a,g,64)
    a['natural_step'][-1]-=1
    with pytest.raises(AssertionError):validate(a,g,64)

def test_contact_reconstruction_detects_fake_grasp():
    a,g=recording();a['grasp'][65]=True
    with pytest.raises(AssertionError):validate(a,g,64)
