"""Synthetic physical counterexamples, with no simulator or research-data access."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('rel_audit', Path(__file__).resolve().parents[1]/'scripts/audit_relational_state_availability.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def state():
    x=np.zeros((1,5,476),np.float32)
    b=x[...,:104].reshape(1,5,8,13)
    b[...,3]=1
    x[...,196:205]=np.eye(3).reshape(9)
    return x


def test_global_translation_does_not_change_relations():
    x=state();x[...,0]=.1;x[...,26]=.3;x[...,184]=.2
    y=x.copy();b=y[...,:104].reshape(1,5,8,13)
    shift=np.array([4,-3,2]);b[...,:3]+=shift;y[...,184:187]+=shift
    a,b=audit.relations(x),audit.relations(y)
    np.testing.assert_allclose(a['target_plate_position'],b['target_plate_position'],atol=5e-7)
    np.testing.assert_allclose(a['target_gripper_local_position'],b['target_gripper_local_position'],atol=5e-7)


def test_relative_frame_rotates_position_and_quaternion_sign_is_equivalent():
    x=state();x[...,0]=1
    R=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    x[...,196:205]=R.reshape(9);x[:,2,3]=-1
    r=audit.relations(x)
    np.testing.assert_allclose(r['target_gripper_local_position'][0],np.tile([0,-1,0],(5,1)),atol=1e-7)
    np.testing.assert_allclose(r['gripper_relative_rotation_change'],0,atol=1e-7)


def test_recontact_is_not_first_contact_from_root():
    x=state();x[...,168]=[1,0,1,0,1]
    e=audit.relations(x)['events']
    assert e['sampled_target_contact_onset'][0] and e['sampled_target_contact_loss'][0]
    assert not e['first_sampled_contact_from_uncontacted_root'][0]
    assert not e['all_samples_target_contact'][0]


def test_lift_proxy_requires_initial_contact_final_gripper_and_rise():
    x=state();x[...,2]=[0,0,.002,.006,.01];x[0,0,104+5]=1
    # An object flying upward without robot contact is not lift evidence.
    assert not audit.relations(x)['events']['sampled_detach_and_rise_5mm'][0]
    x[0,-1,168]=1
    assert audit.relations(x)['events']['sampled_detach_and_rise_5mm'][0]
    x[0,-1,104+5]=1
    assert not audit.relations(x)['events']['sampled_detach_and_rise_5mm'][0]


def test_contact_or_stationary_object_does_not_create_strong_event():
    x=state();x[...,168]=1
    e=audit.relations(x)['events']
    assert e['all_samples_target_contact'][0]
    assert not any(name in e for name in ['stable_grasp','lift_and_hold','released_stable_placement'])
    assert all('unavailable' in v for v in audit.SCHEMA['strong_events'].values())


def test_review_sampling_covers_roots_before_repeating_one():
    items=[{'root_id':r,'candidate_id':j} for r in ['a','b','c'] for j in range(3)]
    picked=audit.diverse(items,4)
    assert [r['root_id'] for r in picked]==['a','b','c','a']
