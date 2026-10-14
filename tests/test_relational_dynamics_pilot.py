import numpy as np
import torch
from src.probes.relational_dynamics_pilot import features,causal_window,anchors,normalization,Dynamics,relation,score


def record(n=24):
    a=dict(pose=np.zeros((n,8,13)),contacts=np.zeros((n,8,8)),gripper_contact=np.zeros((n,8)),
        pad_contact=np.zeros((n,2,8)),non_gripper_contact=np.zeros((n,8)),grip_pos=np.zeros((n,3)),
        grip_rot=np.tile(np.eye(3),(n,1,1)),grip_velocity=np.zeros((n,6)),qpos=np.zeros((n,48)),
        qvel=np.zeros((n,43)),on=np.zeros((n,2)),actions=np.zeros((n-1,7)),step=np.arange(n))
    return a,np.full((n,2,5),-1)


def test_no_future_observation_or_action_enters_current_features():
    a,v=record();x,_=features(a,v)
    # State t follows action t-1; action t is still a future plan.
    for k in a:
        if k not in ('step','actions'):a[k][13:]=3
    a['actions'][12:]=5;v[13:]=1
    changed,_=features(a,v)
    np.testing.assert_array_equal(x[:13],changed[:13])
    assert not np.array_equal(x[13:],changed[13:])


def test_history_padding_and_prefix_once():
    x=np.arange(20).reshape(10,2)
    w=causal_window(x,2,5)
    np.testing.assert_array_equal(w[:,-1],[0,0,1,1,1])
    np.testing.assert_array_equal(w[-1,:-1],x[2])
    assert anchors(32,0)[:5]==[0,8,16,24,32]
    assert anchors(32,49)[0]==32
    assert max(anchors(96,50))+8<=300


def test_relation_order_and_gripper_coordinate_transform():
    a,v=record();a['pose'][:,0,:3]=[1,2,3];a['pose'][:,1,:3]=[4,5,6]
    a['pose'][:,2,:3]=[.5,1,1.5];a['grip_pos'][:]=[1,1,1]
    np.testing.assert_allclose(relation(a)[0],[.5,1,1.5,0,1,2,3.5,4,4.5,3,4,5])


def test_development_changes_cannot_affect_fit_normalization():
    rng=np.random.default_rng(7);x=rng.normal(size=(4,11,6));x[:,:,-1]=1
    a=rng.normal(size=(4,8,7));y=rng.normal(size=(4,12));fit=np.array([1,1,0,0],bool);w=np.ones(4)
    old=normalization(x,a,y,fit,w);x[~fit]=1e5;a[~fit]=2e5;y[~fit]=3e5
    new=normalization(x,a,y,fit,w)
    for k in old:np.testing.assert_array_equal(old[k],new[k])


def test_no_action_comparison_is_same_capacity_and_same_state():
    torch.manual_seed(1);model=Dynamics(8).eval()
    x=torch.zeros(3,11,8);a=torch.randn(3,8,7)
    no=model(x,torch.zeros_like(a))
    torch.testing.assert_close(no[0],no[1]);torch.testing.assert_close(no[0],no[2])
    assert not torch.equal(model(x,a)[0],model(x,a)[1])


def test_unknown_labels_excluded_and_changes_not_accuracy():
    d=dict(delta=np.zeros((2,12)),binary=np.zeros((2,18),np.int8),current=np.zeros((2,18),np.int8))
    d['binary'][0,4]=-1;d['binary'][1,4]=1
    m=[dict(root='r'),dict(root='r')]
    s=score(np.zeros((2,12)),np.zeros((2,18)),d,m,np.ones(2,bool))
    assert s['binary']['target/attachment']['n']==1
    assert s['binary']['target/attachment']['onset']['fn']==1
