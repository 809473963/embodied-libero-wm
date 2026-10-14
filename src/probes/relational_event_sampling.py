"""Causal state strata and fixed H24 open-loop plans; no outcome selection."""
import numpy as np
from src.probes.relational_event_labels import labels
NAMES=('a0_h8_then_hold','hold','open','close','lift_close','lower_then_open')
STRATA=('near_without_contact','bilateral_not_detached','detached_attachment','released_on_with_history')

def select(a,parameters):
    v=labels(a,parameters)['values'];grip=a['gripper_contact'][:,0]
    both=a['pad_contact'][:,:,0].all(1);ext=a['non_gripper_contact'][:,0]
    distance=np.linalg.norm(a['pose'][:,0,:3]-a['grip_pos'],axis=1)
    seen=np.maximum.accumulate(v[:,1]==1)
    masks=[(~grip)&(distance<=.08),both&(v[:,1]!=1),v[:,1]==1,
           (~grip)&ext&a['on'][:,0]&seen]
    out={}
    for name,mask in zip(STRATA,masks):
        eligible=[t for t in range(16,min(len(mask)-24,241),8) if mask[t]]
        out[name]=eligible[0] if eligible else None
    return out

def candidates(a0,amplitude=.15):
    assert a0.shape==(8,7) and np.isfinite(a0).all() and (abs(a0)<=1).all()
    out=np.zeros((6,24,7),np.float32)
    # Root-time planned A0 grip is known; no action beyond its first planned H8 is read.
    hold=1. if a0[0,6]>=0 else -1.
    out[:,:,6]=hold
    out[0,:8]=a0;out[0,8:,6]=a0[-1,6]
    out[2,:,6]=-1;out[3,:,6]=1
    out[4,:8,2]=amplitude;out[4,:,6]=1
    out[5,:8,2]=-amplitude;out[5,8:,6]=-1
    return out
