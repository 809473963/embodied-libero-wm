"""Finite autonomous development chain helpers; no confirmation scenes."""
import numpy as np
from src.probes.relational_event_labels import labels

def select_anchors(a,parameters):
    v=labels(a,parameters)['values'];grip=a['gripper_contact'][:,0]
    xy=np.linalg.norm(a['pose'][:,0,:2]-a['pose'][:,2,:2],axis=1)
    masks=dict(near_without_contact=(~grip)&(np.linalg.norm(a['pose'][:,0,:3]-a['grip_pos'],axis=1)<=.08),
        bilateral_not_detached=a['pad_contact'][:,:,0].all(1)&(v[:,1]!=1),
        held_far=(v[:,1]==1)&(xy>.06),held_near=(v[:,1]==1)&(xy<=.06),
        released_on_history=(~grip)&a['on'][:,0]&a['non_gripper_contact'][:,0]&np.maximum.accumulate(v[:,1]==1))
    result={}
    for name,mask in masks.items():
        ts=[t for t in range(16,241,8) if mask[t]];result[name]=ts[0] if ts else None
    return result
