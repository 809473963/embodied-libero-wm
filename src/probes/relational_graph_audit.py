"""Geometry consistency diagnostics; rotations must be explicitly supplied."""
import numpy as np

def to_world(relations,gripper_rotation):
    x=np.asarray(relations).reshape(-1,4,3).copy()
    for k in (1,3):x[:,k]=np.einsum('nij,nj->ni',gripper_rotation,x[:,k])
    return x

def closure(world_relations):
    x=np.asarray(world_relations)
    return x[:,0]-x[:,2]-x[:,1]+x[:,3]

def reconcile(world_relations):
    """Equal-weight orthogonal projection onto one vector closure constraint.
    Algebraic consistency is guaranteed, not evidence of learned accuracy.
    """
    x=np.asarray(world_relations).copy();e=closure(x)/4
    for k,sign in enumerate((1,-1,-1,1)):x[:,k]-=sign*e
    return x

def macro_error(pred,true,roots):
    e=np.asarray(pred)-np.asarray(true);norm=np.linalg.norm(e,axis=-1)*1000
    per={str(r):float(norm[np.asarray(roots)==r].mean()) for r in sorted(set(roots))}
    return dict(scene_macro_mm=float(np.mean(list(per.values()))),per_scene_mm=per,
                scene_macro_axis_mae_mm=np.mean([abs(e[np.asarray(roots)==r]).mean(0)*1000 for r in sorted(set(roots))],axis=0).tolist())
