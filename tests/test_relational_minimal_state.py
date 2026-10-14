import numpy as np
import pytest
from src.probes.relational_minimal_state import project, REPRESENTATIONS

# Independent compact schema, including excluded nuisance fields.
SCHEMA=[]
for name,n in [('entity_world_pose_velocity',104),('bowls_relative_plate_and_gripper',12),('contacts',64),('gripper_contact',8),('pad_contact',16),('non_gripper_contact',8),('grip_velocity',6),('on',2),('qpos',64),('event_value_unknown_zero',10),('contact_height_reference_and_seen_detached',6),('previous_executed_action',7)]:
    start=SCHEMA[-1]['stop'] if SCHEMA else 0
    SCHEMA.append(dict(name=name,start=start,stop=start+n))

@pytest.mark.parametrize('rep,dim',list(zip(REPRESENTATIONS,[26,34,40,40])))
def test_projection_excludes_privileged_shortcuts_and_preserves_time(rep,dim):
    x=np.arange(2*11*(SCHEMA[-1]['stop']+1)).reshape(2,11,-1).astype(float)
    y,indices=project(x,SCHEMA,rep)
    assert y.shape==(2,1 if rep=='memory_current' else 11,dim)
    excluded=[s for s in SCHEMA if s['name'] in ('entity_world_pose_velocity','contacts','on','qpos','event_value_unknown_zero')]
    perturbed=x.copy()
    for s in excluded: perturbed[...,s['start']:s['stop']]+=100000
    np.testing.assert_array_equal(project(perturbed,SCHEMA,rep)[0],y)
    np.testing.assert_array_equal(y[...,-1],x[:,-1:,-1] if rep=='memory_current' else x[...,-1])
    if rep=='memory_current':
        perturbed=x.copy();perturbed[:,:-1,:]+=20000
        np.testing.assert_array_equal(project(perturbed,SCHEMA,rep)[0],y)
    if rep!='geometry_history':
        pad=next(s['start'] for s in SCHEMA if s['name']=='pad_contact')
        assert [i-pad for i in indices if pad<=i<pad+16]==[0,1,8,9]
