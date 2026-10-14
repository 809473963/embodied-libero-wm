"""H24 contact and synchronized-frame validation, derived from frozen H8 checks."""
import numpy as np

def validate(a,geometry,anchor,rgb=True):
    n=anchor+25
    assert np.array_equal(a['step'],np.arange(n)) and a['actions'].shape==(n-1,7)
    assert a['contact_offsets'].shape==(n+1,) and a['contact_offsets'][0]==0
    assert (np.diff(a['contact_offsets'])>=0).all()
    for k in ('contact_geom_pairs','contact_wrenches','contact_frames','contact_distances'):assert len(a[k])==a['contact_offsets'][-1]
    assert all(np.isfinite(v).all() for v in a.values())
    owners={int(k):v for k,v in geometry['geom_entity_owner'].items()}
    grip=set(geometry['gripper_ids']);pads=[set(geometry['left_pad_ids']),set(geometry['right_pad_ids'])]
    rebuilt=np.zeros_like(a['pad_contact']);anycontact=np.zeros_like(a['gripper_contact']);external=np.zeros_like(a['non_gripper_contact']);contact=np.zeros_like(a['contacts'])
    for t in range(n):
        for g1,g2 in a['contact_geom_pairs'][a['contact_offsets'][t]:a['contact_offsets'][t+1]]:
            i,j=owners.get(int(g1)),owners.get(int(g2))
            if i is not None and j is not None:contact[t,i,j]=contact[t,j,i]=True
            for target,other in ((i,int(g2)),(j,int(g1))):
                if target is not None:
                    anycontact[t,target]|=other in grip
                    for side in range(2):rebuilt[t,side,target]|=other in pads[side]
                    external[t,target]|=other not in grip and owners.get(other)!=target
    for x,y in [(rebuilt,a['pad_contact']),(anycontact,a['gripper_contact']),(external,a['non_gripper_contact']),(contact,a['contacts'])]:np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(a['grasp'],rebuilt[:,:,0].all(1))
    expected=np.r_[0,np.arange(anchor-10,anchor+25)] if rgb else np.array([],int)
    np.testing.assert_array_equal(a['natural_step'],expected)
    assert a['rgb'].shape==(len(expected),128,128,3) and a['rgb'].dtype==np.uint8
    assert a['proprio'].shape==(len(expected),12)
