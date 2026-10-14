"""Schema-driven causal projections of the frozen H24 history dataset."""
import numpy as np

REPRESENTATIONS = ('geometry_history', 'contact_history', 'memory_history', 'memory_current')

def project(x, schema, representation):
    if representation not in REPRESENTATIONS:
        raise ValueError(representation)
    fields = {s['name']: s for s in schema}
    indices = []
    def add(name, offsets=None):
        s = fields[name]
        offsets = range(s['stop']-s['start']) if offsets is None else offsets
        for offset in offsets:
            assert 0 <= offset < s['stop']-s['start']
            indices.append(s['start']+offset)
    add('bowls_relative_plate_and_gripper')
    add('grip_velocity')
    add('previous_executed_action')
    if representation != 'geometry_history':
        add('gripper_contact', (0,1))
        add('pad_contact', (0,1,8,9))
        add('non_gripper_contact', (0,1))
    if representation in ('memory_history', 'memory_current'):
        add('contact_height_reference_and_seen_detached')
    assert x.shape[-1] == max(s['stop'] for s in schema)+1
    indices.append(x.shape[-1]-1)  # History-valid flag, always last for normalization.
    assert len(set(indices)) == len(indices)
    result = x[..., indices].copy()
    if representation == 'memory_current':
        result = result[:, -1:, :]
    assert np.isfinite(result).all()
    return result, indices
