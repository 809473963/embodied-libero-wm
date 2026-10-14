import numpy as np
from scripts.audit_unified_object_state_v3 import angle_degrees

def test_dtype_alignment_preserves_real_observation_and_detects_changes():
 raw=np.array([2.551444162,.001],dtype=np.float64);stored=raw.astype(np.float32)
 np.testing.assert_array_equal(stored,raw.astype(stored.dtype))
 changed=raw.copy();changed[0]+=.001
 assert not np.array_equal(stored,changed.astype(stored.dtype))
 np.testing.assert_allclose(angle_degrees(np.eye(3)[None],np.eye(3)[None]),0)
