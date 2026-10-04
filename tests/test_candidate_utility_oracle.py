from pathlib import Path
import io
import pickle
import zipfile

import numpy as np
import pytest

from src.probes.candidate_utility_oracle import candidate_subsets, decision_summary, selective_init_states, t1_select, identity


def test_subsets_include_gripper_and_complete_antithetic_pairs():
    previous = set()
    for name, ids in candidate_subsets().items():
        assert len(ids) == int(name[1:]) and len(ids) == len(set(ids))
        assert previous <= set(ids)
        assert {0,49,50,51} <= set(ids)
        for item in ids:
            if 1 <= item <= 48:
                partner = item+1 if item%2 else item-1
                assert partner in ids
        previous=set(ids)


def test_selector_boolean_priority_a0_tie_and_subset():
    assert t1_select(np.array([False,True,True]),np.array([100.,.1,.2]),.01)==2
    assert t1_select(np.array([True,True,True]),np.array([.195,.1,.2]),.01)==0
    assert t1_select(np.array([False,False,False]),np.array([.0,.198,.2]),.01)==1
    assert t1_select(np.array([False,True,True]),np.array([100.,.1,.2]),.01,[0,1])==1
    with pytest.raises(ValueError):
        t1_select(np.array([False]),np.array([np.nan]),.01)


def test_selective_loader_stops_before_forbidden_arrays(tmp_path: Path):
    # Protocol2 matches the pinned numpy serialization. A forbidden object's
    # __setstate__ would raise if this reader continued past the admitted boundary.
    arrays=[np.arange(5,dtype=np.float64)+index for index in range(35)]
    class EndSentinel:
        def __reduce__(self):
            return (eval,('1/0',))
    payload=pickle.dumps([*arrays,EndSentinel()],protocol=2)
    path=tmp_path/'init.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('archive/data.pkl',payload)
    selected,ledger=selective_init_states(path)
    assert set(selected)=={31,32,33,34}
    for index,value in selected.items():
        assert np.array_equal(value,arrays[index])
    assert ledger['last_ndarray_record_parsed']==34
    assert ledger['forbidden_init_indices_materialized']==[]
    with pytest.raises(ValueError):
        selective_init_states(path,(40,))


def test_decision_metrics_distinguish_opportunity_capture_and_harm():
    y=np.zeros((4,64),bool)
    y[0,2]=True  # rescue captured
    y[1,3]=True  # rescue missed
    y[2,0]=True  # A0 successful, intervention harms
    y[3,0]=True  # retain A0
    m=decision_summary(y,np.array([2,1,1,0]))
    assert m['rescue_roots']==2 and m['captured_rescue_roots']==1
    assert m['harm_roots']==1 and m['net_success_gain_roots']==0
    assert m['capture_fraction']==.5 and m['override_precision']==pytest.approx(1/3)
    assert m['mean_regret']==.5


def test_selective_matrix_reader_never_constructs_full_numeric_matrix(tmp_path: Path):
    data=np.arange(50*7,dtype=np.float64).reshape(50,7)
    path=tmp_path/'init.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('archive/data.pkl',pickle.dumps(data,protocol=2))
    rows,ledger=selective_init_states(path)
    for index,value in rows.items():
        assert np.array_equal(value,data[index])
        assert value.shape==(7,)
    assert ledger['forbidden_init_indices_materialized']==[]
    assert ledger['storage_format'].startswith('one C-contiguous matrix')


def test_no_opportunity_and_no_override_are_not_positive_evidence():
    m=decision_summary(np.zeros((2,64),bool),np.zeros(2,int))
    assert m['capture_fraction'] is None and m['override_precision'] is None
    assert m['net_success_gain_roots']==0


def test_identity_requires_exact_natural_inputs_and_tolerant_sim_state():
    reference={k:np.zeros(3) for k in ['agentview_image','robot0_eef_pos','robot0_gripper_qpos','robot0_joint_pos','sim_state']}
    actual={k:v.copy() for k,v in reference.items()}
    actual['sim_state'][0]=1e-12
    assert identity(reference,actual)['pass']
    actual['robot0_eef_pos'][0]=1e-12
    assert not identity(reference,actual)['pass']


def test_registered_roots_and_protocol_are_balanced_and_frozen():
    import json
    from pathlib import Path
    from src.probes.candidate_utility_oracle import sha256
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root/'configs/candidate_utility_oracle_protocol.json').read_text())
    rows = json.loads((root/'configs/candidate_utility_oracle_roots.json').read_text())['roots']
    assert len(rows) == len({r['root_id'] for r in rows}) == 40
    for task in range(10):
        selected = [r for r in rows if r['task_id'] == task]
        assert {r['init_index'] for r in selected} == {31,32,33,34}
        assert {r['prefix_steps'] for r in selected} == {0,32,64,96}
    assert protocol['subsets'] == candidate_subsets()
    assert not protocol['scope']['hdf5_access']
    assert not protocol['scope']['training']
    assert not protocol['scope']['learned_controller']
    for name,digest in protocol['frozen_files'].items():
        assert sha256(root/name) == digest
