#!/usr/bin/env python3
"""Select only causal strata on existing exposed A0 recordings; no simulation."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.relational_event_sampling import select
from src.probes.relational_dynamics_pilot import sha

def main():
    cfg=json.loads((ROOT/'configs/relational_event_h24_selection.json').read_text())
    assert sha(ROOT/'src/probes/relational_event_sampling.py')==cfg['code_sha256']
    parameters=json.loads((ROOT/'configs/relational_event_labels_v1.json').read_text())['parameters']
    records=json.loads((ROOT/'results/relational_coverage_expansion/coverage.json').read_text())['records']
    source={r['root_id']:r for r in records if r['candidate_id']==0};cases=[];missing=[];counts={}
    for root in cfg['root_ids']:
        r=source[root];assert sha(ROOT/r['source'])==r['source_sha256']
        with np.load(ROOT/r['source'],allow_pickle=False) as f:
            a={k:f[k] for k in ('step','pose','grip_pos','grip_rot','pad_contact','gripper_contact','non_gripper_contact','on')}
        selected=select(a,parameters);merged={}
        for stratum,t in selected.items():
            if t is None:missing.append(dict(root_id=root,stratum=stratum));continue
            counts[stratum]=counts.get(stratum,0)+1;merged.setdefault(t,[]).append(stratum)
        for t,strata in sorted(merged.items()):cases.append(dict(root_id=root,anchor=t,strata=strata,source=r['source'],source_sha256=r['source_sha256']))
    result=dict(selection_sha256=sha(ROOT/'configs/relational_event_h24_selection.json'),cases=cases,missing=missing,stratum_root_counts=counts,
        maximum_runs=6*len(cases)+3,maximum_environment_steps=6*sum(c['anchor']+24 for c in cases)+3*(cases[0]['anchor']+24),independent_new_scenes=False)
    path=ROOT/'configs/relational_event_h24_cases.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','missing')},ensure_ascii=False))
if __name__=='__main__':main()
