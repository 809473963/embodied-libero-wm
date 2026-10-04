#!/usr/bin/env python3
"""Analyze only complete, hash-verified utility-oracle root artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.probes.candidate_utility_oracle import (
    array_hash, atomic_json, candidate_subsets, decision_summary, sha256, t1_select,
)


def main():
    directory=ROOT/'results/candidate_utility_oracle'
    protocol_path=ROOT/'configs/candidate_utility_oracle_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    digest=sha256(protocol_path)
    roots=json.loads((ROOT/'configs/candidate_utility_oracle_roots.json').read_text())['roots']
    records=[]
    for root in roots:
        path=directory/'roots'/f"{root['root_id']}.json"
        record=json.loads(path.read_text())
        if not record['complete'] or record['protocol_sha256']!=digest or record['root']!=root:
            raise ValueError('root completion/protocol identity mismatch')
        arrays=directory/'arrays'/f"{root['root_id']}.npz"
        if sha256(arrays)!=record['arrays_sha256']:
            raise ValueError('root sidecar hash mismatch')
        with np.load(arrays,allow_pickle=False) as a:
            if array_hash(a['candidates'])!=record['candidates_sha256'] or array_hash(a['prefix'])!=record['prefix_sha256']:
                raise ValueError('candidate/prefix hash mismatch')
        if [x['candidate_id'] for x in record['rows']]!=list(range(64)):
            raise ValueError('missing or duplicate candidate ID')
        if not all(x['identity']['pass'] and x['final_step']==300 for x in record['rows']):
            raise ValueError('invalid branch identity or time budget')
        if not record['a0_repeat_pass'] or not record['intervention_repeat_pass']:
            raise ValueError('repetition precondition failed')
        records.append(record)
    eligible=np.asarray([r['reference']['first_success_step'] is None for r in records])
    tasks=np.asarray([r['root']['task_id'] for r in records])
    y=np.asarray([[x['success'] for x in r['rows']] for r in records],bool)
    goals=np.asarray([[x['h8']['on'] for x in r['rows']] for r in records],bool)
    margins=np.asarray([[x['h8']['m2'] for x in r['rows']] for r in records],float)
    selected=np.asarray([t1_select(g,m,protocol['epsilon_goal']) for g,m in zip(goals,margins)])
    assert selected.tolist()==[r['t1_selected_id'] for r in records]
    primary=decision_summary(y[eligible],selected[eligible])
    per_task={str(t):decision_summary(y[eligible&(tasks==t)],selected[eligible&(tasks==t)]) for t in range(10)}
    rescue=(~y[:,0])&y.any(axis=1)&eligible
    rescue_tasks=int(len(np.unique(tasks[rescue])))
    gains=y[np.arange(40),selected].astype(float)-y[:,0]
    # Fixed-task scene-block bootstrap; eligibility is retained within each block.
    rng=np.random.default_rng(protocol['statistics']['seed'])
    indices=np.arange(40).reshape(10,4)
    repetitions=protocol['statistics']['bootstrap_repetitions']
    draws=rng.integers(0,4,size=(repetitions,10,4))
    sampled=indices[np.arange(10)[None,:,None],draws].reshape(repetitions,40)
    denominator=eligible[sampled].sum(axis=1)
    usable=denominator>0
    gain_boot=(gains[sampled]*eligible[sampled]).sum(axis=1)[usable]/denominator[usable]
    oracle_boot=rescue[sampled].sum(axis=1)[usable]/denominator[usable]
    ci={'net_success_gain_rate':np.quantile(gain_boot,[.025,.975]).tolist() if len(gain_boot) else None,
        'oracle_rescue_rate':np.quantile(oracle_boot,[.025,.975]).tolist() if len(oracle_boot) else None,
        'unit':'task-stratified root/scene block; fixed 10 tasks, four scenes/task; pilot only'}
    subset_results={}
    for name,ids in candidate_subsets().items():
        choice=np.asarray([t1_select(g,m,protocol['epsilon_goal'],ids) for g,m in zip(goals,margins)])
        # Summary's oracle must be over the named subset, while selected IDs
        # remain in the original 64-space. Zero out excluded columns.
        restricted=np.zeros_like(y)
        restricted[:,ids]=y[:,ids]
        subset_results[name]=decision_summary(restricted[eligible],choice[eligible])
    cheap=decision_summary(y[eligible],np.full(int(eligible.sum()),49,int))
    coverage=bool(eligible.sum()>=30 and len(np.unique(tasks[eligible]))>=8)
    opportunity=bool(primary['rescue_roots']>=4 and rescue_tasks>=3)
    t1_valid=bool(opportunity and primary['capture_fraction']>=.5 and primary['harm_roots']<=1 and primary['net_success_gain_roots']>0)
    simple_saturates=bool(opportunity and cheap['captured_rescue_roots']==primary['rescue_roots'] and cheap['harm_roots']==0)
    if not coverage:
        verdict='D_COVERAGE_INCONCLUSIVE'
        next_step='ONE_PREREGISTERED_COVERAGE_EXTENSION_OR_PAUSE'
    elif not opportunity:
        verdict='C_INSUFFICIENT_PILOT_CANDIDATE_OPPORTUNITY'
        next_step='ONE_BOUNDED_CANDIDATE_HORIZON_OR_REPLAN_DIAGNOSTIC_BEFORE_LEARNING'
    elif simple_saturates:
        verdict='S_SIMPLE_GRIPPER_RULE_SATURATES_PILOT_HEADROOM'
        next_step='VALIDATE_SIMPLE_REPAIR_ON_NEW_SCENES; NO_COMPLEX_EVALUATOR_TRAINING'
    elif t1_valid:
        verdict='A_CANDIDATE_OPPORTUNITY_WITH_USEFUL_T1'
        next_step='OBSERVABILITY_LADDER_THEN_MINIMAL_STATE_CONSEQUENCE_MODEL'
    else:
        verdict='B_CANDIDATE_OPPORTUNITY_BUT_T1_UTILITY_FAILS'
        next_step='CONTINUATION_AWARE_UTILITY_AND_OBSERVABILITY_LADDER'
    result={'schema_version':1,'protocol_sha256':digest,'complete':True,
            'root_accounting':{'registered':40,'complete':40,'eligible_unfinished':int(eligible.sum()),
                               'ceiling_previously_successful':int((~eligible).sum()),'invalid':0},
            'primary':primary,'per_task':per_task,'rescue_task_count':rescue_tasks,
            'ci95':ci,'subsets':subset_results,'c0_binarized':cheap,
            'short_horizon':{'on_changed_vs_a0_roots':int((np.any(goals!=goals[:,:1],axis=1)&eligible).sum()),
                             'm2_spread_above_epsilon_roots':int(((np.ptp(margins,axis=1)>protocol['epsilon_goal'])&eligible).sum()),
                             'root_grasp_count':int(sum(r['reference']['diagnostic']['grasp'] for r in records))},
            'gates':{'G1a_technical':True,'G1b_coverage':coverage,'G1c_opportunity':opportunity,
                     'G1d_t1':t1_valid,'simple_rule_saturates':simple_saturates},
            'verdict':verdict,'next_step':next_step,
            'new_learning_authorized':bool(coverage and opportunity and not simple_saturates),
            'learned_controller_authorized':False,
            'scope':'calibration-only diagnostic; fixed C0 continuation; no project-level blind confirmation',
            'root_sha256':{r['root']['root_id']:sha256(directory/'roots'/f"{r['root']['root_id']}.json") for r in records},
            'access_ledger_sha256':sha256(directory/'access_ledger.json')}
    atomic_json(directory/'metrics.json',result)
    atomic_json(directory/'decision.json',{k:result[k] for k in ['protocol_sha256','gates','verdict','next_step','new_learning_authorized','learned_controller_authorized','scope']})
    print(json.dumps({k:result[k] for k in ['root_accounting','primary','gates','verdict','next_step']},indent=2))


if __name__=='__main__':
    main()
