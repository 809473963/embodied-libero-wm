#!/usr/bin/env python3
"""Apply frozen causal labels to existing recordings. No simulator or training."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hashlib
import json
import numpy as np
from src.probes.relational_event_labels import EVENTS, labels, future_targets


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    protocol=ROOT/'configs/relational_event_labels_v1.json'
    cfg=json.loads(protocol.read_text())
    for p,digest in cfg['frozen_files'].items():
        assert sha(ROOT/p)==digest,p
    manifest=json.loads((ROOT/cfg['manifest']).read_text())
    out=ROOT/'results/relational_event_v1';(out/'arrays').mkdir(parents=True,exist_ok=True)
    summaries=[];prefixes={};source_samples=set();repeat={}
    for row in manifest['branches']:
        source=ROOT/row['dense']['arrays'];assert sha(source)==row['dense']['sha256']
        assert all(row['identity'].values())
        assert row['dense']['geometry']['control_freq']==cfg['control_hz']
        # Explicit field whitelist: success and terminal metadata cannot enter label extraction.
        with np.load(source,allow_pickle=False) as f:
            a={k:f[k] for k in cfg['source_fields']}
        root=row['root']['root_id'];p=row['root']['prefix_steps'];cid=row['candidate_id']
        ds=[labels(a,cfg['parameters'],i) for i in (0,1)]
        values=np.stack([d['values'] for d in ds],axis=1)
        onset=np.stack([d['onset'] for d in ds],axis=1)
        if (root,cid) in repeat:
            assert np.array_equal(values,repeat[root,cid]);continue
        repeat[root,cid]=values
        if root in prefixes:np.testing.assert_array_equal(prefixes[root],values[:p+1])
        else:prefixes[root]=values[:p+1].copy()
        for t in range(len(values)):
            source_samples.add((root, 'prefix' if t<=p else cid,t))
        path=out/'arrays'/f'{root}_c{cid:02d}.npz'
        np.savez_compressed(path,step=a['step'],values=values,onset=onset,
            future_values=np.stack([future_targets(d['values'],cfg['horizons']) for d in ds],axis=1),
            lift_reference_step=np.stack([d['lift_reference_step'] for d in ds],axis=1),
            local_position=np.stack([d['local_position'] for d in ds],axis=1),
            pair_position=ds[0]['pair_position'],pair_velocity=ds[0]['pair_velocity'],
            contacts=a['contacts'],pad_contact=a['pad_contact'],gripper_contact=a['gripper_contact'],
            non_gripper_contact=a['non_gripper_contact'],on=a['on'],actions=a['actions'])
        summary=dict(root_id=root,candidate_id=cid,root_step=p,source=str(source.relative_to(ROOT)),
                     source_sha256=row['dense']['sha256'],arrays=str(path.relative_to(ROOT)),sha256=sha(path),events={})
        for j,name in enumerate(EVENTS):
            v=values[:,0,j];ends=np.flatnonzero(v==1)
            summary['events'][name]=dict(positive_samples=int((v==1).sum()),
                negative_samples=int((v==0).sum()),unknown_samples=int((v<0).sum()),
                positive_window_ends=ends.tolist(),
                fully_after_root_window_ends=ends[ends-cfg['parameters']['window_intervals']>=p].tolist(),
                observed_onsets=np.flatnonzero(onset[:,0,j]==1).tolist(),
                other_object_positive_samples=int((values[:,1,j]==1).sum()))
        summaries.append(summary)
    coverage={}
    for name in EVENTS:
        rows=[r for r in summaries if r['events'][name]['positive_samples']]
        coverage[name]=dict(branches=len(rows),roots=sorted({r['root_id'] for r in rows}),
            unknown_samples=sum(r['events'][name]['unknown_samples'] for r in summaries),
            other_positive_branches=sum(r['events'][name]['other_object_positive_samples']>0 for r in summaries),
            fully_post_root_branches=sum(bool(r['events'][name]['fully_after_root_window_ends']) for r in summaries))
    result=dict(status='OPERATIONAL_LABEL_V1_COMPUTABLE_COVERAGE_INSUFFICIENT_FOR_TRAINING',
        protocol_sha256=sha(protocol),branches=len(summaries),roots=len(prefixes),
        samples_excluding_duplicate_runs=sum(len(v) for v in repeat.values()),
        samples_after_shared_prefix_dedup=len(source_samples),
        dedup_note='Distinct post-root branches may still coincide; this count is not independent data.',
        events=list(EVENTS),coverage=coverage,records=summaries,
        tests='7 focused tests passed before freeze; see validation.json for rerun',
        no_success_used_for_labels=True,simulation_run=False,training_run=False,
        limitations=['All roots previously exposed; no new scene generalization.',
          'No aligned dense natural RGB; not natural-estimator training data.',
          'Operational stability evidence, not validated load-bearing or geometric clearance.',
          'Other-bowl attachment lacks positive coverage; synthetic identity test alone is insufficient.',
          'Thresholds are engineering conventions on previously inspected recordings, not blind validation.'])
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
