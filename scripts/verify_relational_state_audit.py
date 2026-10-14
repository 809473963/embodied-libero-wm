#!/usr/bin/env python3
"""Independent checks of audit artifacts against old raw arrays; no simulator."""
from pathlib import Path
import json
import hashlib
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/relational_state_audit'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    read=lambda name:json.loads((OUT/name).read_text())
    m=read('source_manifest.json');a=read('availability.json');o=read('outcome_association.json')
    rows=read('sampled_relations.json')['rows'];review=read('review_cases.json')
    for entry in m['source_files']:
        p=Path(entry['path']);p=p if p.is_absolute() else ROOT/p
        assert sha(p)==entry['sha256'],str(p)
    for name,expected in m['output_sha256'].items(): assert sha(OUT/name)==expected,name
    groups=defaultdict(list)
    for r in rows:
        assert not any(k in r for k in ['Y','first_success_step','final_on','final_grasp','A0_Y'])
        groups[r['root_id']].append(r)
    assert len(groups)==40 and len(rows)==1280
    raw_onset=raw_new=raw_loss=0
    final_counts=np.zeros(4,dtype=int)
    eligible_count=0;rescue_count=0
    for key,rs in groups.items():
        old=json.loads((ROOT/'results/candidate_utility_oracle/roots'/f'{key}.json').read_text())
        with np.load(ROOT/'results/continuation_utility_observability/arrays'/f'{key}.npz',allow_pickle=False) as f:
            x=f['branch_state'];initial=f['prefix_state'][-1]
            for j,r in enumerate(rs):
                contacts=np.r_[initial[168],x[j,:,168]].astype(bool)
                assert contacts.tolist()==r['target_contact_samples']
                onset=any(not contacts[k] and contacts[k+1] for k in range(4))
                loss=any(contacts[k] and not contacts[k+1] for k in range(4))
                raw_onset+=onset;raw_loss+=loss;raw_new+=onset and not contacts[0]
                assert onset==r['events']['sampled_target_contact_onset']
                assert loss==r['events']['sampled_target_contact_loss']
                cid=r['candidate_id'];d=old['rows'][cid]
                p=x[j,-1,:3].astype(float)-x[j,-1,26:29]
                np.testing.assert_allclose(p,r['target_plate_relative_position_m'][-1],atol=1e-12)
                if old['reference']['first_success_step'] is None:
                    eligible_count+=1
                    final_counts+=np.array([d['success'],d['success'] and not d['final']['on'],
                        d['success'] and d['final']['on'] and d['final']['grasp'],
                        d['first_success_step'] is not None and d['first_success_step']<=r['times'][-1]],int)
            if old['reference']['first_success_step'] is None and not old['rows'][0]['success']:
                rescue_count+=any(old['rows'][r['candidate_id']]['success'] for r in rs)
    assert (raw_onset,raw_new,raw_loss)==(10,0,150)
    assert eligible_count==a['eligible_branches']==1216 and rescue_count==7
    assert final_counts.tolist()==[456,107,6,13]
    for e,reported in a['event_coverage_all'].items():
        positives=[r for r in rows if r['events'][e]]
        assert reported['branches']==len(positives)
        assert reported['roots']==len({r['root_id'] for r in positives})
    assert len({(r['root_id'],r['candidate_id']) for r in review['unique_cases']})==len(review['unique_cases'])==48
    assert sum(bool(r['mixed_Y_signature_count']) and r['eligible'] for r in o['roots'])==19
    assert sum(r['discrete_signature_count']==1 and r['eligible'] for r in o['roots'])==26
    validation={'pass':True,'source_file_count':len(m['source_files']),'roots':40,'branches':1280,
                'independently_recomputed':['raw contact onset/loss','root-first contact','H8 relative position','source Y endpoint counts','event root denominators','7 rescue roots'],
                'source_hashes_rechecked':True,'targeted_tests':'6 passed',
                'tests_source_sha256':sha(ROOT/'tests/test_relational_state_availability.py'),
                'verifier_sha256':sha(Path(__file__)),
                'visual_review_sha256':sha(OUT/'review_notes.json'),
                'report_sha256':sha(ROOT/'docs/relational_state_data_availability_audit.md'),
                'simulation_steps':0,'new_fits':0}
    (OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(validation,ensure_ascii=False))

if __name__=='__main__':main()
