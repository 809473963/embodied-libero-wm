#!/usr/bin/env python3
"""Read-only event coverage review, separating observed release from window maturation."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.candidate_utility_oracle import atomic_json,sha256
OUT=ROOT/'results/relational_event_h24'

def support(rows):
    return dict(branches=len(rows),roots=sorted({r['root'] for r in rows}),anchors=sorted({f"{r['root']}/t{r['anchor']}" for r in rows}))

def main():
    cfg=json.loads((ROOT/'configs/relational_event_h24_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    analysis=json.loads((OUT/'analysis.json').read_text());validation=json.loads((OUT/'validation.json').read_text())
    assert validation['analysis_sha256']==sha256(OUT/'analysis.json')
    manifest=json.loads((OUT/'manifest.json').read_text());records={r['name']:r for r in manifest['records']}
    rows=analysis['rows'];physical={}
    for r in rows:
        record=records[r['name']];path=ROOT/record['dense']['arrays'];assert sha256(path)==record['dense']['sha256']
        with np.load(path,allow_pickle=False) as a:
            t=r['anchor'];physical[r['name']]=dict(grip=bool(a['gripper_contact'][t,0]),on=bool(a['on'][t,0]),external=bool(a['non_gripper_contact'][t,0]))
    checks=0
    for key,c in analysis['summary'].items():
        group,h,event=key.split('/');j=['bilateral_attachment','detached_attachment','lift_hold_evidence','released_on_quiet','release_after_attachment'].index(event)
        rr=[r for r in rows if group=='all' or group in r['strata']]
        now=np.array([r['current'][0][j] for r in rr]);after=np.array([r['endpoints'][h[1:]][0][j] for r in rr])
        assert c['branches']==len(rr) and c['positive']==int((after==1).sum()) and c['onset']==int(((now==0)&(after==1)).sum()) and c['offset']==int(((now==1)&(after==0)).sum())
        checks+=1
    release={}
    for h in ('8','16','24'):
        positive=[r for r in rows if r['current'][0][4]==0 and r['endpoints'][h][0][4]==1]
        release[h]=dict(all=support(positive),from_detached=support([r for r in positive if r['current'][0][1]==1]),
            from_current_grip=support([r for r in positive if physical[r['name']]['grip']]),
            already_released_on_at_anchor=support([r for r in positive if not physical[r['name']]['grip'] and physical[r['name']]['on']]))
    splits={}
    for name,init in [('fit_source','31'),('development_source','34')]:
        rr=[r for r in rows if r['root'].endswith(init)]
        splits[name]=dict(**support(rr),events={})
        for j,event in enumerate(['attachment','detached','lift','quiet_on','release']):
            splits[name]['events'][event]=support([r for r in rr if r['current'][0][j]==0 and r['endpoints']['24'][0][j]==1])
        splits[name]['events']['detached_to_release']=support([r for r in rr if r['current'][0][1]==1 and r['endpoints']['24'][0][4]==1])
    held_rows=[r for r in rows if r['current'][0][1]==1 and r['endpoints']['24'][0][4]==1]
    result=dict(status='H24_RELEASE_WINDOW_COVERAGE_IMPROVED_ACTIVE_RELEASE_EVIDENCE_LIMITED',date='2026-10-08',
        source_sha256=sha256(OUT/'analysis.json'),script_sha256=sha256(Path(__file__)),recomputed_summary_groups=checks,
        release_origins_by_horizon=release,splits=splits,held_to_release_cases=held_rows,
        scoped_roots_missing_all_strata=sorted(set(cfg['root_ids'])-{r['root'] for r in rows}),
        no_training=True,no_new_scene_test=True)
    atomic_json(OUT/'scientific_review.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='held_to_release_cases'},ensure_ascii=False,indent=2))
    print('held release cases',[(r['root'],r['anchor'],r['variant']) for r in held_rows])
if __name__=='__main__':main()
