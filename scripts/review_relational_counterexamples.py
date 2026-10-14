#!/usr/bin/env python3
"""Read-only post-collection review: grouped support, sensitivity and replayed metrics."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.relational_dynamics_pilot import score,sha
from src.probes.candidate_utility_oracle import atomic_json
OUT=ROOT/'results/relational_counterexamples'

def main():
    cfg=json.loads((ROOT/'configs/relational_counterexamples_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha(ROOT/p)==h,p
    analysis=json.loads((OUT/'analysis.json').read_text())
    validation=json.loads((OUT/'validation.json').read_text());assert validation['analysis_sha256']==sha(OUT/'analysis.json')
    manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['complete']
    for r in manifest['records']:assert sha(ROOT/r['dense']['arrays'])==r['dense']['sha256']
    meta=json.loads((OUT/'samples.json').read_text())
    with np.load(OUT/'model_evaluation_data.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    dev=np.array([m['split']=='development' for m in meta]);checks=0
    for name in ('unchanged','no_action','action'):
        if name=='unchanged':delta=np.zeros_like(d['delta']);prob=np.where(d['current']>=0,d['current'],np.nan)
        else:
            with np.load(OUT/f'frozen_{name}_predictions.npz',allow_pickle=False) as f:delta=f['delta'];prob=f['probability']
        for split,mask in [('development',dev),('fit_exposed',~dev)]:
            expected=analysis['groups'][split][name]
            for k,v in score(delta,prob,d,meta,mask).items():assert v==expected[k],(name,split,k)
            checks+=1
    groups=analysis['groups']['development'];a=groups['action'];b=groups['unchanged'];n=groups['no_action']
    roots=sorted(a['per_root']);leaveout={};types={}
    for root in roots:
        leaveout[root]={}
        for name,g in [('action',a),('unchanged',b)]:
            pairs=[p for p in g['pairs'] if p['root']!=root]
            leaveout[root][name]=float(np.mean([p['error_mm'] for p in pairs]))
    for variant in sorted({p['variant'] for p in a['pairs']}):
        ar=[p for p in a['pairs'] if p['variant']==variant];br=[p for p in b['pairs'] if p['variant']==variant]
        types[variant]=dict(pairs=len(ar),material_pairs=sum(p['truth_mm']>1 for p in ar),action_mm=float(np.mean([p['error_mm'] for p in ar])),unchanged_mm=float(np.mean([p['error_mm'] for p in br])))
    support={}
    for j,name in [(4,'attachment'),(8,'release')]:
        support[name]={}
        for start,end,label in [(0,1,'onset'),(1,0,'offset')]:
            ii=np.flatnonzero(dev&(d['current'][:,j]==start)&(d['binary'][:,j]==end))
            support[name][label]=dict(branches=len(ii),roots=sorted({meta[i]['root'] for i in ii}),anchors=sorted({(meta[i]['root'],meta[i]['anchor']) for i in ii}))
    review=dict(status='CONTINUOUS_ACTION_SIGNAL_BROADER_SUPPORT_EVENT_COVERAGE_STILL_LIMITED',
        metric_groups_recomputed=checks,source_sha256=sha(OUT/'analysis.json'),script_sha256=sha(Path(__file__)),
        gains_over_no_action_pct={k:100*(1-a[k]/n[k]) for k in ['root_macro_target_plate_mm','root_macro_target_gripper_mm','pair_error_all_mm','material_error_mm']},
        root_wins={k:sum(a['per_root'][r][k]<n['per_root'][r][k] for r in roots) for k in ['target_plate_mm','target_gripper_mm']},
        leave_one_root_out_pair_errors=leaveout,leave_one_root_out_action_better_count=sum(v['action']<v['unchanged'] for v in leaveout.values()),
        per_variant=types,event_transition_support=support,training=False,new_scenes=False)
    atomic_json(OUT/'scientific_review.json',review)
    print(json.dumps(review,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
