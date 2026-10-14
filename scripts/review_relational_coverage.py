#!/usr/bin/env python3
"""Read-only scientific coverage review; no training/simulator or outcome fitting."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=ROOT/'results/relational_coverage_expansion'
    cfg=json.loads((ROOT/'configs/relational_coverage_expansion_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha(ROOT/p)==h,p
    data=json.loads((out/'coverage.json').read_text())
    assert data['unique_roots']==40 and data['unique_branches']==120
    groups={};events=list(data['coverage']);counts={};paired=[]
    for row in data['records']:
        for path,h in [(row['source'],row['source_sha256']),(row['labels'],row['labels_sha256'])]:assert sha(ROOT/path)==h,path
        root=row['root_id'];cid=row['candidate_id'];p=row['root_step']
        with np.load(ROOT/row['labels'],allow_pickle=False) as f:v=f['values']
        with np.load(ROOT/row['source'],allow_pickle=False) as f:
            rel=f['pose'][:,0,:3]-f['pose'][:,2,:3]
            actions=f['actions']
        groups.setdefault(root,[]).append((row,v,rel,actions))
        split='development' if root.endswith('34') else 'fit'
        for j,name in enumerate(events):
            for group in [split,f"task{row['task_id']:02d}"]:
                c=counts.setdefault(group,{}).setdefault(name,dict(positive_roots=set(),negative_roots=set(),onset_roots=set(),offset_roots=set(),future_valid=0,future_changed=0))
                if (v[:,0,j]==1).any():c['positive_roots'].add(root)
                if (v[:,0,j]==0).any():c['negative_roots'].add(root)
                if ((v[:-1,0,j]==0)&(v[1:,0,j]==1)).any():c['onset_roots'].add(root)
                if ((v[:-1,0,j]==1)&(v[1:,0,j]==0)).any():c['offset_roots'].add(root)
                # Planned chunk boundaries after root only; actions for these H8 are known at start.
                for t in range(p,293,8):
                    a,b=int(v[t,0,j]),int(v[t+8,0,j])
                    if a>=0 and b>=0:
                        c['future_valid']+=1;c['future_changed']+=int(a!=b)
    for root,rows in groups.items():
        assert {r[0]['candidate_id'] for r in rows}=={0,49,50}
        p=rows[0][0]['root_step'];byid={r[0]['candidate_id']:r for r in rows};a=byid[0]
        for cid in (49,50):
            b=byid[cid]
            paired.append(dict(root_id=root,candidate_id=cid,
                action_difference_max=float(np.abs(a[3][p:p+8]-b[3][p:p+8]).max()),
                target_plate_H8_difference_m=float(np.linalg.norm(a[2][p+8]-b[2][p+8])),
                events_disagree=[events[j] for j in range(len(events)) if a[1][p+8,0,j]>=0 and b[1][p+8,0,j]>=0 and a[1][p+8,0,j]!=b[1][p+8,0,j]]))
    for g in counts.values():
        for c in g.values():
            for key in ('positive_roots','negative_roots','onset_roots','offset_roots'):c[key]=sorted(c[key])
            c['unchanged_baseline_accuracy']=1-c['future_changed']/c['future_valid'] if c['future_valid'] else None
    summary=dict(status='READY_FOR_BOUNDED_PRIVILEGED_PILOT_DESIGN_NOT_GENERALIZATION',
        source_sha256=sha(out/'coverage.json'),code_sha256=sha(Path(__file__)),groups=counts,H8_pairs=paired,
        roots_with_H8_event_disagreement=sorted({r['root_id'] for r in paired if r['events_disagree']}),
        roots_with_H8_position_difference_over_1mm=sorted({r['root_id'] for r in paired if r['target_plate_H8_difference_m']>.001}),
        descriptive_position_threshold_m=.001,
        no_change_denominator='Known target labels at post-root planned H8 boundaries, correlated branch samples; not independent trials.',
        split='init31-33 fit (30 roots), init34 development (10); retain already-successful roots for physical dynamics, unlike old utility eligible29/9.',
        caveats=['All old roots previously exposed.','No other-bowl attachment positives.','Event predicates can persist despite different physical states.','Recorded future feedback actions beyond a planned chunk are not available at root.'])
    (out/'scientific_review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('groups','H8_pairs')},ensure_ascii=False,indent=2))
    for split in ('fit','development'):
        print(split,{k:{'positive_roots':len(v['positive_roots']),'onset_roots':len(v['onset_roots']),'unchanged_accuracy':v['unchanged_baseline_accuracy']} for k,v in counts[split].items()})

if __name__=='__main__':main()
