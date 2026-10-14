#!/usr/bin/env python3
"""Descriptive, root-balanced localization; no model selection or retraining."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.probes.candidate_utility_oracle import atomic_json
P=ROOT/'results/relational_natural_state'

def main():
    with np.load(P/'dataset.npz',allow_pickle=False) as f:y=f['target'];history=f['history']
    with np.load(P/'rgb_proprio_ensemble_prediction.npz',allow_pickle=False) as f:pred=f['state']
    meta=json.loads((ROOT/'results/relational_autorun/models/samples.json').read_text())
    anchors={}
    for i,m in enumerate(meta):
        if m['split']=='new_development':anchors[(m['root'],m['anchor'])]=int(history[i,-1])
    rows=[]
    for (root,t),idx in sorted(anchors.items()):
        truth=y[idx,:12].reshape(2,2,3);p=pred[idx,:12].reshape(2,2,3)
        errors=np.linalg.norm(p-truth,axis=2)*1000
        swapped=np.linalg.norm(p[::-1]-truth,axis=2)*1000
        e=(p-truth)*1000
        rows.append(dict(root=root,anchor=t,index=idx,error_mm=errors.tolist(),target_plate_axis_abs_mm=abs(e[0,0]).tolist(),target_plate_axis_signed_mm=e[0,0].tolist(),
          oracle_identity_swap_better=bool(swapped.mean()<errors.mean()),original_all_relations_mm=float(errors.mean()),oracle_best_identity_all_relations_mm=float(min(errors.mean(),swapped.mean()))))
    groups=sorted({r['root'] for r in rows})
    macro=lambda key:np.mean([np.mean([r[key] for r in rows if r['root']==root],axis=0) for root in groups],axis=0).tolist()
    summary=dict(type='post_hoc_descriptive_only',unique_roots=len(groups),unique_anchors=len(rows),target_plate_axis_mae_mm=macro('target_plate_axis_abs_mm'),target_plate_axis_bias_mm=macro('target_plate_axis_signed_mm'),original_all_relations_mm=macro('original_all_relations_mm'),oracle_best_identity_all_relations_mm=macro('oracle_best_identity_all_relations_mm'),swap_better_anchors=sum(r['oracle_identity_swap_better'] for r in rows),rows=rows,source_prediction_sha256=hashlib.sha256((P/'rgb_proprio_ensemble_prediction.npz').read_bytes()).hexdigest())
    atomic_json(P/'perception_error_localization.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
