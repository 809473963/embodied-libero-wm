#!/usr/bin/env python3
"""Post-hoc OOF localization task checks; no fitting or selection."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.candidate_utility_oracle import atomic_json
P=ROOT/'results/relational_localization_cv'
def main():
 assert json.loads((P/'pipeline.json').read_text())['status']=='COMPLETE'
 with np.load(P/'dataset.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
 meta=json.loads((P/'metadata.json').read_text());roots=np.array([r['root'] for r in meta['rows']]);w=np.array([1/(roots==r).sum() for r in roots]);preds=[];pred_masks=[]
 baseline=np.zeros_like(d['points']);baseline_masks=np.zeros_like(d['masks'])
 for fold in range(3):
  fit=d['fold']!=fold;test=~fit;fw=w[fit]/w[fit].sum()
  baseline[test]=(d['points'][fit]*fw[:,None]).sum(0);baseline_masks[test]=(d['masks'][fit]*fw[:,None,None,None]).sum(0)
 for seed in [11,23,47]:
  pred=np.zeros_like(d['points']);mask=np.zeros_like(d['masks'])
  for fold in range(3):
   with np.load(P/f'fold{fold}_localized_seed{seed}_prediction.npz',allow_pickle=False) as f:
    pred[f['indices']]=f['points'];mask[f['indices']]=f['mask']
  preds.append(pred);pred_masks.append(mask)
 def metrics(pred,mask):
  uv=np.linalg.norm((pred[:,:6]-d['points'][:,:6]).reshape(-1,3,2),axis=2)
  dep=abs(pred[:,6:]-d['points'][:,6:])*1000
  mse=((mask-d['masks'])**2).mean((2,3))
  macro=lambda a:np.mean([a[roots==r].mean(0) for r in sorted(set(roots))],0).tolist()
  return dict(projected_origin_error_px=macro(uv),depth_mae_mm=macro(dep),mask_fraction_mse=macro(mse))
 out=dict(type='post_hoc_descriptive',object_order=['bowl1','bowl2','plate'],localized=metrics(np.mean(preds,0),np.mean(pred_masks,0)),training_scene_mean=metrics(baseline,baseline_masks))
 atomic_json(P/'localization_diagnostics.json',out)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
