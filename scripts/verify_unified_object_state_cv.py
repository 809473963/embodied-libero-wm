#!/usr/bin/env python3
"""Independently recompute CV headline metrics and check fold/ensemble accounting."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.candidate_utility_oracle import atomic_json,sha256
OUT=ROOT/'results/unified_object_state_cv'
def main():
 runtime=json.loads((OUT/'runtime.json').read_text());assert runtime['status']=='COMPLETE' and runtime['completed_fits']==18
 result=json.loads((OUT/'evaluation.json').read_text());edges=[tuple(e) for e in result['edges']];key=result['key_edges']
 with np.load(ROOT/'results/unified_object_state_audit_v3/node_supervision.npz',allow_pickle=False) as f:truth=f['relations'];roots=f['roots'];fold=f['fold'];positions=f['positions']
 checks=0
 for family in ['shared','independent']:
  members=[]
  for suffix in ['seed11','seed23','seed47','ensemble']:
   name=f'{family}_{suffix}'
   with np.load(OUT/f'oof_{name}.npz',allow_pickle=False) as f:p=f['edges'];nodes=f['nodes']
   if suffix!='ensemble':
    members.append(p)
    seen=np.zeros(len(roots),int)
    for k in range(3):
     with np.load(OUT/f'fold{k}_{name}_prediction.npz',allow_pickle=False) as f:
      ix=f['indices'];np.testing.assert_array_equal(ix,np.flatnonzero(fold==k));np.testing.assert_array_equal(f['edges'],p[ix]);seen[ix]+=1
    assert (seen==1).all()
   else:np.testing.assert_allclose(p,np.mean(members,0),atol=0,rtol=0)
   error=np.sqrt(np.sum((p-truth)**2,axis=-1))*1000
   macro=lambda x:float(np.mean([np.mean(x[roots==r]) for r in set(roots)]))
   expected=result['metrics'][name]
   for metric,value in [('key_mean_mm',macro(error[:,key].mean(1))),('worst_key_mm',macro(np.max(error[:,key],axis=1)))]:
    np.testing.assert_allclose(value,expected[metric],rtol=1e-12,atol=1e-12);checks+=1
   if family=='shared':np.testing.assert_allclose(p,np.stack([nodes[:,i]-nodes[:,j] for i,j in edges],1),atol=3e-7,rtol=0)
   np.testing.assert_allclose(nodes[:,8],positions[:,8],atol=1e-7,rtol=0)
 atomic_json(OUT/'validation.json',dict(status='PASS',headline_checks=checks,fold_accounting='each row exactly one held-out fold per seed',ensemble_verified=True,evaluation_sha256=sha256(OUT/'evaluation.json')))
 print('PASS:',checks,'headline metrics; all fold/ensemble accounting')
if __name__=='__main__':main()
