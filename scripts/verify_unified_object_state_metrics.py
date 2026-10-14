#!/usr/bin/env python3
"""Independent physical-unit and scene-macro recomputation of all CV metrics."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.candidate_utility_oracle import sha256,atomic_json

def main():
 out=ROOT/'results/unified_object_state_cv';r=json.loads((out/'evaluation.json').read_text());edges=[tuple(e) for e in r['edges']]
 with np.load(ROOT/'results/unified_object_state_audit_v3/node_supervision.npz',allow_pickle=False) as f:xyz=f['positions'];y=f['relations'];roots=f['roots'];fold=f['fold']
 unique=sorted(set(roots));macro=lambda v: np.stack([v[roots==scene].mean(axis=0) for scene in unique]).mean(axis=0)
 checked=0
 for name,m in r['metrics'].items():
  with np.load(out/f'oof_{name}.npz',allow_pickle=False) as f:n=f['nodes'];p=f['edges']
  ne=np.sqrt(((n-xyz)**2).sum(-1))*1000;ee=np.sqrt(((p-y)**2).sum(-1))*1000
  np.testing.assert_allclose(macro(ne),m['node_mm'],atol=1e-10,rtol=1e-12);np.testing.assert_allclose(macro(ee),m['edge_mm'],atol=1e-10,rtol=1e-12);checked+=45
  for g,ix in r['groups'].items():np.testing.assert_allclose(macro(ee[:,ix].mean(-1)),m['groups_mm'][g],atol=1e-10,rtol=1e-12);checked+=1
  if name=='mean_position':
   for k in range(3):
    with np.load(out/f'fold{k}_normalization.npz',allow_pickle=False) as f:center=f['shared']
    np.testing.assert_array_equal(n[fold==k,:8],np.broadcast_to(center,n[fold==k,:8].shape))
  if name.startswith('independent'):
   reconstructed=np.stack([p[:,edges.index((i,8))]+n[:,8] for i in range(8)],1)
   np.testing.assert_allclose(n[:,:8],reconstructed,atol=3e-7,rtol=0)
  tri=np.stack([np.linalg.norm(p[:,edges.index((i,j))]+p[:,edges.index((j,k))]-p[:,edges.index((i,k))],axis=-1)*1000 for i in range(9) for j in range(i+1,9) for k in range(j+1,9)],1)
  np.testing.assert_allclose(macro(tri.mean(1)),m['triangle_closure_mm'],atol=1e-5,rtol=1e-6);checked+=1
 wins=[]
 for seed in [11,23,47]:
  s=r['metrics'][f'shared_seed{seed}'];d=r['metrics'][f'independent_seed{seed}']
  wins.append(s['key_mean_mm']/d['key_mean_mm']<=.9 and s['worst_key_mm']/d['worst_key_mm']<=.9 and max(s['edge_mm'][k]/d['edge_mm'][k] for k in r['key_edges'])<=1.1)
 assert wins==r['seed_gate'] and (sum(wins)>=2)==r['shared_prioritized']
 atomic_json(out/'detailed_validation.json',dict(status='PASS',metric_checks=checked,gate_recomputed=True,mean_baseline_verified=True,source_sha256=sha256(out/'evaluation.json'),script_sha256=sha256(Path(__file__))))
 print('PASS',checked,'node/edge/group/closure values and gate')
if __name__=='__main__':main()
