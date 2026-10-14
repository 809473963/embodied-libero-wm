#!/usr/bin/env python3
"""Recompute key research outputs without refitting or opening any other scene set."""
import sys,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.diagnose_multi_object_perception import inputs,read_npz,OUT,CV,EDGES,KEY
from src.probes.candidate_utility_oracle import atomic_json,sha256

def main():
 d,n,_=inputs();roots=n['roots'];unique=sorted(set(roots));macro=lambda x:np.mean([x[roots==r].mean(0) for r in unique],0);checks=0
 baseline=json.loads((OUT/'baseline_manifest.json').read_text())
 for p,h in baseline['sources'].items():assert sha256(ROOT/p)==h,p
 analysis=json.loads((OUT/'object_error_analysis.json').read_text())['models']
 for name,a in analysis.items():
  p=read_npz(CV/f'oof_{name}.npz');delta=(p['nodes']-n['positions'])*1000;err=np.linalg.norm(delta,axis=2)
  np.testing.assert_allclose(macro(err),a['euclidean_mm']['scene_macro_mean'],atol=1e-9,rtol=0);np.testing.assert_allclose(macro(abs(delta)),a['axis_mae_mm']['scene_macro_mean'],atol=1e-9,rtol=0);checks+=36
 identity=json.loads((OUT/'identity_diagnosis.json').read_text())
 for family in ['shared','independent']:
  p=read_npz(CV/f'oof_{family}_ensemble.npz');gt=n['positions'][:,:2]
  original=np.linalg.norm(p['nodes'][:,:2]-gt,axis=2);swapped=np.linalg.norm(p['nodes'][:,[1,0]]-gt,axis=2);choose=swapped.sum(1)<original.sum(1)
  oracle=np.where(choose[:,None],swapped,original);np.testing.assert_allclose(macro(oracle)*1000,identity[f'{family}_ensemble']['per_frame_best']['node_mm'][:2],atol=1e-9,rtol=0);checks+=2
 frames=read_npz(OUT/'diagnostic_frames.npz');np.testing.assert_array_equal(frames['visible_pixels'],d['masks'].sum((2,3))*256)
 geom=read_npz(OUT/'geometry_predictions.npz');report=json.loads((OUT/'geometry_feasibility.json').read_text())
 for f in range(3):
  fit=d['fold']!=f;z=np.median([np.median(n['positions'][roots==r,:5,2],axis=0) for r in sorted(set(roots[fit]))],0)
  np.testing.assert_array_equal(geom['height_priors'][~fit],np.broadcast_to(z,geom['height_priors'][~fit].shape))
 for noise,runs in report['noise_results'].items():
  for rep,result in enumerate(runs):
   pred=geom[f'plane_oracle_uv_noise{noise}_repeat{rep}'];err=np.linalg.norm(pred-n['positions'][:,:5],axis=2)*1000
   np.testing.assert_allclose(macro(err),result['errors']['scene_macro_mean'],atol=1e-9,rtol=0);checks+=5
 for key,field in [('plane_existing_oof_uv','natural_existing_oof_uv'),('existing_oof_uv_predicted_depth','existing_oof_uv_predicted_depth')]:np.testing.assert_allclose(macro(np.linalg.norm(geom[key]-n['positions'][:,:3],axis=2)*1000),report[field]['scene_macro_mean'],atol=1e-9,rtol=0);checks+=3
 sens=json.loads((OUT/'input_sensitivity.json').read_text());pred=read_npz(OUT/'sensitivity_predictions.npz')
 for name,metric in sens['models'].items():
  np.testing.assert_allclose(macro(np.linalg.norm(pred[name]-n['positions'],axis=2)*1000),metric['node_mm'],atol=1e-9,rtol=0);checks+=9
 pn=read_npz(OUT/'pnp_oracle_predictions.npz');pr=json.loads((OUT/'pnp_oracle.json').read_text());ix=pn['indices'];assert set(roots[ix])==set(unique) and len(ix)==104
 for noise,runs in pr['results'].items():
  for rep,result in enumerate(runs):
   e=np.linalg.norm(pn[f'noise{noise}_repeat{rep}']-n['positions'][ix,:2],axis=2)*1000
   expected=np.mean([e[roots[ix]==r].mean(0) for r in unique],0)
   np.testing.assert_allclose(expected,result['errors']['scene_macro_mean'],atol=1e-9,rtol=0);checks+=2
 cases=json.loads((OUT/'visual_case_manifest.json').read_text())['cases'];assert len(cases)==39
 for c in cases:assert (OUT/c['image']).exists()
 scripts=['scripts/diagnose_multi_object_perception.py','scripts/probe_multi_object_inputs_geometry.py','scripts/probe_multi_object_pnp_oracle.py','scripts/supplement_multi_object_strata.py','scripts/visualize_multi_object_diagnosis.py','scripts/verify_multi_object_perception_diagnosis.py','tests/test_multi_object_perception_diagnosis.py','configs/multi_object_perception_diagnosis_protocol.json','configs/multi_object_pnp_oracle_protocol.json']
 atomic_json(OUT/'validation.json',dict(status='PASS',independent_numeric_checks=checks,old_sources_unchanged=True,fit_only_height_priors=True,figures_and_cases=39,statistical_unit='13 old scenes; no new independent confirmation',source_hashes={p:sha256(ROOT/p) for p in scripts},result_hashes={p.name:sha256(p) for p in OUT.glob('*.json') if p.name!='validation.json'},training_executed=False,new_development_opened=False,confirmation_opened=False,limitations='Hash records describe current reproducible implementation, not retroactive proof of immutable preregistration; no claim that all process file access was OS-audited.'))
 print('PASS',checks,'numeric values plus source/fold/prior/case checks')
if __name__=='__main__':main()
