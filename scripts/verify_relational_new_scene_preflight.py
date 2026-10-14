#!/usr/bin/env python3
"""Recheck persisted replay arrays and scene novelty without simulator access."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.relational_new_scene import near,distance,fingerprint,probe_actions
from src.probes.candidate_utility_oracle import atomic_json,sha256
OUT=ROOT/'results/relational_new_scene_preflight'

def main():
    cfg=json.loads((ROOT/'configs/relational_new_scene_preflight_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    analysis=json.loads((OUT/'analysis.json').read_text());rt=json.loads((OUT/'runtime.json').read_text())
    assert rt['status']=='COMPLETE' and rt['confirmation_seeds_executed']==0
    records=analysis['records'];allowed={(r['task_id'],r['seed']) for r in cfg['scenes']}
    assert {(r['task_id'],r['seed']) for r in records}==allowed and len(records)==20
    old=[]
    for r in json.loads((ROOT/'results/relational_coverage_expansion/coverage.json').read_text())['records']:
        if r['candidate_id']==0:
            assert sha256(ROOT/r['source'])==r['source_sha256']
            with np.load(ROOT/r['source'],allow_pickle=False) as f:old.append((r['root_id'],f['pose'][0]))
    accepted=[];checked=0;comparisons=[]
    for r in records:
        assert r['status']=='REPLAY_VALID' and r['split']=='development'
        assert sha256(ROOT/r['initial'])==r['initial_sha256']
        with np.load(ROOT/r['initial'],allow_pickle=False) as f:initial={k:f[k] for k in f.files}
        assert r['layout_fingerprint']==fingerprint(initial['pose'])
        reference=None
        for run in r['runs']:
            assert sha256(ROOT/run['path'])==run['sha256']
            with np.load(ROOT/run['path'],allow_pickle=False) as f:a={k:f[k] for k in f.files}
            assert a['pose'].shape==(25,8,13) and all(np.isfinite(v).all() for v in a.values())
            np.testing.assert_array_equal(a['actions'],probe_actions())
            for x,y in [('initial_state','sim_state'),('initial_rgb','rgb'),('initial_proprio','proprio')]:np.testing.assert_array_equal(a[x],initial[y])
            np.testing.assert_array_equal(a['pose'][0],initial['pose'])
            if reference is None:reference=a
            else:
                assert set(a)==set(reference)
                for k in a:np.testing.assert_array_equal(a[k],reference[k])
            checked+=1
        pose=initial['pose'];matches=[name for name,p in old if near(pose,p)];newmatches=[name for name,p in accepted if near(pose,p)]
        closest=min(((name,distance(pose,p)) for name,p in old),key=lambda z:z[1]['max_position_m'])
        comparisons.append(dict(seed=r['seed'],task=r['task_id'],eligible_for_development=not matches and not newmatches,old_near_matches=matches,new_near_matches=newmatches,nearest_old=[closest[0],closest[1]]))
        if not matches and not newmatches:accepted.append((str(r['seed']),pose))
    assert comparisons==analysis['comparisons'] and len(accepted)==20 and checked==60
    values=[c['nearest_old'][1]['max_position_m']*1000 for c in comparisons]
    result=dict(status='PASS',date='2026-10-08',replay_arrays_checked=checked,eligible_development_scenes=20,old_layouts_compared=40,
        old_nearest_max_object_position_distance_mm_range=[min(values),max(values)],confirmation_generated=False,
        analysis_sha256=sha256(OUT/'analysis.json'),verifier_sha256=sha256(Path(__file__)),no_simulation_or_training=True)
    atomic_json(OUT/'validation.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
