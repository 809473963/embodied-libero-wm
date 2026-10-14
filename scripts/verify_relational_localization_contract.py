#!/usr/bin/env python3
"""Verify saved rendering equality and projection math without constructing environments."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.candidate_utility_oracle import sha256,atomic_json
from src.probes.camera_projection_contract import official_float_projection
P=ROOT/'results/relational_localization_contract_v2'

def main():
    cfg=json.loads((ROOT/'configs/relational_localization_contract_v2_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    result=json.loads((P/'evaluation.json').read_text());assert result['verdict']=='STATIC_LOCALIZATION_RENDER_CONTRACT_VALIDATED'
    camera=json.loads((ROOT/'results/camera_projection_contract/camera_inventory.json').read_text())['camera_reference']
    checks=0;digests={}
    with np.load(P/'projected_points.npz',allow_pickle=False) as points:
        for i,r in enumerate(result['rows']):
            path=ROOT/r['source'];assert sha256(path)==r['source_sha256']
            with np.load(path,allow_pickle=False) as source:
                t=r['step'];j=int(np.flatnonzero(source['natural_step']==t)[0]);rgb=source['rgb'][j]
                xyz=np.r_[source['pose'][t,:3,:3],source['grip_pos'][t][None]]
                np.testing.assert_array_equal(xyz,points['xyz'][i])
            render=P/f"{r['root']}_t{t:03d}_render.npz"
            with np.load(render,allow_pickle=False) as saved:np.testing.assert_array_equal(saved['rgb'],rgb)
            digests[render.name]=sha256(render)
            uv,_=official_float_projection(xyz,np.array(camera['world_to_raw_pixel']));uv[:,1]=127-uv[:,1]
            np.testing.assert_allclose(uv,r['uv128'],atol=1e-10,rtol=0)
            np.testing.assert_allclose((uv-.5)/2,r['uv64'],atol=1e-10,rtol=0);checks+=1
    atomic_json(P/'validation.json',dict(status='PASS',frames_recomputed=checks,render_sha256=digests,report_sha256=sha256(ROOT/'docs/relational_localization_contract_v2_audit.md'),confirmation_opened=False))
    print(f'PASS: {checks} frames verified')
if __name__=='__main__':main()
