#!/usr/bin/env python3
"""Describe visible-mask centroid versus body-origin projection on audited old frames."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/relational_localization_contract_v2'

def main():
    cfg=json.loads((ROOT/'configs/relational_localization_contract_v2_protocol.json').read_text())
    cases={r['root_id']:r for r in cfg['cases']}
    result=json.loads((P/'evaluation.json').read_text());assert result['verdict']=='STATIC_LOCALIZATION_RENDER_CONTRACT_VALIDATED'
    rows=[];objects=['akita_black_bowl_1','akita_black_bowl_2','plate_1']
    for r in result['rows']:
        names=cases[r['root']]['dense']['geometry']['geom_names']
        with np.load(P/f"{r['root']}_t{r['step']:03d}_render.npz",allow_pickle=False) as a:seg=a['segmentation']
        for k,name in enumerate(objects):
            ids=[i for i,g in enumerate(names) if g and g.startswith(name+'_')];assert ids
            mask=(seg[...,0]==5)&np.isin(seg[...,1],ids);yx=np.argwhere(mask);uv=np.array(r['uv128'][k]);n=len(yx)
            rows.append(dict(root=r['root'],step=r['step'],object=name,visible_pixels=n,projected_body_origin_uv=uv.tolist(),visible_centroid_uv=None if not n else yx[:,::-1].mean(0).tolist(),centroid_offset_px=None if not n else float(np.linalg.norm(yx[:,::-1].mean(0)-uv)),distance_to_visible_mask_px=None if not n else float(np.linalg.norm(yx[:,::-1]-uv,axis=1).min())))
    summary={}
    for name in objects:
        v=[r for r in rows if r['object']==name]
        summary[name]=dict(samples=len(v),zero_visible=sum(r['visible_pixels']==0 for r in v),median_visible_pixels=float(np.median([r['visible_pixels'] for r in v])),median_centroid_offset_px=float(np.median([r['centroid_offset_px'] for r in v if r['centroid_offset_px'] is not None])))
    (P/'label_semantics.json').write_text(json.dumps(dict(type='post_hoc_descriptive',summary=summary,rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
