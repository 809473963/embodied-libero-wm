#!/usr/bin/env python3
"""Validate H24 recordings and summarize causal event transitions by root/stratum."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.relational_event_recording import validate
from src.probes.relational_event_sampling import candidates
from src.probes.relational_event_labels import labels,EVENTS
from src.probes.candidate_utility_oracle import atomic_json,sha256
OUT=ROOT/'results/relational_event_h24'

def main():
    cfg=json.loads((ROOT/'configs/relational_event_h24_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['complete'] and len(manifest['records'])==6*len(cfg['cases'])
    lc=json.loads((ROOT/'configs/relational_event_labels_v1.json').read_text())['parameters']
    summary={};rows=[];seen={}
    for r in manifest['records']:
        assert sha256(ROOT/r['dense']['arrays'])==r['dense']['sha256']
        assert sha256(ROOT/r['source'])==r['source_sha256']
        with np.load(ROOT/r['dense']['arrays'],allow_pickle=False) as f:a={k:f[k] for k in f.files}
        t=r['anchor'];validate(a,r['dense']['geometry'],t)
        with np.load(ROOT/r['source'],allow_pickle=False) as f:
            plan=candidates(f['actions'][t:t+8],cfg['translation_action_offset'])[r['variant']]
            np.testing.assert_array_equal(a['actions'][:t],f['actions'][:t])
            for field in ('pose','qpos','qvel','pad_contact','gripper_contact','non_gripper_contact','on','grip_pos','grip_rot','grip_velocity'):
                np.testing.assert_array_equal(a[field][:t+1],f[field][:t+1])
        np.testing.assert_array_equal(a['actions'][t:],plan)
        v=np.stack([labels(a,lc,j)['values'] for j in (0,1)],axis=1)
        with np.load(OUT/'labels'/f"{r['name']}.npz",allow_pickle=False) as f:np.testing.assert_array_equal(v,f['values'])
        key=(r['root_id'],t)
        if key in seen:np.testing.assert_array_equal(v[:t+1],seen[key])
        else:seen[key]=v[:t+1].copy()
        row=dict(name=r['name'],root=r['root_id'],anchor=t,strata=r['strata'],variant=r['variant_name'],current=v[t].tolist(),endpoints={str(h):v[t+h].tolist() for h in (8,16,24)})
        rows.append(row)
        for group in ['all']+r['strata']:
            for h in (8,16,24):
                for j,event in enumerate(EVENTS):
                    sk=f'{group}/H{h}/{event}';c=summary.setdefault(sk,dict(branches=0,known=0,positive=0,onset=0,offset=0,onset_roots=set(),onset_anchors=set(),positive_roots=set(),other_positive=0))
                    now,after=int(v[t,0,j]),int(v[t+h,0,j]);c['branches']+=1;c['known']+=int(after>=0)
                    if after==1:c['positive']+=1;c['positive_roots'].add(r['root_id'])
                    if now==0 and after==1:c['onset']+=1;c['onset_roots'].add(r['root_id']);c['onset_anchors'].add(f"{r['root_id']}/t{t}")
                    c['offset']+=int(now==1 and after==0);c['other_positive']+=int(v[t+h,1,j]==1)
    for c in summary.values():
        for k in ('onset_roots','onset_anchors','positive_roots'):c[k]=sorted(c[k])
    result=dict(status='H24_EVENTS_COLLECTED_COVERAGE_REVIEW_REQUIRED',branches=len(rows),anchors=len(seen),source_roots=len({k[0] for k in seen}),
        missing_strata=cfg['missing_strata'],new_scenes=False,training=False,H24_model_evaluated=False,summary=summary,rows=rows)
    atomic_json(OUT/'analysis.json',result)
    lines=['# 状态分层H24事件采集结果','', '本轮完成物理/自然观测采集及标签复算，无训练，也未将旧H8模型直接用于H24。', '',
        f"{len(rows)}条分支、{len(seen)}个起点时刻、{result['source_roots']}个旧场景；各层缺失记录保留。",'',
        '| 事件 | H8形成：分支/场景 | H16形成：分支/场景 | H24形成：分支/场景 |','|---|---|---|---|']
    for event in EVENTS:
        v=[summary[f'all/H{h}/{event}'] for h in (8,16,24)]
        lines.append('| '+event+' | '+' | '.join(f"{c['onset']}/{len(c['onset_roots'])}" for c in v)+' |')
    lines+=['','形成仅计当前明确为0而终点为1；未知不作负例。窗口内发生后又消失的事件可能不计入终点形成，完整标签保留用于后续分析。',
        '各层可共享起点，分支共享前缀和场景，不作为独立场景样本。正例仍缺失时不能声称该类鲁棒性。',
        '新场景仅登记待验证seed，未生成、未确认独立性、未执行。', '', '[完整结果](../results/relational_event_h24/analysis.json)']
    (ROOT/'docs/relational_event_h24_audit.md').write_text('\n'.join(lines)+'\n')
    atomic_json(OUT/'validation.json',dict(status='PASS',raw_hashes_verified=True,contact_reconstruction=True,labels_recomputed=True,prefix_identity_verified=True,analysis_sha256=sha256(OUT/'analysis.json')))
    print(json.dumps({k:v for k,v in result.items() if k not in ('summary','rows','missing_strata')}),flush=True)
if __name__=='__main__':main()
