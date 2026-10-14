#!/usr/bin/env python3
"""Verify new raw records and apply frozen event v1 to all forty old roots."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.analyze_dense_relation_preflight import validate_contacts
from src.probes.relational_event_labels import EVENTS,labels


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=ROOT/'results/relational_coverage_expansion'
    protocol=ROOT/'configs/relational_coverage_expansion_protocol.json';cfg=json.loads(protocol.read_text())
    for p,h in cfg['frozen_files'].items():assert sha(ROOT/p)==h,p
    runtime=json.loads((out/'runtime.json').read_text());assert runtime['status']=='COMPLETE'
    new=json.loads((out/'manifest.json').read_text());assert new['protocol_sha256']==sha(protocol)
    assert new['complete'] and len(new['branches'])==109 and new['environment_steps']==32700
    old=json.loads((ROOT/'results/dense_relation_preflight/manifest.json').read_text())
    label_cfg=json.loads((ROOT/'configs/relational_event_labels_v1.json').read_text())
    (out/'labels').mkdir(exist_ok=True)
    rows=[];seen={};prefix={};unique_samples=set()
    for r in old['branches']+new['branches']:
        assert all(r['identity'].values())
        source=ROOT/r['dense']['arrays'];assert sha(source)==r['dense']['sha256']
        with np.load(source,allow_pickle=False) as f:a={k:f[k] for k in f.files}
        assert np.array_equal(a['step'],np.arange(301)) and a['actions'].shape==(300,7)
        assert all(np.isfinite(v).all() for v in a.values())
        assert a['contact_offsets'].shape==(302,) and a['contact_offsets'][0]==0
        assert (np.diff(a['contact_offsets'])>=0).all()
        for key in ('contact_geom_pairs','contact_wrenches','contact_frames','contact_distances'):
            assert len(a[key])==a['contact_offsets'][-1]
        validate_contacts(a,r['dense']['geometry'])
        assert hashlib.sha256(np.ascontiguousarray(a['actions']).tobytes()).hexdigest()==r['applied_actions_sha256']
        success=np.flatnonzero(a['success'])
        assert (int(success[0]) if len(success) else None)==r['first_success_step']
        root=r['root']['root_id'];cid=r['candidate_id'];p=r['root']['prefix_steps']
        with np.load(ROOT/'results/candidate_utility_oracle/arrays'/f'{root}.npz',allow_pickle=False) as f:
            assert np.array_equal(a['actions'][:p],f['prefix'])
            assert np.array_equal(a['actions'][p:p+8],f['candidates'][cid])
        key=(root,cid)
        if key in seen:
            with np.load(ROOT/seen[key],allow_pickle=False) as previous:
                assert set(previous.files)==set(a) and all(np.array_equal(previous[k],a[k]) for k in a)
            continue
        seen[key]=r['dense']['arrays']
        values=np.stack([labels(a,label_cfg['parameters'],i)['values'] for i in (0,1)],axis=1)
        if root in prefix:np.testing.assert_array_equal(prefix[root],values[:p+1])
        else:prefix[root]=values[:p+1].copy()
        for t in range(301):unique_samples.add((root,'prefix' if t<=p else cid,t))
        path=out/'labels'/f'{root}_c{cid:02d}.npz';np.savez_compressed(path,values=values,step=a['step'])
        rows.append(dict(root_id=root,task_id=r['root']['task_id'],candidate_id=cid,root_step=p,
            source=r['dense']['arrays'],source_sha256=r['dense']['sha256'],labels=str(path.relative_to(ROOT)),
            labels_sha256=sha(path),
            events={name:dict(positive_samples=int((values[:,0,j]==1).sum()),
                unknown_samples=int((values[:,0,j]<0).sum()),
                fully_post_root_positive_samples=int((values[p+10:,0,j]==1).sum()),
                other_positive_samples=int((values[:,1,j]==1).sum())) for j,name in enumerate(EVENTS)}))
    assert len(seen)==120 and len(prefix)==40
    coverage={name:dict(positive_branches=sum(r['events'][name]['positive_samples']>0 for r in rows),
        positive_roots=sorted({r['root_id'] for r in rows if r['events'][name]['positive_samples']}),
        positive_tasks=sorted({r['task_id'] for r in rows if r['events'][name]['positive_samples']}),
        other_positive_roots=sorted({r['root_id'] for r in rows if r['events'][name]['other_positive_samples']})) for name in EVENTS}
    result=dict(status='COVERAGE_MEASURED_REQUIRES_REVIEW_BEFORE_TRAINING',protocol_sha256=sha(protocol),
        unique_roots=40,unique_branches=120,new_runs=109,new_env_steps=32700,
        samples_after_shared_prefix_dedup=len(unique_samples),independent_sample_count_claimed=False,
        coverage=coverage,records=rows,training_run=False,natural_rgb_collected=False,
        new_scene_generalization_test=False,success_used_for_labels=False)
    (out/'coverage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 关系事件覆盖扩展结果','', '日期：2026-10-07。40个旧起点、120条唯一分支；其中36个起点本轮补录。',
      '', '**仅完成物理记录与操作性标签覆盖检查；未训练，也不是新场景泛化验证。**', '',
      '| 标签 | 正例分支 | 正例起点 | 正例任务 | 另一只碗正例起点 |','|---|---:|---:|---:|---:|']
    for name,c in coverage.items():lines.append(f"| {name} | {c['positive_branches']} | {len(c['positive_roots'])} | {len(c['positive_tasks'])} | {len(c['other_positive_roots'])} |")
    lines+=['','新增109次完整运行、32,700控制步；所有旧分支身份、接触重建和两批A0重复检查通过。',
      '正例分支共享起点和prefix，不是独立成功试验。所有阈值沿用v1，没有按success调整。',
      '', '下一步先审查按任务与物体的覆盖、事件反例及训练/开发分组，再冻结最小privileged动力学试验。',
      '自然观测训练仍缺同步RGB；支撑承重、表面净空和未来稳定性仍未获得真值证明。',
      '', '[逐分支结果与来源哈希](../results/relational_coverage_expansion/coverage.json)']
    (ROOT/'docs/relational_coverage_expansion_audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
