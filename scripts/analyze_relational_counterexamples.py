#!/usr/bin/env python3
"""Verify prospective branch data and test already-frozen models; no refitting."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.probes.relational_counterexamples import candidates,validate,NAMES
from src.probes.relational_dynamics_pilot import features,causal_window,relation,binary_state,Dynamics,score,sha
from src.probes.relational_event_labels import labels
from src.probes.candidate_utility_oracle import atomic_json
OUT=ROOT/'results/relational_counterexamples'

def main():
    cfg=json.loads((ROOT/'configs/relational_counterexamples_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha(ROOT/p)==h,p
    m=json.loads((OUT/'manifest.json').read_text());assert m['complete'] and len(m['records'])==360
    labelcfg=json.loads((ROOT/'configs/relational_event_labels_v1.json').read_text())['parameters']
    xx=[];aa=[];yy=[];bb=[];cc=[];meta=[];cases=[];seen={};plans={}
    for r in m['records']:
        assert sha(ROOT/r['dense']['arrays'])==r['dense']['sha256']
        with np.load(ROOT/r['dense']['arrays'],allow_pickle=False) as f:a={k:f[k] for k in f.files}
        t=r['anchor'];validate(a,r['dense']['geometry'],t)
        v=np.stack([labels(a,labelcfg,j)['values'] for j in (0,1)],axis=1)
        with np.load(OUT/'labels'/f"{r['name']}.npz",allow_pickle=False) as f:np.testing.assert_array_equal(v,f['values'])
        assert sha(ROOT/r['source'])==r['source_sha256']
        with np.load(ROOT/r['source'],allow_pickle=False) as f:oldactions=f['actions']
        expected=candidates(oldactions[t:t+8],cfg['translation_action_offset'])[r['variant']]
        np.testing.assert_array_equal(a['actions'][t:],expected)
        feat,_=features(a,v);window=causal_window(feat,t);key=(r['root_id'],t)
        if key in seen:np.testing.assert_array_equal(seen[key],window)
        else:seen[key]=window
        plans.setdefault(key,set()).add(expected.tobytes())
        rel=relation(a);binary=binary_state(a,v)
        xx.append(window);aa.append(expected);yy.append(rel[t+8]-rel[t]);bb.append(binary[t+8]);cc.append(binary[t])
        meta.append(dict(root=r['root_id'],anchor=t,variant=r['variant'],split='development' if r['root_id'].endswith('34') else 'fit_exposed'))
        cases.append(dict(name=r['name'],other_attachment_positive=bool((v[:,1,0]==1).any()),target_endpoint_events=v[t+8,0].tolist(),target_current_events=v[t,0].tolist()))
    data=dict(x=np.array(xx,np.float32),actions=np.array(aa,np.float32),delta=np.array(yy,np.float32),binary=np.array(bb,np.int8),current=np.array(cc,np.int8))
    np.savez_compressed(OUT/'model_evaluation_data.npz',**data);atomic_json(OUT/'samples.json',meta)
    torch.set_num_threads(2)
    with np.load(ROOT/'results/relational_dynamics_pilot/normalization.npz',allow_pickle=False) as f:st={k:f[k] for k in f.files}
    x=torch.tensor((data['x']-st['x_mean'])/st['x_std'],dtype=torch.float32)
    act=torch.tensor((data['actions']-st['action_mean'])/st['action_std'],dtype=torch.float32)
    predictions={'unchanged':(np.zeros_like(data['delta']),np.where(data['current']>=0,data['current'],np.nan))}
    for family in ('no_action','action'):
        outputs=[]
        for seed in (11,23,47):
            ck=torch.load(ROOT/f'results/relational_dynamics_pilot/{family}_seed{seed}.pt',map_location='cpu',weights_only=True)
            model=Dynamics(ck['feature_dimension']);model.load_state_dict(ck['state_dict']);model.eval()
            with torch.no_grad():outputs.append(model(x,torch.zeros_like(act) if family=='no_action' else act))
        delta=np.mean([o[:,:12].numpy()*st['delta_scale'] for o in outputs],axis=0)
        prob=np.mean([torch.sigmoid(o[:,12:]).numpy() for o in outputs],axis=0)
        predictions[family]=(delta,prob)
        np.savez_compressed(OUT/f'frozen_{family}_predictions.npz',delta=delta,probability=prob)
    groups={}
    for split in ('fit_exposed','development'):
        mask=np.array([r['split']==split for r in meta]);groups[split]={}
        for name,(delta,prob) in predictions.items():
            metrics=score(delta,prob,data,meta,mask);pairs=[]
            for key in seen:
                ix={r['variant']:i for i,r in enumerate(meta) if (r['root'],r['anchor'])==key and mask[i]}
                if not ix:continue
                for cid in range(1,9):
                    truth=data['delta'][ix[cid],:3]-data['delta'][ix[0],:3];pred=delta[ix[cid],:3]-delta[ix[0],:3]
                    pairs.append(dict(root=key[0],anchor=key[1],variant=NAMES[cid],truth_mm=float(np.linalg.norm(truth)*1000),error_mm=float(np.linalg.norm(pred-truth)*1000),predicted_mm=float(np.linalg.norm(pred)*1000)))
            metrics['pairs']=pairs
            metrics['pair_error_all_mm']=float(np.mean([p['error_mm'] for p in pairs]))
            material=[p for p in pairs if p['truth_mm']>1];small=[p for p in pairs if p['truth_mm']<=1]
            metrics['material_pairs']=len(material);metrics['material_roots']=len({p['root'] for p in material})
            metrics['material_error_mm']=float(np.mean([p['error_mm'] for p in material])) if material else None
            metrics['small_difference_error_mm']=float(np.mean([p['error_mm'] for p in small])) if small else None
            groups[split][name]=metrics
    result=dict(status='COLLECTED_AND_FROZEN_MODELS_EVALUATED',new_training=False,new_scene=False,
        unique_source_roots=20,anchor_states=40,branches=360,distinct_action_plans=sum(len(v) for v in plans.values()),
        other_attachment_positive_branches=sum(c['other_attachment_positive'] for c in cases),groups=groups,cases=cases,
        note='Prospective intervention consequences at exposed scenes; not a blind scene test. Counterexample categories are observed, not guaranteed by recipes.')
    atomic_json(OUT/'analysis.json',result)
    lines=['# 动作后果与反例采集结果','', '20个旧场景、40个固定时刻、360条H8分支。同步自然RGB/proprio与物理记录已核验。',
        '', '仅评估此前冻结的模型，没有重训。init31属于旧拟合集、init34属于已暴露开发集；本轮仍不是独立新场景验证。','',
        '| 开发分支方法 | 目标—盘误差 mm | 全部候选差分误差 mm | >1mm候选差分误差 mm |', '|---|---:|---:|---:|']
    for name,g in groups['development'].items():lines.append(f"| {name} | {g['root_macro_target_plate_mm']:.3f} | {g['pair_error_all_mm']:.3f} | {g['material_error_mm']} |")
    lines+=['',f"有>1mm候选差异的开发对：{groups['development']['unchanged']['material_pairs']}，覆盖{groups['development']['unchanged']['material_roots']}个原始场景。另一只碗出现持续抓持证据的分支：{result['other_attachment_positive_branches']}。", '',
        '下一步先审查变化/不变、事件发生/消失及分物体结果，确定是否仍需补充失败模式。不能用更多相关分支代替独立scene确认。',
        '', '[完整指标与逐分支记录](../results/relational_counterexamples/analysis.json)']
    (ROOT/'docs/relational_counterexamples_audit.md').write_text('\n'.join(lines)+'\n')
    atomic_json(OUT/'validation.json',dict(status='PASS',all_source_hashes=True,all_contact_reconstructed=True,labels_recomputed=True,root_histories_equal=True,analysis_sha256=sha(OUT/'analysis.json')))
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','cases')}),flush=True)

if __name__=='__main__':main()
