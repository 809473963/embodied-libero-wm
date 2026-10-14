#!/usr/bin/env python3
"""Oracle substitutions and fixed geometry-noise sensitivity; never fit a model."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.probes.candidate_utility_oracle import atomic_json,sha256
from src.probes.relational_dynamics_pilot import score
from scripts.evaluate_relational_autorun import paired
from scripts.run_relational_natural_state import evaluate
SOURCE=ROOT/'results/relational_natural_state'
OUT=SOURCE/'diagnosis'

def main():
    cfg=json.loads((ROOT/'configs/relational_natural_diagnosis_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    assert json.loads((SOURCE/'runtime.json').read_text())['status']=='COMPLETE'
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    with np.load(SOURCE/'dataset.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    roots=np.array(json.loads((SOURCE/'inventory.json').read_text())['roots'])
    with np.load(SOURCE/'rgb_proprio_ensemble_prediction.npz',allow_pickle=False) as f:p=f['state']
    variants=dict(oracle_all=d['target'].copy(),oracle_geometry=p.copy(),oracle_velocity=p.copy())
    variants['oracle_geometry'][:,:12]=d['target'][:,:12]
    variants['oracle_velocity'][:,12:]=d['target'][:,12:]
    results={}
    for name,pred in variants.items():results[name]=evaluate('diagnosis_'+name,pred,d,roots)
    with np.load(SOURCE/'diagnosis_oracle_all_bridge.npz',allow_pickle=False) as f:oracle=f['delta']
    with np.load(ROOT/'results/relational_minimal_state/geometry_history/action_ensemble_predictions.npz',allow_pickle=False) as f:
        np.testing.assert_allclose(oracle,f['delta'],rtol=1e-5,atol=1e-6)
    # Same normalized random field is scaled across noise levels; shared observations share perturbations.
    for seed in (101,202,303):
        rng=np.random.default_rng(seed);noise=rng.normal(size=(len(p),12)).astype(np.float32)
        for mm in (1,2,5,10):
            pred=d['target'].copy();pred[:,:12]+=noise*(mm/1000)
            name=f'noise_{mm}mm_seed{seed}';results[name]=evaluate('diagnosis_'+name,pred,d,roots)
    ev=json.loads((SOURCE/'evaluation.json').read_text())['models'];results['natural']=ev['rgb_proprio_ensemble']
    def vals(m):return [m['absolute_endpoint']['root_macro_target_plate_mm'],m['absolute_endpoint']['root_macro_target_gripper_mm'],m['pairs']['root_macro_error_mm']]
    lines=['# 自然状态桥的误差来源与噪声敏感性','', '日期：2026-10-08。固定已训练模型的事后机制诊断；不训练、不选阈值、不执行控制。','',
      '| 输入状态 | 绝对未来目标—盘 mm | 绝对未来目标—夹爪 mm | 候选差分 mm |','|---|---:|---:|---:|']
    for name in ('oracle_all','natural','oracle_geometry','oracle_velocity'):
        lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in vals(results[name]))+' |')
    lines += ['', 'oracle_geometry将12维关系替换为真实值，仍预测速度；oracle_velocity只替换6维夹爪速度。它们是诊断上限，不是可部署成绩。全部真实输入已与原geometry_history预测核对一致。', '',
      '## 人工几何噪声','', '| 每坐标噪声标准差 | 未来目标—盘 mm | 未来目标—夹爪 mm | 候选差分 mm |','|---|---:|---:|---:|']
    noise_summary={}
    for mm in (1,2,5,10):
        arr=np.array([vals(results[f'noise_{mm}mm_seed{s}']) for s in (101,202,303)])
        noise_summary[mm]=dict(mean=arr.mean(0).tolist(),min=arr.min(0).tolist(),max=arr.max(0).tolist())
        lines.append(f'| {mm} mm | '+' | '.join(f'{v:.3f}' for v in arr.mean(0))+' |')
    gains={name:(np.array(vals(results['natural']))-np.array(vals(results[name]))).tolist() for name in ('oracle_geometry','oracle_velocity')}
    lines += ['', '扰动为每坐标独立高斯噪声；同一去重观测共用扰动，三个固定随机场跨幅度配对。它不模拟有偏、遮挡、身份交换或时间相关的真实感知错误；不能从表格推导部署允许误差或总体置信区间。', '',
      f"仅修复几何带来的三项误差减少为 {gains['oracle_geometry']} mm；仅修复速度为 {gains['oracle_velocity']} mm。负值表示该项反而变差，不能强行解释为单一误差来源。", '',
      '后续投入应依据实际误差来源：几何修复有效则优先定位/身份/尺度；速度修复有效则优先使用合法机器人历史估计运动；两者均无明显修复则检查自然误差与动力学训练分布的交互。不能通过只换成真实起点却继续称自然模型来掩盖误差。', '',
      '此诊断保持confirmation封存，未使用任何结果选择新checkpoint或训练数据。']
    report=ROOT/'docs/relational_natural_state_diagnosis.md';report.write_text('\n'.join(lines)+'\n')
    atomic_json(OUT/'evaluation.json',dict(models=results,noise_summary=noise_summary,oracle_repair_improvement_mm=gains,confirmation_opened=False,training=False))
    # Independent recomputation from saved bridges; original natural metrics verified by pilot reporter.
    with np.load(ROOT/'results/relational_autorun/models/dataset.npz',allow_pickle=False) as f:dyn={k:f[k] for k in f.files}
    meta=json.loads((ROOT/'results/relational_autorun/models/samples.json').read_text());mask=~dyn['fit'];checks=0
    for name,m in results.items():
        if name=='natural':continue
        with np.load(SOURCE/f'diagnosis_{name}_bridge.npz',allow_pickle=False) as f:
            assert score(f['delta'],f['probability'],dyn,meta,mask)==m['delta']
            assert score(f['endpoint_delta'],f['probability'],dyn,meta,mask)==m['absolute_endpoint']
            assert paired(f['delta'],dyn,meta,mask)==m['pairs'];checks+=3
    atomic_json(OUT/'validation.json',dict(status='PASS',metric_groups_recomputed=checks,oracle_equivalence=True,report_sha256=sha256(report),source_prediction_sha256=sha256(SOURCE/'rgb_proprio_ensemble_prediction.npz')))
    print(json.dumps(dict(status='COMPLETE',groups_recomputed=checks,gains=gains)),flush=True)
if __name__=='__main__':main()
