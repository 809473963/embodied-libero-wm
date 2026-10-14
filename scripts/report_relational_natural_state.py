#!/usr/bin/env python3
"""Saved-prediction verification and bounded natural-state pilot report."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.candidate_utility_oracle import sha256,atomic_json
from src.probes.relational_dynamics_pilot import score
from scripts.evaluate_relational_autorun import paired
from scripts.run_relational_natural_state import stats
OUT=ROOT/'results/relational_natural_state';SRC=ROOT/'results/relational_autorun'

def main():
    cfg=json.loads((ROOT/'configs/relational_natural_state_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    with np.load(OUT/'dataset.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    with np.load(OUT/'normalization.npz',allow_pickle=False) as f:
        for k,v in stats(d).items():np.testing.assert_array_equal(f[k],v)
    inv=json.loads((OUT/'inventory.json').read_text());roots=np.array(inv['roots']);dev=~d['fit']
    with np.load(SRC/'models/dataset.npz',allow_pickle=False) as f:dyn={k:f[k] for k in f.files}
    meta=json.loads((SRC/'models/samples.json').read_text());mask=~dyn['fit']
    ev=json.loads((OUT/'evaluation.json').read_text())['models'];checks=0;hashes={}
    for name,m in ev.items():
        for suffix in ('prediction','bridge'):
            path=OUT/f'{name}_{suffix}.npz';hashes[path.name]=sha256(path)
        with np.load(OUT/f'{name}_prediction.npz',allow_pickle=False) as f:pred=f['state']
        err=np.linalg.norm((pred[:,:12]-d['target'][:,:12]).reshape(-1,4,3),axis=2)*1000
        scene={r:err[(roots==r)&dev].mean(0).tolist() for r in sorted(set(roots[dev]))}
        assert scene==m['perception_per_scene_mm'];np.testing.assert_array_equal(np.mean(list(scene.values()),0),m['perception_scene_macro_mm']);checks+=1
        with np.load(OUT/f'{name}_bridge.npz',allow_pickle=False) as f:
            np.testing.assert_array_equal(f['endpoint_delta'],f['delta']+pred[d['history'][:,-1],:12]-dyn['x'][:,-1,104:116])
            assert score(f['delta'],f['probability'],dyn,meta,mask)==m['delta'];checks+=1
            assert score(f['endpoint_delta'],f['probability'],dyn,meta,mask)==m['absolute_endpoint'];checks+=1
            assert paired(f['delta'],dyn,meta,mask)==m['pairs'];checks+=1
    oracle=json.loads((ROOT/'results/relational_minimal_state/geometry_history/evaluation.json').read_text())['models']['action_ensemble']
    # Constant current-state extrapolation has the same endpoint error as true delta under oracle current state.
    unchanged=json.loads((SRC/'evaluation.json').read_text())['models']['unchanged']
    def values(m):return [m['absolute_endpoint']['root_macro_target_plate_mm'],m['absolute_endpoint']['root_macro_target_gripper_mm'],m['pairs']['root_macro_error_mm']]
    seed_checks=[]
    for seed in (11,23,47):
        a=np.array(values(ev[f'rgb_proprio_seed{seed}']));n=np.array(values(ev[f'proprio_seed{seed}']))
        b=np.array([unchanged['new_development']['root_macro_target_plate_mm'],unchanged['new_development']['root_macro_target_gripper_mm'],unchanged['new_pairs']['root_macro_error_mm']])
        seed_checks.append(dict(seed=seed,joint_pass=bool(np.all(a<.9*n) and np.all(a<b))))
    passed=sum(s['joint_pass'] for s in seed_checks)>=2
    verdict='NATURAL_STATE_PILOT_BRIDGE_SUPPORTED' if passed else 'NATURAL_STATE_PILOT_BRIDGE_NOT_ESTABLISHED'
    lines=['# 自然观测→关系状态→冻结动力学：首轮试验','', '日期：2026-10-08。**'+verdict+'**。','',
      '目的：把感知误差实际接入已学到的动力学，而不是只看视觉回归分数。该结果仅针对固定小CNN和当前数据，不证明视觉信息存在或缺失的普遍结论。','',
      f"数据：旧{inv['fit_scenes']}场景、{inv['fit_frames']}条去重观测训练；新{inv['development_scenes']}场景、{inv['development_frames']}条观测开发评估。相同场景的共享分支前缀去重；帧数不是独立场景数。无确认集访问。",'',
      '图像模型输入初始自然RGB与当前自然RGB（固定缩至64×64）、当前12维合法proprio及上一7维动作；对照只用proprio与上一动作。输出两碗相对盘/夹爪12维位置与夹爪6维速度。每个时刻只用该时刻及之前输入，训练监督可以来自旧轨迹后段；未来图像不进入当前预测。', '',
      '每种输入3种子×40轮，无最佳epoch选择。预测状态按11步组装，替换26维geometry_history模型的18个状态量，上一动作及有效位不变；冻结3个动力学模型并平均其预测，不重新拟合。', '',
      '| 输入 | 当前目标—盘误差 mm（帧集场景宏平均） | 未来目标—盘误差 mm | 未来目标—夹爪误差 mm | 候选差分 mm |','|---|---:|---:|---:|---:|']
    o=oracle['new_development'];lines.append(f"| 真实关系状态参考 | 0 | {o['root_macro_target_plate_mm']:.3f} | {o['root_macro_target_gripper_mm']:.3f} | {oracle['new_pairs']['root_macro_error_mm']:.3f} |")
    for name in ('proprio_ensemble','rgb_proprio_ensemble'):
        m=ev[name];lines.append(f"| {name} | {m['perception_scene_macro_mm'][0]:.3f} | "+' | '.join(f'{v:.3f}' for v in values(m))+' |')
    lines += ['', '未来误差包含起点位置估计误差：预测当前关系＋预测变化，对比真实当前关系＋真实变化；只比较Δ会隐藏感知偏差。当前感知指标来自所有记录帧，未来指标来自原132条分支，二者样本分布不同。', '',
      '## 固定继续线','', '至少2/3视觉种子在未来目标—盘、目标—夹爪与候选差分三项均较对应proprio模型改善10%，且低于真实当前状态的不变基线。该线衡量是否值得继续此自然输入桥，不是控制安全、语言理解或跨任务泛化认证。', '', '`'+json.dumps(seed_checks)+'`。', '',
      '## 事件与误动','', '| 输入 | 近静止分支绝对未来误差 mm | On发生 TP/FP/FN | release发生 TP/FP/FN |', '|---|---:|---:|---:|']
    for name in ('proprio_ensemble','rgb_proprio_ensemble'):
        m=ev[name]['absolute_endpoint'];b=m['binary'];counts=lambda k:'/'.join(str(b['target/'+k]['onset'][s]) for s in ('tp','fp','fn'))
        lines.append(f"| {name} | {m['unchanged_target_plate_mm']:.3f} | {counts('on')} | {counts('release')} |")
    lines += ['', '释放标签发生包含稳定窗口形成；本轮不新增真正释放场景。即使平均连续误差通过，事件误报仍不能被当作可靠动作价值。', '',
      '## 解释范围与后续','', '两个碗按固定物理身份槽位监督，初始图像用于保留场景参考；未输入语言，也没有独立证明从相似碗视觉恢复持久身份。所有标签是训练时的特权监督；推理特征不含真实几何/接触/事件。夹爪速度也由自然输入预测，没有偷偷保留真实速度。', '',
      '该初始+当前帧CNN不具备可学习的完整时序记忆；下游11步动力学历史不等于感知层已解决遮挡身份。因此失败可能来自容量、观测、身份或数据覆盖，不能重启“frozen backbone无信息”的结论。', '',
      ('通过继续线后，下一步应在旧训练场景内部评估误差分层与拒绝干预规则，再用独立注册数据确认；本轮不会自动接controller。' if passed else '未通过时保留该负结果，下一轮应先分解当前几何/身份误差与动力学分布偏移；用旧场景内部的输入对照诊断，不以新开发集反复调模型，也不直接增加仿真或打开确认集。'), '',
      f'验证：从保存预测复算{checks}组指标、精确复算fit-only归一化及绝对未来关系公式；协议冻结文件哈希一致。', '',
      '复现复核：`env -u PYTHONPATH .venv/bin/python scripts/report_relational_natural_state.py`。']
    report=ROOT/'docs/relational_natural_state_audit.md';report.write_text('\n'.join(lines)+'\n')
    atomic_json(OUT/'validation.json',dict(status='PASS',groups_recomputed=checks,prediction_sha256=hashes,report_sha256=sha256(report)))
    atomic_json(OUT/'decision.json',dict(verdict=verdict,seed_checks=seed_checks,confirmation_opened=False,controller_executed=False))
    print(verdict,flush=True)
if __name__=='__main__':main()
