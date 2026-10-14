#!/usr/bin/env python3
"""Recompute metrics from saved predictions; no fitting or simulator."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.relational_dynamics_pilot import sha,score,pair_score,normalization
OUT=ROOT/'results/relational_dynamics_pilot'

def main():
    runtime=json.loads((OUT/'runtime.json').read_text());assert runtime['status']=='COMPLETE' and runtime['completed_fits']==6
    cfg=json.loads((ROOT/'configs/relational_dynamics_pilot_protocol.json').read_text())
    for path,h in cfg['frozen_files'].items():assert sha(ROOT/path)==h,path
    for path,h in json.loads((OUT/'artifact_manifest.json').read_text()).items():assert sha(ROOT/path)==h,path
    with np.load(OUT/'dataset.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    meta=json.loads((OUT/'samples.json').read_text());result=json.loads((OUT/'evaluation.json').read_text())
    assert result['protocol_sha256']==sha(ROOT/'configs/relational_dynamics_pilot_protocol.json')
    stats=normalization(d['x'],d['actions'],d['delta'],d['fit'],d['weights'])
    with np.load(OUT/'normalization.npz',allow_pickle=False) as f:
        for key,v in stats.items():np.testing.assert_array_equal(v,f[key])
    checks=0
    for name,metrics in result['evaluations'].items():
        if name=='unchanged':delta=np.zeros_like(d['delta']);prob=np.where(d['current']>=0,d['current'],np.nan).astype(float)
        else:
            with np.load(OUT/f'{name}_predictions.npz',allow_pickle=False) as f:delta=f['delta'];prob=f['probability']
        for s,mask in [('fit',d['fit']),('development',~d['fit'])]:
            assert score(delta,prob,d,meta,mask)==metrics[s];checks+=1
        assert pair_score(delta,d,meta,~d['fit'])==metrics['development_pairs'];checks+=1
        if name.startswith('action_seed'):
            with np.load(OUT/f'{name}_root_action_permuted.npz',allow_pickle=False) as f:
                assert pair_score(f['delta'],d,meta,~d['fit'])==metrics['permuted_development_pairs'];checks+=1
    ev=result['evaluations'];pairs=ev['unchanged']['development_pairs']
    lines=['# 动作条件关系动力学：小型特权状态试验','',
        '日期：2026-10-07。**'+result['verdict']+'**。', '',
        '六次固定拟合已完成（两种模型×三个种子）。本次仅使用已有物理记录，无新仿真、视觉模型或控制器执行。', '',
        '## 开发集主要结果','',
        '| 方法 | 目标—盘位移误差（mm，root宏平均） | 目标—夹爪位移误差（mm，root宏平均） | 有效候选差分误差（mm） |',
        '|---|---:|---:|---:|']
    for name in ['unchanged']+[f'{f}_seed{s}' for s in cfg['seeds'] for f in ['no_action','action']]+['no_action_ensemble','action_ensemble']:
        e=ev[name];v=e['development'];p=e['development_pairs']['material_mean_error_mm']
        lines.append(f"| {name} | {v['root_macro_target_plate_mm']:.3f} | {v['root_macro_target_gripper_mm']:.3f} | {p:.3f} |")
    lines+=['',f"候选差分开发集共{pairs['n_pairs']}对、{pairs['n_roots']}个root；真实目标—盘差异>1mm的有效分层只有{pairs['material_pairs']}对、{pairs['material_roots']}个root。差分分数不构成独立泛化证明。",'',
        '冻结继续线：至少2/3种子同时满足两项连续关系误差低于无动作与不变基线，且有效候选差分误差改善至少10%。该线是有界pilot投入规则，不是统计显著性结论。', '',
        '```json',json.dumps(result['seed_checks'],ensure_ascii=False,indent=2),'```','',
        '## 事件及变化检查','',
        '普通accuracy不作主要成绩。完整结果逐类列出接触/事件的precision、recall、Brier、发生/消失计数及未知mask；当前状态未知时，不变基线拒绝预测，因此其二元分母可能不同。',
        '连续预测另外报告>1mm变化与未变化分层、每root误差、另一只碗误差。部分稀有事件只有少数正例，不能只展示总体均值。', '',
        '| 模型 | 目标抓持发生 TP / FP / FN | 目标释放发生 TP / FP / FN |',
        '|---|---|---|']
    for name in ['unchanged','no_action_ensemble','action_ensemble']:
        b=ev[name]['development']['binary'];a=b['target/attachment']['onset'];r=b['target/release']['onset']
        lines.append(f"| {name} | {a['tp']} / {a['fp']} / {a['fn']} | {r['tp']} / {r['fp']} / {r['fn']} |")
    lines+=['','## 解释边界与下一步','',
        '输入是丰富的真实物理状态（包含qpos/qvel、接触和已知目标身份）、11采样点历史、因果接触记忆；不是从RGB估计的最小状态。预测的是已规划H8块内后果，不提前使用块后C0反馈动作。',
        '30/10旧起点划分不含新的盲测；仅三个候选，另一个碗的抓持无正例。该结果不能证明跨物体、跨任务或自然观测泛化。',
        '即使本轮为正，也只支持进一步检验物理状态压缩、覆盖与自然感知；不能直接升级为候选选择或闭环收益。若未通过，先根据拟合/开发差距、动作置换和同root差分分析瓶颈，不换视觉backbone来解释。', '',
        f'验证：13项输入/标签测试在冻结前通过；本次重新计算{checks}组指标并逐项完全一致，fit-only归一化复算一致。未重跑全仓库测试。', '',
        '[冻结协议](relational_dynamics_pilot_protocol.md) · [完整指标](../results/relational_dynamics_pilot/evaluation.json) · [运行状态](../results/relational_dynamics_pilot/runtime.json)']
    (ROOT/'docs/relational_dynamics_pilot_audit.md').write_text('\n'.join(lines)+'\n')
    (OUT/'validation.json').write_text(json.dumps(dict(status='PASS',recomputed_metric_groups=checks,
        frozen_inputs_verified=True,artifact_hashes_verified=True,normalization_fit_only_recomputed=True,
        verifier_sha256=sha(Path(__file__)),report_sha256=sha(ROOT/'docs/relational_dynamics_pilot_audit.md')),indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],metric_checks=checks),ensure_ascii=False))

if __name__=='__main__':main()
