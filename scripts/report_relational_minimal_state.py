#!/usr/bin/env python3
"""Recompute every ablation metric and report fixed resource-decision rules."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.probes.candidate_utility_oracle import atomic_json,sha256
from src.probes.relational_dynamics_pilot import normalization,score,prf
from src.probes.relational_minimal_state import project,REPRESENTATIONS
from scripts.evaluate_relational_autorun import paired
SOURCE=ROOT/'results/relational_autorun';OUT=ROOT/'results/relational_minimal_state'

def main():
    cfg=json.loads((ROOT/'configs/relational_minimal_state_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    with np.load(SOURCE/'models/dataset.npz',allow_pickle=False) as f:original={k:f[k] for k in f.files}
    meta=json.loads((SOURCE/'models/samples.json').read_text());schema=json.loads((SOURCE/'models/schema.json').read_text())
    reference=json.loads((SOURCE/'evaluation.json').read_text())['models']
    coverage=json.loads((SOURCE/'coverage.json').read_text())['rows']
    cov={(r['root'],r['anchor'],r['variant']):r for r in coverage}
    # Require the operational detached-attachment label at the root, not mere contact.
    held=np.array([not original['fit'][i] and cov[(m['root'],m['anchor'],m['variant'])]['current_events'][0][1]==1 for i,m in enumerate(meta)])
    keys=('root_macro_target_plate_mm','root_macro_target_gripper_mm')
    def errors(m):return [m['new_development'][k] for k in keys]+[m['new_pairs']['root_macro_error_mm']]
    def subset(prob):
        mask=held & (original['binary'][:,8]>=0)
        out=prf(original['binary'][mask,8]==1,prob[mask,8]>=.5)
        out['n']=int(mask.sum());out['roots']=len({meta[i]['root'] for i in np.flatnonzero(mask)})
        return out
    summary={};checks=0;digests={}
    lines=['# 最小关系状态消融：结果与下一步','', '日期：2026-10-08。本轮只复用已有数据，无新增仿真或确认集访问。','',
      '旧13场景228分支用于拟合；新9个可用开发场景132分支用于评估。这批开发布局已参与设计判断，不能称为独立确认。每组动作/无动作各3种子、固定80轮，共24次拟合，无最佳epoch或种子筛选。','',
      '| 输入 | 维度×历史 | 目标—盘 mm | 目标—夹爪 mm | 候选差分 mm | 静止误差 mm | 压缩继续线通过种子 |', '|---|---:|---:|---:|---:|---:|---:|']
    full=reference['action_ensemble']
    lines.append('| full_reference | 522×11 | '+' | '.join(f'{v:.3f}' for v in errors(full))+f" | {full['new_development']['unchanged_target_plate_mm']:.3f} | — |")
    for rep in REPRESENTATIONS:
        directory=OUT/rep;d=dict(original);d['x'],indices=project(original['x'],schema,rep)
        with np.load(directory/'normalization.npz',allow_pickle=False) as f:
            for k,v in normalization(d['x'],d['actions'],d['delta'],d['fit'],d['weights']).items():np.testing.assert_array_equal(f[k],v)
        result=json.loads((directory/'evaluation.json').read_text())['models']
        for name,m in result.items():
            if name=='unchanged':delta=np.zeros_like(d['delta']);prob=np.where(d['current']>=0,d['current'],np.nan)
            else:
                path=directory/f'{name}_predictions.npz';digests[str(path.relative_to(ROOT))]=sha256(path)
                with np.load(path,allow_pickle=False) as f:delta=f['delta'];prob=f['probability']
            for split,mask in [('old_fit',d['fit']),('new_development',~d['fit'])]:
                assert score(delta,prob,d,meta,mask)==m[split];checks+=1
            assert paired(delta,d,meta,~d['fit'])==m['new_pairs'];checks+=1
        seed_checks=[]
        for seed in (11,23,47):
            a=np.array(errors(result[f'action_seed{seed}']));n=np.array(errors(result[f'no_action_seed{seed}']));b=np.array(errors(result['unchanged']));f=np.array(errors(reference[f'action_seed{seed}']))
            signal=bool(np.all(a[:2]<np.minimum(n,b)[:2]) and a[2]<.9*min(n[2],b[2]))
            retention=bool(np.all(a<=1.1*f))
            seed_checks.append(dict(seed=seed,action_signal=signal,within_10_percent_full=retention,joint_pass=signal and retention))
        ens=result['action_ensemble'];v=ens['new_development'];passed=sum(s['joint_pass'] for s in seed_checks)
        with np.load(directory/'action_ensemble_predictions.npz',allow_pickle=False) as f:held_metrics=subset(f['probability'])
        summary[rep]=dict(shape=list(d['x'].shape),seed_checks=seed_checks,compression_supported=passed>=2,errors_mm=errors(ens),static_error_mm=v['unchanged_target_plate_mm'],held_release=held_metrics,
          on_onset=v['binary']['target/on']['onset'],release_onset=v['binary']['target/release']['onset'])
        lines.append(f"| {rep} | {d['x'].shape[-1]}×{d['x'].shape[1]} | "+' | '.join(f'{z:.3f}' for z in errors(ens))+f" | {v['unchanged_target_plate_mm']:.3f} | {passed}/3 |")
    supported=[rep for rep in REPRESENTATIONS if summary[rep]['compression_supported']]
    verdict='MINIMAL_STATE_RETENTION_SUPPORTED' if supported else 'MINIMAL_STATE_RETENTION_NOT_ESTABLISHED'
    lines += ['',f'**{verdict}**。通过既定资源继续线：'+(', '.join(supported) if supported else '无')+'。', '',
      '继续线：至少2/3种子同时保持原动作优势，并且三项连续/差分误差均不超过同种子完整状态模型的110%。这是有限投入规则，不是统计等价性证明；不因本次结果调整门槛。', '',
      '## 事件可靠性单独报告','', '| 输入 | On发生 TP/FP/FN | release标签 TP/FP/FN | 真正持有起点 release TP/FP/FN |', '|---|---:|---:|---:|']
    for rep,s in summary.items():
        def counts(v):return f"{v['tp']}/{v['fp']}/{v['fn']}"
        lines.append(f"| {rep} | {counts(s['on_onset'])} | {counts(s['release_onset'])} | {counts(s['held_release'])} |")
    lines += ['', '真正持有起点按v1 detached_attachment成立定义；release正例仍只有6条、3场景。同起点多动作不是独立证据。标签未知不算负例；完整release标签还包括已释放后的稳定窗口形成。', '',
      '## 输入解释与限制','',
      'geometry_history包含两碗相对盘/夹爪的位置12维、夹爪速度6维、上一动作7维及有效位1维。contact_history增加两碗的任意夹爪接触、双侧指垫接触与外部接触8维。memory_history再加因果高度参考、参考已知位及曾有detached证据6维。memory_current只使用最后一步40维，保留因果记忆；不是无历史信息输入。', '',
      '所有压缩组都不直接输入官方On、当前事件标签、全场景绝对pose、qpos/qvel或全接触矩阵。记忆中的seen_detached仍由特权事件计算；它的自然观测可恢复性尚未验证。接触历史和几何历史不等于视觉感知已经实现。', '',
      '完整模型保留原结果，本轮不重复训练。每组GRU宽度相同，但输入维度改变会改变参数量；差异可能混合信息、容量和优化效应，不能据一次消融判定某变量物理上不必要。四组共用相同数据和种子，无确认集选择。', '', '## 后续决策','']
    if supported:
        lines += ['已找到保留连续动作信号的紧凑输入候选。下一项应检查这些变量的自然观测恢复能力：固定同场景分组，用已有RGB/proprio/history预测必要几何/接触，比较真实状态与预测状态输入下的后果误差。先分解感知误差，不把本轮通过解释为事件安全或控制收益。']
    else:
        lines += ['当前压缩未保留完整模型的全部优势。下一项应在旧数据内部按场景分组检查相对姿态、速度/机器人状态等被移除信息；不立即投入自然视觉训练，不利用确认集筛选输入。不能据此认定关系表示不可行。']
    lines += ['', '静止误动和On误报未作为这次连续信号继续线的替代指标；它们必须在任何动作选择实验前另行解决。没有新增奖励、控制器、在线override或确认集实验。', '',
      f'验证：{checks}组指标从保存预测逐一复算，4组fit-only归一化精确一致，协议冻结源哈希一致。', '',
      '复算：`env -u PYTHONPATH .venv/bin/python scripts/report_relational_minimal_state.py`。机器证据：[review.json](../results/relational_minimal_state/review.json)。']
    report=ROOT/'docs/relational_minimal_state_audit.md';report.write_text('\n'.join(lines)+'\n')
    atomic_json(OUT/'review.json',dict(verdict=verdict,representations=summary,metric_groups_recomputed=checks,confirmation_executed=False,new_data_used_for_fitting=False))
    atomic_json(OUT/'validation.json',dict(status='PASS',metric_groups_recomputed=checks,prediction_sha256=digests,report_sha256=sha256(report)))
    print(verdict,flush=True)
if __name__=='__main__':main()
