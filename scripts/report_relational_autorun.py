#!/usr/bin/env python3
"""Final independent metric recomputation and bounded scientific report."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.candidate_utility_oracle import atomic_json,sha256
from src.probes.relational_dynamics_pilot import score,normalization
from scripts.evaluate_relational_autorun import paired
OUT=ROOT/'results/relational_autorun'

def main():
    cfg=json.loads((ROOT/'configs/relational_autorun_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    result=json.loads((OUT/'evaluation.json').read_text());coverage=json.loads((OUT/'coverage.json').read_text());checks=0
    lines=['# 新development场景：连续执行链结果','', '日期：2026-10-08。**'+result['verdict']+'**。','',
        '自动完成C0轨迹、当前状态与盘相对位置分层、固定H24分支、数据核验、旧H24拟合及新development评估。确认集未打开，无学习式控制器执行。','',
        f"实际有可用起点的新场景：{coverage['new_scenes_with_anchors']}/20；新分支：{coverage['new_branches']}。没有满足采样层的场景记录在branches/cases.json，不静默替换。"]
    if 'models' in result:
        with np.load(OUT/'models/dataset.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
        meta=json.loads((OUT/'models/samples.json').read_text());fit=d['fit'];dev=~fit
        stats=normalization(d['x'],d['actions'],d['delta'],fit,d['weights'])
        with np.load(OUT/'models/normalization.npz',allow_pickle=False) as f:
            for k,v in stats.items():np.testing.assert_array_equal(f[k],v)
        for name,m in result['models'].items():
            if name=='unchanged':delta=np.zeros_like(d['delta']);prob=np.where(d['current']>=0,d['current'],np.nan)
            else:
                with np.load(OUT/f'models/{name}_predictions.npz',allow_pickle=False) as f:delta=f['delta'];prob=f['probability']
            for s,mask in [('old_fit',fit),('new_development',dev)]:assert score(delta,prob,d,meta,mask)==m[s];checks+=1
            assert paired(delta,d,meta,dev)==m['new_pairs'];checks+=1
        lines+=['','| 方法 | 目标—盘误差 mm | 目标—夹爪误差 mm | 候选差分误差 mm（场景宏平均） |','|---|---:|---:|---:|---:|']
        for name in ('unchanged','no_action_ensemble','action_ensemble'):
            m=result['models'][name];s=m['new_development'];lines.append(f"| {name} | {s['root_macro_target_plate_mm']:.3f} | {s['root_macro_target_gripper_mm']:.3f} | {m['new_pairs']['root_macro_error_mm']:.3f} |")
        lines+=['','固定种子判定：`'+json.dumps(result['seed_checks'],ensure_ascii=False)+'`。六次拟合均仅用旧H24数据，未用新development数据归一化、拟合或选择epoch。']
    lines+=['','## 实际释放与稳定确认','', '| 来源 | 释放标签形成：分支/场景 | 从持有到释放：分支/场景 |','|---|---:|---:|']
    for split,g in coverage['events'].items():
        release=g['release_after_attachment'];held=g['held_to_release'];lines.append(f"| {split} | {release['onset_branches']}/{len(release['onset_roots'])} | {held['branches']}/{len(held['roots'])} |")
    lines+=['','事件预测是否可靠必须结合逐类precision/recall、发生/消失分母及场景数。释放标签形成不自动等于起点正在持有后的实际释放；小场景支持下不作泛化保证。',
        '', '## 解释边界','',
        '该链检验同任务/物体下的新development布局及动作后果。布局仅与40个已审计旧布局去重，未证明C0历史从未见过；旧训练数据也只有13个场景，失败不能自动归因为视觉。',
        '输入仍是丰富特权状态。同步RGB/proprio已留存，但没有训练自然感知。H24模型不等于动作价值模型，预测误差改善不授权直接选动作或接控制器。',
        '不根据结果自动调阈值或重跑。后续应据误差和事件分层决定压缩状态、自然感知或补充指定反例；confirmation继续封存。',
        '', f'验证：保存预测的{checks}组指标已完全复算；旧fit归一化复算一致。完整指标见evaluation.json，分层与缺失见coverage.json、branches/cases.json。']
    report=ROOT/'docs/relational_autorun_audit.md';report.write_text('\n'.join(lines)+'\n')
    atomic_json(OUT/'validation.json',dict(status='PASS',metric_groups_recomputed=checks,report_sha256=sha256(report),evaluation_sha256=sha256(OUT/'evaluation.json'),confirmation_executed=False))
if __name__=='__main__':main()
