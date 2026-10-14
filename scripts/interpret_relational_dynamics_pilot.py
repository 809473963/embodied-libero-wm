#!/usr/bin/env python3
"""Descriptive sensitivity audit; no refitting, selection, or gate changes."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.relational_dynamics_pilot import sha


def main():
    out=ROOT/'results/relational_dynamics_pilot'
    result=json.loads((out/'evaluation.json').read_text());ev=result['evaluations']
    a=ev['action_ensemble'];n=ev['no_action_ensemble'];b=ev['unchanged']
    metric_names=['root_macro_target_plate_mm','root_macro_target_gripper_mm']
    gains={key:100*(1-a['development'][key]/n['development'][key]) for key in metric_names}
    roots=list(a['development']['per_root'])
    wins={key:sum(a['development']['per_root'][r][key]<n['development']['per_root'][r][key] for r in roots)
          for key in ['target_plate_mm','target_gripper_mm']}
    leaveout={}
    for root in roots:
        aa=[r['error_mm'] for r in a['development_pairs']['rows'] if r['root']!=root]
        bb=[r['error_mm'] for r in b['development_pairs']['rows'] if r['root']!=root]
        leaveout[root]=dict(action_all_pairs_mm=float(np.mean(aa)),unchanged_all_pairs_mm=float(np.mean(bb)))
    absent=[r for r in a['development_pairs']['rows'] if r['truth_norm_mm']<=1]
    extra=dict(protocol_verdict_unchanged=result['verdict'],posthoc_descriptive_only=True,
        improvement_over_no_action_percent=gains,root_wins_out_of_10=wins,leave_one_root_out_all_pairs=leaveout,
        small_difference_pairs=len(absent),small_difference_mean_prediction_mm=float(np.mean([r['prediction_norm_mm'] for r in absent])),
        small_difference_max_prediction_mm=float(max(r['prediction_norm_mm'] for r in absent)),
        action_permutation={str(s):dict(correct_mm=ev[f'action_seed{s}']['development_pairs']['mean_error_mm'],
            permuted_mm=ev[f'action_seed{s}']['permuted_development_pairs']['mean_error_mm']) for s in (11,23,47)},
        evaluation_sha256=sha(out/'evaluation.json'),script_sha256=sha(Path(__file__)))
    (out/'interpretation.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2)+'\n')
    text=f'''# 关系动力学pilot：结果解释与剩余瓶颈

日期：2026-10-07。本页是固定训练之后的描述性复核，不修改原协议、阈值、模型或继续线。

**初步支持“真实关系历史＋动作能够预测部分关系变化”，尚未支持可靠的事件预测或候选选择。** 三个种子全部通过预先冻结的pilot继续线。已完成6次拟合，耗时68.48秒；13项输入/标签测试通过，30组保存预测指标完全复算一致。

## 动作信号确实有用

三seed平均相对无动作平均模型：目标—盘误差从5.686降至4.997mm（改善{gains[metric_names[0]]:.1f}%），目标—夹爪误差从10.953降至8.268mm（改善{gains[metric_names[1]]:.1f}%）。开发10起点中分别有{wins['target_plate_mm']}和{wins['target_gripper_mm']}个起点改善。

同root交换候选动作后，三个seed的全部候选差分误差均从约5.6–6.2mm上升到13.6–14.5mm。相同状态配正确动作优于错误动作，是有用的动作条件证据；这仍只针对当前固定动作池及已暴露开发起点。

## 正结果有多窄

真实目标—盘候选差异>1mm的开发对只有3对，来自task03/07/08，且都为候选50相对A0。对应真实差异分别约112.0、2.0、30.4mm；模型差分误差分别约49.2、6.2、15.0mm。方向点积虽均为正，但task07的误差比零差分更大；方向对不代表幅度对或动作更好。

所有20对的平均差分误差7.266→5.617mm，但去掉task03的两对后，零差分基线约{leaveout['task03_init34']['unchanged_all_pairs_mm']:.3f}mm、动作模型约{leaveout['task03_init34']['action_all_pairs_mm']:.3f}mm，优势反转。这个事后敏感性检查不能重定义继续线，却说明总体候选差分收益受一个大变化起点明显影响。

真实差异≤1mm的17对，预测差分平均约{extra['small_difference_mean_prediction_mm']:.3f}mm，最大约{extra['small_difference_max_prediction_mm']:.3f}mm。因此模型还会为几乎没有差别的动作预测出假差别，不能直接用于保守override。

## 事件和静止场景仍是瓶颈

目标抓持发生：动作ensemble TP/FP/FN=8/8/10，无动作=10/11/8。释放发生：动作=14/18/9，无动作=12/11/11。事件并未随连续误差改善而全面变好，且这些是相关窗口数，不是独立事件试验。

目标—盘变化>1mm的窗口误差：不变22.093mm、无动作9.907mm、动作8.253mm；未变化窗口不变只有0.069mm，而动作模型2.617mm。模型在真实运动上有信息，但静止场景仍有虚假运动预测。

动作ensemble拟合集目标—盘误差约0.995mm，开发约4.997mm；泛化缺口仍明显。另一只碗的抓持正例和task04/07的目标稳定事件正例仍缺失。丰富qpos/qvel和真实接触是特权输入，不能把本轮成绩当RGB能力。

## 下一步应解决什么

当前下一优先项是**动作后果差分与事件变化的覆盖/可靠性**，不是立即接控制器或继续换视觉encoder。建议先冻结一个独立的反例与采集设计：纳入更多有真实分支分歧的起点、接触形成/失去、动作不同但物体不动、错误物体等情况；起点和动作规则在看结果之前注册，不能只挑oracle显示有益的案例。保持成功与失败两类，另登记真正独立的新场景确认。

自然RGB/proprio/history同步记录可以随这批数据补齐，但在开始学习感知前，先明确最低需要保留哪些关系/记忆及误差容忍度。小模型物理状态压缩对照可以帮助确定目标，不能替代扩大差分证据。事件置信度/拒绝规则须另立拟合与验证，不能在当前开发数据上调阈值直到零harm。

此次没有启动上述新增采集或训练，也没有授权学习式控制器。本轮可交付结论是有边界的动作条件预测正信号及可定位的剩余失误。

[完整实验报告](relational_dynamics_pilot_audit.md) · [敏感性数值](../results/relational_dynamics_pilot/interpretation.json)
'''
    (ROOT/'docs/relational_dynamics_pilot_interpretation.md').write_text(text)
    print(json.dumps(extra,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
