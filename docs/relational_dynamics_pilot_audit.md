# 动作条件关系动力学：小型特权状态试验

日期：2026-10-07。**ACTION_SIGNAL_PILOT_SUPPORTED**。

六次固定拟合已完成（两种模型×三个种子）。本次仅使用已有物理记录，无新仿真、视觉模型或控制器执行。

## 开发集主要结果

| 方法 | 目标—盘位移误差（mm，root宏平均） | 目标—夹爪位移误差（mm，root宏平均） | 有效候选差分误差（mm） |
|---|---:|---:|---:|
| unchanged | 9.096 | 23.111 | 48.149 |
| no_action_seed11 | 7.462 | 14.697 | 48.149 |
| action_seed11 | 6.868 | 10.140 | 21.051 |
| no_action_seed23 | 7.214 | 13.894 | 48.149 |
| action_seed23 | 6.240 | 11.170 | 26.733 |
| no_action_seed47 | 7.712 | 15.045 | 48.149 |
| action_seed47 | 6.154 | 10.155 | 23.445 |
| no_action_ensemble | 5.686 | 10.953 | 48.149 |
| action_ensemble | 4.997 | 8.268 | 23.461 |

候选差分开发集共20对、10个root；真实目标—盘差异>1mm的有效分层只有3对、3个root。差分分数不构成独立泛化证明。

冻结继续线：至少2/3种子同时满足两项连续关系误差低于无动作与不变基线，且有效候选差分误差改善至少10%。该线是有界pilot投入规则，不是统计显著性结论。

```json
[
  {
    "seed": 11,
    "continuous_both_better": true,
    "material_pair_improvement_10pct": true,
    "joint_pass": true
  },
  {
    "seed": 23,
    "continuous_both_better": true,
    "material_pair_improvement_10pct": true,
    "joint_pass": true
  },
  {
    "seed": 47,
    "continuous_both_better": true,
    "material_pair_improvement_10pct": true,
    "joint_pass": true
  }
]
```

## 事件及变化检查

普通accuracy不作主要成绩。完整结果逐类列出接触/事件的precision、recall、Brier、发生/消失计数及未知mask；当前状态未知时，不变基线拒绝预测，因此其二元分母可能不同。
连续预测另外报告>1mm变化与未变化分层、每root误差、另一只碗误差。部分稀有事件只有少数正例，不能只展示总体均值。

| 模型 | 目标抓持发生 TP / FP / FN | 目标释放发生 TP / FP / FN |
|---|---|---|
| unchanged | 0 / 0 / 18 | 0 / 0 / 23 |
| no_action_ensemble | 10 / 11 / 8 | 12 / 11 / 11 |
| action_ensemble | 8 / 8 / 10 | 14 / 18 / 9 |

## 解释边界与下一步

输入是丰富的真实物理状态（包含qpos/qvel、接触和已知目标身份）、11采样点历史、因果接触记忆；不是从RGB估计的最小状态。预测的是已规划H8块内后果，不提前使用块后C0反馈动作。
30/10旧起点划分不含新的盲测；仅三个候选，另一个碗的抓持无正例。该结果不能证明跨物体、跨任务或自然观测泛化。
即使本轮为正，也只支持进一步检验物理状态压缩、覆盖与自然感知；不能直接升级为候选选择或闭环收益。若未通过，先根据拟合/开发差距、动作置换和同root差分分析瓶颈，不换视觉backbone来解释。

验证：13项输入/标签测试在冻结前通过；本次重新计算30组指标并逐项完全一致，fit-only归一化复算一致。未重跑全仓库测试。

[冻结协议](relational_dynamics_pilot_protocol.md) · [完整指标](../results/relational_dynamics_pilot/evaluation.json) · [运行状态](../results/relational_dynamics_pilot/runtime.json)
