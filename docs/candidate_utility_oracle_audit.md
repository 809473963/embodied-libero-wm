# Candidate Utility Oracle：真实候选效用审计

结果：**B：候选有机会，T1效用失败**。仅为calibration pilot，不构成独立新场景确认。

完成时间：2026-10-05 16:29（Asia/Shanghai）。主要结果、数组hash及复验记录已独立核查；见[最终核验记录](../results/candidate_utility_oracle/final_verification.json)。

**即使直接读取真实H8后果，按冻结T1规则仍漏掉6/7个可救回起点。** 这使“先把RGB学生学到更接近T1，再期待候选排名改善”的投入依据不足。N64在此固定C0续行设定下有实际机会；下一步需要修改utility/horizon，并检查自然观测能否预测有用后果。它没有证明任何一种视觉架构必然成功。

协议 SHA-256：`de6e77d0405a370f397e362c8fe3d6d52c9ef6fa5cd6077c93fc81a64cfbcb7e`。冻结实现提交：`4f5cf66`。

已完成40个注册root和2,560条主候选分支；有效未完成root=38，此前成功root=2。

| 方法 | 成功root | 对A0救回 | 伤害 | 净变化（百分点） |
|---|---:|---:|---:|---:|
| A0 + 固定C0 continuation | 16/38 | — | — | — |
| 集合内outcome oracle | 23/38 | 7 | 0 | +18.42 |
| H8 T1-selected | 17/38 | 1 | 0 | +2.63 |
| c0_binarized固定规则 | 16/38 | 0 | 0 | 0.00 |

rescue覆盖6个任务；T1捕获1/7（14.29%），override=17个root，其中只有1次改善；mean regret=6/38（15.79%）。Oracle的23/38是离线集合内上限，不是已训练方法的成功率。0个harm仅是这38个root的观测，不能证明满足部署要求的1%伤害上界。

T1净success变化的95%描述性区间：0.00至+7.89个百分点；oracle rescue区间：+7.89至+30.56个百分点。T1区间包含0，不能声称稳定正收益。按任务分层重采样root，不把64个候选当作独立样本。每任务仅4个root，区间不覆盖训练seed、任务选择或全部场景分布的不确定性。

| Gate | 结果 |
|---|---|
| G1a_technical | True |
| G1b_coverage | True |
| G1c_opportunity | True |
| G1d_t1 | False |
| simple_rule_saturates | False |

| 任务 | 有效root | A0成功 | Oracle成功 | T1成功 | Rescue | Harm |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 3 | 3 | 3 | 0 | 0 |
| 1 | 4 | 3 | 4 | 3 | 1 | 0 |
| 2 | 4 | 3 | 4 | 3 | 1 | 0 |
| 3 | 3 | 3 | 3 | 3 | 0 | 0 |
| 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 4 | 1 | 3 | 2 | 2 | 0 |
| 6 | 4 | 1 | 2 | 1 | 1 | 0 |
| 7 | 4 | 0 | 0 | 0 | 0 | 0 |
| 8 | 4 | 2 | 3 | 2 | 1 | 0 |
| 9 | 4 | 0 | 1 | 0 | 1 | 0 |

| 子集 | Oracle成功 | T1成功 | Rescue | T1 harm |
|---|---:|---:|---:|---:|---:|
| N8 | 20 | 16 | 4 | 0 |
| N16 | 21 | 16 | 5 | 0 |
| N32 | 23 | 16 | 7 | 0 |
| N64 | 23 | 17 | 7 | 0 |

N32与N64在本pilot覆盖同样的7个oracle rescue，但N32的T1没有救回任何root。这说明增加候选数量与选中有用候选是两个问题。N32可以作为后续冻结的预算对照，不能把这个开发集上的相等结果直接推广到新scene。

七个rescue起点的具体核查：

| Root | Prefix步数 | T1候选ID | T1是否救回 |
|---|---:|---:|---|
| task01_init32 | 64 | 25 | 否 |
| task02_init34 | 32 | 0 | 否 |
| task05_init32 | 64 | 61 | 是 |
| task05_init34 | 0 | 0 | 否 |
| task06_init33 | 0 | 0 | 否 |
| task08_init32 | 32 | 0 | 否 |
| task09_init34 | 0 | 0 | 否 |

五个rescue起点被T1规则保留A0，另一个选到失败候选。task01/task02的成功候选均来自continuous类别；task08/task09均来自structured gripper类别。其他三个root有两类成功候选，不能将收益全归因于gripper或某一几何维度。完整成功候选IDs及forms见最终核验记录；未事后搜索新的廉价规则。

按预定root时刻分层：

| Prefix步数 | 注册root | 有效root | Root已抓持（全部注册） | A0成功 | Oracle成功 | Rescue | T1救回 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 10 | 0 | 5 | 8 | 3 | 0 |
| 32 | 10 | 10 | 0 | 4 | 6 | 2 | 0 |
| 64 | 10 | 10 | 3 | 3 | 5 | 2 | 1 |
| 96 | 10 | 8 | 4 | 4 | 4 | 0 | 0 |

G1b通过的是38个未完成root、10任务的覆盖计数；以上时刻不是reach/grasp/transport/place真值。每scene只有一个root，时间与scene仍有混杂；这轮没有证明覆盖所有关键操作阶段，也不能由96步无rescue推断晚期干预普遍无用。

下一动作：`CONTINUATION_AWARE_UTILITY_AND_OBSERVABILITY_LADDER`。本次收集结束；没有自动启动下一阶段训练或learned controller。

所有候选从fresh env及同seed/init/prefix起步，H8后使用同一冻结C0至总300步。官方success首次命中为primary；最终On与H8几何为诊断。G2身份从初始scene确定并保持。A0和预声明T1/rescue分支通过重复动作及结果检查。

最终核验复算所有主指标并检查40个数组文件、候选范围、root RGB身份、300步预算、80条A0复验和50条干预复验的动作hash/首次成功时间/结果及预声明复验ID；均通过。10项针对性测试通过；本轮未重跑全仓库测试。

执行曾因系统重启中断。30个完成root通过校验，最后一批8条未完成root的输出文件为空，已保留原文件并按同一候选重算，通过恢复后的身份与artifact hash检查。见[恢复事件](../results/candidate_utility_oracle/recovery/20261005_interrupted_batch/event.json)。重算没有改变root、candidate、T1或gate，也没有将同一分支计为额外独立样本。

未打开HDF5；官方init容器只将31–34行构造为数值数组。旧协议和运行授权保持原样；没有训练encoder、value或WM。

各次运行已记录耗时合计：38482.5秒；中断期间不计入。技术preflight另见结果。

[完整指标](../results/candidate_utility_oracle/metrics.json)、[决策](../results/candidate_utility_oracle/decision.json)、[preflight](../results/candidate_utility_oracle/preflight.json)、[访问记录](../results/candidate_utility_oracle/access_ledger.json)、[运行状态](../results/candidate_utility_oracle/runtime.json)。
