# 定位辅助：全旧场景拟合与固定动力学开发评估

日期：2026-10-09。**LOCALIZATION_DEVELOPMENT_BRIDGE_NOT_ESTABLISHED**。

固定上一轮架构、监督权重和40轮训练，旧13场景4,843帧拟合；原9开发场景2,812帧仅评估，动力学后果评估为132分支。两方法×3种子=6次训练，无epoch/seed选择。确认集未打开。

| 方法 | 当前目标—盘 mm | 当前目标—夹爪 mm | 未来目标—盘 mm | 未来目标—夹爪 mm | 候选差分 mm |
|---|---:|---:|---:|---:|---:|
| direct_ensemble | 30.083 | 23.450 | 30.309 | 23.733 | 9.203 |
| localized_ensemble | 29.130 | 19.756 | 28.785 | 20.185 | 9.253 |

当前误差来自去重观测；未来误差来自132条分支且包含起点感知偏差。二者样本分布不同。固定三模型geometry_history动力学均不重训，所有预测关系与夹爪速度均由自然输入恢复。

继续线：至少2/3种子两项当前关系与两项绝对未来关系误差均比direct下降10%，候选差分不退化超过10%。这是开发资源线，不是控制效用证明。

`[{"seed": 11, "perception_pass": false, "bridge_pass": false}, {"seed": 23, "perception_pass": false, "bridge_pass": false}, {"seed": 47, "perception_pass": false, "bridge_pass": true}]`。

| 方法 | 静止分支绝对未来误差 mm | On发生 TP/FP/FN | release标签发生 TP/FP/FN |
|---|---:|---:|---:|
| direct_ensemble | 23.275 | 6/11/1 | 23/9/2 |
| localized_ensemble | 21.418 | 3/12/4 | 21/9/4 |

release标签不全是真正持有后释放；稀少的真实事件覆盖不因模型训练增加。没有运行controller、candidate ranking或任务成功实验。

此9场景开发集已多次用于诊断，不能称为独立确认。与三折结果比较时须考虑拟合场景数和评估群体不同。本轮不得用于事后选择新辅助权重或最佳checkpoint。

三折辅助收益未满足本轮匹配开发桥的联合要求。保留全部结果，不追加训练追求过线；下一判断应定位哪些关系或场景退化，而非继续随机调loss。

验证：32组指标从保存预测重算，两个ensemble精确复算，归一化仅旧fit、原输入数组与定位数据对齐验证通过。

状态：[runtime.json](../results/relational_localization_bridge/runtime.json)。机器指标：[evaluation.json](../results/relational_localization_bridge/evaluation.json)。
