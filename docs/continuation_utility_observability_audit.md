# 阶段2A：真实H8后果的continuation utility读出审计

交互式说明：[瓶颈到底在哪里](project_bottleneck_explainer.html)。用浏览器离线打开，可查看真实候选图像、救回/误伤、泛化差距及概念解释；只展示已有结果。

结果：**`NO_USEFUL_DEVELOPMENT_READOUT`**。这是已暴露pilot的工程筛查，不是独立确认或可部署候选排名。

协议SHA-256：`23f2eebee36a75ed032d685f1bdafa1af37ebb71d0a51944acd0792c18eeb0a6`。40个root、固定N32已补录；有效fit=29、development=9；排除2个此前成功root。

运行于2026-10-05 18:43:59（Asia/Shanghai）结束。最终成功运行耗时3,734.57秒；此前桌面API preflight失败耗时16.12秒，已保留记录。主补录执行71,680个prefix/H8 env steps，技术preflight另执行336步；没有新增300步continuation。

所有输入止于真实H8；标签复用阶段1的固定C0至总300步续行结果。本轮没有重跑续行。每条分支已通过root RGB/proprio/sim-state、H8 RGB与旧几何/接触诊断的一致性检查。

G使用8个物体/fixture的世界坐标pose、velocity、接触，机器人状态及qpos/qvel历史；N使用initial/prefix/H8图像的固定CLIP特征、12D proprio与动作历史。两组均有语言、时间和剩余预算。G包含persistent referent角色，N不接收物体身份或privileged字段。

| 开发集方法 | 成功root | 救回 | harm | override | 净gain | regret |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 4/9 | 0 | 0 | 0 | +0 | 0.3333 |
| 集合内outcome oracle | 7/9 | 3 | 0 | — | +3 | 0.0000 |
| T1 N32 | 4/9 | 0 | 0 | 4 | +0 | 0.3333 |
| c0_binarized | 4/9 | 0 | 0 | 9 | +0 | 0.3333 |
| G ensemble threshold0 | 4/9 | 1 | 1 | 9 | +0 | 0.3333 |
| G ensemble dev-selected | 4/9 | 0 | 0 | 0 | +0 | 0.3333 |
| N ensemble threshold0 | 4/9 | 1 | 1 | 9 | +0 | 0.3333 |
| N ensemble dev-selected | 4/9 | 0 | 0 | 0 | +0 | 0.3333 |

开发集有3个oracle rescue机会。两组threshold0 ensemble均捕获1个，却同时损伤1个A0原本成功的root；预注册阈值选择最终均取1.0，保留所有A0，未通过正收益筛查。部分单seed净gain为+1，说明结果有seed差异；不事后挑选最佳seed来替换预注册ensemble结论。

## Fit-development差距

| 输入ensemble | Fit accuracy | Development accuracy | Fit同root宏AUC | Development同root宏AUC |
|---|---:|---:|---:|---:|
| G privileged | 99.25% | 51.04% | 0.9971 | 0.5682 |
| N natural history | 100.00% | 70.14% | 1.0000 | 0.4638 |

6个单seed模型在fit上都捕获全部4个rescue且没有harm，但开发选择不稳定。自然输入的普通accuracy约70%仍未转化为可靠的候选差分选择。当前表现是明显的fit-development差距；scene覆盖不足、状态表示、读出与目标敏感性尚未被受控拆分，不能将其中任何一项写成已经隔离的根因。AUC只对mixed-label root平均（fit14个、development7个），不把branch行当作独立场景。

## 模型与seed差异

每组固定3个seed；固定最后epoch，不在开发集选checkpoint。fit root上的BCE加同root正负候选ranking loss。阈值在同一开发集选择后报告，存在开发集选择偏差；通过筛查只支持继续收集新scene。

| 输入 | seed | 参数 | 开发救回 | harm | 净gain | Brier | N32读出ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| G | 11 | 97345 | 1 | 1 | +0 | 0.4531 | 1.646 |
| G | 23 | 97345 | 2 | 1 | +1 | 0.4928 | 1.525 |
| G | 47 | 97345 | 1 | 0 | +1 | 0.4849 | 1.772 |
| N | 11 | 100417 | 1 | 0 | +1 | 0.3112 | 2.062 |
| N | 23 | 100417 | 1 | 0 | +1 | 0.2929 | 1.612 |
| N | 47 | 100417 | 1 | 1 | +0 | 0.2941 | 2.015 |

延迟只计缓存特征之后的单个小读出器；三seed ensemble还需三次读出，未包含图像编码、仿真或部署通信。

有效root的1216条N32分支中，13条已在H8前命中success。完整指标另报告排除这些分支后的classification/calibration；首次成功时间只用于分层，不作为特征。

## 开发root逐项结果

| root | A0 | Oracle | T1(candidate/Y) | G(candidate/Y) | N(candidate/Y) |
|---|---:|---:|---|---|---|
| task01_init34 | 1 | 1 | 0/1 | 0/1 | 0/1 |
| task02_init34 | 0 | 1 | 0/0 | 0/0 | 0/0 |
| task03_init34 | 1 | 1 | 17/1 | 0/1 | 0/1 |
| task04_init34 | 0 | 0 | 7/0 | 0/0 | 0/0 |
| task05_init34 | 0 | 1 | 0/0 | 0/0 | 0/0 |
| task06_init34 | 1 | 1 | 0/1 | 0/1 | 0/1 |
| task07_init34 | 0 | 0 | 2/0 | 0/0 | 0/0 |
| task08_init34 | 1 | 1 | 56/1 | 0/1 | 0/1 |
| task09_init34 | 0 | 1 | 0/0 | 0/0 | 0/0 |

## 解释与下一动作

先检查scene覆盖、读出、状态与分支目标；暂不启动2B或WM。

privileged输入加有限小读出器不是数学信息上限。失败只约束当前状态表示、模型、数据和标签；不能推出视觉没有信息。自然输入较好也不能证明不需要身份或接触。

全部40个root在项目历史中已经暴露。init31–33只用于fit，init34只用于开发；所有候选与历史跟随root分区，归一化仅在有效fit root估计。branch数量不作为独立样本数；本轮开发每任务至多一个scene，不报告伪精确的分支bootstrap区间。

真实H8是执行后输入，不能直接用这些分数在root选择动作。未启动2B、独立新scene、WM、reranker或learned controller；旧阶段授权文件未改动。

未打开HDF5。官方init容器只将31–34行构造为数值数组；未启动demo36–39评估，未访问demo40–49数组或组元数据。

## 完成后只读核验

[核验脚本](../scripts/verify_continuation_utility_observability.py)检查40个root的输入/来源/feature/checkpoint SHA-256、1,280条分支的旧标签关联和H8 RGB、动作与时间边界、G/N字段及fit-only归一化；重新加载6个checkpoint计算预测，并独立复算各seed与ensemble的开发决策。所有检查通过，详见[final_verification.json](../results/continuation_utility_observability/final_verification.json)。此次核验没有新增fit、simulator或数据分区访问。

实施阶段完整仓库测试记录为345 passed、21条依赖弃用警告，见[implementation_validation.json](../results/continuation_utility_observability/implementation_validation.json)；本次完成后未重跑整个套件。测试和哈希核验验证实现及复算一致，不验证研究命题成立。

[完整指标](../results/continuation_utility_observability/metrics.json)、[运行状态](../results/continuation_utility_observability/runtime.json)、[preflight](../results/continuation_utility_observability/preflight.json)、[训练记录](../results/continuation_utility_observability/training_manifest.json)、[协议](../configs/continuation_utility_observability_protocol.json)。
