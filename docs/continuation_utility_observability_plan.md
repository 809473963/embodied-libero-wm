# 阶段2A：固定C0续行效用与输入上限诊断

日期：2026-10-05。依据：[阶段1正式报告](candidate_utility_oracle_audit.md)，结果快照 `ea289a3`。

状态：**用户已授权执行阶段2A；实施已冻结，先做重放preflight。** [独立protocol](../configs/continuation_utility_observability_protocol.json)与[split](../configs/continuation_utility_observability_split.json)固定特征白名单、读出器、3个seed、指标和停止规则。阶段1protocol、标签及门槛保持原样。实时进度见[runtime.json](../results/continuation_utility_observability/runtime.json)。

**先问：在真实H8后果已经给定时，哪些观测能预测固定C0的后续成功？再学习动作执行前的候选后果。** 本轮先解决效用读出，不启动完整WM或learned controller。

## 1. 为什么现在做这个实验

38个有效root中，候选集合能救回7个、覆盖6任务；精确H8 T1只捕获1个。这已证明此pilot有学习动作选择的价值，但以T1为唯一目标会漏掉多数机会。下一步应使用实际续行结果，并拆分状态信息、自然观测与动作后果预测的误差。

N32保留现有7个oracle rescue，可用于首轮降低预算。它是在已暴露pilot上选择的接口，不保证与N64在新scene等价；后续结果的上限必须按实际N32计算。

## 2. 效用定义和时间边界

对root时刻t、候选H8动作A，定义经验标签：

```text
Y(s_t,A,l) = 执行A后，按冻结C0每8步重算，
             到episode总300步是否至少一次命中官方success

DeltaY(A,A0) = Y(s_t,A,l) - Y(s_t,A0,l)
```

只对root前未成功的scene做主分析。保留剩余步数作为已知上下文，不能把不同prefix的剩余预算隐去。success首次命中沿用阶段1定义；最终On、稳定接触和耗时作辅助结果，T1作诊断，不事后改动Y。

Y是固定policy、固定预算的实际分支结果；单次0/1标签不能直接称为已校准的成功概率。若训练输出概率，另报告Brier/calibration与复验稳定性。

对H8前已经成功与H8时仍未成功的分支分别报告，避免把已命中的目标当成“预测未来成功”。从现有结果核查，N32的1,216条有效root分支中有13条在H8前命中，来自3个A0本已成功的root；全部7个rescue root都需要H8之后的续行才成功。因此只识别H8终态不能捕获这些机会。首次成功时间用于生成标签及分层，不作为读出器特征。

必须区分两种输入时刻：

| 实验 | 输入 | 用途 |
|---|---|---|
| **本次2A：H8后果效用诊断** | 真实H8状态/图像、截至H8的合法历史、语言、剩余预算 | 隔离后果已知时的效用读出及观测问题；只能用于离线诊断 |
| 后续2B：动作执行前的候选预测 | root自然观测/历史、proprio、语言、候选动作及A0、剩余预算 | 预测固定续行结果或相对A0收益；这是可以用于root选择的输入契约 |

**真实H8输入不能直接用于动作执行前选候选。** 2A有效不等于已经得到可部署的reranker，更不等于WM通过。

## 3. 首项工程工作：补齐输入，不重跑全部continuation

现有NPZ有prefix动作、完整N64动作、root RGB/proprio/sim-state及H8 RGB；JSON有H8物体位置、On/M2、grasp/reach及续行标签。缺少H8 proprio、较完整的robot/contact/velocity状态、初始与prefix观测历史。

对原40个注册root补录固定N32，保留2个ceiling root作账目与技术检查，训练主分析排除它们：

1. fresh env、同seed/init、精确重放原prefix，不重新生成动作。
2. 每root保存初始及每8步prefix的自然RGB/proprio/action历史；这些是自然模型的身份记忆输入，不传入G2 physical ID。
3. 执行原candidate H8，补录H8 proprio、EEF pose/velocity、gripper状态、目标/盘/其他相关物体pose/velocity、抓取/支撑/接触及必要场景状态。字段、坐标系、单位和对象顺序先冻结；不能把少量目标坐标称为完整state。
4. 如补录H8内部2/4/6/8步自然历史，验证额外观测读取没有改变最终后果。H8 RGB必须与原sidecar一致，On/M2及已存物体几何满足冻结容差；root输入与原reference一致。
5. 在身份、动作hash及H8后果检查通过后，按原root/candidate ID关联已有Y。失败则停止标签复用并诊断，不能把不一致分支强行接上旧标签。

本次不重新执行300步续行。全部40×32次prefix+H8重放约71,680 env steps，另加技术检查；实际耗时需先测吞吐量，不预先承诺GPU小时数。自然特征可缓存，避免每个训练seed重复编码。

原demo访问边界保持；不打开HDF5或demo40–49数组/组元数据，不启动新的demo36–39评估。

## 4. 只比较两个固定小读出器

| 条件 | 输入 | 检验的问题 |
|---|---|---|
| G：privileged参考输入 | H8 object/robot/contact/velocity、persistent referent角色、语言、预算 | 较充分状态能否在小数据下支持continuation utility读出？ |
| N：自然观测输入 | 真实H8 RGB/history + proprio/action history + prefix/initial memory + 语言/预算 | 相同后果下，自然观测保留了多少决策信号？ |

先固定一种现有视觉特征提取方式与小型时间聚合器，不扫描backbone；两组采用尽可能匹配的读出预算，记录参数量。首轮每组3个训练seed，共6次主要fit，不做宽泛超参数搜索。

禁止输入final300几何、first-success step、success标志、candidate oracle赢家、文件名/init ID/scene ID或其他label后果。G组的privileged字段不能混入N组。H8 On/M2可作辅助诊断，不能单独替代完整效用标签。

训练可预测Y；选择时只比较同root候选相对A0的预测收益。预测差异不足或不确定时保留A0，阈值只在开发数据选定并冻结。场景多数标签或proprio阶段信号不能单独证明候选选择有效。

## 5. 数据划分与对照

原40root已全部暴露，仅用于工程筛查。可固定init31–33的30个root拟合、init34的10个root做开发，标注为“拟合时未见scene、项目已暴露开发资料”；不宣称独立确认。归一化只在fit部分估计。所有candidate、prefix帧及反向/配对数据跟随完整root进入同一个split。

除A0、T1、c0_binarized外，检查同root置乱模型分数、候选间常量分数和只依赖语言/预算的基线。它们不使用success来选择部署候选；若普通classification高但动作区分与regret无改善，不能宣布效用学习通过。

标签对两组相同。比较以下root级指标，而不是以大量branch行制造样本量：

- oracle rescue及模型捕获数/比例；报告每个开发scene的可用机会。
- 相对A0净gain、harm、override precision和候选top-1 regret。
- 每task、prefix、grasp/contact模式、三个训练seed的结果。
- 同root候选区分能力、概率校准与实际推理延迟。

初筛只回答是否值得进入新scene数据与模型开发。正面筛查至少要看到相对A0正gain、regret低于T1且没有观测到harm，并列出seed差异；这是投入判断，不是统计或部署证明。小样本不能证明harm≤1%或稳健+5pp收益。

## 6. 如何根据结果推进

| 结果 | 下一动作 |
|---|---|
| G读出有用，N明显损失 | 用额外身份/几何监督或受控输入分叉定位损失；比较memory、contact/proprio及GT referent定位，最多选择一个主要修复 |
| G与N都能捕获收益 | 进入2B，训练root观测+候选动作→continuation outcome的direct consequence基线；随后才比较structured dynamics是否必要 |
| G也不好 | 先查状态缺项、C0对完整场景视觉的依赖、scene数量、读出容量/优化与标签敏感性；不能升级为“状态没信息”或“视觉必败” |
| 仅普通accuracy好，候选regret仍差 | 停止扩大网络，修同root差分目标和数据覆盖，保留诊断负结果 |

“privileged参考输入”加有限读出器不是数学上的信息上限。失败只能约束这组输入、读出与数据量；若需要归因，必须增加受控对照，不能重复旧backbone归因错误。

## 7. 进入主模型开发后再收新scene

按总计划先注册每task4个fit、2个development新scene（共60）；生成规则、seed、机器人初态和policy-neutral settling先做技术preflight。检查source关系、G2身份与布局重复，invalid scene按预先规则记账，不根据success筛选或补入“好例子”。独立seed不自动等于独立layout。

使用冻结N32收集固定C0 continuation；60×32=1,920条主体分支、至多576,000 env steps，另加重放/复验。报告这个接口自己的oracle机会；不假装测了N64上限。最终确认scene另注册并保留。

首个root选择模型采用direct consequence预测，再检验structured WM是否带来同预算收益。初次控制验证优先“单次H8干预→恢复C0”，与Y的续行policy契约一致；反复干预和提前打断需要另做分布与闭环验证。

这些结果仅针对原十个已知任务与注册scene分布；正确指令上的收益不自动证明counterfactual grounding、开放词汇或新任务泛化。

## 8. 本次实施交付

```text
configs/continuation_utility_observability_protocol.json   # 首次补录/拟合前冻结
configs/continuation_utility_observability_split.json
scripts/collect_continuation_observability.py              # 仅补录prefix/H8
scripts/run_continuation_utility_readout.py                # G/N两组小读出器
scripts/analyze_continuation_utility_readout.py            # root级决策指标
results/continuation_utility_observability/
docs/continuation_utility_observability_audit.md
```

上述代码与配置已实施；运行链为preflight→补录→固定CLIP缓存→G/N各3次fit→分析报告。preflight或分支身份检查失败时停止标签复用。最终报告尚待运行完成，本方案不将原40root重新变盲，不自动启动controller，不以T1准确率替代实际候选效用。
