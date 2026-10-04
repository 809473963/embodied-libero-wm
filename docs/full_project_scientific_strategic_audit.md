# Full Project Scientific & Strategic Audit
## LIBERO-Spatial Prediction-Verified Embodied Agent

日期：2026-10-04。审计基线：`9c2aed21ec88e2314475fce60116be7e64ef75a6`。

## 1. Executive verdict

**Overall project verdict：B — project is scientifically promising but current architecture is wrong。Confidence：medium。**

这里的“wrong”指当前强制研究分解——先把单帧 RGB 蒸馏成跨场景 T1 比较器，过所有门槛后才允许检查真实候选收益——缺乏必要性，不是已经证明所有 prediction-verified control 架构无效。

**继续按当前路线投入一个月，不是最聪明的选择。建议 MAJOR ARCHITECTURAL PIVOT：先验证干预机会和效用，再决定需要学习什么状态与动力学。** 对“停止继续盲换 frozen encoder”的建议置信度高；对最终采用哪种感知架构的判断置信度中等。

- 最重要的缺口是 **尚未证明 N64 在真实、策略访问的状态上包含有用替代动作，也未证明 T1 会选中它们**。这比选哪个 backbone 更上游。
- T1 是明确的目标关系描述与近终态几何代理，**不是经过验证的动作价值、成功概率或剩余控制代价**。
- 新近失败真实存在，但它们隔离的是“小数据、固定读出、固定目标下的可读出性”，没有隔离视觉信息本身。
- Stage I/E 实际拟合数据只有 **30 条轨迹、3,756 帧**；选择集只有 **10 条轨迹、1,288 帧**。62,868 个训练 pair 不构成 62,868 个独立场景。
- 最近 cross-match 控制只有 **16 个不重复图像对、5 个任务**；其中 task2 占 9 对。不能让这一项独自承载整个 backbone 的机制归因。
- 单帧需要同时恢复历史 referent 身份、接触状态和毫米级三维关系；这些要求尚未做观测上限拆分。**单帧不充分是强候选解释，尚非已证定理。**
- C0 的因果时序修正确实带来明显控制改善；此前 WM 也确实使用了 action。不能把这些历史结果改写成“策略不可用、WM 不可学”。
- 原始目标是“预测偏差是否值得触发提前重规划”，后来演变成了“学习通用语义 value 并进行候选排名”。**后者并不是前者必须经过的证明链。**
- 下一步只选一个实验：**calibration-only 的真实 N64 分支上限审计，联合检查 H8 的 T1 后果和固定 C0 后续执行收益**。本报告只提出该实验，不执行，不解除现有运行禁令。

## 2. 审计边界、证据等级与可复现性

这是研究判断报告，不是新训练或新执行 protocol。未改动任何旧 teacher、gate、checkpoint、结果或 authorization 文件；未运行 simulator、候选分支、WM、controller；未打开 HDF5 或 demo36--39 的数值 sidecar；未读取 demo40--49 数组或组元数据。读取了旧文档中已经公开的历史汇总，它们与重新打开受限数据不是同一行为。

证据分三类：**R** 为现存代码/JSON/报告核查；**C** 为本次仅使用已有 calibration sidecar 的复算；**I** 为从定义推出的机制解释或待检验判断。后文建议中的阈值是未来投入决策标准，不是已经通过的结果。

本次复算输出：[evidence.json](../results/full_project_scientific_strategic_audit/evidence.json)；复算脚本：[audit_project_scientific_evidence.py](../scripts/audit_project_scientific_evidence.py)。输出保留显式输入 SHA-256、样本数量、控制对去重统计、独立端点 referent 校验以及 Stage I/E 主指标复算。脚本不调用任何训练或 simulator 入口。

复现命令：

```bash
env -u PYTHONPATH .venv/bin/python scripts/audit_project_scientific_evidence.py
```

另运行现有 Stage I/E 主指标和 probe 的四项复算测试：**4 passed**。未重跑完整 326 项套件；历史全套通过仅是原报告记录，不能被表述成本次验证。测试通过验证计算与约定一致，不验证研究命题正确。

主要证据索引：

| ID | 控制性来源 | 本次核查用途 |
|---|---|---|
| E01 | [原始立项计划](Embodied_VLM_WorldModel_LIBERO_Execution_CN.md) | 原始问题、8GB 资源边界、预测校验与重规划目标 |
| E02 | [C0 因果修正](c0_v2_causal_alignment_audit.md)、[dataset 源码](../src/c0/causal_dataset.py) | label offset、旧训练分区、控制改善 |
| E03 | [C1-v2](c1_v2_rebase_and_wm_distribution_audit.md)、[WM action audit](c1_v2_2_wm_action_conditioning_diagnosis.md)、[真实旧分支](local_candidate_counterfactual_dynamics_audit.md) | reranking 与 action sensitivity 的历史范围 |
| E04 | [N64 报告](expanded_candidate_interface_contract_restart.md)、[结果 JSON](../results/expanded_candidate_contract_restart/diagnosis.json)、[源码](../src/probes/expanded_candidate_contract_restart.py) | imitation headroom、候选构造和 gripper 影响 |
| E05 | [temporal progress](goal_conditioned_pairwise_progress_audit.md) | 时间监督与机器人阶段相关性 |
| E06 | [G2 报告](robust_same_scene_referent_grounding_audit.md)、[grounding 源码](../src/probes/robust_grounding.py) | 正确指令与反事实 grounding 的证据差别 |
| E07 | [T1 完整评估](t1_semantic_goal_progress_teacher_full_evaluation.md)、[冻结定义](../configs/t1_teacher_frozen_definition.json)、[官方 On](../third_party/LIBERO/libero/libero/envs/object_states/base_object_states.py) | teacher 定义、epsilon、非 tie 密度、接触语义 |
| E08 | [global student 报告](goal_conditioned_semantic_value_distillation_audit.md)、[结果](../results/semantic_value_evaluation.json)、[cross controls](../results/semantic_cross_demo_controls.json)、[sampler 与 metric 源码](../src/probes/semantic_value_distillation.py) | 概率、CF、跨 demo 匹配及排序目标 |
| E09 | [Stage R](goal_conditioned_spatial_visual_representation_audit.md)、[Stage I 协议](../configs/clip_internal_layer_protocol.json)、[读出器](../src/probes/clip_internal_layer.py)、[runner](../scripts/run_clip_internal_layer_audit.py) | 读出容量、layer 比较、probe 实际任务 |
| E10 | [Stage E 协议](../configs/alternate_encoder_protocol.json)、[结果](../results/alternate_encoder_calibration.json)、[probe](../results/alternate_encoder_information_probes.json) | 两个替代 backbone 的受限负结果 |
| E11 | [camera](camera_projection_contract_resolution.md)、[footprint](spatial_visual_target_design_study.md)、[area extraction](relevant_object_visible_area_natural_extraction_audit.md) | 已排除的实现问题、绝对回归与差分决策的断裂 |
| E12 | [Stage I ledger](../results/clip_internal_layer/access_ledger.json)、[Stage E ledger](../results/alternate_frozen_encoder/access_ledger.json)、[历史边界事件](semantic_goal_progress_teacher_audit.md) | 本阶段未访问与全项目历史暴露的区别 |
| E13 | [本次 evidence.json](../results/full_project_scientific_strategic_audit/evidence.json) | C 类派生事实与输入哈希 |

核查了 git 历史及 annotated tags；例如 `expanded-candidate-interface-valid` 指向 `5d38525`，`frozen-visual-backbone-research-blocked` 指向 `9c2aed2`。这些证明冻结版本存在，不赋予科学结论额外真值。未从原始环境重现所有历史 rollout；历史控制成功率在此按现有报告证据处理，不冒充独立复现。

## 3. What has actually been solved?

| Component | Claimed status | Audited status | Confidence | Evidence |
|---|---|---|---|---|
| causal action timing | causal-aligned | 此数据/采样契约下 offset 修正有强支持；连同必要的归一化变化 | 高 | E02：development 11→45/100；代码 `k+1:k+9` |
| C0 proposal quality | usable prior | 比旧版强，非所有任务可靠；是合理 baseline，非已解决 manipulation | 高 | E02；E03 的匹配复跑 44/100，不能混成统一 45/100 |
| candidate diversity | N64 valid | 有限候选契约有效、动作空间更广；语义覆盖未知 | 高 | E04：64 个合法候选；C48_G15 构造 |
| candidate task-value headroom | 常被简称 headroom solved | **UNTESTED**；只验证了专家动作 MSE oracle | 高 | E04 明确未模拟 N64 future |
| grounding | robust valid | privileged G2 在本分布正确指令上可靠；不是 RGB grounding | 高/中 | E06：40/40 + 40/40；反事实无独立身份真值 |
| persistent referent | fixed identity | teacher 身份保存正确；视觉身份记忆尚未建立 | 高 | E06/E07；student 不接收 initial state/memory |
| semantic teacher | valid goal progress | 有定义的目标关系监督有效；控制 utility 未验证 | 高 | E07：On+M2，不含 reach/grasp/recoverability |
| single-RGB observability | 隐含充分 | 未隔离；存在几何与历史身份歧义风险 | 中 | E07 定义 + E09 输入契约，I 类判断 |
| language conditioning | insufficient | 有真实语言效应；准确 grounding 与强动作偏好仍不足 | 高 | E08：63.99% CF direction；18.57% reversal both-correct |
| visual representation | backbone limit | 固定小读出、少量数据下失败；信息缺失归因不足 | 高 | E09/E10/E13 |
| cross-demo invariance | scene shortcut | 这批难例上失败可信；总体强度、归因不稳定 | 高/中 | E13：32 rows=16 pairs，只覆盖5任务 |
| value distillation | blocked | 已测 S0/S1/S1.5/S2 均未全过旧 gate；不是所有 value 学习失败 | 高 | E08 |
| world model | action sensitivity block | 旧 latent WM 有 action 信号，局部差分不足；structured object dynamics 未决 | 高 | E03；旧 mediator 有部分 proprio 差分能力 |
| real candidate ranking | blocked | **新 N64/T1 链未执行**；旧 N3 C1 已执行且未改善总体收益 | 高 | E03/E04 |
| controller | false | 新链未授权、未验证；旧 C0/C1/C2 控制实验实际存在 | 高 | E01/E02/E03；不能说全项目从未运行 controller |

### Observation → interpretation → gate → authorization

| 观测 A | 可支持解释 B | 旧 gate C | 审计后的授权含义 D |
|---|---|---|---|
| L9 CF 比 L12-proj +4.96 pp，cross controls 更差 | 末层不是唯一问题；可读出 tradeoff | I joint gate fail | 保留旧 block；不能升级为全 backbone 不含几何 |
| DINO/SigLIP 在 P1 下失败 | 换冻结 backbone 未解决该设定 | E +5 pp 等条件 fail | 停止此扫描合理；不禁止独立科学 oracle 诊断的设计 |
| T1 非 tie 57.91%、CF material change 29.59% | 专家轨迹上有语言相关关系标签 | teacher pass | 可以作为监督对象；不能直接授权 candidate utility |
| N64 MSE 改善68.10% | 候选中有更像专家的动作块 | imitation gate pass | 可以研究真实后果；未证明实际规划收益 |
| S2 accuracy 69.81% | 有信号、整体不够可靠 | 70% point gate fail | 按旧规则失败；0.19 pp 不是机制边界 |

## 4. 当前真正的 bottleneck

**研究决策的 root blocker：干预机会 × 效用定义尚未闭合。** 不是说已经知道真实物理瓶颈是 candidate generation，而是尚未确认最终优化问题值得解；在此之前继续优化中间 evaluator 的投资依据不足。

**当前监督学习的 bottleneck：几何/身份的可观测性、数据覆盖、读出压缩和目标分辨率纠缠。** 不能把这四项统称为 frozen backbone。其优先级如下：

1. N64 在 C0 访问分布上能否 rescue，T1 是否识别这些机会：高影响、几乎未测。
2. 必要 reach/grasp/release 与 T1 的短视目标是否冲突：定义上就存在风险，控制后果未测。
3. initial referent memory、contact、相对高度等状态信息是否可由部署观测恢复：未做上限。
4. 30 个拟合 episode 是否足以跨场景泛化：很可疑，但无 scene-count learning curve，不能断言不可学。
5. P1 只有单个 24-D 注意力 slot、24-D mean、中心与熵，再接小 head：它本身也可能扔掉对象间几何。

固定相机有助于利用桌面先验恢复 2-D→3-D 关系，因此不能仅凭“单目”断言任务不适定。反过来，固定相机也不能让遮挡接触、被抓物体姿态和历史身份自动可见。**可观测性需要构造同观测/近观测异标签例，或用输入阶梯检验，而不是从一个失败 probe 推断。**

## 5. What have we actually learned from failures?

| 可以合理排除或降优先级 | 仍不能排除 |
|---|---|
| 仅修正最后 CLIP projection/LN 就足以修复完整联合 gate | trainable encoder、不同读出、历史输入、更多独立场景 |
| temporal ordering 本身已经是语言条件控制价值 | temporal features 与合法 proprio 对真实控制有帮助 |
| 旧 N3、小扰动的 imitation 搜索空间已经足够 | N64 是否有真实 task utility、是否需要改变 H |
| attention 随语言变化就代表找对物体 | 通过额外定位监督能否改善身份与关系估计 |
| 绝对 area 回归好就保证候选差分排序好 | 直接学习分支差分、contact 或 relative pose 是否有效 |
| 旧 global latent WM 完全忽略 action | 旧训练分布之外能否可靠预测局部操作后果 |
| 单一 threshold FAIL 足以证明方向错误 | 失败是目标、统计设计、容量或数据量造成 |

历史上最有价值的学习不是“模型又失败了”，而是：**普通预测精度、局部差分保真和控制决策收益是三个不同量。** 可见面积实验已经展示 MAE 改善约41–44%、R²约0.665，却只有约50–51%候选方向准确率。此次审计不应再把普通 T1 accuracy 当作最后决策的充分统计量。[E11]

## 6. T1 判决：KEEP AS DIAGNOSTIC ONLY

### 6.1 它是什么，不是什么

`T1=[On, min(1-d_xy/0.03, Δz/0.03)]` 是终态关系及其数值不等式 slack。On 正确保留 contact；M2 不包含 grasp、reach、支撑稳定性、障碍绕行、动作代价或从当前状态实现目标的可达性。[E07]

因此 **terminal relation proximity** 比“general goal progress value”更准确。它并不等于 `V^π(s,l)` 或 `Q^π(s,A,l)`；没有固定后续 policy 和 horizon 时，“更好状态”甚至未被完整定义。

四个直接反例/限制是定义推论，尚不是已测控制损失：

- 碗和盘不动时，手接近、对齐和闭合可以改变可抓取性，但 T1 完全不变。
- 把碗从抽屉/支撑物中取出可能先增加平面距离；M2 可能惩罚必要操作。
- 当水平 slack 是 min 的较小项，继续抬高碗可能完全不改变 M2；另一些区域，高于盘但未接触可获得比已放稳前的几何 margin 更高的值。
- 不同抓取、速度、障碍和剩余动作预算可有相同 On/M2，却有不同后续成功概率。T1 tie 是 **teacher 等价**，不是 **control 等价**。

local non-tie 57.91%证明标签密度；不能证明必要早期操作有梯度。尤其 T2 只有在 T1 非 tie 密度不足时才启动，实际上用“标签够密”替代了“阶段覆盖充分”的判断。[E07]

### 6.2 几何精度错配

冻结 `epsilon=0.0040655`。当比较由同一归一化几何分量主导时，它对应约 **0.122 mm** slack 变化；cross-difference 的 `0.05` 对应 **1.5 mm**。这是单位换算，不是说所有非 tie 图像对都只相差0.122 mm，也不是已证明视觉必须达到该绝对定位误差。[E13]

但它揭示了目标设计问题：用 simulator 很容易标出数值差异，不代表 RGB 能稳定看见，也不代表机器人控制值得响应。应该在新实验同时报告差分幅度、观测噪声与决策 regret；不能事后修改旧 epsilon 来“救过关”。

### 6.3 Language signal 的强度

29.59% preference change 主要由 **non-tie→tie 28.17%**组成；真正反转只有1.29%。这不是无语言信息，而是演示通常只移动一个目标碗：更换语言后常在评价一个基本没动的碗。学习“相关/不相关”与学习“在两种可行动作之间反向选择”不同。[E07]

G2 对正确指令有构造真值；eligible counterfactual 的 precision 没有同样的独立真值。100% correct grounding 不能移植为100% counterfactual label correctness。可以保留 G2 作为明确定义的反事实 teacher，但需把“服从 resolver”与“人类语义正确”分开。[E06]

| T1 角色 | 判决 | 使用条件 |
|---|---|---|
| perception teacher | KEEP | 保留 On/M2 作为输出诊断；最好同时监督更可解释的相对几何/身份/接触 |
| state value teacher | KEEP AS DIAGNOSTIC ONLY | 不称成功概率或完整 manipulation value |
| candidate utility teacher | KEEP AS DIAGNOSTIC ONLY | 与 continuation outcome 比较后决定 MODIFY/REPLACE |
| world-model objective | MODIFY | 预测相对状态和接触等充分变量；T1 为可派生的辅助量，不是唯一预测靶 |

potential shaping 的理论也不能为这里的短视选择背书：保留原 reward 并加入 `γΦ(s')−Φ(s)` 的策略不变性，不等于贪心最大化 `Φ(s_{t+8})` 最优。[Ng、Harada、Russell 原论文](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

## 7. 单帧、历史信息、proprio 和最小状态

| 信息 | 单帧 RGB 能提供什么 | 未解决的风险 | 最小补充候选 |
|---|---|---|---|
| referent identity | 当前可见外观和空间关系 | 指令描述的是初始位置；移动后两只相似碗需要身份记忆 | 初始 grounding + recurrent identity/belief |
| XY relative geometry | 固定相机桌面上通常有较强线索 | perspective、遮挡、离开桌面平面 | 深度上限/相机标定/多帧 |
| relative Z | 视觉轮廓和支撑先验 | 碗高度、倾斜、遮挡造成别名 | 短历史、多视角或合法 depth（若部署允许） |
| grasp/contact/support | 手指接近、物体共运动等间接线索 | 像素接近不等于接触；静态图未必区分被抓/悬空/支撑 | proprio + action history + 多帧 |
| recoverability | 几何布局提供部分线索 | 速度、夹爪状态、障碍与预算缺失 | 历史状态估计和动作条件后果 |

**不应继续把禁止 proprio 当作部署原则。** 旧 proprio-only≈96%暴露的是时间监督容易被阶段解释，不是 proprio 不合法。连 C0 已合法使用12-D proprio。合理做法是比较 RGB-only、proprio-only、RGB+proprio、history+proprio，并用同观测换语言、同物体状态换机器人姿态、跨场景分组、真实候选 regret 检验是否依赖错误捷径。

优先考虑的最小 structured state 不是完整 scene graph：**persistent target belief、目标相对盘的位置/高度及不确定性、抓取/支撑模式、EE相对目标关系**。哪些变量必要由 branch utility 决定。若只需检测动作块失效，甚至无需恢复精确 M2。

训练时 privileged state 可以监督这些量，部署时仍只用自然 RGB/proprio/history。这一范式有先例，但先例并不证明当前标签或小数据设置足够；没有直接文献依据能免除本项目的 observability 测试。[Asymmetric Actor Critic 原论文](https://www.roboticsproceedings.org/rss14/p08.pdf)

## 8. 数据、pair construction、控制集与 gates 的审计

### 8.1 有多少独立数据？

本次从 calibration sidecar 复算：fit=30 episodes/3,756 frames；selection=10 episodes/1,288 frames；合计40 episodes/5,044 frames。每任务拟合3条、选择1条。这里的40是 episode 数，**未验证40个独立 scene layout**；不能把相邻帧、反向 pair、counterfactual 扩增计为新独立场景。[E13]

后来的 global student 在全部40条上 refit，并以另外40条36--39做开发评估。与此不同，C0 历史训练使用 demo0--44、验证45--49；不能宣称整个项目都只用40条数据，更不能宣称40--49在项目历史中从未暴露。[E02/E08/E12]

因此N64的demo36--39是候选设计的开发评估，但不是C0 actor未见过的demonstrations；“held out”必须说明对哪个组件。历史controller init-state indices与HDF5 demo IDs是不同命名空间，不因数字相同就等价。最近阶段ledger记录零访问，也不能消除旧C0/WM训练与早期metadata incident的历史暴露。

Stage I/E 的 encoder 比较复用了同一个 demo34 选择集。不同模型固定一个 seed，没有跨训练 seed 的方差估计，也没有新 scene 上独立重复。大量试验共享相同选择集，会逐步让架构决策适应这十条轨迹；“每次预注册”并不能把共享开发数据重新变盲。

### 8.2 Cross-demo control 实际测了什么？

源码 `make_cross_demo_controls` 用相同 task 的循环下一个 demo，16 个均匀 anchor，对 target trajectory 搜索匹配：

- semantic match 要求同 On 且 `|ΔM2|<=epsilon`，选最接近者；不是粗略 normalized-time matching。
- semantic difference 要求 On 不同或 margin 达到0.05，**选满足条件但最不极端的 target**。这是刻意困难的近边界集，不是随机任务差异集。
- 每对添加反向项；反向项保证方向平衡，但不产生独立样本。
- I/E 只保留 source demo34，因此另一端是 demo31。未见场景对训练场景的测试仍有价值，但不是两个未见场景之间的泛化。

| Stage I/E 控制 | 报告行数 | 不重复无向图像对 | 主要限制 |
|---|---:|---:|---|
| semantic match | 32 | 16 | 5任务；task2=9对、task5=4对，其余3任务各1对 |
| semantic difference | 320 | 160 | 每任务16对；同一轨迹帧可复用 |
| 其中 M2 same-On | 266 | 133 | shared scene pair、近边界抽样 |
| 其中 On 不同 | 54 | 27 | 极少量跨目标状态对 |

以上匹配控制相对于 **T1** 是定义正确的；不是相对于完整 manipulation state。不能因为 grasp/robot state 不同就声称 T1 标签错了，也不能把 T1 等价当成控制价值必须相等。[E08/E13]

代码还有一个潜在复用风险：`compare_pairs` 默认用 a 所在 episode 的 referent 索引评价两端。本次以两端各自 referent 重新算 I/E cross labels，**mismatch=0**，所以没有证据将当前结果归因为该问题。未来若把不同指令/不同身份索引跨场景配对，必须改成双端点解析并另立测试，不能盲目复用。

旧 S2 的 cross-match 为94行，即47次正反成对记录；控制量仍小。不能直接将0.1681 vs0.15解读为稳定的总体机制阈值。对应 cross-diff 为1,280行，证据更广，但仍有 episode/帧复用。[E08]

建议未来以 scene pair 为 cluster，报告 task-macro、每任务有效 scene 数、差异幅度分层和不确定区间；只按 pair 行 bootstrap 会伪造精度。现有普通/CF 指标用 demo-block bootstrap是正确方向，但每任务4个block的重采样不能估计未见场景模式、训练seed或架构选择的不确定性。

### 8.3 Probes 能支持多强的结论？

referent probe 的输入是一张图与两条语言，标签是“是否换了同一个 physical referent”；不是识别哪个碗、检测框、像素位置或 track identity。每个 episode 的语言对标签在8帧重复，2,176行只对应272个 episode×language-pair 组合。[E09/E13]

本次增加一个**事后描述性负控制**：只用 demo31--33 的“无序语言对→多数标签”列联表，在 demo34 得到 **69.49%**，不使用图像、task ID 或状态。原 L6 81.48%仍高于它；这不证明 probe 忽略图像，却说明它的成绩不能全归为视觉 referent 恢复。该检查未选择新模型、未打开36--39，不是新的 confirmatory 结果。[E13]

M2 probe 共享 P1 单 slot 压缩，并非任意充分读出器。低分证明这种读出未泛化；不证明原始 patch 中不存在关系信息。DINO/SigLIP低于50%还提示可能有系统性错序/分布偏移；不能简单当作随机无信号。把输出事后翻号也不构成可部署修复。

### 8.4 Gates 分类

| Gate/要求 | 类型 | 审计意见 |
|---|---|---|
| causal alignment、禁止数据泄漏、部署输入契约 | scientifically necessary | 否则因果或泛化解释失效 |
| 真正同图换语言、反向一致性、明确 tie 语义 | scientifically necessary（构造层面） | 必须明确它测什么；不能用语言响应替代正确 grounding |
| 跨 scene evaluation、按 scene 聚类 | scientifically necessary（泛化主张） | 具体 pair sampler 不具有唯一必要性 |
| ordinary≥70%、CF≥70%、+4/+5 pp | arbitrary conservative threshold | 可作工程继续/停止规则，不是机制识别 |
| cross-match≤0.15、same-ref mean≤0.10 | reasonable heuristic，数值保守且人为 | 依赖温度和样本分布；需同时报告区分能力 |
| M2 probe≥65%、goal AUC≥0.8 | reasonable heuristic | 不能据此把 readout fail 映射为 information absent |
| teacher local non-tie≥30%、7任务可用 | reasonable heuristic | 只验证密度；不验证动作可控性或阶段覆盖 |
| proprio-only<90% | arbitrary conservative threshold | 低于90%不等于排除捷径，更不等于应禁用 proprio |
| imitation oracle改善≥10% | reasonable heuristic | 衡量 imitation 空间，无权代表 task-value headroom |
| 必须 RGB value 先过关才能诊断真实候选 | 无科学必要性的研究顺序 | 应把 oracle diagnostic 与 learned controller 授权分开 |

不建议回写旧 gate 或追认失败模型有效；建议在下一项不同问题的实验中换用与决策相连的 estimand。

### 8.5 Pairwise accuracy 与规划目标

70%普通 pair accuracy 对 candidate top-1 **既可能太严，也可能太弱**：模型可以错很多无关近 tie，但始终选对最好候选；也可以正确排序多数差候选，却总把一个危险候选排第一。平均 accuracy 无法分辨这两者。

CF direction 用 `(p_alt-p_orig)*(y_alt-y_orig)>0`，极小的正确方向变化也计成功；image-only固定输出因此必为0，而不是通常分类意义的50%。所以“比image-only高63.99点”证明非零目标相关语言响应，不能当作63.99点控制增益。same-ref false flip也按0.5两侧变号计数，极小概率变化可触发；应搭配幅度/拒绝区报告。[E08]

未来 primary metrics 应是 **best-vs-A0 可改善机会、top-1 regret、override precision、误干预率、按场景 rescue rate 与延迟成本**。T1 pair accuracy可以保留为感知诊断。

## 9. Hidden assumptions

| # | 假设 | 状态 | 原因 |
|---|---|---|---|
| 1 | 当前数据需要 obs[k]→a[k+1:k+9] | SUPPORTED | 时序契约与历史受控改善 |
| 2 | C0是可以比较的默认策略 | SUPPORTED | 明确 checkpoint 与可复用 rollout；并非每任务可靠 |
| 3 | N64有 imitation headroom | SUPPORTED | 68.10% MSE oracle gain |
| 4 | N64有真实 task-value headroom | UNTESTED | 新候选没有真实效用分支 |
| 5 | N64是最小必要预算 | PARTIAL | 只在测试过的构造/模仿准则成立 |
| 6 | H8足以看见有用动作差异 | UNTESTED | 对 reach/grasp/release 未证实 |
| 7 | T1密度足以证明control utility | CONTRADICTED | 定义上可对必要机器人动作完全不敏感 |
| 8 | 精确官方On已被teacher保留 | SUPPORTED | 直接使用contact与几何官方判定 |
| 9 | G2反事实含义全面正确 | PARTIAL | 几何可解释但缺独立counterfactual真值 |
| 10 | 当前RGB足以辨别initial referent | UNTESTED | teacher有历史身份，student无身份记忆 |
| 11 | 单帧恢复毫米级几何足够稳定 | UNTESTED | 未做观测噪声与输入上限 |
| 12 | P1小probe失败意味着backbone无信息 | CONTRADICTED | 推论不成立；有容量/优化/数据混杂 |
| 13 | 当前scene量足以学目标invariance | UNTESTED | 每任务3条fit，无scene scaling |
| 14 | counterfactual语言方向正确意味着动作选择正确 | CONTRADICTED | 大多为non-tie→tie，非竞争动作反转 |
| 15 | proprio应在deployment value中禁用 | CONTRADICTED | 捷径诊断不能推出非法观测；机器人状态本就合法 |
| 16 | 必须先独立学RGB value再研究WM | UNTESTED | 是协议安排，不是必要依赖 |
| 17 | WM预测global latent是正确靶 | PARTIAL | 有动作信号，但局部task-relevant差分弱 |
| 18 | 新机制的36--39仍是全局盲test | CONTRADICTED | 历史teacher/student/candidate研究已经使用 |
| 19 | 语义比较失败就说明prediction verification不值得 | CONTRADICTED | 失配触发重规划不必经过通用语义排序 |
| 20 | structured dynamics一定优于direct consequence | UNTESTED | 模块化解释性与累积误差之间尚未比较 |

**最危险的五项：#4 候选真实收益、#7 T1控制效用、#10 历史身份可由当前帧恢复、#13 场景覆盖足够、#16 value必须先于WM。** 其中#4/#7决定项目是否值得继续，其余决定怎样实现。

## 10. Failure / evidence tree

```text
目标：prediction-verified intervention 有净控制收益（尚未建立）
|
+-- ROOT DECISION BLOCKER：是否存在可利用的干预机会？
|   +-- N64 imitation headroom：已证
|   +-- N64 on-policy semantic / continuation headroom：未测
|   +-- 提前replan是否比继续A0有收益：新C0链未充分测
|
+-- UTILITY BLOCKER：哪些后果值得选择？
|   +-- official On：正确终态规则
|   +-- T1/M2：可解释关系代理
|   +-- reach/grasp/release与recoverability：缺失
|
+-- SECONDARY：观测、表征、统计可识别性
|   +-- initial referent记忆 / contact / Z：观测充分性未隔离
|   +-- fit只有30条episode：scene diversity未隔离
|   +-- 单slot、小head、单seed：readout/optimization未隔离
|   +-- cross-match只有16对：gate与总体结论混淆
|
+-- DOWNSTREAM：task-state dynamics / ranking / override / controller
|   +-- 旧WM有action信号，旧N3 rerank未收益
|   +-- 新N64/T1 bridge、WM、controller：未验证
|
+-- POSSIBLY FALSE DIAGNOSES
    +-- backbone缺信息（证据只到受限可读出失败）
    +-- language-insensitive（有真实语言效应，未达要求）
    +-- world modeling失败（旧预测目标/候选/分布的局部失败）
    +-- holdout仍全新（只对最近阶段未开启，不是项目级盲测）
```

## 11. Where the current research narrative may be wrong

历史闭环负结果也要保留正确尺度：旧D22在task0的C0/C1/C2均0/20，同时C2提前结束约90%的动作计划；这能诊断过度重规划，却不能在共同floor下识别收益差异。C0-v2之后的旧N3 C1-v2与匹配C0均44/100，也没有证明“所有verification无用”。参见[旧D22](d22_task0_ablation_audit.md)和E03。

1. **把语义标签的可定义性升级成控制价值的正确性。** On/M2有效描述当前关系，不等于会促进长期操作。
2. **把“检测不到”升级成“不存在”。** I/E共同采用单slot读出、同样少的scene与相同teacher；换encoder不是独立排除所有混杂。
3. **把 matching control 的条件含义说大了。** 当前只要求T1等价；未来若utility包含抓取状态，要求完全中立反而可能错误。
4. **把稀疏不均匀控制集的点阈值当作全局机制证据。** 16个图像对不足以支撑强泛化语言。
5. **把候选数量与任务搜索覆盖等同。** N64没有continuous×gripper组合；真实改善可能需要同时修正手和夹爪。
6. **把teacher历史记忆“免费”留在标签端，让student从当前帧猜。** 问题可能是输入不对称，不是encoder能力不够。
7. **把proprio的阶段预测能力当作必须消除的污染。** 对时间监督是捷径；对控制状态估计可能是必需信息。
8. **忘记原始问题。** 检测失配并提前replan与通用N64 semantic ranking是不同干预；多轮value研究不自动服务原始主张。
9. **把每一轮冻结理解为项目级独立验证。** 同一个开发集合反复引导新假设，不能重置其暴露历史。
10. **把工程门禁的必要性扩张为科学门禁。** 不应部署未验证evaluator；但无需先学会evaluator，才允许离线测它是否值得学。

camera历史提供一个具体反例：旧gate曾判projection block，后续紧致landmark验证发现正确坐标契约。问题曾在验证设计，而非相机本身。现在不能再次只尊重标签而不审查构造。[E11]

## 12. N64 判决：REVALIDATE

保留 C48_G15 作为冻结候选池和比较基准，停止称它“已经解决candidate generation”。[E04]

本次核查旧JSON：H8 raw MSE由0.028485降至0.009088，改善68.0957%；按维度独立oracle，gripper改善95.07%、translation31.95%、rotation27.84%。**各维度oracle可能选不同候选，不能把这些百分数当作同一候选的贡献分解。** raw action MSE还把尺度不同的控制坐标直接混合，因此尤其不能当作物理价值。

完整MSE oracle赢家中，structured gripper约39.64%、continuous约57.99%、A0约2.37%；不能简单说总改善全部来自gripper。候选协方差有效秩约3.89说明变化集中，**不意味着只有4个有效语义候选**。

需要比较的不是更多MSE：应对同一root state测A0、完整N64、预先固定的N8/N16/N32子集及gripper类别的真实后果；只有真实utility随N增加，才值得支付N64推理成本。类别分解先诊断，不在本轮重新搜索候选以追求过gate。

## 13. Value should come before World Model? ONLY A SIMPLE VALUE

需要先明确效用是什么；**不需要先训练一个高精度通用RGB value student**。可以先用 privileged state 与真实环境构成 oracle diagnostic，再研究自然观测下能否估计有用状态和动作后果。

建议的条件性架构：

```text
自然 RGB history + proprio + action history + 初始语言grounding
                         ↓
        带不确定性的最小task-state / referent memory
                         ↓
        小型 action-conditioned structured dynamics
                         ↓
    简单经验证的utility / continuation-aware consequence score
                         ↓
   相对A0的保守override；预计收益不足或不确定时保留A0
                         ↓
执行中校验真正相关的状态偏差，必要时重规划
```

structured state不自动使value简单：若utility审计表明reach/grasp需要长程后果，仍可能需要小的continuation model或阶段状态；不能换个表示就假定M2再次正确。

如果E类direct consequence model能以更低成本获得同等regret，接受它。只有预测结构本身提高解释性、样本效率、泛化或可校验性，才保留独立WM。论文不应为了标题硬保留无收益模块。

## 14. Major strategic options：按可证伪假设比较

以下成本为个人项目的预算级估计，未实测runtime；GPU日不是承诺。以现有8GB约束做小模型、缓存与分阶段运行。每条路线只有在前置oracle显示存在有用任务后才进入训练。

| Option | 科学理由与支持证据 | 反证/主要未测假设 | 最小实验、信息增益 | 算力/实现成本、歧义风险 | 证伪条件与controller连接 |
|---|---|---|---|---|---|
| A privileged-supervised object perception | G2/几何标签可得；oracle footprint有信息；定位可解释 | 过去area监督也未保留差分；2D中心未必足够；当前帧可能无身份 | 输入上限阶梯后，只学target+plate+contact必要项；高信息增益 | 中：缓存后小decoder/encoder适配，预计1–3 GPU日；实现中高；歧义中 | GT几何/身份仍不能支持utility则否定此目标；预测误差导致regret不降则停；给WM提供state |
| B trainable task-aligned visual | 三种frozen特征失败不排除task adaptation | 极少scene易过拟合；优化T1可能仍是错靶 | 相同输入、scene、reader，对冻结/末block/低秩适配做一项固定对照；中信息增益 | 中：约1–3 GPU日；实现中；歧义高 | 新scene与seed无稳定regret改善即停；直接评分或state encoder |
| C temporal/multi-frame + proprio | 身份记忆、共运动、抓取线索正是当前缺项 | 历史也可能只强化phase；init referent需要长期而非仅短窗 | 同scene count下RGB_t、RGB_t+proprio、4-frame+action、initial memory对照；高 | 中：缓存特征+小GRU，约0.5–2 GPU日；实现中；歧义中 | proper CF/branch控制下不优于单帧且上限无益则降级；作为belief state |
| D structured state + structured WM | 避免global latent保真任务；旧physical mediator部分有效 | contact dynamics难、误差累积；state是否充分未知 | 先GT current state→GT next state，配action shuffle与no-change；高 | 中：小MLP/ensemble，数据收集更贵；实现高；歧义中 | privileged dynamics都不能降低candidate regret则不做自然整链；预测后果接simple utility |
| E direct candidate consequence supervision | 直接对齐部署的同起点action选择；绕过通用state value | 分支数据昂贵、N64分布专用、恢复能力目标未定 | oracle分支上拟合(obs/history,A,l)→后果，与structured pipeline匹配成本；高 | 收集中高、拟合低中；实现中；歧义中 | scene-heldout不优于A0/简单gripper rule且regret不降即停；可直接rerank |
| F return WM with task geometry targets | 旧WM确实有action信号；目标应为task-relevant差分 | 只是换输出可能无法补足输入状态或off-policy覆盖 | 固定自然state encoder，用Δrelative pose/contact替代CLIP latent，检查差分regret；中高 | 中：约1–2 GPU日+分支标签；实现中高；歧义中高 | 绝对RMSE改善但decision fidelity不升则停；D的较小实现变体，非独立万能路线 |
| G more independent scenes | 当前3 demos/task fit明显少；大pair量不增独立性 | 没有scaling曲线，更多expert轨迹未必覆盖错误状态 | 新seed registry，固定架构做每任务4/8/16 scene nested curve；高 | 采集中高、拟合低中；实现中；歧义中 | 独立scene翻倍两次但目标decision指标平台且上限有余量，改结构/观测；补足泛化 |
| H recover original verification question | 原始目标是提前重规划，不要求完整value；旧C2过度中断 | 可能简单固定频率replan已一样好；需要净收益机会 | 同root下继续chunk vs立即重算C0 vs oracle interruption，固定后续policy；高 | 收集中、无需大训练；实现中；歧义低中 | clairvoyant interruption无收益或廉价schedule已饱和则放弃learned verification；回到原目标 |

当前不选A为默认主线。**首选的是跨A–H都要回答的oracle诊断；通过后偏向C+D/E，再由输入阶梯决定A/B/G投入。** D与F高度重叠，不应包装成两条彼此独立的“新方向”。

## 15. Top 5 next experiments：expected information gain / cost

这是排序，不是五个并行启动。只执行第1项之前，还需独立冻结新诊断范围；本审计不做执行。

| Rank | Experiment | Hypothesis tested | Expected info gain | Cost | Stop condition |
|---:|---|---|---|---|---|
| 1 | **真实N64分支utility上限：H8 T1 + 固定C0 continuation** | imitation≠utility？T1短视？候选本身无机会？ | 很高；能直接否定月级感知投入的前提 | 40 roots、2,560候选，无训练；主要成本是仿真与重放 | 无可复现branch identity则技术停止；无utility机会则不训练该N64 verifier |
| 2 | Observability / representation upper-bound ladder | 几何输入、深度、身份、历史、读出谁造成损失？ | 高；区分A/B/C的必要性 | 复用标签；少量GT投影/自然小读出；禁止大encoder搜索 | privileged目标都不能支持utility则回到目标定义；不是继续学视觉 |
| 3 | 固定小模型的scene-count×history对照 | 数据稀少 vs 单帧信息不足 | 高；可比较更多scene与更多时间信息哪个值钱 | 新独立scene，4/8/16 per-task curve；2输入条件 | 双翻倍无增益且history无增益，停止该模型族 |
| 4 | privileged structured dynamics vs direct consequence | 可预测状态充分性、分解是否必要 | 高但依赖前3项 | 分支数据复用、小MLP；少量GPU时间 | GT输入下action-aware模型仍不优于no-change/A0则停止自然WM链 |
| 5 | oracle/cheap replan intervention对照 | 原始verification是否有独立价值，还是固定重规划足够 | 高；评估H路线 | C0 continuation分支，可复用第1项root | oracle或learned verifier净收益不超过固定schedule则取消WM verification主张 |

### 上限阶梯的具体解释

阶梯应包括：A privileged完整object/contact state+语言；B privileged 2-D centers+语言；C privileged depth/3-D points+语言；D GT referent heatmap+RGB；E GT referent+predicted plate；F predicted referent+plate；G RGB+language。另加initial identity memory、短历史和proprio的受控分叉。

这些输入**不是严格嵌套的信息包含关系**，不能期待分数单调。必须固定相同scene split、监督、读出预算与输出指标；报告坐标尺度、可见性、接触与身份分层。A里直接放On/M2再重算T1只是sanity check，不能当作学习成功；真正要看完整状态能否支持candidate utility，以及压缩到哪些观测后丢失它。

若目的是立即定位当前RGB/T1失败，它可能是最高信息增益实验；**若目的是决定下个月项目投不投入，第1项候选utility上限更优先**。否则可能非常精确地找到了一个不值得预测的量的感知瓶颈。

## 16. The single best next experiment

**If I controlled this project, the next experiment I would run is: a calibration-only real-branch N64 utility upper-bound audit, comparing H8 T1 ordering with fixed-C0 continuation outcomes.**

中文：**用真实环境替代感知模型和WM，先问：完美知道候选后果，是否有值得选的动作？T1会不会选错？**

### 为什么先做

它同时移除视觉读出误差和WM预测误差。若依然无收益，训练更好的视觉student无法救该候选接口；若有收益却T1选错，应改目标/时域；若T1选对，才有理由把资源投入感知与动力学。仅测“max T1−A0 T1”会循环论证，因此**必须同时测固定后续policy的outcome**。

### Exact inputs / data / supervision / outputs

| 项目 | 精确定义（未来提案，尚未执行） |
|---|---|
| Root population | 全10任务，controller-development **init-state indices31--34**，每个init一个root，共40个；这是init索引，不冒称与demo同号一一对应 |
| Root selection | 对四个init依次预先固定C0 prefix长度0、32、64、96；按时间选择，不按T1、success或candidate好坏选择；报告实际阶段覆盖，不宣称覆盖所有manipulation phase |
| State access | fresh env + 相同seed + 精确prefix replay；同root的RGB/proprio/state与controller内部chunk状态一致。G2从初始scene选身份后保持，不能在root重新按source phrase选碗 |
| Candidate input | 冻结C0-v2、相同自然RGB/proprio/language、精确C48_G15，H=8、N=64；A0不变；固定seed规则；不重新挑candidate |
| Supervision | 无训练。真实On、M2、目标/盘相对几何、接触/抓取诊断；执行H8后用同一个冻结C0继续，到该episode总步数300；记录官方success首次命中与末端状态，二者分别报告 |
| Training data | none；没有训练集、拟合、温度选择或模型selection |
| Evaluation split | 仅上述calibration roots。demo36--39/对应受保留开发实验不启动；demo40--49仍禁止。后续确认须使用新注册scene，不把本轮40root说成独立确认 |
| Outputs | root×candidate的H8后果、完整continuation结果、身份/重放校验、运行步数/时间；A0、T1-selected、clairvoyant outcome-oracle的配对差别 |
| T1 selection | 先最大On，再最大M2；若A0同On且距最大M2≤旧epsilon，保留A0；否则在最大值epsilon内按固定candidate index打破tie。不得利用continuation选择T1候选 |
| Oracle selection | 仅离线看所有固定continuation，计算集合内成功机会；不是可部署policy，也不是全动作空间最优值 |
| Extra baselines | A0；冻结候选中的c0_binarized单一规则；按预先固定order抽取的N8/N16/N32子集；不事后搜索“最好的便宜规则” |

已经成功或invalid的root应保留其状态与原因，不静默替换；success可能造成ceiling，应分别报告全体与root尚未成功子集，后者是预声明分层。四种prefix与init ID绑定会混杂阶段和scene，此pilot不能估计阶段因果效应；若阶段结论重要，后续新scene做平衡设计。

若官方API在执行中产生reward/done，收集范围与隔离用途必须在新protocol中明确；这些是本报告建议新增的诊断监督，不会因为旧阶段禁用而自动合法，也不会因为本报告建议就自动运行。

### Primary metrics

1. **Opportunity/rescue**：A0 continuation失败但至少一个候选成功的root比例；全体和A0失败条件下都报告。每root64次尝试只算一个root，不按2,560个独立样本计算CI。
2. **T1 captured gain**：T1-selected的continuation success减A0；与集合内oracle可得增益并列，报告regret `max_j Y_j−Y_selected`。
3. **Override precision / harm**：T1选择非A0时，改善、持平、使A0本可成功变失败的比例；success相同下的稳定接触/耗时仅作预声明次指标。
4. **Short-horizon diagnostic**：H8 T1 non-tie机会、On变化、M2增量、分阶段密度，与continuation收益的一致性。不得把M2增益百分比当success增益。
5. **Compute**：总simulation steps、每root候选预算、自然模型未来允许的延迟。只能在真实硬件测后给wall-clock，不预先虚构GPU小时数。

本轮最多2,560条候选分支；fresh replay+到300总步，候选主体上限约768,000 env steps，另加默认轨迹、身份重复检查和初始化开销。可逐root流式保存紧凑状态与结果，避免为每条continuation编码或保存完整视频。先做重复A0的技术preflight，若不确定性不可忽略，oracle结果须作为随机环境下的单次optimistic上限，不能当稳定成功概率。

没有同root expert action标签时，不伪造expert-MSE与utility的相关系数。旧imitation oracle是在expert demos上，本轮是C0访问状态；分布差异本身就是要审计的桥，不能把两者假装一一配对。

### Failure criteria 与结果解释

技术失败：重放不能复原相同root、动作/时序不一致、身份丢失，则停止并报告implementation block，不推导科学负结论。

| 结果 | 含义与下一动作 |
|---|---|
| A：集合内oracle有明显rescue，T1也捕获大部分且少有harm | 支持候选与简单utility；做输入上限阶梯，随后优先history/state+dynamics；不等于部署已通过 |
| B：集合内oracle有rescue，但T1捕获差/有harm；或H8 T1几乎无差而continuation差异大 | 主要问题是utility/horizon；停止T1作为独立控制value，转continuation-aware consequence或阶段状态 |
| C：oracle也几乎无rescue | 该C0+H8+N64接口缺机会；先查root coverage与统计上限，再pivot候选/时域或replan问题，不能让感知为它背锅 |

作为**pilot资源分配线**，预先建议：至少4/40个rescue root且跨至少3任务才进入月级学习投资；T1需至少捕获一半这些rescue且harm≤1/40，才保留为第一版utility。它们是可修改但必须在执行前冻结的工程线，不是显著性证明。40root不足以一锤定音：0/40在独立同分布Bernoulli近似下95%单侧上限仍约7.2%，而实际按任务聚类可能更宽。

因此未通过pilot投资线意味着“不开始一个月模型工程”，不是宣告所有prediction verification失败。若需hard kill，使用后文更充分的新scene确认预算。

## 17. Three-step roadmap（未来1–2个月，条件推进）

| Step | 工作与预算 | Gate |
|---|---|---|
| Step 1 | 第1周：冻结并完成上述真实候选utility上限；先通过replay preflight，无新大模型 | 有跨任务可复现oracle机会；同时判定T1是否保留。没有机会则停止该接口，转H或结束 |
| Step 2 | 第2–3周：根据结果只选一条，做输入上限阶梯与最小state/history estimator；必要时注册新scene数据，比较direct consequence与structured dynamics | 在scene-disjoint开发集，决策regret优于A0/简单规则，误干预受控；GT→自然观测误差不会吞没可用headroom |
| Step 3 | 第4–8周：冻结唯一方案，独立新scene闭环比较C0、廉价replan/规则、learned intervention；报告成功、harm、latency、校验能力 | 达到预先约定最低实际收益，优于同预算便宜基线；若不达则触发kill/pivot，不再启动encoder巡回 |

这不是承诺八周产出positive result。Step1/2失败时应缩短项目并交付负结果，而不是用后续阶段填满时间。

## 18. Kill criteria 与 pivot criteria

hard kill针对**当前资源约束下的这个prediction-verified control方案**，不是声称世界模型在科学上不可能。建议在新protocol冻结最低实际收益 `δ=5 pp success` 和允许harm `≤1 pp`；这两个数来自项目投入判断，不追认旧gate。

### 三个 hard kill

1. **不存在可利用的干预上限。** 新独立scene确认，覆盖失败与关键阶段、至少约100个root并检查聚类有效样本；在一次预声明候选/时域修正预算之后，clairvoyant intervention相对C0的净收益95%上界仍低于δ。此时停止此候选/干预机制。区间过宽只能称inconclusive，不能假装满足kill；预算用尽则按资源理由暂停。
2. **没有超越简单基线的价值。** 固定频率replan或冻结gripper规则已捕获oracle可用收益；learned verifier在同延迟/仿真预算的独立闭环中，其相对便宜基线增益上界低于δ。停止“WM verification带来独立贡献”路线，即使组件RMSE继续改善。
3. **收益被错误干预与成本吞没，且达到修复上限。** 在最多两次预声明实现修正与一个月训练预算后，冻结方法的harm仍显著高于允许线，或在真实延迟预算下无法达到正净收益；停止这套learned override，保留C0与可复用研究证据。不能无限改threshold直到某个seed好看。

### 三个 pivot（不是继续原路线）

1. **Oracle有收益、T1选择差**：改utility/horizon或direct consequence；不要优先换encoder。
2. **3-D/contact/history上限明显好、2-D/当前帧差**：改观测/记忆/状态估计；不要要求frozen单帧恢复不存在的线索。
3. **上限足够、增加独立scene稳定改善而加capacity无益**：改数据采集与分布覆盖；停止以pair augmentation代替scene diversity。

另一个可接受出口：若N64 ranking无价值而oracle interruption有效，回到原始prediction-triggered replan问题；若两者都无价值，CHANGE THE CORE RESEARCH QUESTION或PAUSE，不必保护“World Model”标题。

## 19. Scientific story：现在能写什么？

LIBERO原本面向lifelong robot learning及知识迁移；这个项目使用Spatial子集开展控制机制研究是可以的，但10个固定任务与反复使用的开发scene不支持开放词汇、跨任务广泛泛化或SOTA主张。[LIBERO 原论文](https://arxiv.org/abs/2306.03310)

| Story | 可能的贡献 | 当前支持 | 缺失证据/新颖性风险 |
|---|---|---|---|
| Best-case | 在有限资源下，以可观测task-state和不确定性校验实现有净收益的prediction-verified intervention；用oracle ladder解释何时有效 | 当前只有可复用基础与负结果 | 真实分支oracle、独立scene控制改善、廉价基线、预测模块必要性全部要补；模块拼装本身不新颖 |
| Realistic | 决策导向评测：imitation headroom、普通视觉回归、语义pair score与candidate regret之间的系统性断裂；提供可复现诊断方法 | 绝对area与candidate差分断裂已有证据；其余桥部分缺失 | 需要用同root consequence把几个断裂真正连接，不能只堆失败表格 |
| Negative-result | 在明确数据量/读出容量/输入条件下，冻结表征对细粒度language-conditioned manipulation relation的可读出性限制 | 当前最接近已有证据 | 需加强scene/seed、采样控制、观测上限；不能声称“冻结视觉模型没有空间信息” |

**目前最可信的是有边界的negative-result story；最值得争取的是realistic decision-audit story。** 最初闭环agent story仍是假设，不是结果。单纯“换了很多encoder都不行”不足以构成强论文；把失败机制区分清楚、可复现地揭示proxy到decision的断裂，才可能形成科学贡献。是否达到NeurIPS/ICLR标准不能从现有材料保证。

也要考虑项目原始交付是可复现研究与硕士复试展示，而非不惜成本追逐顶会。可靠的C0因果修正、清楚的失败机制与一个有判别力的oracle实验，比继续堆积授权标签更有交付价值。[E01]

## 20. Final recommendation

**MAJOR ARCHITECTURAL PIVOT — confidence: medium。**

保留C0-v2、因果数据契约、G2与官方On、精确N64构造、已有replay校验与负结果。停止继续当前 frozen encoder→通用T1 student→等待bridge授权的线性路线。T1保留为诊断，N64重新验证，学习目标由真实候选后果和干预收益决定。

**下一笔算力花在“完美预测是否值得”上，而不是“还能否多拿几个T1 accuracy点”上。** 如果oracle显示值得，优先解决历史身份、最小状态与动作后果；如果不值得，及时改候选、时域或研究问题。

本次审计交付的是可改变研究顺序的判断与可复核证据。它没有执行上述下一实验，也没有修改任何现有运行授权。
