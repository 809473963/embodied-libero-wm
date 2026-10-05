# LIBERO-Spatial 解决方案与执行计划

日期：2026-10-04。

依据：[Full Project Scientific & Strategic Audit](full_project_scientific_strategic_audit.md)。

状态：**阶段1与阶段2A均于2026-10-05完成。** 阶段1判决B：40个root与2,560条主分支完成；38个有效未完成root中，oracle救回7个、覆盖6任务，T1仅救回1个；G1a/b/c通过，G1d失败。见[阶段1报告](candidate_utility_oracle_audit.md)。阶段2A完成1,280条N32输入补录及6次fit，判决 `NO_USEFUL_DEVELOPMENT_READOUT`：两组threshold0 ensemble在9个有效开发root均救回1个、harm1个；开发选定阈值全部保留A0。见[阶段2A报告](continuation_utility_observability_audit.md)。2B、WM、新scene与learned controller均未启动。本文保留原设计内容，历史运行授权不变。

下一步按B分支推进：固定后续C0的continuation outcome为主效用，相对A0的gain/regret为选择指标，T1只保留诊断用途。先注册阶段2的输入上限与scene split，再比较完整task-state/contact/action和自然history/proprio输入。当前40个pilot root均为开发证据，不能再作为独立确认集。N32在本pilot保留全部7个oracle rescue，可作为后续预算对照；这不是N32已在新scene验证的结论。

已完成的具体实施见[阶段2A：续行效用与输入上限诊断](continuation_utility_observability_plan.md)。当前先诊断fit-development差距及scene覆盖、状态和分支目标，不进入动作执行前的direct consequence训练。真实未来H8输入仅供离线诊断；本次负结果只约束当前输入、有限读出与数据，不能升级为backbone信息不存在。

## 1. 解决方案总览

**把主线改为：先验证真实干预收益，再确定效用与必要观测，最后学习后果并验证闭环。**

保留 C0-v2 为默认策略；保留 N64 为待验证的候选池；保留 G2、On、M2 为诊断工具。停止继续扫描 frozen encoder，也不立即训练 localization network。

三个阶段只沿结果支持的一条分支推进：

```text
阶段1：真实分支oracle
       是否有好动作可选？T1能否选中？
          |
          +-- 无机会/证据不足 → 检查覆盖；仅一次候选或时域修正，或停止
          |
          +-- 有机会，T1错 → 改utility，不继续蒸馏原T1
          |
          +-- 有机会，T1可用 → 暂留简单utility
                         ↓
阶段2：输入上限 + 最小可学习模型
       哪些状态必须观测？自然输入能否保留决策信号？
                         ↓
阶段3：保守干预闭环
       是否优于C0和廉价规则，且满足延迟与误干预要求？
```

**本计划的成功标准是得到可用的控制改善，或及时证明当前方案不值得投入。不是让某个旧 gate 变成 PASS。**

## 2. 问题与对应修复

| 已识别问题 | 解决动作 | 验证方式 |
|---|---|---|
| imitation headroom 被误当 task utility | 在同起点执行真实候选，并固定后续 C0 | oracle rescue、相对A0的净收益 |
| T1忽略必要操作阶段 | 同时记录H8 T1和完整continuation结果 | T1捕获机会比例、harm、regret、阶段分层 |
| 单帧缺历史身份/接触信息 | 输入上限阶梯加入initial memory、历史和proprio | 相同数据/预算的受控输入对照 |
| 少量scene、大量相关pair | 按完整scene划分，增加独立scene，不靠pair扩增补数量 | scene learning curve、scene级区间 |
| probe失败被归因为backbone缺信息 | 将观测、监督、读出容量分别控制 | GT state、GT localization、自然输入之间的性能落差 |
| 普通accuracy与最终决策脱节 | primary metric改为候选选择效用 | best-vs-A0、regret、override precision/harm |
| 全套WM可能过度复杂 | 先建立direct consequence基线，再检验structured dynamics必要性 | 同数据、同候选、同推理预算比较 |
| 新模型不断消耗开发集 | 独立注册新scene、一次冻结后评估、限定修正次数 | split registry、访问记录、终止判据 |

## 3. 数据和工程边界

### 3.1 保持冻结

- C0-v2 checkpoint、因果动作契约、H8、图像方向与归一化。
- 原 C48_G15 的64个候选及其构造规则；第一阶段不重新搜索候选参数。
- G2、persistent referent、官方On、M2、epsilon。
- 历史协议、结果、标签和旧authorization全部保留。

### 3.2 新诊断与旧链分离

后续若实施，单独建立 `candidate_utility_oracle` 命名空间。协议应明确新增的只读诊断监督：真实物体状态、官方success、分支后果；这些不得进入部署模型输入。

本阶段不学习evaluator、不训练WM、不启用learned controller。success可用来评价固定分支，不用来事后选择root、调整候选或筛掉失败任务。

禁止打开demo40--49的数组或组元数据。第一阶段也不启动新的demo36--39评估。controller init-state IDs和HDF5 demo IDs分别记账，绝不因编号相同就视为相同样本。

旧36--39只保留为已暴露的开发证据，不再称全项目盲测。后续新的确认集使用显式注册、独立生成的场景；注册不依赖枚举受禁demonstrations。新场景测试与原官方固定评测分开报告。

### 3.3 执行前冻结的内容

root清单、seed、prefix长度、候选子集、success定义、终止行为、continuation policy、tie-break、统计单位、预算、结果到决策的映射，必须在第一次效用结果出现前写入新protocol并形成git快照。

这是一次性实验准备，不是每处理一个root都再请示。当前请求只生成计划，所以本轮不创建运行授权或执行分支。

## 4. 阶段1：Candidate Utility Oracle

### 4.1 要回答的三个问题

1. 在C0实际访问状态上，N64是否存在能把失败改为成功的候选？
2. 若存在，H8的T1能捕获多少，是否会破坏A0本可完成的任务？
3. 更小候选池或固定gripper规则是否已获得同等收益？

这轮消除学习式感知和WM误差；oracle只代表“此候选集合 + 此后续策略 + 此预算”的上限，不能代表所有策略的最优值。

### 4.2 Root与执行定义

| 项目 | 计划定义 |
|---|---|
| 场景 | 10任务 × controller-development init31--34，共40个root |
| 默认trajectory | 冻结C0-v2、H8执行，保存原始动作prefix |
| Root时间 | `{0,32,64,96}`步；每task各使用一次；按task编号循环置换与四个init的对应关系，在清单中逐项固定 |
| Root选择原则 | 不按T1、success、误差或候选结果选择；循环置换只减少跨任务的init/时间固定关联，不消除单scene只有一个时间点的限制 |
| 分支恢复 | fresh environment + 相同seed + 精确prefix replay；不默认mid-episode state restore足够 |
| 目标身份 | 初始scene以G2确定referent，所有分支保留其physical identity |
| 候选 | 每root一次生成完整冻结C48_G15，保存动作和hash，再复制到各分支 |
| Continuation | 执行候选8步后，从分支自然观测重新调用同一C0；清空旧动作缓存，后续仍每8步重算；总episode预算300步 |
| Primary outcome | root之前尚未成功的episode，root后截至300步是否至少一次命中官方success；所有候选相同定义 |
| Secondary outcome | H8的On/M2、最终On、成功后是否失去目标关系、耗时、接触诊断 |
| 提前终止 | root之前已成功标记为ceiling，保留记录；真正不可继续的环境终止/异常单独记录。不能把无法取得H8状态的分支补成T1标签 |

对于success即终止还是继续到固定horizon，runner必须在preflight确认实际环境契约，并在效用收集前冻结。若支持成功后继续，H8与终态诊断保留；若不支持，标记缺失，不改变primary success定义。所有方法必须一致。

主分析先报告全部40个root的状态账，再报告有效且root前未成功的population；覆盖不足时报告inconclusive，不静默替换root。

### 4.3 修正小候选池对照的潜在混杂

**不能直接取原候选列表的前8/16/32项。** 原列表先排连续候选、后排gripper；简单截断会把“N的作用”与“是否包含gripper”混在一起。

从同一完整N64中预先固定嵌套子集：

| 子集 | A0 | 连续候选 | Gripper候选 | 合计 |
|---|---:|---:|---:|---:|
| N8 | 1 | 4（2对） | 3 | 8 |
| N16 | 1 | 10（5对） | 5 | 16 |
| N32 | 1 | 22（11对） | 9 | 32 |
| N64 | 1 | 48（24对） | 15 | 64 |

连续项按冻结的antithetic pair顺序选取；gripper按冻结15项顺序选取，前三项包含`c0_binarized/all_negative/all_positive`。不生成新动作、不按结果重排。所有子集分析复用同一次N64收集，不增加候选simulation。

另单列A0与`c0_binarized`，避免复杂planner的收益其实来自一个廉价二值化修复。上述子集是本计划对审计建议的明确化，尚未注册或执行。

### 4.4 T1选择规则

先保留On最大的候选，再找其中最大M2。若A0的On相同且M2距该最大值不超过旧epsilon，则保留A0；否则在最大值epsilon范围内按固定candidate ID选择。

规则只看H8结果，不看continuation；有缺失H8状态的root不能混入完整T1排序主分析，应单列并解释。无需先训练S2或进行N64 pairwise tournament。

### 4.5 指标与统计

令 `Y_ij` 为root i的candidate j在固定continuation下的success，j=0为A0。

| 指标 | 定义与用途 |
|---|---|
| Oracle opportunity | `max_j Y_ij - Y_i0`；测是否存在可救回机会 |
| Rescue count/rate | A0失败且集合内至少一候选成功；同时报全population和A0失败条件下比例 |
| Selected gain | `Y_i,selected - Y_i0`；正确改善与破坏分别统计 |
| Selection regret | `max_j Y_ij - Y_i,selected`；无需以T1为最终真值 |
| Capture fraction | rescue roots中被T1-selected救回的比例；与净gain分开 |
| Override precision | override中改善的比例，持平和伤害分别报告；无override时为N/A |
| Harm | A0成功、selected失败；同时报占全部有效root及override的比例 |
| N与类别曲线 | N8/N16/N32/N64及固定gripper baseline的oracle/selected收益 |
| 诊断指标 | T1非tie、margin幅度、阶段与task分层、运行时间 |

以root/scene为单位，报告每task结果与macro汇总。2,560个candidate rows不是2,560个独立样本。由于每task只有4root，区间仅作pilot不确定性描述，不声称估计所有任务分布。

若环境或policy存在随机性，单次best-of64有乐观选择偏差。先检查重复A0；对拟用于正面机制结论的分支，在协议预声明的验证预算内复验，并把探索与复验分开。若重复不稳定，不得把单次oracle解释为成功概率上限的可靠估计。

### 4.6 第一阶段gate

以下是**拟议资源分配线**，必须在执行前冻结，不能见结果后改。

| Gate | 条件 | 未满足怎么办 |
|---|---|---|
| G1a 技术有效 | 同root重放一致、candidate hash一致、动作时序正确、referent一致、无越界访问 | 停止收集，修实现；不作科学负结论 |
| G1b 覆盖可解释 | 至少30个有效未完成root、至少8任务；说明reach/grasp/transport/place实际覆盖 | 标记pilot覆盖不足；只允许一次预注册覆盖补充，不直接kill |
| G1c 候选值得继续 | 至少4个rescue roots，跨至少3任务；机会不完全由不稳定重放造成 | 不开始月级感知训练；检查覆盖，转一次候选/时域或replan诊断 |
| G1d T1可暂留utility | 捕获至少一半rescue，harm最多1root，且净gain为正 | 保留T1诊断；改用continuation-aware目标 |

同时公布所有分母和区间。4个rescue不是显著性证明，0个rescue也不证明机会不存在。若廉价gripper规则已经捕获全部可用机会，优先保留该修复，停止为了N64而训练复杂evaluator。

### 4.7 预算与实施工作包

40×64=2,560条分支；每分支fresh replay加执行至episode总300步，主体上限约768,000 env steps，另加默认轨迹、preflight和复验开销。先以少量技术preflight测实际step throughput，再给wall-clock预算。收集紧凑状态和结果，只保留少量事先指定的诊断视频。

| 顺序 | 工作包 | 交付与验收 |
|---:|---|---|
| 1 | 冻结新protocol、明确diagnostic授权范围 | 输入/输出白名单、root清单、哈希、预算、gate完整 |
| 2 | 编写纯函数候选子集和T1 selector | 嵌套性、antithetic完整性、A0优先tie、缺失处理单元测试 |
| 3 | 抽取已有fresh-replay能力，做新runner | 不直接运行旧脚本的历史默认分区；env/policy动作缓存复原测试 |
| 4 | 技术preflight | 不开展gate选择；重复A0一致性、计步和success行为明确 |
| 5 | 可恢复的N64收集 | 每root完成标记、原子写入、hash校验；断点续跑不覆盖完成结果 |
| 6 | 独立analysis入口 | 只读收集结果，计算机会/捕获/harm/regret及完整账目 |
| 7 | 写结果和decision | A/B/C/inconclusive明确落一类；不自动启动下阶段 |

建议新增文件（均为待实现，不表示现在已存在）：

```text
configs/candidate_utility_oracle_protocol.json
configs/candidate_utility_oracle_roots.json
src/probes/candidate_utility_oracle.py
scripts/collect_candidate_utility_oracle.py
scripts/analyze_candidate_utility_oracle.py
tests/test_candidate_utility_oracle.py
results/candidate_utility_oracle/{preflight,manifest,metrics,decision}.json
docs/candidate_utility_oracle_audit.md
```

可复用的已有代码：[候选构造](../src/probes/expanded_candidate_contract_restart.py)、[fresh replay审计](../scripts/audit_local_candidate_counterfactual.py)、[G2](../src/probes/robust_grounding.py)。复用函数，不继承旧实验的数据访问范围或默认执行行为。

## 5. 阶段1之后只选一条路线

| 结果 | 主行动 | 暂停的工作 |
|---|---|---|
| A：有真实机会，T1捕获有效 | 进入观测上限；优先最小state/history + 简单utility | 不再进行无边界encoder扫描 |
| B：有真实机会，T1错误 | 首先定义continuation-aware consequence；比较H8后果与更长时域 | 停止以原T1为唯一训练目标 |
| C：机会少且覆盖有效 | 一次有限candidate/horizon诊断，或回到提前replan问题 | 不训练新perception/WM为无机会接口服务 |
| D：覆盖或重复性不足 | 只修相应诊断设计/工程；保留inconclusive | 不扩张科学结论 |

候选/时域修正最多一次，必须针对结果揭示的机制，例如同时改变连续动作和gripper，或延长干预可见时域。不能同时放开N、H、候选分布和utility做大搜索。

## 6. 阶段2：观测上限与最小学习方案

### 6.1 先定位必要信息

在已确认有用的utility上建立输入对照；T1仍作为辅助诊断，但不再独占主指标。

| 输入条件 | 要区分的解释 |
|---|---|
| 完整privileged object/robot/contact state + persistent identity + action | 是否存在可学的状态到后果映射；不是把On/M2直接输入再预测自己 |
| GT 2-D projected centers + identity + action | 单纯平面位置是否足够 |
| GT 3-D relative geometry/contact + action | 2-D不足是否来自深度/接触缺失 |
| GT target heatmap + RGB/proprio + action | 主要损失是否来自找错目标 |
| 自然RGB当前帧 + language/proprio + action | 与历史单帧方案可比的合法部署基线 |
| 自然RGB短历史 + proprio/action history + initial identity memory | 历史是否补足身份、抓取和运动信息 |

它们不是严格嵌套，不能用“排名单调”作为gate。所有比较保持相同scene split、监督及尽可能匹配的读出预算；普通T1精度、contact/geometry误差与candidate regret一起报告。

### 6.2 如何按结果修复

- **GT3D/contact好，GT2D差**：重点是几何/接触，不把只预测2D框当解决方案。
- **GT heatmap好，自然输入差**：才投资privileged-supervised localization；保留初始身份与跟踪。
- **历史好，当前帧差**：使用小GRU/时序聚合和initial memory，避免要求当前RGB推断历史身份。
- **上限好、小数据差，scene增加改善**：扩独立scene；不能只扩pair。
- **完整state也无法在新scene支持utility预测**：先检查目标、action覆盖及随机性；不把问题推给视觉。

### 6.3 首先建立哪个学习模型

**先做小型direct consequence基线，再决定是否需要structured WM。**

```text
baseline:
  RGB/history + proprio + language + candidate actions
      → 预测continuation outcome / task-relevant consequence
      → 相对A0选择或拒绝干预

conditional structured variant:
  自然观测 → task-state belief
      → action-conditioned state change/contact mode
      → 已验证的simple utility或continuation head
```

若阶段1支持T1，可在structured variant中使用T1相关状态；若阶段1否定T1控制意义，不得仅因几何可预测就再次让M2主导选择。

MSE/交叉熵用于训练，但模型选择以开发scene上的regret、相对A0净gain和harm为主。预测success必须报告校准；若拒绝区按不确定性设置，只能在开发集选择并冻结。不得用privileged state作为自然部署模型输入。

### 6.4 新数据方案

pilot数据全部视为已暴露的开发资料，可用于debug或训练，不能再充当确认集。

建议首批新收集以每任务4个fit scene、2个development scene为起点；仅在学习曲线表明值得时将fit扩至8、16个。scene种子与生成规则先注册，完整episode及所有分支只属于一个split。没有必要为每个新scene都跑N64：由阶段1结果决定并冻结足够的小候选池，但记录其与完整N64的区别。

最终确认scene单独保留，不因architecture失败反复开启。按可用simulation预算计算实际数量；如果区间无法排除有意义的收益/损失，就如实报告inconclusive，不把大量pair当补救。

### 6.5 阶段2gate与预算

**G2：自然输入模型在scene-disjoint开发集上，优于A0及冻结的廉价规则，降低candidate regret，且误干预与延迟处于预冻结预算内。** 同时检验正面结果不只来自单任务或一次seed。

首轮只比较两个有机制差异的候选方案，每方案3个训练seed，最多6次主要fit；不做宽泛超参数搜索。优先缓存、低维state和小网络，适配现有8GB条件。两轮scene数量扩充均无改善时，停止该模型族，依据输入上限决定换观测还是结束。

没有足够开发证据通过G2，不进入在线learned override。

## 7. 阶段3：闭环收益验证

### 7.1 必需对照

| Arm | 作用 |
|---|---|
| Frozen C0-v2 | 所有收益的默认参照 |
| 廉价规则 | 阶段1冻结的gripper规则或固定频率replan；控制简单修复的贡献 |
| Learned intervention | 唯一冻结的最终方案，自然观测输入 |
| 必要消融 | 若使用WM，移除WM/预测校验，其余尽量相同；检验预测模块是否贡献收益 |

所有arm用相同scene、seed、300步预算和一致success定义。比较同等延迟/算力成本；默认C0动作不能因为增加模型调用而获得不公平不同的执行语义。

干预采取保守策略：只有相对A0预期收益超过开发集冻结门槛且不确定性可接受时才override；否则执行A0。执行中校验是否提前replan是额外机制，必须单独有证据，不能在ranking试验后自动加上。

### 7.2 最终验收

拟议最低实际收益为相对C0 **+5 pp success**，允许harm目标 **≤1 pp**；这些是投入决策线，不是保证可由小样本证明的统计门槛。

报告paired success差异及区间、task-macro、A0-only/learned-only成功数、override频率、harm、延迟和失效模式。只有区间与样本量支持时才能声称达到收益或harm上界；点估计达标不等于完成确认。

必须优于廉价规则，才能将收益归于复杂预测机制。若只证明gripper二值化有效，应交付该简单修复，不把结果包装成WM成功。

## 8. 时间安排与停止规则

| 时间预算 | 交付 | 决策 |
|---|---|---|
| 第1周 | protocol、preflight、pilot分支与utility报告 | 决定有无学习价值，T1保留/降级 |
| 第2–3周 | 输入上限、最小模型、新scene开发证据 | 只选一个有支持的方案，或停止 |
| 第4–8周 | 冻结方案、独立scene闭环、消融与完整报告 | positive/negative/inconclusive，结束本轮而不是无限续训 |

时间为工程预算，不是实测运行承诺；实际收集时间以preflight吞吐量更新。若第一阶段否定投资前提，不消耗后面的训练预算。

**三个硬停止条件：**

1. 在一次预声明候选/时域修正后，充分新scene上的oracle净收益上界仍低于最低实际收益：停止此干预接口。
2. 廉价规则已获取可用收益，learned verifier没有有意义的额外收益：停止复杂WM/verification主张。
3. 最多两次实现修正、约一个月模型预算后，harm或真实延迟仍使净收益不成立：停止learned override。

如果数据量不足导致区间过宽，结果是inconclusive；预算耗尽可以暂停，但不能改写为科学上“已证无效”。

## 9. 下一次实施的具体起点

**只启动阶段1的protocol与runner实现。** 第一交付是可审查的新诊断协议、40-root清单、候选子集/selector测试及replay preflight设计。随后按冻结规则完成真实分支收集和分析；结果决定阶段2走哪条路。

当前不应直接开始：新的encoder训练、T1 redistillation、完整object-centric WM、N64 learned reranker或新的闭环controller。

整个计划要解决的是：**先找到值得预测和干预的差异，再为这些差异学习最小必要表示。**
