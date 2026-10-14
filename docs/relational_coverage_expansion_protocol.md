# 关系记录覆盖扩展：冻结执行协议

日期：2026-10-07。用户授权：按步骤顺序继续推进。机器协议：[JSON](../configs/relational_coverage_expansion_protocol.json)。

**目的：在原有40起点范围内检查关系事件的覆盖，暂不训练。** 事件标签v1可计算，但4起点中正例集中于2个起点，尚不足以判断模型可学性。

## 范围与预算

- 原先已完成的4个起点、12条唯一分支直接复用；补录注册表内剩余36个起点，保持注册表顺序，不根据success、事件或候选表现挑选。
- 每起点固定候选`0/49/50`，第一新增起点额外重复一次A0。新增109次完整运行、32,700个环境控制步为硬上限；不自动重试、不重置预算。
- 所有运行从原初始状态复现prefix，执行冻结H8候选，再由同一冻结C0按原契约续行到总计300步；prefix计入预算。
- 只读取已允许init-state数值行31–34，覆盖10任务；这是controller-init命名空间，不是HDF5 demo编号。不打开HDF5、不访问新的场景分区。
- 记录逐控制步物体/夹爪物理状态、接触及wrench、有序动作、官方On/success。success仅用于旧分支一致性检查，不用于标签或阈值选择。
- 不新增密集RGB。当前要补齐物理关系覆盖，自然观测训练另需同步采集协议。

## 技术停止条件

root/H8身份、完整动作哈希、首次及最终success诊断必须与旧分支一致。第一新增起点两次A0所有密集数组必须完全一致。之后逐分支重建接触掩码并检查有限数值、时间、偏移、动作。任意失败保留证据，停止后续步骤；不解释为关系学习科学失败。

新文件独立输出至`results/relational_coverage_expansion`，旧协议、脚本、结果不修改。冻结125个代码与数据文件的哈希；协议在启动之前写入。已有preflight数据早已暴露，本次不是盲确认。

## 自动执行顺序

1. `run_relational_coverage_expansion.py`完成限定采集。
2. `finalize_relational_coverage_expansion.py`核验新增与复用记录，用冻结v1计算两只碗的标签，输出40起点/120唯一分支的覆盖报告。
3. 停在`awaiting_scientific_coverage_review`：下一项训练必须依据实际覆盖、身份反例和按root分组情况设计，当前后台任务不会自动训练。

驱动：[continue_relational_coverage.py](../scripts/continue_relational_coverage.py)。监视[采集状态](../results/relational_coverage_expansion/runtime.json)与[整体步骤状态](../results/relational_coverage_expansion/pipeline.json)。日志位于`/tmp/relational_coverage_expansion_driver.log`。

后处理输出：`results/relational_coverage_expansion/coverage.json`和`docs/relational_coverage_expansion_audit.md`，仅在完整采集及核验成功后生成。

## 解释边界

40起点仍全部属于历史开发数据。多分支、相邻窗口和共享prefix不算新独立场景；按root/task报告正例覆盖，不按帧数制造精度。正例多不等于训练/开发均覆盖，也不等于可泛化。

固定三个候选是本次有界覆盖筛查，不意味着动作永远限于C0，也不能证明已覆盖所有失败方式。若错误物体、滑落、未抓起等仍缺失，下一步先设计相应数据来源，不用缺失类的全负预测宣称鲁棒性。
