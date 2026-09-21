# C0 数据契约（LIBERO-Spatial）

版本：`2`。本文件在实现 C0 前冻结，用来防止训练、验证、在线控制之间出现输入不一致或 oracle 泄漏。

## 范围与样本单位

- 基准：官方 `libero_spatial` 的 10 个任务，数据版本固定在 `configs/source_versions.yaml`。
- 训练数据仅为官方 demonstrations；一个训练样本是 `(task_id, demo_id, t)` 的 transition。
- C0 预测从时刻 `t` 开始的连续 `H=8` 个 7-D 动作；不足 8 步的末尾 transition 不产生样本。训练步长为 1。
- 训练/验证按完整 demonstration 划分，而非随机 transition 划分：每个任务 `demo_0`–`demo_44` 为训练，`demo_45`–`demo_49` 为验证。验证集只能用于模型选择和阈值选择，绝不用于官方测试成绩。
- 在线评测每个任务执行 20 个固定 episode，种子在 `configs/eval_seeds_spatial.json`。C0/C1/C2 必须读同一份文件；训练随机种子固定为 `[11, 23, 47]`。

## 策略可见输入

下表同时适用于离线数据加载和在线 `Policy.act(observation, language)`。任何 C0 及之后的策略实现都只能由此表中的输入构造网络特征。

| 逻辑字段 | demonstration HDF5 字段 | 在线 observation 字段 | dtype / shape | 用途 |
|---|---|---|---|---|
| RGB | `obs/agentview_rgb[t]` | `agentview_image` | `uint8[128,128,3]` | 唯一的外部视觉输入 |
| 任务语言 | LIBERO task metadata 的 `task.language` | `task.language` | UTF-8 string | 冻结语言编码器的输入 |
| 末端位置 | `obs/ee_pos[t]` | `robot0_eef_pos` | `float64[3]` | 机器人本体感觉 |
| 夹爪状态 | `obs/gripper_states[t]` | `robot0_gripper_qpos` | `float64[2]` | 本体感觉 |
| 关节状态 | `obs/joint_states[t]` | `robot0_joint_pos` | `float64[7]` | 本体感觉 |
| 行为标签 | `actions[t:t+8]` | 不适用 | `float64[8,7]` | 监督目标；不作为输入 |

离线 `ee_ori` 是 3-D 表示，而在线 MuJoCo observation 为 4-D 四元数。经同一仿真状态核验，两者不能在不额外标定的前提下直接等价；因此 C0 **不使用朝向字段**，最终 `proprio_dim=12`。未来若要加入朝向，必须先保存从在线四元数到离线目标表示的可复现标定和单元测试，再提高契约版本并重新训练 C0/C1/C2。

为避免相机语义不一致，C0 只使用固定外部相机 `agentview`；不使用 `eye_in_hand_rgb`。官方 demonstration `agentview_rgb` 与在线 `agentview_image` 已同为正向 RGB，因此两侧均**不得竖直翻转**；只保持 `uint8` RGB 后交给同一 CLIPProcessor，并在特征 manifest 中记录。

## 严禁作为策略输入的字段

以下字段可能可用于环境重放或离线完整性审计，但不得被模型、归一化器、候选筛选器或 world model 读取：

- `states`（92-D MuJoCo 全局状态）及 `data/demo_*/model_file`、`init_state`；
- 任何对象位置、四元数、相对位姿、物体名称编码或 BDDL goal 条件；
- `robot_states`：它不属于已声明的最小本体感觉字段，先整体禁用；
- `obs/ee_ori`：不是 oracle，但当前 C0 禁用它，以避免 3-D 离线表示与在线四元数之间的未验证转换；
- `rewards`、`dones`、`env_infos`、`check_success()` 返回值和所有 success 标签；
- 测试 episode 的初始状态、随机种子、未来 RGB 帧、未来动作或未来 latent；
- `obs/eye_in_hand_rgb`，除非未来新版本的该契约显式新增并对 C0/C1/C2 重新公平训练。

审计脚本会读取上述字段的**元数据/shape**来发现数据损坏，但永不将其导出到训练特征。

## 动作与执行约定

动作 `a ∈ R^7` 直接沿用 LIBERO demonstration 中的 OSC-POSE 控制格式。训练前必须从**训练 demonstrations**单独计算每维归一化统计；验证/测试不得参与统计。在线动作裁剪到环境允许范围 `[-1, 1]`，并记录裁剪比例。C0 一次输出一个 `8×7` 动作块；环境以 receding horizon 执行，默认在下一次规划前最多执行该块。后续 C1/C2 的候选数、重排和提前中断只可改变“选哪个块/何时丢弃块”，不得改变动作尺度、观察字段或测试种子。

## 必须写入每个训练样本/特征 manifest 的元数据

`task_id`、`task_name`、`task_language`、`demo_id`、`t`、`chunk_horizon`、`source_hdf5_relative_path`、数据集 revision、数据契约版本、图像预处理版本、训练/验证 split。这个元数据用于重建样本来源，不送入网络。

## 审计命令

```bash
cd /home/luo/embodied-libero-wm
env -u PYTHONPATH .venv/bin/python scripts/audit_spatial_dataset.py \
  --output artifacts/stage1/data_audit.json
```

该脚本只扫描 HDF5 结构和动作数组，输出总 transition 数、可用 `H=8` 样本数、动作范围与 schema 检查结果；不会复制 RGB 数据。

冻结特征由 `scripts/extract_features.py` 生成。它下载固定 revision 的官方 OpenAI CLIP 权重到项目忽略的 `.cache/huggingface/`，并将训练/验证分片、仅训练集动作统计、来源 hash 写入 `artifacts/stage1/features/clip_vit_b32_h8_p12_upright/feature_manifest.json`。该特征集仍不包含 oracle 字段。
