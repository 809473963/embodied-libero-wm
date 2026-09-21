# Pre-D13 Foundation Audit & Fix

审计日期：2026-09-22。范围是 D12 C0-v1 的可复现性、LIBERO 的时间语义、离线/在线 representation parity，以及不会改变 C0 架构的必要工程修复。本轮**没有**开始 D13 world model、C1 或 C2；没有增加 GRU、第二相机、`K>1`，没有重抽特征、重训 C0，也没有根据 fixed evaluation episode 调参。

## 结论

**GO：可以进入 D13 的设计/实现阶段，但 D13 的 transition 数据契约必须使用 `z_obs[k], action[k+1], z_obs[k+1]`，不可复用 C0 的同索引 imitation-learning 标签。**

这个结论基于固定 LIBERO converter 的逐行因果语义，以及 3 task × 3 demo × 3 timestep 的独立 replay 对照。回放的绝对误差受到未写入 HDF5 的 robosuite controller-goal 内部状态影响，因而不是唯一证据；但它与 converter 自己的 `states[j+1]` playback check 一致，并支持 shifted action 的物理解释。

进入 D13 不等于多任务 C0 已经足够好。当前 C0 的 all-task 结果仍是 17/200（macro 0.085），而本轮 validation-only 审计确认 task 间 validation MSE 与偏差差异明显。这是后续的 baseline 解释与世界模型比较需要保留的事实，不是本轮改 C0 或调 test 的理由。

## 1. D12 冻结快照

冻结发生在任何 C0 源码修改前。

| 项 | 值 |
|---|---|
| root commit | `28206b4b1623657b32df366f71727e2d5c4a0b0f` |
| commit message | `D12 C0 baseline before world model` |
| annotated tag | `d12-c0-baseline` |
| pinned LIBERO gitlink | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| upstream URL | `https://github.com/Lifelong-Robot-Learning/LIBERO.git` |
| dataset revision | `f13aa24a3da8c43c7225569f28c562979fa0e35a` |

该 initial commit 纳入源码、配置、tests、README、docs、LIBERO gitlink、轻量环境/数据/特征 metadata，以及 JSON result summaries。冻结前通过 staged-path audit 确认没有将 `data/`、`.venv/`、`.cache/`、任意 `.hdf5` 或 `.pt` 纳入 Git；提交后工作树只有刻意忽略的运行时/大对象，未有未跟踪源码。

未入 Git 的 D12 关键对象在 [d12_excluded_artifact_manifest.json](../configs/d12_excluded_artifact_manifest.json) 中逐文件记录 `path`、`bytes`、`sha256`、`run`、`config_path`。它包含：10 个官方 dataset HDF5（总计 6,237,050,764 bytes）、三组 feature stores，以及所有 C0 / probe checkpoint。环境和模型下载 cache 不入 Git，但环境版本、LIBERO commit、dataset revision、CLIP revision 记录在 [source_versions.yaml](../configs/source_versions.yaml)。

特别地，canonical feature store 的哈希为：

| 文件 | SHA-256 |
|---|---|
| `artifacts/stage1/features/clip_vit_b32_h8_p12_upright/train.hdf5` | `5ac7307278ae2e0a7b692bb0cbe8c00f5a605b8111f0b6a1f1e3d5d3b56849dc` |
| `artifacts/stage1/features/clip_vit_b32_h8_p12_upright/validation.hdf5` | `931982e293753ce44001dce9fd1dc8b9b3fd93fc3899f55569effd5482265244` |
| all-task D12 best checkpoint `results/c0/seed11/best.pt` | `8666019d8572f0334965a560c41a3a3442b37692a4b65847375caf3f2f24bbb4` |

本次 audit 产生的 JSON 也保持在 Git 忽略的 `artifacts/stage1/`，可通过以下 hash 追溯：

| 审计产物 | bytes | SHA-256 |
|---|---:|---|
| `temporal_alignment.json` | 47,199 | `d08cf9cf1c799b8b299f50c4badf78a285a60052814036218d576c4af18c263b` |
| `offline_online_feature_parity.json` | 62,709 | `041393a7d12c56d95285159478f6d7baa5ed53a91646cd5eed3818af33be3272` |
| `c0_validation_diagnostic.json` | 81,868 | `8549deee9ca44c0228a3c4f09b491657db745231a512c7afa67f2bfe57b5d5cb` |

## 2. LIBERO action / observation 时间语义

### 固定 converter 的源码证据

审计的是固定 commit `8f1084…` 的 `third_party/LIBERO/scripts/create_dataset.py`。

1. 它设定 `cap_index = 5`（line 123）。
2. 对原始 replay index `j`，先执行 `obs, reward, done, info = env.step(action)`（lines 175–177）。
3. converter 随后以 `||states[j+1] - simulator_state_after_step||` 检查回放（lines 179–188）。
4. 对 `j >= 5`，它把这个 **step 后** `obs`（包括 agentview RGB/proprio）append，并把 `j` append 到 `valid_index`（lines 190–220）。
5. 循环结束后才做 `states = states[valid_index]` 和 `actions = actions[valid_index]`（lines 225–227），随后写入 processed HDF5（lines 248–261）。

因此，令 `k=0,1,...` 是 processed HDF5 的 index，`j=k+5`：

```text
raw state[j] -- raw action[j] --> raw state[j+1]
       |                               |
       |                               +-- processed obs[k]
       +-- processed states[k]

processed actions[k] = raw action[j]
processed states[k+1] = raw state[j+1]  (即 processed obs[k] 的 simulator state)
```

由此导出：

```text
processed obs[k] + processed actions[k+1] -> processed obs[k+1]
```

pre-conversion 原始 HDF5 本体不在本机保留；上述 raw-to-processed 追踪来自固定 converter，而其 state-after-action 自检是它自身写入 processed HDF5 前已执行的语义断言。`scripts/verify_temporal_alignment.py` 将这个映射、`cap_index` 和 `valid_index` 明确写入 JSON，并只读地在发布后的 processed HDF5 上再做 replay。

### empirical replay

命令：

```bash
env -u PYTHONPATH LIBERO_CONFIG_PATH="$PWD/.libero" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  .venv/bin/python scripts/verify_temporal_alignment.py \
  --output artifacts/stage1/temporal_alignment.json
```

覆盖 task `[0, 4, 8]`、每 task demo `[0, 17, 44]`、每 demo processed timestep `[0, 5, 10]`，合计 27 个观测转移。每个候选都从 `processed_states[k+1]` 恢复（即 `obs[k]` 所对应的 state），并以相同的 `processed_states[k+2]`、`obs[k+1]` 作 target：

| 指标（median，27 对） | Candidate 1：`obs[k]+a[k]→obs[k+1]` | Candidate 2：`obs[k]+a[k+1]→obs[k+1]` |
|---|---:|---:|
| simulator state L2 | 0.38449 | **0.36517** |
| 12-D proprio L2 | 0.013958 | **0.013270** |
| RGB pixel MAE | 3.00163 | **2.92598** |
| RGB PSNR dB | **26.2289** | 25.9866 |
| CLIP RGB cosine | 0.988871 | **0.989141** |
| state-L2 win count | 6/27 | **21/27** |

补充的 converter 对应关系 `states[k]+actions[k] -> obs[k] / states[k+1]` 的 median state L2 为 0.31201，proprio L2 为 0.011679，RGB MAE 为 2.73769，CLIP cosine 为 0.991486。RGB/PSNR 并非每条都严格单调：`sim.set_state_from_flattened` 不保存 robosuite OSC controller 的 goal internals，独立 restore 后的一步控制会引入绝对误差。该限制已在 JSON 的 `controller_caveat` 中记录；它不改变 converter 中 `action[j]` 到 `states[j+1]` 的明确因果语义。

### 四个必须区分的 pairing

1. **官方 IL loader：** `libero/lifelong/datasets.py` 创建 `SequenceDataset(dataset_keys=["actions"], load_next_obs=False)`；obs 和 action 用同一 sequence index。因此官方 imitation-learning convention 是 `obs[k] -> actions[k]`。
2. **真实观察 transition：** `obs[k] + actions[k+1] -> obs[k+1]`。
3. **C0：** 保持 `obs[k] -> action_chunks[k:k+H]`，因为 C0-v1 需要保持官方 IL 标签 convention；本审计不授权对现有 C0 label 做 shift。
4. **D13 world model：** 若预测 next observed latent，使用 `z_obs[k], action[k+1], z_obs[k+1]`；最后一个没有 next observation 的 index 不能形成 transition。若 D13 还要建模 pre-action state，则应显式另建 `states[k], action[k], obs[k]` 语义，不能混称为 observation transition。

## 3. Offline / online CLIP parity

审计脚本 [verify_offline_online_feature_parity.py](../scripts/verify_offline_online_feature_parity.py) 不复制 feature extraction 逻辑，而是实例化 deployed `C0ActorPolicy`，用其本地 snapshot、`CLIPProcessor`、autocast 和 L2 normalization 对 canonical source RGB 重算 feature。它从 train 与 validation 各 task 选 3 个等距 feature row，共 60 个图像，并检查全部 10 个 task text embedding。

两侧均固定为：正向 `uint8 RGB[128,128,3]`、`vertical_flip=false`、`openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`；60/60 个 online input 与 source RGB 字节相等。差异与 feature store 的 float16 保存一致：

| 比较 | cosine mean (min) | max abs error mean (max) | mean abs error | L2 error mean |
|---|---:|---:|---:|---:|
| image，60 samples | 0.99999913 (0.99999750) | 0.00033485 (0.00074536) | 0.00004460 | 0.00132901 |
| text，10 tasks | 0.99999959 (0.99999946) | 0.00023463 (0.00041419) | 0.00003312 | 0.00097054 |

离线 image norm 平均 1.00000360、online 0.99999998；离线 text norm 平均 0.99997095、online 0.99999997。结论是：D12 canonical feature store 与 C0 online representation 没有实质 latent mismatch；微小数值误差由 float16 quantization 解释。

## 4. C0 multi-task 低分的 validation-only 诊断

脚本 [diagnose_c0_validation.py](../scripts/diagnose_c0_validation.py) 只载入已保存的 all-task checkpoint 和 canonical single-task task0 checkpoint，在 train/validation feature stores 上计算 raw predictions。没有运行 simulator、fixed test 或 optimizer。

### 样本、validation MSE 与共享 actor 平均偏差

下表 `bias L2` 是 all-task actor 的每维 raw prediction mean 减 target mean 的 7-D L2；它是偏向的诊断值，不是 success-rate 因果证明。

| task | train rows | val rows | all-task val raw MSE | val bias L2 |
|---:|---:|---:|---:|---:|
| 0 | 4,298 | 420 | 0.053750 | 0.05336 |
| 1 | 5,724 | 633 | 0.037823 | 0.02361 |
| 2 | 4,933 | 599 | 0.040791 | 0.07026 |
| 3 | 4,255 | 447 | 0.032392 | 0.05088 |
| 4 | 6,426 | 703 | 0.035271 | 0.09290 |
| 5 | 4,934 | 512 | 0.032172 | 0.03531 |
| 6 | 5,291 | 671 | 0.053559 | 0.11930 |
| 7 | 6,060 | 701 | 0.042057 | 0.07337 |
| 8 | 5,005 | 608 | 0.048557 | 0.04386 |
| 9 | 5,905 | 625 | 0.029140 | 0.06011 |

总 validation weighted raw MSE 是 0.0405226，macro task MSE 是 0.0405513。saved best 的选择准则是 overall weighted raw MSE；因此它在该 checkpoint 上覆盖了 0.029140–0.053750 的 task 差异（task 0/6 最差）。由于每 epoch 没有保存 per-task checkpoint history，本审计不能声称哪个 epoch 对某一 task 最优；只能确认 overall selection 不会直接优化最差 task。

### 每 task 动作分布

每个 task 的完整 7-D raw / normalized target `mean, std, min, max` 在 `artifacts/stage1/c0_validation_diagnostic.json` 的 `multitask.*_per_task.*_distribution` 中。下表为 train target 的 7-D raw mean 和使用 all-task train-derived normalization 后的 7-D mean，显示任务间动作先验确实不同（向量顺序是 LIBERO 的 7-D OSC-POSE action）：

| task | raw mean | normalized mean |
|---:|---|---|
| 0 | `[.179,.117,-.194,.007,-.011,-.025,.035]` | `[.070,-.061,-.039,.298,.017,-.105,-.121]` |
| 1 | `[.161,.138,-.151,-.008,-.015,-.016,.062]` | `[.026,-.002,.045,-.088,-.036,.051,-.094]` |
| 2 | `[.178,.154,-.184,-.003,-.015,-.025,.191]` | `[.067,.043,-.020,.045,-.036,-.094,.037]` |
| 3 | `[.190,.183,-.193,-.018,-.017,-.002,-.024]` | `[.096,.127,-.036,-.343,-.061,.300,-.181]` |
| 4 | `[.091,.149,-.190,-.008,-.014,.002,.216]` | `[-.140,.030,-.032,-.073,-.021,.363,.062]` |
| 5 | `[.158,.116,-.160,-.003,-.012,-.056,.286]` | `[.020,-.063,.027,.059,.003,-.633,.133]` |
| 6 | `[.134,.152,-.186,-.008,-.024,-.020,.073]` | `[-.036,.039,-.022,-.070,-.165,-.006,-.082]` |
| 7 | `[.143,.113,-.166,.003,.001,-.019,.443]` | `[-.015,-.073,.016,.207,.185,.005,.292]` |
| 8 | `[.157,.109,-.178,.004,-.011,-.047,-.029]` | `[.017,-.083,-.007,.232,.020,-.468,-.186]` |
| 9 | `[.136,.156,-.149,-.014,-.008,.005,.186]` | `[-.032,.049,.048,-.238,.058,.419,.032]` |

### task/language mapping 与 task0 对比

- Feature metadata 的 10 个 task ID 唯一，10 个 language string 唯一，且 10/10 与 fixed LIBERO benchmark 的 task name/language 完全一致；未发现 task_id/language 错位。
- 语言 embedding 两两 off-diagonal cosine 为 0.82168–0.98890（mean 0.89950）：语句相近但不是同一个 embedding；第 3 节进一步证明它们在 offline/online 一致。
- task0 multi-task raw MSE：train 0.0402824，validation 0.0537501；canonical single-task checkpoint：train 0.0347744，validation 0.0474534。multi/single validation 比为 **1.13269**。这与 multi-task shared actor 的容量/干扰问题相容，但不能仅凭 offline MSE 或 task0 fixed test 推断具体结构改动。

## 5. P2 工程修复与验证

### Resume historical-best 保护

`scripts/train_c0.py` 的新 checkpoint schema 为 2，并在 `last.pt` 与 `best.pt` 都保存 `best_validation`。resume 时恢复该值；如果给的是旧 D12 schema-1 checkpoint（没有这个字段），脚本明确失败而不是静默回退到 `inf`。这是一项 fail-closed 兼容性策略，保证不会因为恢复旧 checkpoint 意外覆盖历史 `best.pt`；现有 D12 checkpoint 没有被写入或改动。

`tests/test_c0_resume_best.py` 创建一个 historical best=0.125 的临时 checkpoint，恢复后输入更差的 validation=0.250，验证 `best.pt` SHA 未变化。结果：`best_checkpoint_preserved=True`。

### Production-equivalent training step

保留原来的 `tests/test_c0_actor.py` wiring test，并新增 `tests/test_c0_production_step.py`。它直接使用 canonical train feature HDF5 和 train-derived proprio/action statistics，执行真实的：12-D proprio normalization、normalized action target、`SmoothL1Loss`、forward、backward、gradient clipping、AdamW step，并验证全部 finite。

本机执行结果：CUDA AMP enabled，batch inputs 为 image `[64,512]`、language `[64,512]`、proprio `[64,12]`、target `[64,8,7]`；output `[64,8,7]`；loss `0.4284583`；18 个 gradient tensors 均 finite；clip 前 gradient norm `0.0895455`。

### Action clipping audit

现有规则没有改变：执行动作仍裁剪到 `[-1,1]`。`C0ActorPolicy` 现在在内部保留 denormalized pre-clip chunk 的统计接口，但 public `plan_batch` 仍返回历史相同的 clipped action chunk。正式 all-task evaluator `scripts/evaluate_c0_all.py` 将以下字段写入每 episode、每 task 和 run summary：

- `planned_action_elements`
- `elements_below_minus_one`
- `elements_above_one`
- `elements_outside_minus_one_to_one`
- `clipped_element_ratio`
- pre-clip `raw_min` / `raw_max`

`tests/test_c0_clipping_stats.py` 用 14 个合成 raw action elements 验证统计；其中 5 个越界，episode/run 比例都是 `5/14 = 0.3571428571`。这验证 logging，不改变 C0 推理或执行数值。

### 正式 evaluator 数值协议

新增 [evaluation_protocol_c0.json](../configs/evaluation_protocol_c0.json)，并更新 [data_contract.md](data_contract.md)。正式 C0/C1/C2 主结果固定：

- `parallel_envs=1`；`parallel_envs>1` 只可用于吞吐诊断，不能直接与正式结果比较。
- `max_steps=300`，每 task 20 episodes。
- seed 和 init-state 固定为 [eval_seeds_spatial.json](../configs/eval_seeds_spatial.json)；episode `i` 使用官方 init state `i`。
- 唯一视觉输入是正向、未翻转的 `agentview` `uint8 RGB[128,128,3]`；规划时激活 camera，chunk 的中间 open-loop actions 关闭它；CLIP revision 固定。

## 6. 验证记录与后续边界

本轮通过了：Python compile、resume regression、clipping accounting、production-equivalent CUDA AMP step、既有 actor wiring test、既有 online policy adapter test。常见的 robosuite private-macro warning 和 Gym deprecation warning 没有影响这些结果。

D13 可以开始的前提已经满足，但实现必须先把本报告第 2 节的 transition pairing 写成 versioned data contract，并从 observation-aligned rows 构造 train/validation transitions；不得擅自 shift C0 监督标签，也不得把 `states` / object pose / reward / success 等审计字段泄漏进 policy 或 latent。D13 开始前也不应因当前 fixed test 结果再调 C0 的超参数。
