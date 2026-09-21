# D12 实现事实审计（只读）

审计日期：2026-09-22。审计范围为当前工作区中实际存在的文件、代码、HDF5 数据、配置、日志和结果。除本报告外，未修改源码、配置、checkpoint、特征或评测结果；为确认 shape 和反向传播而运行的 Python 探针只在内存/GPU 中执行，未落盘。

## 1. 结论先行

- D1–D11 的主要产物都存在；D11 的 `seed=11` 全任务 C0 评测已完成 200/200 个 episode，17 次成功，宏平均成功率为 0.085。
- 当前只实现了 C0：冻结 CLIP 特征 + MLP 动作块回归。**没有**世界模型、`z_t` 预测器、C1/C2 候选生成、候选重排、阈值选择或闭环提前重规划。
- 项目根目录的 Git 状态是严重的复现缺口：`main` 分支存在，但没有任何 commit / `HEAD`；根目录下的 `.gitignore`、`README.md`、`configs/`、`docs/`、`results/`、`scripts/`、`src/`、`tests/`、`third_party/` 全部是未跟踪文件。因此不存在可报告的“当前项目 commit hash”。只有 `third_party/LIBERO` 有固定 commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`。
- 当前 feature store 的索引实现确实按相同整数 `t` 存储 `obs[t]` 的视觉/本体字段和 `actions[t:t+8]`。本次只读抽查了 3 条跨任务样本，字段逐元素相等；但没有发现能从官方 demo 语义上严格证明 `actions[t]` 必然对应 `obs[t]` 而非某个相邻帧的测试。因此“实现索引一致”已验证，“官方数据语义无 off-by-one”**未验证**。
- D12 的最低 G1 要求（10 个任务可结束评测、原始 episode 结果存在）达到；但“测试种子从未参与人工调参”不能仅由代码/文件完全证明，且根 Git 无 commit 使源代码版本不可锁定。

## 2. 当前进度和 Git 状态

### Git 快照

审计开始时执行：

```text
$ git rev-parse HEAD
fatal: ambiguous argument 'HEAD': unknown revision or path not in the working tree.

$ git branch --show-current
main

$ git status --short --ignored
?? .gitignore
?? README.md
?? configs/
?? docs/
?? results/
?? scripts/
?? src/
?? tests/
?? third_party/
!! .cache/
!! .libero/
!! .venv/
!! artifacts/
!! data/
```

根仓库 `git log -1` 同样报错：`your current branch 'main' does not have any commits yet`。因此：

| 项目 | 实际值 |
|---|---|
| 根仓库 commit | 不存在（无 `HEAD`） |
| 根仓库 branch | `main` |
| 根仓库状态 | 所有项目文件未跟踪；`data/` 和 `artifacts/` 被 `.gitignore` 忽略 |
| LIBERO 外部源码 commit | `8f1084e3132a39270c3a13ebe37270a43ece2a01`，branch `master`，审计时无短状态输出 |

这意味着 `configs/source_versions.yaml`、结果 JSON 和 checkpoint 虽然存在，但不能用 Git commit 唯一绑定本项目的 C0 源码版本。

### D0–D12 实际状态

| 计划项 | 实际状态 | 现有证据 / 说明 |
|---|---|---|
| D1 | 完成 | `artifacts/stage0/gpu_probe.txt`：WSL2、RTX 4070 Laptop 8188 MiB、Torch CUDA 可用。 |
| D2 | 完成 | `.venv`、`artifacts/stage0/env_freeze.txt`、`configs/source_versions.yaml`。 |
| D3 | 完成 | 10 个官方 `libero_spatial` HDF5；`artifacts/stage0/data_manifest.json` 记录 HF revision 和 SHA-256。 |
| D4 | 完成 | `artifacts/stage0/smoke_run.json` 和 PNG：reset、128×128 RGB、7-D step、`check_success()`。 |
| D5 | 完成（有限证据） | `scripts/run_smoke.sh` 与 `artifacts/stage0/repeat/` 第二次 smoke 输出存在；并非完整 benchmark 重跑。 |
| D6 | 完成 | `docs/blockers.md` 记录无 active blocker；这不是签名式 gate 记录。 |
| D7 | 完成 | `docs/data_contract.md` 冻结 C0 输入、禁止字段、demo 划分和固定评测 seeds。 |
| D8 | 完成 | canonical feature store 是 `artifacts/stage1/features/clip_vit_b32_h8_p12_upright/`；有 manifest 和 verification。较早的 `clip_vit_b32_h8/`、`clip_vit_b32_h8_p12/` 仍存在，是历史版本，不应混用。 |
| D9 | 完成（实现/探针层面） | `C0Actor`、`FeatureDataset`、训练器、checkpoint/恢复代码、`tests/test_c0_actor.py` 和 `artifacts/stage1/c0_backward_probe/` 存在。测试不是完整训练。 |
| D10 | 完成 | 单任务 task 0、seed 11 两个预处理版本均有完整 20 episode JSON；canonical upright 版本为 11/20。 |
| D11 | 完成 | `results/c0/seed11/summary.json`：10/10 task、200/200 episode、17 成功、`complete=true`。仅完成训练随机种子 11；23、47 是计划 D25，不是 D11 门槛。 |
| D12 | 最低 G1 条件完成；严格公平性部分未验证 | `configs/d12_g1_checklist.json` 和 D11 原始 episode 结果存在。该 JSON 自评为 pass；本报告对人工调参、异常重试、Git 溯源作出更保守判断。 |
| D13+ | 未开始 | 未发现 `configs/world_model.yaml`、world-model class/test、ensemble checkpoint、C1/C2 实现或世界模型结果。 |

## 3. D0–D12 相关目录树

下树是审计时实际存在的相关文件；6.24 GB 原始数据和较大的 HDF5/checkpoint 以集合表示，未省略其逻辑位置。

```text
embodied-libero-wm/
├── .gitignore                         # 忽略 data/、artifacts/、.venv/、.cache/
├── README.md
├── configs/
│   ├── c0_actor.yaml                   # C0 模型、数据、优化配置
│   ├── d12_g1_checklist.json           # D12/G1 审计清单（本审计前已创建）
│   ├── eval_seeds_spatial.json         # 10×20 个固定评测 seed
│   └── source_versions.yaml            # LIBERO、数据、运行时、CLIP 版本
├── data/LIBERO/datasets/libero_spatial/
│   └── 10 × *_demo.hdf5                # 每任务 50 demos；原始数据，Git ignored
├── artifacts/
│   ├── stage0/
│   │   ├── gpu_probe.txt, env_freeze.txt
│   │   ├── data_manifest.json, data_sha256.txt, dataset_download.log
│   │   ├── smoke_run.json, smoke_agentview.png, smoke_stdout_stderr.log
│   │   ├── repeat/{smoke_run.json, smoke_agentview.png, smoke_stdout_stderr.log}
│   │   └── teleop_gui_probe.log, policy_viewer_*.log
│   └── stage1/
│       ├── data_audit.json, image_alignment.json
│       ├── c0_backward_probe/{config.yaml,run_metadata.json,metrics.jsonl,best.pt,last.pt}
│       ├── c0_p12_backward_probe/... , c0_task0_filter_probe/...
│       ├── eval_camera_gated_task0.json, eval_camera_gated_serial_task0.json
│       └── features/
│           ├── clip_vit_b32_h8/{train.hdf5,validation.hdf5,feature_manifest.json,verification.json}
│           ├── clip_vit_b32_h8_p12/{...}
│           └── clip_vit_b32_h8_p12_upright/{train.hdf5,validation.hdf5,feature_manifest.json,verification.json}
├── docs/
│   ├── data_contract.md
│   ├── experiment_log.md
│   ├── blockers.md
│   ├── Embodied_VLM_WorldModel_LIBERO_Execution_CN.md -> 外部计划文档的符号链接
│   └── d12_implementation_fact_audit.md   # 本报告
├── results/c0/
│   ├── task0_seed11/{config.yaml,run_metadata.json,metrics.jsonl,best.pt,last.pt,eval_task0_seed11.json}
│   ├── task0_seed11_upright/{同上}
│   ├── task0_preprocessing_ablation.json
│   └── seed11/{config.yaml,run_metadata.json,metrics.jsonl,best.pt,last.pt,summary.json}
├── scripts/
│   ├── audit_image_alignment.py        # 离线/在线 RGB 方向检查；有索引缺陷，见 §17
│   ├── audit_spatial_dataset.py         # 原始 HDF5 schema/数量/动作范围审计
│   ├── compare_c0_preprocessing.py      # 两个 20-episode 单任务结果逐 index 比较
│   ├── evaluate_c0.py                  # 单任务、逐 episode C0 evaluator
│   ├── evaluate_c0_all.py              # 10 任务 evaluator，partial JSON 与 --resume
│   ├── extract_features.py             # 原始 HDF5 → 冻结 CLIP feature HDF5
│   ├── run_policy_viewer.py/.sh         # GUI viewer，加载 policy adapter
│   ├── run_smoke.sh, smoke_libero.py    # EGL smoke test
│   ├── run_teleop.sh, teleop_libero.py  # WSLg/GLFW 键盘遥操作
│   ├── train_c0.py                     # C0 训练、验证、checkpoint/resume
│   └── verify_feature_store.py          # feature hash/schema/norm/split 审计
├── src/
│   ├── c0/
│   │   ├── actor.py                    # `C0Actor`：三路 MLP 融合并输出 H×7
│   │   ├── dataset.py                  # `FeatureDataset` 与训练集统计量计算
│   │   └── __init__.py
│   └── policies/
│       ├── c0_actor_policy.py          # 在线 CLIP + C0 checkpoint + action-chunk buffer
│       ├── policy_api.py               # viewer `LivePolicy` Protocol
│       ├── zero_policy.py              # 非学习的接线 fixture
│       └── __init__.py
├── tests/
│   ├── test_c0_actor.py                # CPU、B=2 的 forward/backward/step smoke test
│   └── test_c0_policy_adapter.py       # task0 checkpoint 的 adapter output shape/finite test
└── third_party/LIBERO/                 # 单独 Git 仓库，固定于 8f1084...
```

## 4. LIBERO 数据读取与时间对齐

### 原始格式和实际字段

数据来源为 Hugging Face `yifengzhu-hf/LIBERO-datasets`、revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`；文件清单与 SHA-256 在 `artifacts/stage0/data_manifest.json`。目录中有 10 个 `*_demo.hdf5`，每个 `data` group 有 50 个 `demo_0` 到 `demo_49`。

对 task 0 的 `data/demo_0` 实测：

```text
actions:       float64 [98, 7]
states:        float64 [98, 92]          # 禁止作为模型输入
robot_states:  float64 [98, 9]           # 禁止作为模型输入
rewards/dones: uint8   [98]              # 禁止作为模型输入
obs/agentview_rgb:     uint8   [98, 128, 128, 3]
obs/eye_in_hand_rgb:   uint8   [98, 128, 128, 3]
obs/ee_pos:            float64 [98, 3]
obs/ee_ori:            float64 [98, 3]   # 当前禁止
obs/ee_states:         float64 [98, 6]   # 当前禁止
obs/gripper_states:    float64 [98, 2]
obs/joint_states:      float64 [98, 7]
```

原始 `data` 的 `env_args` 记录 `camera_names=["robot0_eye_in_hand", "agentview"]`、128×128、OSC_POSE controller、`control_freq=20`。C0 只读取外部 `agentview_rgb`，不使用 wrist camera。

语言不从每条 demo 读取；`scripts/extract_features.py:83-98` 通过 `benchmark.get_benchmark_dict()["libero_spatial"]()` 获得每个 `task.language`，再写入 feature store 的 `task_metadata/task_language`。在线评测同样使用 `task.language`（`scripts/evaluate_c0_all.py:111-145`）。

### 当前实际 sample 的定义

抽取器的 `DemoRecord` 是 Python dataclass：`task_id`、`task_name`、`task_language`、源 HDF5 `path`、`demo_id`、`transitions`。它不是将整个 episode 常驻内存的数据结构。特征 HDF5 的每行额外保存 `task_id`、`demo_id`、`timestep`，用于来源溯源。

对于长度为 `T` 的一个 demo，`scripts/extract_features.py:248-296` 生成 `T - H + 1` 行；当前 `H=8`，所以起点为 `t=0,...,T-8`。每行实际构造为：

```text
image feature      = CLIP(obs/agentview_rgb[t])
proprio_raw        = concat(obs/ee_pos[t], obs/gripper_states[t], obs/joint_states[t])
action_chunks      = actions[t:t+8]   # a[t] 到 a[t+7]，shape [8,7]
```

它不读取、保存或监督 `obs[t+1]`；也没有 transition mask、padding 或跨 demo 拼接。尾部少于 8 个 future action 的起点不产生 sample。

本次审计直接将 canonical train feature HDF5 的三行回查到源 HDF5：

| feature row | task / demo / t | `proprio_raw == source obs[t]` | `action_chunks == source actions[t:t+8]` |
|---:|---|---|---|
| 0 | 0 / 0 / 0 | `true` | `true` |
| 12,345 | 2 / 21 / 48 | `true` | `true` |
| 52,830 | 9 / 44 / 123 | `true` | `true` |

这是对**当前抽取实现的整数索引**的强证据。`scripts/audit_spatial_dataset.py:35-83` 还验证了所有原始 observation 数组与 `actions` 都有相同长度，且动作维度为 7。

但是，没有找到通过环境 replay、官方录制器文档或 `obs[t] --a[t]--> obs[t+1]` 状态比对来证明 demo 的语义索引约定的测试。故以下结论必须区分：

- `obs` 和 `actions` 的数组长度相等：**已验证**。
- 特征行采用同一数值索引 `t`：**已验证**。
- 官方 demo 中 `actions[t]` 在语义上就是从 `obs[t]` 发出的 action（不存在整体 ±1 偏移）：**未验证**。

`scripts/audit_image_alignment.py` 也不能填补这一缺口：它在第 39 行只读取 `agentview_rgb[0]`，但第 40–46 行却与 `states[0]`、`states[1]`、`states[2]` 的渲染比较。它支持“直接图像比竖直翻转更接近”这一方向判断，却没有为 `t=1,2` 使用对应的 stored image，因此不是完整的逐时刻时间对齐测试。

## 5. Data Contract 与实际遵守情况

`docs/data_contract.md` 的版本是 2。当前 policy/actor 允许的逻辑输入只有：

| 输入 | 离线字段 | 在线字段 | 实际维度 |
|---|---|---|---:|
| 外部 RGB | `obs/agentview_rgb[t]` | `agentview_image` | `[128,128,3] uint8` |
| 语言 | `task.language` | `task.language` | 一个 task 一个 string / CLIP 512-D feature |
| 末端位置 | `obs/ee_pos[t]` | `robot0_eef_pos` | 3 |
| 夹爪 | `obs/gripper_states[t]` | `robot0_gripper_qpos` | 2 |
| 关节位置 | `obs/joint_states[t]` | `robot0_joint_pos` | 7 |

本体状态合计为 12-D。禁止字段包括 `states`、`robot_states`、object pose/relative pose、`model_file`、`init_state`、奖励、`dones`、`env_infos`、success、未来 frame/action/latent、`eye_in_hand_rgb` 和当前未校准的 `ee_ori`。

代码层面的遵守证据：

- 抽取器只创建 `image_features`、`proprio_raw`、`action_chunks`、IDs/timestep、task metadata；见 `scripts/extract_features.py:202-226`。`verify_feature_store.py` 拒绝顶层 forbidden name。
- `FeatureDataset.__getitem__()` 只返回 image/language/proprio/action/task_id；`src/c0/dataset.py:69-90`。`demo_id`、`timestep` 虽在 HDF5 中，训练 item 不返回它们。
- 在线 evaluator 的 `policy_observation()` 只传 4 个键；`scripts/evaluate_c0_all.py:40-46`。虽然原始 environment observation 在局部变量中可能还包含 object fields，但 `C0ActorPolicy.plan_batch()` 只接受这 4 个 required keys；`src/policies/c0_actor_policy.py:103-129`。
- `C0Actor.forward()` 仅接收 three tensors；无 reward、success、object 或 state 参数。

保守限制：`task_id` 会由 dataset 用于索引 `language_features[task_id]`，但 actor forward 不接收 `task_id`。这是把允许的 task language 映射为预计算语言 embedding，不是将 object/state metadata 送入 actor。

## 6. 冻结视觉/语言特征

离线 extraction 已实现，canonical store 为 `artifacts/stage1/features/clip_vit_b32_h8_p12_upright/`。

- Encoder：`openai/clip-vit-base-patch32`（CLIP ViT-B/32），revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`；不是 SigLIP。
- `scripts/extract_features.py:153-168` 调用 `eval()` 并对 CLIP 全参数 `requires_grad_(False)`；image/text feature 在 `torch.inference_mode()` 中提取。因此 Actor 训练不会向 encoder 反向传播。
- 输入为单 camera 的 raw `uint8 [128,128,3]`，不竖直翻转。实测 `CLIPProcessor` 输出 `pixel_values: float32 [1,3,224,224]`；单条语言 tokenizer 输出 `input_ids` 和 `attention_mask` 均为 `int64 [1,21]`（不同 instruction 的 token length 可不同）。
- 对 image/text projection 分别 L2 normalize（`normalized_features()`），projection feature 为 512-D。没有多 camera，故没有 multi-camera fusion。
- image feature HDF5 dtype 是 `float16 [N,512]`；language feature 是 `float16 [10,512]`；`FeatureDataset` 读取时转换为 `torch.float32`。verification 文件报告 image feature L2 norm 在 train `[0.999774,1.000221]`、validation `[0.999790,1.000212]`。
- 语言只对 10 个 task instruction 批量编码一次，然后写为 `task_metadata/language_features`；不是每个 transition 重算。在线 adapter 以 language string 为 key 做进程内 cache（`_language_cache`）。
- feature file 为 gzip level 4 + shuffle 的 HDF5。train：52,831 行、55,710,168 bytes、SHA-256 `5ac730...b56849dc`；validation：5,919 行、6,460,620 bytes、SHA-256 `931982...2265244`。

当前代码中没有名称为 `z_t` 的 latent；若把冻结 image feature 称作表征，则其形状是 `[512]` / `[B,512]`，但它不是可学习世界模型 latent。

## 7. Actor Dataset：实际 sample、batch 与划分

对 `FeatureDataset(train.hdf5)` 进行只读实测：

```text
单 sample:
image_features    torch.float32 [512]
language_features torch.float32 [512]
proprio_raw       torch.float32 [12]
action_chunks     torch.float32 [8, 7]
task_id           torch.int64   []   （抽样行值为 0）

DataLoader(batch_size=64):
image_features    torch.float32 [64, 512]
language_features torch.float32 [64, 512]
proprio_raw       torch.float32 [64, 12]
action_chunks     torch.float32 [64, 8, 7]
task_id           torch.int64   [64]
```

实际 Dataset 长度为 52,831，`horizon=8`、`action_dim=7`、`proprio_dim=12`。没有 `vision_ctx`、`proprio_ctx`、mask、`episode_id` 或 `timestep` 返回值；`demo_id`/`timestep` 存在 feature HDF5，却不从 `__getitem__()` 返回。故当前 context length `K` 是隐式的 **1**（单个时刻），不是序列模型。

train/validation 按完整 demo 切分：每个 task 的 demo 0–44 训练、45–49 验证。抽取循环逐 `DemoRecord` 写入，每个 record 单独计算可用起点，故不会跨 episode 取 action chunk；无 padding/mask。只要 source demo id 没有被错误标记，单一 demo 不会同时进入两个 split。`verify_feature_store.py` 实际检查 train 的 demo IDs `[0..44]` 与 validation `[45..49]` 无交集。

## 8. Action normalization

训练实际使用 standardization，而非 min-max：

```text
normalized_target = (raw_action_chunk - action_mean) / action_std
```

`train_c0.py:200-212, 277-284` 从 **train.hdf5 的 `action_chunks`** 调用 `compute_action_stats()`；每个 overlap action 在不同 8-step chunk 中会被重复计数。这是 checkpoint 中真正使用的统计，而不是 feature manifest 中从原始、未重叠 `actions` 统计的 `action_statistics_train_only`。两者均无 test episode 参与，但数值不完全相同。

全任务 seed-11 checkpoint 的实际训练统计（来自 `results/c0/seed11/run_metadata.json`）是：

```text
mean = [ 0.149619, 0.138424,-0.174151,-0.004802,-0.012218,-0.019336, 0.154436]
std  = [ 0.421533, 0.351998, 0.513157, 0.038773, 0.072078, 0.058623, 0.988003]
```

`proprio_raw` 也在 train feature rows 上计算 mean/std。validation 使用同一套 train-derived statistics。在线 `C0ActorPolicy.plan_batch()` 从 checkpoint 加载统计，把 actor 的 normalized output 反标准化，然后 `np.clip(...,-1,1)`；all-task evaluator 又执行一次同区间 clip。没有从 evaluation/test episodes 计算任何 statistic。

风险：`docs/data_contract.md` 说应记录在线 action clipping ratio，但现有 evaluator/result JSON 没有该字段。另一个容易误读处是 feature manifest 的 raw-action stats 与训练 checkpoint stats 不是同一个分布计数口径。

## 9. Actor 模型、loss 和训练

### 结构和 shape trace

实际实例采用 `feature_dim=512`、`proprio_dim=12`、`fusion_dim=256`、`H=8`、`action_dim=7`；参数量实测 739,512。没有 GRU、Transformer、attention、Gaussian head 或 sampling。

```text
image_features       [B,512]  → Linear(512,256) → LayerNorm(256) → GELU = [B,256]
language_features    [B,512]  → Linear(512,256) → LayerNorm(256) → GELU = [B,256]
normalized proprio   [B,12]   → Linear(12,128)  → LayerNorm(128) → GELU = [B,128]
concat                                                     = [B,640]
→ Linear(640,512) → GELU                              = [B,512]
→ Linear(512,256) → GELU                              = [B,256]
→ Linear(256,56)                                      = [B,56]
→ view                                                  = [B,8,7]
```

定义在 `src/c0/actor.py:10-59`。输出是 deterministic normalized action chunk。构造函数的默认 `proprio_dim=15` 与当前实际 contract 的 12 不同，但训练和 checkpoint reconstruction 都显式传入 12；这是陈旧默认值风险，不影响已审计 run。

### 训练事实

| 项 | 实际值 |
|---|---|
| training loss | `torch.nn.SmoothL1Loss()`（默认 beta=1）作用于 normalized action chunk |
| validation / best metric | raw action space MSE，越低越好 |
| optimizer | AdamW |
| learning rate / weight decay | `3e-4` / `1e-4` |
| batch / accumulation | 64 / 1 |
| AMP | CUDA 下 `torch.autocast(float16)` + `torch.amp.GradScaler` |
| clip | `clip_grad_norm_(max_norm=10.0)` |
| configured epochs | 50 |
| actual full seed-11 steps | 41,300 optimizer steps，50 validation events |
| random seed | Python / NumPy / Torch / CUDA 均设为 11；未启用 `torch.use_deterministic_algorithms` |
| saved | 每个 epoch `last.pt`；validation raw MSE 变好时 `best.pt`；`metrics.jsonl`、copied config、metadata |

seed-11 run 的真实日志摘要：epoch 0 / step 1 的 normalized Huber loss `0.4219538`、gradient norm `0.0541408`；最优 validation 是 epoch 4 / step 4,130 的 raw MSE `0.0405226454`；epoch 49 / step 41,300 的 raw MSE 为 `0.0481681442`。`results/c0/seed11/best.pt` 因此不是最后一个 epoch。

resume 代码会校验 config hash、feature-manifest hash、task ID，恢复 model/optimizer/RNG/epoch/step，见 `scripts/train_c0.py:220-247`。但 checkpoint payload **不保存 `best_validation`**；恢复后该变量重新设为 `inf`，下一次 validation 即使更差也会重写 `best.pt`。所以“可恢复训练”存在，但“resume 后仍严格保持历史最优 checkpoint”未实现。

## 10. D9 one-batch forward/backward 证据

已重新执行现有 `tests/test_c0_actor.py`（不写文件），实际输出：

```text
{'shape': [2, 8, 7], 'loss': 0.2281792163848877}
```

该测试确认 output shape、finite loss、`loss.backward()` 后至少一个 gradient 非空、AdamW step 能执行。它运行在 CPU，模型随机初始化但**没有设置随机 seed**，所以打印的 loss 不可复现；并且它用 raw `action_chunks` 作为 MSE target，而正式训练用 normalized target 的 SmoothL1 loss。因此它是 wiring probe，不是正式训练损失的等价测试。

为满足审计 shape/显存要求，另做了未落盘的 CUDA B=64、正式 normalization + SmoothL1 一步探针。实际输出：

```text
device: cuda
output / normalized target: [64, 8, 7]
output dtype under AMP: torch.float16
loss: 0.4287217557
finite loss: true
parameters with gradients: 18
all gradients finite: true
pre-clip gradient norm: 0.0879381
CUDA peak allocated/reserved: 30.660 / 50.000 MiB
```

这个 peak 只包括 feature batch + 739k-parameter actor + optimizer，不包括 CLIP 或 MuJoCo；不能当作完整训练/评测显存峰值。已有 `artifacts/stage1/c0_backward_probe/metrics.jsonl` 也显示两步训练与 validation 成功，但不保存 peak memory 或 Tensor shapes。

## 11. D10 单任务 C0

两次 task-0 / seed-11 评测都存在。早期 flipped-RGB run 是一个保存的预处理消融，不应当与当前 upright canonical run 混淆。

| run | checkpoint SHA-256 | episodes | 成功 / 失败 | success rate | mean episode steps |
|---|---|---:|---:|---:|---:|
| `task0_seed11`（历史 vertical flip） | `4f2c7884...9856287` | 20 | 12 / 8 | 0.60 | 179.90 |
| `task0_seed11_upright`（当前 direct RGB） | `7d4f4108...03bf067` | 20 | 11 / 9 | 0.55 | 183.45 |

两者的固定 evaluation seed 都是 10000–10019，`max_steps=300`；失败均为 300 steps。canonical upright run 的成功 episode 平均 88.09 steps。结果路径分别为 `results/c0/task0_seed11/eval_task0_seed11.json` 与 `results/c0/task0_seed11_upright/eval_task0_seed11.json`；JSON 没有异常字段，且各有 20 条记录。`results/c0/task0_preprocessing_ablation.json` 明确记录 direct RGB 比 flipped run 少 1 个成功，但 canonical direct RGB 仍被保留。

## 12. D11 全部 10 任务结果

checkpoint 为 `results/c0/seed11/best.pt`，SHA-256 `8666019d8572f0334965a560c41a3a3442b37692a4b65847375caf3f2f24bbb4`。`summary.json` 标记 `complete=true`、`parallel_envs=1`、`max_steps=300`、每任务 20 episode，并保留每一条 episode record。

| task | episodes | successes | success rate | mean steps | result file / 异常 |
|---:|---:|---:|---:|---:|---|
| 0 | 20 | 0 | 0.00 | 300.00 | `results/c0/seed11/summary.json`; 无异常字段 |
| 1 | 20 | 0 | 0.00 | 300.00 | 同上 |
| 2 | 20 | 0 | 0.00 | 300.00 | 同上 |
| 3 | 20 | 0 | 0.00 | 300.00 | 同上 |
| 4 | 20 | 6 | 0.30 | 256.75 | 同上 |
| 5 | 20 | 0 | 0.00 | 300.00 | 同上 |
| 6 | 20 | 7 | 0.35 | 250.65 | 同上 |
| 7 | 20 | 0 | 0.00 | 300.00 | 同上 |
| 8 | 20 | 4 | 0.20 | 263.75 | 同上 |
| 9 | 20 | 0 | 0.00 | 300.00 | 同上 |

总计 17/200 = 0.085。183 个失败 record 的 `steps` 都等于 300；17 个成功的 step 在 116–257 之间。每个任务从 `configs/eval_seeds_spatial.json` 读取该 task 的 20 个 seed，并把 episode index 0–19 配到 LIBERO 官方 init states 0–19。

## 13. D12 检查项逐项复核

| 项 | 回答 | 证据与限定 |
|---|---|---|
| A. 10 个任务都正常结束？ | 是 | `summary.json.complete=true`，`completed_task_ids=[0..9]`，10 个 task 各 20 records。 |
| B. 所有失败都有原始记录？ | 是 | 183 个 `success=false` record 都有 episode index、seed、init-state index、steps=300。 |
| C. test/eval seeds 参与调参？ | 未验证 | 代码把 test seed file 只用于 evaluator；checkpoint 是 held-out demo raw MSE 选择。可是没有不可变实验登记、操作日志或 Git history，不能证明人工从未据测试结果调整任何决定。 |
| D. validation 与 test 严格分开？ | 是（代码/数据层） | train demos 0–44，validation demos 45–49；test 是独立 simulator rollout 的固定 init states/seeds。无 test rollout 用于 normalization 的代码。 |
| E. 为提高最终成功率而根据 test 改模型/超参/checkpoint/preprocess？ | 未验证 | 未找到“为了提高 score”而挑选的证据；direct-RGB correction 的 task-0 score 反而从 .60 降至 .55。但 test 结果已在预处理迭代时间线上出现，且没有盲化/审计日志，不能给出绝对否定。 |
| F. silent fallback、异常重试或跳过？ | 是，有 silent retry；无 evaluator skip 代码 | `ControlEnv.reset()` 在上游 `env_wrapper.py:90-101` 对 `RandomizationError` 无日志循环重试。C0 evaluator 对其他异常会退出；未见将失败 episode 跳过或填充为成功的代码。resume 会从未完成 task 重跑，而非跳过。 |
| G. 有人工挑选而非完整协议输出？ | 否（对最终 D11 summary） | final summary 包含全部 10 tasks、全部 200 episode；单任务 D10 ablation 是另行保存的，不是 D11 主表。 |

## 14. C0 evaluation 的真实执行链

以下为 D11 `parallel_envs=1` 的真实执行链；D10 `evaluate_c0.py` 的主流程相近，但不使用 camera gating 或 `plan_batch()`。

```text
scripts/evaluate_c0_all.py:main
  → 读取 eval_seeds_spatial.json，建立 LIBERO suite
  → create_policy(checkpoint, device)
    → C0ActorPolicy.__init__（src/policies/c0_actor_policy.py:25-86）
      加载 checkpoint、normalization、固定本地 CLIP snapshot；冻结 CLIP
  → evaluate_task(...)
    → trusted_init_states(task_suite, task_id)
    → OffScreenRenderEnv(... 128×128 ...)
    → env.seed(seed)
    → env.reset()
       上游 ControlEnv.reset 对 RandomizationError 重试
    → env.set_init_state(init_states[episode_index])
       上游 regenerate_obs_from_state: set_state → sim.forward → _check_success
       → _post_process → force update observables → raw observation
    → disable_camera_observable()，仅在中间 7 步不渲染 agentview
    → 每个 8-step chunk 边界：refresh_policy_observation()
       激活 agentview → force observables → _get_observations → 关闭 camera
    → policy_observation()
       仅保留 agentview_image、robot0_eef_pos、robot0_gripper_qpos、robot0_joint_pos
    → C0ActorPolicy.plan_batch()
       CLIPProcessor: uint8 HWC → pixel_values [B,3,224,224]
       frozen CLIP image encoder → normalized image features [B,512]
       cached frozen CLIP text encoder → normalized language features [B,512]
       concat 3+2+7 → proprio [B,12] → train-stat normalization
       C0Actor → normalized action chunk [B,8,7]
       checkpoint action mean/std 反标准化 → clip [-1,1]
    → evaluator 取 chunk 的一个 [7] action，再 clip [-1,1]
    → env.step(action)
    → env.check_success()
       成功：停止该 episode 并写 success/steps；否则继续，最多 300 步
  → task 完成后把 partial summary 原子替换 output；全部完成时 complete=true
```

reward、done、info 虽由 `env.step` 返回，却被赋给 `_`，不传入 policy 或 success 逻辑。

## 15. 当前 Tensor shape 总表

| 名称 | 当前位置 / dtype | 实际 shape |
|---|---|---|
| raw agentview RGB | 原始 HDF5 / `uint8` | `[128,128,3]` |
| raw wrist RGB | 原始 HDF5 / `uint8`，不使用 | `[128,128,3]` |
| CLIP processed RGB | `CLIPProcessor` / `float32` | `[B,3,224,224]` |
| offline image feature | HDF5 / `float16` | `[N,512]` |
| actor image input | Dataset / `float32` | `[B,512]` |
| language feature HDF5 | `float16` | `[10,512]` |
| actor language input | Dataset / `float32` | `[B,512]` |
| raw proprio | `float32` | `[B,12]` |
| normalized proprio | `float32` outside AMP | `[B,12]` |
| actor sample target | `float32` | `[8,7]` |
| actor batch target | `float32` | `[B,8,7]` |
| visual/lang hidden | AMP dtype in training | `[B,256]` each |
| proprio hidden | AMP dtype in training | `[B,128]` |
| fused hidden | AMP dtype in training | `[B,640]` |
| action head intermediate | AMP dtype in training | `[B,512] → [B,256] → [B,56]` |
| actor output | AMP `float16` in CUDA probe | `[B,8,7]` |
| environment action | NumPy `float32` | `[7]` |

## 16. 与原项目设计的偏差

| 设计要求 | 判定 | 事实 |
|---|---|---|
| 冻结 visual/language encoder | 一致 | CLIP 两塔均 `eval` + `requires_grad_(False)`；actor 只接收 feature。 |
| 只用 RGB、language、允许 proprio | 一致 | canonical feature store 和 online filter 都只保留该契约。 |
| 不用 object ground-truth pose | 一致 | raw HDF 有 state/object-related observation，但 feature/dataset/policy actor 不读取。 |
| 轻量 Actor | 一致 | 739,512 parameter 三路 MLP；无 sequence module。 |
| action chunk H=8 | 一致 | offline target、actor output、online buffer、evaluator horizon 全部为 8。 |
| 固定 evaluation seeds | 一致 | JSON seed record，D11 summary 保存其绝对路径。 |
| 官方 10-task LIBERO-Spatial | 一致 | 10 HDF5、10 task summary、官方 BDDL/init state API。 |
| 官方 success 判定 | 一致 | 所有 C0 evaluator 调 `env.check_success()`，其 wrapper 调上游 `_check_success()`。 |
| train/validation/test 隔离 | 部分一致 | demo train/validation split 和 test rollout 代码隔离；但 test 驱动人工决策的绝对隔离未验证。 |
| 每 run 保存 config / commit / raw episode result | 部分一致 | config、metadata、feature manifest hash、checkpoint、raw results 有；根项目 commit 不存在，运行 metadata 不记录源码 hash。 |
| 不把单条视频当结果 | 一致 | D11 是完整 JSON；GUI/teleop 只是 wiring/interaction probe。 |
| 正向 RGB | 有意改变 / 已纠正 | 早期 feature 版本做 vertical flip；经 `image_alignment.json` 后 canonical store 改为 direct RGB，旧 run 保留为 ablation。方向审计自身有时间索引缺陷，见 §4。 |

## 17. 已知问题、技术债和风险

1. **根 Git 没有 commit（高）**：无法锁定代码；所有关键源码/结果未跟踪或忽略。独立审查不能用 commit 复原本项目状态。
2. **demo action/observation 语义 off-by-one 未严格验证（高）**：只验证了 shape 与本实现的同 index copy；没有 replay `obs[t] + action[t]` 至 `obs[t+1]` 的语义测试。
3. **`audit_image_alignment.py` 时间索引缺陷（中）**：循环 states 0/1/2，却恒定使用 stored RGB 0；不能作为多时刻对齐证据。
4. **训练 resume 会丢失历史 best metric（中）**：`best_validation` 没有存进 checkpoint，恢复后可用更差 validation 覆盖原 `best.pt`。
5. **D9 unit test 与真实训练 loss 不同（中）**：test 是 CPU、无 seed、raw-target MSE；训练是 AMP、normalized-target SmoothL1。它不能验证正式 objective 的数值正确性。
6. **统计口径易混淆（中）**：manifest 是 raw actions 的 train-only stats；真正训练 checkpoint stats 来自重叠 `action_chunks`，每个 interior action 重复计数。二者均无 test 泄漏，但不应互换。
7. **在线 action clipping ratio 未记录（中）**：contract 要求记录，result JSON 没有；无法量化 actor 是否频繁饱和。
8. **上游 reset silent retry（中）**：RandomizationError 被无限重试且无计数/日志；评测没有 episode skip，但 reset 失败次数不可审计。
9. **私有环境 API（中）**：all-task evaluator 使用 `env.env._observables`、`_update_observables`、`_get_observations` 进行 camera gating，依赖 robosuite/LIBERO 内部实现。已做一个 task 的 aggregate success 检查，但逐 episode step 不完全一致。
10. **batch FP16 行为差异（中）**：`parallel_envs=4` 的 camera-gated task0 artifact 为 12/20；single-stream artifact 为 11/20。代码因此默认 `parallel_envs=1`，D11 确实使用 1；这说明 batch-size 是行为相关的数值协议变量。
11. **训练/部署 encoder parity 未逐值测试（中）**：offline feature 保存为 float16；online CLIP realtime 重算且转 float32。方向检查存在，但未验证每个源 state 的 offline/online embedding 相等。
12. **C0 低分（非实现错误，但研究风险高）**：全任务仅 8.5%，7/10 task 为 0/20；不能把 C0 描述为有效全任务操控策略。
13. **只跑一个 training seed（预期阶段限制）**：seed 23/47 尚未跑；D11 合格，但不是最终主表。
14. **默认参数/文字陈旧（低）**：`C0Actor` 默认 proprio 15，但实际为 12；adapter error string 写“contract-v1 12-D”，实际 contract 文档为 v2。
15. **feature extraction 的网络依赖（低）**：`extract_features.py` 的 `snapshot_download()` 不设 `local_files_only=True`；在线 policy 已强制 local-only。重新抽取会受网络/proxy 影响，尽管 revision 固定。
16. **hard-coded 路径（低）**：默认项目根、`.libero`、feature output、D9 test checkpoint、task0 data path 均硬编码；适合当前工作区，不适合可移植 package。
17. **无代码 TODO/FIXME 不代表无风险**：搜索只发现 intentional `pass`（zero policy 与 keyboard callback），没有世界模型 TODO；因为 C1 根本未开始。

## 18. 独立审查者的最小复现/复核命令

以下命令在 `/home/luo/embodied-libero-wm` 执行。完整 eval 会运行很久；优先用只读复核命令。`env -u PYTHONPATH` 避免继承 ROS 等外部 Python path。

### 只检查已有产物（最快）

```bash
cd /home/luo/embodied-libero-wm

git branch --show-current
git rev-parse HEAD || true
git status --short --ignored

.venv/bin/python -m json.tool results/c0/seed11/summary.json >/dev/null
.venv/bin/python -m json.tool configs/d12_g1_checklist.json >/dev/null

env -u PYTHONPATH .venv/bin/python - <<'PY'
import json
x = json.load(open('results/c0/seed11/summary.json'))
assert x['complete'] and x['completed_task_ids'] == list(range(10))
assert x['episodes'] == 200 and x['successes'] == 17
assert all(len(t['episode_results']) == 20 for t in x['tasks'])
print('D11 verified:', x['successes'], '/', x['episodes'], x['macro_success_rate'])
PY
```

### 环境、原始数据、feature store 和 actor wiring

```bash
cd /home/luo/embodied-libero-wm
cat artifacts/stage0/gpu_probe.txt

env -u PYTHONPATH LIBERO_CONFIG_PATH="$PWD/.libero" \
  .venv/bin/python scripts/audit_spatial_dataset.py \
  --output /tmp/libero_data_audit.json

env -u PYTHONPATH .venv/bin/python scripts/verify_feature_store.py \
  --manifest artifacts/stage1/features/clip_vit_b32_h8_p12_upright/feature_manifest.json \
  --output /tmp/c0_feature_verification.json

env -u PYTHONPATH .venv/bin/python tests/test_c0_actor.py

env -u PYTHONPATH LIBERO_CONFIG_PATH="$PWD/.libero" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  .venv/bin/python tests/test_c0_policy_adapter.py
```

### 新建临时输出的 smoke / 单任务 / 全任务重跑

```bash
cd /home/luo/embodied-libero-wm

SMOKE_OUTPUT_DIR="$(mktemp -d)" ./scripts/run_smoke.sh

env -u PYTHONPATH LIBERO_CONFIG_PATH="$PWD/.libero" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  .venv/bin/python scripts/evaluate_c0.py \
  --checkpoint results/c0/task0_seed11_upright/best.pt \
  --task-id 0 --output /tmp/task0_c0_audit.json

# 完整重跑很慢，并会从头计算 200 episode；不要指向已有 summary.json。
env -u PYTHONPATH LIBERO_CONFIG_PATH="$PWD/.libero" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  .venv/bin/python scripts/evaluate_c0_all.py \
  --checkpoint results/c0/seed11/best.pt \
  --output /tmp/c0_seed11_full_audit.json \
  --parallel-envs 1 --max-steps 300
```

重跑训练会写新的 checkpoint/metadata，因而不属于本报告的只读审计；如需训练复现，应使用新 output directory，而不能覆盖 `results/c0/seed11/`。

## 19. 关键文件与代码位置索引

| 文件 | 类 / 函数 / 行区间 | 为什么优先看 |
|---|---|---|
| `src/c0/dataset.py` | `FeatureDataset` 14–98；stats 111–154 | 决定训练实际读什么、返回什么，以及 normalization 计数口径。 |
| `scripts/extract_features.py` | `collect_records` 78–150；CLIP 153–189；feature schema 192–227；`write_split` 237–296 | 原始 HDF5 到 feature 的字段、split、t 对齐、dtype 和禁用字段。 |
| `scripts/audit_spatial_dataset.py` | `audit_task` 35–83 | 原始 schema/shape audit；不验证语义 time alignment。 |
| `scripts/audit_image_alignment.py` | 37–47 | 方向 audit，同时暴露 stored-frame index bug。 |
| `src/c0/actor.py` | `C0Actor` 10–59 | 完整 actor 层结构与 `[B,8,7]` 输出。 |
| `scripts/train_c0.py` | `evaluate` 79–102；training 132–325 | normalization、loss、AMP、checkpoint、resume 与 best selection。 |
| `tests/test_c0_actor.py` | 18–46 | 实际 one-batch wiring test；注意其 objective 与正式训练不同。 |
| `src/policies/c0_actor_policy.py` | `C0ActorPolicy` 25–167 | online preprocessing、frozen CLIP、language cache、反标准化、chunk buffer。 |
| `scripts/evaluate_c0.py` | `main` 54–137 | 单任务 reset→policy→step→success 链。 |
| `scripts/evaluate_c0_all.py` | 26–100、103–202、205–318 | fixed seeds/init states、camera gating、raw episode logging、resume/atomic summary。 |
| `configs/eval_seeds_spatial.json` | 全文件 | 10×20 immutable evaluation seed record。 |
| `results/c0/seed11/run_metadata.json` | 全文件 | 实际 run 的 data size、hash、normalization、parameter count。 |
| `results/c0/seed11/metrics.jsonl` | 全文件 | 41,300 train steps 和 50 validation metrics。 |
| `results/c0/seed11/summary.json` | 全文件 | D11 task-level 与 raw episode-level 结果。 |

## 20. 当前 D12 实现事实摘要（20 条以内）

1. 根项目 Git `main` 没有 commit，不能提供当前代码 hash。
2. 第三方 LIBERO 固定在 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。
3. 当前数据是 10 个官方 `libero_spatial` HDF5、每任务 50 demos。
4. C0 只使用 `agentview_rgb`，不使用 wrist camera。
5. C0 实际 proprio 为 12-D：3-D ee position、2-D gripper、7-D joint position。
6. C0 不把 object pose、92-D state、robot_states、reward、done 或 success 传入 actor。
7. 离线 encoder 是冻结 OpenAI CLIP ViT-B/32，vision/text feature 均为 512-D、L2 normalized。
8. canonical feature store 为 direct upright RGB，train/validation 分别有 52,831/5,919 action-chunk samples。
9. 当前 sample 实现为 `obs[t]` feature 对 `actions[t:t+8]`，H=8；该整数索引 copy 已抽查验证。
10. 官方 demo 的 action/observation 语义是否存在整体 off-by-one：未验证。
11. Actor 是 739,512 参数的三路 MLP，没有 GRU、Transformer、world model 或 stochastic head。
12. Actor 输出 deterministic normalized `[B,8,7]`，在线从 checkpoint 反标准化并 clip 到 `[-1,1]`。
13. 正式训练使用 normalized-target SmoothL1；best checkpoint 用 validation raw action MSE 选择。
14. `tests/test_c0_actor.py` 能完成 B=2 forward/backward/step，但它是未设 seed 的 raw-target MSE smoke test。
15. 审计期的 B=64 CUDA 正式-objective probe 产出 finite loss/grad，actor-only peak allocated 30.660 MiB。
16. canonical D10 task0 upright run 为 11/20；早期 flipped ablation 为 12/20，二者均保存。
17. D11 seed-11 summary 完成 200/200 episode，17 成功，宏平均 8.5%。
18. D11 的 183 个失败均保留为 `steps=300` 的原始 episode record；没有 evaluator skip 代码。
19. 上游 reset 对 `RandomizationError` 进行无日志重试，故 reset retry 次数未审计。
20. D13/C1 未开始；任何世界模型、候选动作生成或闭环重规划都不是当前仓库已实现的功能。
