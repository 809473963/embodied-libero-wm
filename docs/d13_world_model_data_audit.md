# D13 World Model Data Audit

审计日期：2026-09-22。本轮只完成 world-model 数据契约、observation-aligned feature store、Dataset 与验收；没有实现 `LatentDynamicsModel`、ensemble、candidate sampler、C1/C2，也没有训练任何世界模型或 C0。

## Verdict

**D13 PASS。** 后续可以进入世界模型阶段，但 D13 本身不含 learned predictor、训练结果或 candidate 结果。若 D14 无法继续满足本报告的 observation/action index 合约，必须停止并标记 `BLOCK D14`。

## Pre-D13 snapshot

| item | value |
|---|---|
| Pre-D13 commit | `013f804dba2cc14955da3762cb77cac2e515df79` |
| Pre-D13 annotated tag | `pre-d13-foundation` |
| D12 baseline tag | `d12-c0-baseline` |
| pinned LIBERO commit | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| dataset revision | `f13aa24a3da8c43c7225569f28c562979fa0e35a` |

`pre-d13-foundation` 指向已经提交的 temporal audit、offline/online parity、resume safeguard、production-equivalent test、clipping logging 与 evaluator protocol。D13 未修改 C0 actor、C0 store、C0 label pairing 或 D12 checkpoint。

## Frozen data contract

版本为 **`d13-v1`**；配置见 [d13_world_model_data.yaml](../configs/d13_world_model_data.yaml)，完整定义见 [world_model_data_contract.md](world_model_data_contract.md)。

- camera：upright `agentview` `uint8[128,128,3]`，`vertical_flip=false`。
- latent：frozen `openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` 的 L2-normalized 512-D image projection。store 为 float16，Dataset 为 float32。
- `proprio0 = ee_pos[3] + gripper_states[2] + joint_states[7] = float32[12]`；language 为 frozen L2-normalized CLIP text `float32[512]`。
- 不使用 wrist camera；不更换或训练 CLIP。

以下两个时间 convention 必须严格分开：

```text
# C0 保持不变：official imitation learning
obs[k] -> actions[k:k+8]

# D13 observation dynamics
(z[k], proprio[k], language, actions[k+1:k+9]) -> z[k+1:k+9]
```

故单 sample 为：

```text
z0          float32[512]    = z_obs[k]
proprio0    float32[12]     = proprio_obs[k]
language    float32[512]    = task language embedding
actions     float32[8,7]    = processed actions[k+1:k+9]
z_future    float32[8,512]  = z_obs[k+1:k+9]
```

episode observation 长度为 `T` 时，合法 `k` 是 `0 <= k <= T-9`，sample 数是 **`T-8`**。这不是 C0 的 `T-8+1`；最后合法 sample 的 `actions[7]` 和 `z_future[7]` 都是 source index `T-1`。

## Observation-aligned store and split

现有 C0 store 是 action-chunk sample store，不能作为完整 world-model trajectory，也不能以 global row 邻接推断 future row。D13 新建独立 HDF5：

```text
tasks/task_{task_id}/demos/demo_{demo_id}/
  vision     [T,512] float16
  proprio    [T,12]  float32
  actions    [T,7]   float32
  timestep   [T]     int32
```

每个 demo group 有 task/demo/split/source-hash/count metadata；root 仅含 `task_metadata` 与 `tasks`。split 沿用完整 episode：每 task demo 0–44 train、45–49 validation；得到 450 train / 50 validation episodes，交集为空。

| task | observations | train obs | val obs | train WM samples | val WM samples |
|---:|---:|---:|---:|---:|---:|
| 0 | 5,068 | 4,613 | 455 | 4,253 | 415 |
| 1 | 6,707 | 6,039 | 668 | 5,679 | 628 |
| 2 | 5,882 | 5,248 | 634 | 4,888 | 594 |
| 3 | 5,052 | 4,570 | 482 | 4,210 | 442 |
| 4 | 7,479 | 6,741 | 738 | 6,381 | 698 |
| 5 | 5,796 | 5,249 | 547 | 4,889 | 507 |
| 6 | 6,312 | 5,606 | 706 | 5,246 | 666 |
| 7 | 7,111 | 6,375 | 736 | 6,015 | 696 |
| 8 | 5,963 | 5,320 | 643 | 4,960 | 603 |
| 9 | 6,880 | 6,220 | 660 | 5,860 | 620 |
| **total** | **62,250** | **55,981** | **6,269** | **52,381** | **5,869** |

每个 task 的完整 observation count 向量按 `[demo_0, ..., demo_49]` 列出如下（前 45 个为 train，最后 5 个为 validation；每个 demo 的 WM sample count 为对应元素减 8）：

| task | observation count `[d0..d49]` |
|---:|---|
| 0 | `[98,84,90,97,105,134,92,92,79,89,93,128,86,88,112,84,89,97,75,99,87,121,96,85,100,91,105,113,92,124,127,151,93,111,167,89,102,168,89,108,95,95,109,96,88,90,85,92,96,92]` |
| 1 | `[152,123,140,175,142,155,145,131,152,127,111,117,114,125,150,128,135,113,112,125,162,134,132,109,189,143,145,116,167,120,154,120,150,117,120,137,120,129,143,124,126,112,137,135,126,133,134,133,141,127]` |
| 2 | `[103,123,122,102,111,111,109,123,111,110,109,101,124,122,114,136,105,164,107,98,117,108,117,117,106,117,143,125,111,110,115,122,119,121,131,140,121,129,114,115,108,106,138,93,100,126,125,117,131,135]` |
| 3 | `[99,101,95,115,103,99,111,95,111,129,89,97,93,99,100,119,101,94,171,100,96,96,96,97,96,98,92,101,95,106,86,99,99,111,89,101,92,97,97,92,95,114,89,102,113,99,103,90,100,90]` |
| 4 | `[153,142,158,129,143,158,167,156,159,132,139,182,145,141,138,150,131,156,148,162,153,163,149,165,140,151,149,132,142,139,171,190,156,147,131,172,163,138,140,169,157,134,127,137,137,169,147,151,134,137]` |
| 5 | `[137,131,121,121,119,100,101,129,123,118,137,100,96,128,154,102,116,120,114,98,118,129,107,138,98,91,98,147,106,116,119,118,111,109,111,128,110,102,98,86,117,128,144,113,142,112,108,110,108,109]` |
| 6 | `[123,137,125,108,123,130,124,123,133,113,127,111,124,124,141,106,118,141,130,113,122,108,124,120,133,146,122,129,155,123,135,141,124,113,115,122,130,125,131,117,110,133,114,107,133,154,122,119,159,152]` |
| 7 | `[155,125,124,191,136,138,133,130,139,145,140,164,124,138,120,137,141,152,137,128,197,124,128,126,144,129,168,132,140,133,135,140,142,133,157,167,148,128,133,128,130,129,176,170,141,142,180,124,136,154]` |
| 8 | `[115,107,138,157,138,113,110,107,104,126,116,125,106,137,97,152,114,165,99,104,125,127,100,110,111,140,109,116,124,173,96,116,95,109,122,111,121,110,105,123,114,120,97,99,117,113,133,113,169,115]` |
| 9 | `[136,138,130,136,145,139,164,119,135,173,142,131,126,134,136,193,142,182,133,135,137,137,145,140,124,141,143,124,141,122,132,125,130,144,145,125,133,138,140,127,139,128,132,128,131,130,139,136,117,138]` |

## Dataset trace and back-checks

`WorldModelDataset` indexes each legal start inside exactly one `(task_id, demo_id)` group. It provides `copy_current_state_prediction(z0) = z0.unsqueeze(0).repeat(8, 1)` for the future D16 copy-current-state baseline only; this is not a learned prediction.

真实 train sample `(task=0, demo=0, k=0)` shape：`z0 [512]`、`proprio0 [12]`、`language [512]`、`actions [8,7]`、`z_future [8,512]`。真实 `DataLoader(B=4)` shape：`z0 [4,512]`、`proprio0 [4,12]`、`language [4,512]`、`actions [4,8,7]`、`z_future [4,8,512]`，以及三个 `[4]` metadata tensors。

独立 back-check 直接读 episode-local observation feature group，actions 同时回读官方 processed source HDF5。以下全部通过 `z0=z[k]`、`actions[0]=source_action[k+1]`、`actions[7]=source_action[k+8]`、`z_future[0]=z[k+1]`、`z_future[7]=z[k+8]`：

| split | `(task,demo,k)` | role |
|---|---|---|
| train | `(0,0,0)` | sample trace |
| train | `(4,17,5)` | middle index |
| train | `(8,44,108)` | final legal train sample (`T=117`) |
| validation | `(0,45,0)` | validation start |
| validation | `(6,47,5)` | validation middle |
| validation | `(9,49,129)` | final legal validation sample (`T=138`) |

最后两条证明 `k=T-9` 时第 8 个 action/target 恰为 `T-1`，不越界、不跨 demo。`tests/test_world_model_dataset.py` 还验证 H=8、action=7、latent=512、proprio=12、train/validation demo 无交集、四个跨 task/demo/timestep direct checks 和 copy baseline `[8,512]`。

## Oracle audit and hashes

递归字段审计检查 `states`、`robot_states`、`object_pose`、`reward(s)`、`done(s)`、`success`、`future_proprio`、`evaluation_episode`，结果为 `forbidden_fields_found=[]`。每个 demo group 的精确 field set 仅为 `actions, proprio, timestep, vision`；future proprio 不进入 Dataset 输入，`z_future` 只作为训练 target。

| object | bytes | SHA-256 |
|---|---:|---|
| `observations.hdf5` | 68,073,856 | `e5599153cf44282242d955be62249bbafc752b782f5ae76a0ec0d58f9146f98e` |
| feature manifest | 84,874 | `7c374d187e06c6ef99851c159134e722fa96acca9d69461bd1541e1a2895b0dc` |
| D13 data audit | 85,420 | `950b554182e7b6671155d03533ce2878069da6742fd4517adbe2fa7221d89ac7` |

新增文件：

- [world_model_data_contract.md](world_model_data_contract.md)
- [d13_world_model_data.yaml](../configs/d13_world_model_data.yaml)
- `src/d13/dataset.py`、`src/d13/__init__.py`
- `scripts/extract_world_model_features.py`、`scripts/audit_world_model_data.py`
- `tests/test_world_model_dataset.py`
- 本报告

`observations.hdf5` 保持 Git ignored，不提交；轻量 `feature_manifest.json` 与 `data_audit.json` 可提交以提供 hash 和 per-demo 追溯。

## Verification

已通过 `py_compile`、`tests/test_world_model_dataset.py` 与 `scripts/audit_world_model_data.py`：52,381 train、5,869 validation samples；450/50 episode split 无交集；6 个 index back-check 全部通过；oracle-field audit 通过。D13 PASS，D14 未开始。
