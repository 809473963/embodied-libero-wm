# D13 World Model Data Contract

版本：`d13-v1`。本文件冻结 D13 的 world-model 数据语义；D13 只建立 observation-aligned feature/transition store 与 `WorldModelDataset`，不定义、实现或训练任何 latent dynamics network。

## 时间语义

令 `k` 为 processed LIBERO HDF5 observation index。Pre-D13 temporal audit 已确认 converter 在执行 `action[j]` 后保存 `obs`，同时在循环结束后用同一个 `valid_index` 选择 `states` / `actions`。因此以下两个语义必须严格分开：

```text
# 保留，不修改 C0 的官方 imitation-learning convention
obs[k] -> actions[k:k+H]

# D13 observation dynamics convention
obs[k] + action[k+1] -> obs[k+1]
```

本契约固定 `H=8`。一个 D13 sample 是：

```text
input:
  z0        = z_obs[k]                 float32[512]
  proprio0  = proprio_obs[k]           float32[12]
  language  = frozen CLIP text feature  float32[512]
  actions   = actions[k+1:k+H+1]       float32[8,7]

target:
  z_future  = z_obs[k+1:k+H+1]         float32[8,512]
```

也即：`(z[k], p[k], l, a[k+1:k+H+1]) -> z[k+1:k+H+1]`。它绝不能使用 `actions[k:k+H]` 来学习 observation dynamics。

对 observation trajectory 长度 `T`，合法 start 是 `0 <= k <= T-H-1`，每 episode 的 sample 数为 **`T-H`**。最后合法样本使用 `actions[k+8]` 和 `z[k+8]`，它们均落在 index `T-1`；`T-H` 不是 C0 action-chunk sample store 的 `T-H+1`。

## Latent 与可见输入

`z_t.shape == [512]`，定义为 frozen `openai/clip-vit-base-patch32` 的 image projection output，revision 为 `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`，随后 L2 normalize。

- 相机唯一为 `agentview`；raw image 是正向、未竖直翻转的 `uint8[128,128,3]`。
- HDF5 feature store 可以以 float16 保存 `vision`；`WorldModelDataset` 必须将它转换成 float32 tensor。
- 不使用 wrist/`eye_in_hand` camera，不更换或训练 CLIP。
- `proprio0` 固定为 `ee_pos[3] + gripper_states[2] + joint_states[7] = 12-D`。
- language 是 frozen CLIP 的 512-D、L2-normalized text embedding。

## Observation-aligned store 与 split

现有 C0 feature store 是 action-chunk **sample store**：它只保存具备完整 `actions[t:t+H]` 的起始 observation，不能作为完整 trajectory 或以相邻 global feature row 推断 future observation。D13 使用独立的 observation-aligned HDF5，逻辑结构为：

```text
tasks/task_{task_id}/demos/demo_{demo_id}/
  vision     [T,512] float16
  proprio    [T,12]  float32
  actions    [T,7]   float32
  timestep   [T]     int32
```

每个 group 带 `task_id`、`demo_id`、source HDF5 path、observation count；root metadata 存放 task name、language、language feature、pinned encoder revision 和 contract version。不得使用相邻 global row 形成 `z_future`。

split 不变且按完整 episode：每 task `demo_0`–`demo_44` 训练、`demo_45`–`demo_49` 验证。禁止随机重新划分，禁止 episode leakage，禁止读取 simulator fixed evaluation episodes。

## 禁止字段与部署边界

store、Dataset、未来模型输入或候选筛选器不得包含 simulator `states`、`robot_states`、object pose、reward、done、success、测试 episode 信息，或**未来 proprio**。`z_future` 仅是训练 target，不能成为部署时输入。Dataset 提供 copy-current-state baseline interface：`z0.unsqueeze(0).repeat(H, 1)`，供将来的 D16 与 learned predictor 公平比较；它不是 learned world model 结果。

## 验收

`tests/test_world_model_dataset.py` 必须验证 sample / batch shape、H=8、index shift、episode boundary、split isolation、oracle field absence，以及来自独立 HDF5 group/source action 的 back-check。D13 只有在这些检查通过后才允许开始 D14；D13 本身不训练任何模型。
