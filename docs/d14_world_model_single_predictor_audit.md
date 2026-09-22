# D14 Single Latent Forward Predictor Audit

## Verdict

**D14 PASS — ready to enter D15 when explicitly requested.**

This verdict covers one learned latent forward predictor only. The
copy-current comparison below is a D14 diagnostic, **not** a D16 G2 result.
No ensemble, uncertainty estimate, candidate sampler, reranker, verifier,
C1/C2 evaluation, or fixed simulator evaluation episode was used.

## Frozen source and scope

- D13 source commit: `e37073e588505b4461d11dc81b174032ff297f2d`
- D13 annotated tag: `d13-wm-data-v1`
- D14 implementation commit: `c276bbcced898b2df7b41cf95e964390dd01fb81`
- D13 contract version: `d13-v1`
- D13 store SHA-256: `e5599153cf44282242d955be62249bbafc752b782f5ae76a0ec0d58f9146f98e`
- Data-contract SHA-256: `e988b7e1078b3c89614bc83225ba677045cfda3b854098a3e1180b85cca32181`
- D14 config SHA-256: `fc32c3c006fec9bc57a5aaeca2a3b2e0655e0dbba17c483826d7c1d52fc76577`
- Train/validation samples: 52,381 / 5,869
- Split remains demo 0–44 / demo 45–49 per task.
- C0 and D13 source files were not modified by D14.

The data pairing remains exactly:

```text
z0         = z[k]
proprio0   = proprio[k]
language   = task language
actions    = actions[k+1:k+9]
z_future   = z[k+1:k+9]
```

`H=8`; no same-index action, future proprio, reward, done, success, simulator
state, object pose, or evaluation/test episode is present.

## Model

`LatentDynamicsModel` has **792,000 trainable parameters**.

```text
z0          [B,512]   -> Linear/LN/GELU -> [B,256]
proprio0    [B,12]    -> Linear/LN/GELU -> [B,64]
language    [B,512]   -> Linear/LN/GELU -> [B,128]
concat                              -> [B,448]
initial-hidden projection           -> [1,B,256]

actions      [B,8,7] -> Linear/LN/GELU -> [B,8,64]
language repeated                       -> [B,8,128]
GRU input                               -> [B,8,192]
GRU output                              -> [B,8,256]
delta head                              -> [B,8,512]
L2Normalize(z0[:,None,:] + delta_z)      -> [B,8,512]
```

Primary loss is only mean cosine distance over batch and horizon:

```text
mean(1 - cosine(pred_z, z_future))
```

## Normalization contract and statistics

`z0` and `language` retain their frozen-CLIP L2 normalization and are not
standardized. Population statistics use train episodes only. Validation uses
the same train-derived statistics.

- Proprio count: **55,981**. Every unique train observation row is included
  exactly once; overlapping WM samples are never expanded for statistics.
- Action count: **55,531**. For each of 450 train episodes, only `actions[1:]`
  is included; `action[0]` is excluded and every remaining raw row is counted
  exactly once.
- Minimum standard deviation: `1e-6` (no dimension required clamping).

| Input | Mean | Standard deviation |
|---|---|---|
| proprio[0:3] | `[-0.02578376, 0.10606899, 1.05645103]` | `[0.11198754, 0.13838893, 0.10302571]` |
| proprio[3:6] | `[0.01967004, -0.01986377, 0.11607296]` | `[0.01709845, 0.01692213, 0.13876829]` |
| proprio[6:9] | `[0.63062408, 0.05883583, -1.65001935]` | `[0.41810828, 0.12199119, 0.57098416]` |
| proprio[9:12] | `[-0.15051613, 2.30217306, 1.11536489]` | `[0.20987855, 0.32549205, 0.47887201]` |
| action[0:7] | `[0.15043248, 0.13491899, -0.15583906, -0.00498378, -0.01116378, -0.01996221, 0.10107868]` | `[0.41536851, 0.34807537, 0.51000369, 0.03836074, 0.07189857, 0.05811741, 0.99487844]` |

The full-precision values, counts, rules, store hash, and contract hash are in
both `run_metadata.json` and every checkpoint.

## Forward/backward verification

Synthetic `B=2`:

- input shapes: `[2,512]`, `[2,12]`, `[2,512]`, `[2,8,7]`
- output: `[2,8,512]`, finite, each latent norm within `1e-5` of 1
- cosine loss: `0.9956287146`, finite
- all parameter gradients finite; AdamW step completed

Real D13 `B=4` using the formal normalization:

- `z0`: `[4,512]`
- `proprio0`: `[4,12]`
- `language`: `[4,512]`
- `actions`: `[4,8,7]`
- `z_future`: `[4,8,512]`
- output: `[4,8,512]`
- pre-training one-batch cosine loss: `0.0138439033`
- backward and AdamW step completed; all gradients finite
- one-batch allocated GPU peak: 33,039,360 bytes

The production training batch size was 256 with the same trailing dimensions.

## Training run

- Seed: 11
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`
- AMP: fp16 enabled
- Gradient clipping: global norm `10.0`
- Epochs / optimizer steps: 15 / 3,075
- Wall time: 275.22 seconds
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- Peak allocated / reserved GPU memory: 75,717,120 / 123,731,968 bytes

| Epoch | Train overall cosine error | Validation overall cosine error |
|---:|---:|---:|
| 1 | 0.01259231 | 0.01147971 |
| 2 | 0.01075776 | 0.01063650 |
| 3 | 0.01003459 | 0.01025582 |
| 4 | 0.00963002 | 0.00997342 |
| 5 | 0.00931473 | 0.00979263 |
| 6 | 0.00909460 | 0.00967032 |
| 7 | 0.00887759 | 0.00955096 |
| 8 | 0.00871355 | 0.00970238 |
| 9 | 0.00857183 | 0.00949817 |
| 10 | 0.00846587 | 0.00947085 |
| 11 | 0.00833223 | 0.00948123 |
| 12 | 0.00820870 | 0.00941561 |
| **13** | **0.00813217** | **0.00937134** |
| 14 | 0.00802593 | 0.00939009 |
| 15 | 0.00792912 | 0.00945917 |

Epochs 8, 11, 14, and 15 did not overwrite the better historical checkpoint.
A resume-load check from `last.pt` recovered step 3,075 and historical best
`0.0093713445`.

## Best validation and copy-current diagnostic

The independently reloaded epoch-13 checkpoint reproduced the stored overall
error exactly: **0.009371344546612412** over all 5,869 validation samples.

| Horizon | Learned error | Copy-current error | Copy − learned |
|---:|---:|---:|---:|
| h1 | 0.00686870 | 0.00702433 | +0.00015562 |
| h2 | 0.00801518 | 0.01017511 | +0.00215992 |
| h3 | 0.00880832 | 0.01295711 | +0.00414879 |
| h4 | 0.00940585 | 0.01541428 | +0.00600843 |
| h5 | 0.00986578 | 0.01758327 | +0.00771749 |
| h6 | 0.01026954 | 0.01946334 | +0.00919380 |
| h7 | 0.01065371 | 0.02108555 | +0.01043184 |
| h8 | 0.01108366 | 0.02265956 | +0.01157589 |
| overall | **0.00937134** | **0.01579532** | **+0.00642397** |

The aggregate learned error is 40.67% below copy-current. All aggregate
horizons improve, although h1 improves only slightly and some individual
samples are worse at individual horizons. This table is diagnostic evidence
only and is not a formal D16 gate.

Five fixed validation examples from task 0 / demo 45:

| start k | Learned mean error | Copy mean error |
|---:|---:|---:|
| 0 | 0.00650341 | 0.00875519 |
| 1 | 0.00606468 | 0.01407964 |
| 2 | 0.00514813 | 0.01065468 |
| 3 | 0.00567679 | 0.01061027 |
| 4 | 0.00530947 | 0.01832637 |

The exact h1–h8 learned and copy errors for 16 validation samples are retained
in `results/world_model/d14_single_seed11/best_validation.json`.

## Checkpoint and finite-value audit

- Best checkpoint: `results/world_model/d14_single_seed11/best.pt`
- Epoch / global step: 13 / 2,665
- Size: 9,546,723 bytes
- SHA-256: `12304c9da129f93325201fad84ce38796ca71fcf427821f64644d4203f6b3888`
- Required state present: model, optimizer, AMP scaler, epoch, global step,
  historical best, RNG, full config and hash, D13 version/store/contract hashes,
  normalization statistics, and source commit.
- NaN/Inf audit: all 26 model-state tensors and all 78 optimizer-state tensors
  are finite; every training loss, gradient norm, and validation metric passed
  runtime finite checks.

`best.pt`, `last.pt`, and `metrics.jsonl` remain ignored and are not committed.
The lightweight run metadata, exact validation diagnostics, and config snapshot
are committed for traceability.

## Files introduced by D14

- `configs/world_model.yaml`
- `src/world_models/__init__.py`
- `src/world_models/latent_dynamics.py`
- `src/world_models/normalization.py`
- `scripts/train_world_model.py`
- `tests/test_world_model.py`
- `results/world_model/d14_single_seed11/config.yaml`
- `results/world_model/d14_single_seed11/run_metadata.json`
- `results/world_model/d14_single_seed11/best_validation.json`
- `docs/d14_world_model_single_predictor_audit.md`

## Git status

The acceptance target is a clean tracked/untracked status after committing this
report and the lightweight run records. Expected ignored files remain: source
datasets, D13 `observations.hdf5`, D14 checkpoints/JSONL, virtual environment,
and caches. No large data, HDF5, checkpoint, or cache is staged.
