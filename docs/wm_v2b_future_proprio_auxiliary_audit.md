# Future-proprio supervision learns local physical dynamics, but does not transfer enough sensitivity to visual prediction

Date: 2026-09-24

Final verdict: `PHYSICAL_SIGNAL_LEARNED_VISUAL_TRANSFER_INSUFFICIENT`

## Executive summary

WM-v2B succeeds at the new physical task but fails its primary visual-local
gate. The auxiliary head predicts standardized future-proprio displacement
with SmoothL1 `0.00187477`, versus `0.03250166` for zero displacement. Correct
actions beat within-task shuffled actions by 2,832.5%, with 98.85% paired
accuracy. Frozen A0/A1/A2 candidates produce a physical trajectory
separation/error ratio of `0.4550`, far above the preregistered `0.05` floor.

That signal does not transfer sufficiently to the visual latent head. Visual
production-candidate separation/error is `0.00515718`: below the absolute
`0.01` gate, only 1.218x the WM-v1 reference, and slightly below WM-v2A's
`0.00530693`. Ordinary visual prediction remains acceptable at `0.00962736`,
only 2.73% above same-seed D14 and well within the 5% safety gate. The model
therefore learned action-conditioned robot motion without making nearby
actions sufficiently distinguishable in its predicted CLIP trajectories.

The result is not a failure of candidate separation, proprio supervision, or
global action conditioning. It isolates the remaining failure to transfer
from the shared dynamics representation into the visual prediction head.
Seed23/47 are not authorized, no ensemble is built, and controller/reranker
work remains blocked.

No simulator or controller rollout was run. Only D13 train demos 0--44 and
validation demos 45--49 were used; no init-state outcome or success metric was
read.

## Frozen provenance and boundaries

WM-v2A is frozen at commit
`8c884ac69bfd76b079181d7ed41d52db48914870` with annotated tag
`wm-v2a-delta-objective-insufficient`. Its verdict
`DELTA_OBJECTIVE_INSUFFICIENT` and the upstream conclusions
`WM_HAS_ACTION_SIGNAL_BUT_LOW_LOCAL_SENSITIVITY`,
`WM_ACTION_SENSITIVITY_BLOCK`, `RERANKER_SCORE_BLOCK`, and
`BLOCK_C1_V2_RERANKER` remain unchanged.

WM-v2B gates, target semantics, calibration rule, model, diagnostic conditions,
and tests were preregistered at commit
`fdafce8d649a48d19ebeb721f4c43a0c961f0504`. A metadata-only Python boolean
fix was committed before calibration as `1ff2811a121d9571ef0cbc3b9a2862deea2d99aa`.
The completed calibration was frozen before training at
`801809f880517cebbe6c5d7ebc8b58a4d5198048`; that exact commit is recorded in
the checkpoint.

| Frozen source | SHA-256 |
|---|---|
| D13 observation store | `e5599153cf44282242d955be62249bbafc752b782f5ae76a0ec0d58f9146f98e` |
| D14 seed11 checkpoint | `12304c9da129f93325201fad84ce38796ca71fcf427821f64644d4203f6b3888` |
| WM-v2A checkpoint | `1e3741c1d5307014b070c6256af0753c8cb86f4c591b78c738324161223e1fe5` |
| WM-v2A diagnosis | `68f5d7b8fb01440918cc2dc7b8d4557eaf752362c8610085e5a6aede2373889c` |
| Frozen C1-v2.1 candidates | `5870337ad96c58b3cedee4e124cf89018408446ed82fc7a2df52626f482398fd` |
| Frozen C1-v2.2 diagnostic sidecar | `24f8a8d870237dda04f63065457c92565f9e4b9b0633a8611b6ec23e089f33fd` |
| WM-v2B config | `e0276dc4b205a2b0aee50a57d1b0465081ae77cb0c7bc914a9f4382a3301596b` |
| WM-v2B protocol | `0f5e1f6e3bd18eb61fe848b28b779fe92519643a454bb15f64de13e81c7473a2` |

Frozen D14/D15/WM-v2A checkpoints and all C1-v2/v2.1/v2.2 artifacts were not
overwritten.

## Future-proprio target remains target-only

The D13 causal input/visual-target contract is unchanged:

```text
(z[k], proprio[k], language, actions[k+1:k+9])
    -> z[k+1:k+9]
```

WM-v2B reads the already observation-aligned episode trajectory to construct:

```text
proprio_future = proprio[k+1:k+9]       float32 [8,12]
p0_norm        = (proprio[k]-mean_p)/std_p
p_future_norm  = (proprio_future-mean_p)/std_p
target_p_delta = p_future_norm - p0_norm[:,None,:]
```

The mean/std are the frozen train-only unique-observation statistics from
55,981 rows in demos 0--44. No overlapping WM window is expanded to recompute
statistics; validation uses the train-derived values. Independent index tests
match `proprio_future[0]` to source `proprio[k+1]` and
`proprio_future[7]` to `proprio[k+8]` across train/validation tasks, demos,
interior timesteps, and final legal samples.

Sample counts remain exactly 52,381 train and 5,869 validation, H=8, with
train demos 0--44 and validation demos 45--49. Future proprio is never passed
to `forward`; the deployment signature remains only `z0`, `proprio0`,
`language`, and `actions`. It is not used by the candidate score, controller,
or any deployment path.

## The auxiliary target is independent of visual cosine loss

WM-v2A's target cancelled algebraically into its visual residual. WM-v2B does
not:

```text
L_z = 1 - cosine(pred_z, z_future)
L_p = SmoothL1(pred_p_delta,
               standardized(proprio_future)-standardized(proprio0))
```

`L_p` uses a separate observed modality, a separate 12-D target, and a
separate output head. Holding `pred_z/z_future` fixed while changing the real
future-proprio target changes `L_p` without changing `L_z`; this is explicitly
covered by the objective-independence test. No wrong-action counterfactual
target is manufactured.

## Architecture changes only by a 34,444-parameter head

The 792,000-parameter D14 visual backbone is unchanged tensor-for-tensor at
seed-11 initialization: z/proprio/language/action projections, initial hidden
projection, GRU, and latent delta head are identical. WM-v2B adds only:

```text
shared GRU hidden [B,8,256]
  -> Linear(256,128)
  -> GELU
  -> Linear(128,12)
  -> pred_p_delta [B,8,12]
```

| Component | Parameters |
|---|---:|
| Frozen-form D14 backbone | 792,000 |
| Future-proprio auxiliary head | 34,444 |
| WM-v2B total | 826,444 |

There is no additional GRU, Transformer, independent dynamics network,
camera, or representation change.

## Gradient-scale calibration fixed lambda before training

Calibration used exactly 1,024 unique train samples selected by
`np.random.default_rng(220501).permutation(52381)[:1024]`, preserved as four
sequential batches of 256. All identities are stored; their SHA-256 is
`f1de5bb5c09c572c5f5b735525d8e039ba5ac3a431de7627b4a6f279844e2cf9`.
Every demo ID is 0--44. Calibration ran in CPU float32 on the seed-11 initial
model with zero optimizer updates and was reproduced byte-for-byte.

For each fixed batch, visual and proprio losses were backpropagated separately
to `model.gru.parameters()` only. The arithmetic mean of the per-batch shared-
GRU L2 gradient norms defines:

```text
g_z = 0.0142945286
g_p = 0.0334823951
lambda_p_raw = g_z / g_p = 0.4269267035
lambda_p = clip(lambda_p_raw, 0.05, 20) = 0.4269267035
```

The value was not clipped and was frozen in
`configs/wm_v2b_calibration_freeze.json` before training. No validation metric
or local-sensitivity result entered calibration.

| Batch | g_z | g_p |
|---:|---:|---:|
| 0 | 0.01433319 | 0.03311120 |
| 1 | 0.01431771 | 0.03336997 |
| 2 | 0.01433145 | 0.03293651 |
| 3 | 0.01419578 | 0.03451190 |

The frozen training objective is:

```text
L_total = L_z + 0.4269267035 * L_p
```

All horizons are equally weighted.

## Training and validation curves

Training matches D14's AdamW, learning rate `3e-4`, weight decay `1e-4`, batch
256, fp16 AMP, gradient clip 10, seed 11, and 15 epochs. The run used an NVIDIA
GeForce RTX 4070 Laptop GPU and peaked at 78,600,704 allocated bytes. The
checkpoint rule was frozen as minimum validation total loss, with visual loss
only as a tie-breaker.

| Epoch | Train visual | Train proprio | Train total | Val visual | Val proprio | Val total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01261449 | 0.01009832 | 0.01692573 | 0.01155051 | 0.00456583 | 0.01349978 |
| 2 | 0.01082739 | 0.00342994 | 0.01229172 | 0.01099555 | 0.00322807 | 0.01237370 |
| 3 | 0.01021454 | 0.00262207 | 0.01133398 | 0.01034510 | 0.00286184 | 0.01156690 |
| 4 | 0.00982170 | 0.00224078 | 0.01077835 | 0.01020929 | 0.00252157 | 0.01128582 |
| 5 | 0.00953708 | 0.00199063 | 0.01038694 | 0.01002632 | 0.00240967 | 0.01105507 |
| 6 | 0.00933248 | 0.00182158 | 0.01011017 | 0.00994116 | 0.00224953 | 0.01090155 |
| 7 | 0.00914591 | 0.00168449 | 0.00986506 | 0.00987288 | 0.00216247 | 0.01079610 |
| 8 | 0.00899554 | 0.00158010 | 0.00967013 | 0.00981592 | 0.00207996 | 0.01070391 |
| 9 | 0.00887003 | 0.00147774 | 0.00950091 | 0.00993965 | 0.00204259 | 0.01081169 |
| 10 | 0.00874843 | 0.00140394 | 0.00934781 | 0.00982685 | 0.00196185 | 0.01066442 |
| 11 | 0.00864543 | 0.00133706 | 0.00921625 | 0.00977811 | 0.00192272 | 0.01059898 |
| 12 | 0.00854298 | 0.00127720 | 0.00908825 | 0.00978812 | 0.00195344 | 0.01062209 |
| 13 | 0.00845016 | 0.00123536 | 0.00897757 | 0.00980524 | 0.00190350 | 0.01061789 |
| **14** | **0.00836522** | **0.00118120** | **0.00886950** | **0.00962736** | **0.00187477** | **0.01042775** |
| 15 | 0.00829492 | 0.00114029 | 0.00878174 | 0.00975019 | 0.00184348 | 0.01053722 |

Epoch 14 / step 2870 is selected. Epoch 15 has slightly lower proprio error
but worse frozen total loss, so historical-best protection correctly retains
epoch 14.

## Ordinary visual prediction remains within the safety gate

| Metric | Result | Gate/reference | Pass |
|---|---:|---:|---|
| Correct-action visual cosine error | 0.00962736 | — | — |
| Relative change vs D14 seed11 | +2.732% | <=+5% | yes |
| Copy-current error | 0.01579531 | learned must be lower | yes |
| D15 ensemble error | 0.00891906 | report-only | — |

| Horizon | WM-v2B visual | Copy current | Proprio SmoothL1 | Zero-delta proprio |
|---:|---:|---:|---:|---:|
| h1 | 0.00711130 | 0.00702430 | 0.00030497 | 0.00141953 |
| h2 | 0.00825129 | 0.01017508 | 0.00071286 | 0.00564130 |
| h3 | 0.00905598 | 0.01295711 | 0.00118891 | 0.01256149 |
| h4 | 0.00964796 | 0.01541429 | 0.00165870 | 0.02199080 |
| h5 | 0.01010687 | 0.01758325 | 0.00210885 | 0.03345862 |
| h6 | 0.01052346 | 0.01946334 | 0.00254218 | 0.04660538 |
| h7 | 0.01092741 | 0.02108558 | 0.00298374 | 0.06122613 |
| h8 | 0.01139455 | 0.02265953 | 0.00349798 | 0.07711016 |

h1 visual is slightly worse than copy-current, while h2--h8 are better; the
overall model still comfortably beats copy-current.

## Future-proprio prediction is learned across all groups

Correct-action proprio SmoothL1 is `0.00187477`, 94.23% below the zero-delta
baseline `0.03250166`. Per-group errors in standardized units are:

| Group | Dimensions | SmoothL1 |
|---|---|---:|
| ee position | 0--2 | 0.00052121 |
| gripper | 3--4 | 0.00432357 |
| joints | 5--11 | 0.00175519 |

Per-dimension errors `[ee_pos(3), gripper(2), joints(7)]` are:

```text
[0.00054957, 0.00021405, 0.00080000,
 0.00396452, 0.00468262,
 0.00174721, 0.00081099, 0.00264317, 0.00145461,
 0.00207270, 0.00301829, 0.00053933]
```

Every task beats its own zero-delta baseline. Task 7 has the largest proprio
error (`0.00413777`), but remains far below its zero baseline (`0.03299476`).

## The auxiliary head is genuinely action-conditioned

Only raw physical actions change; initial visual/proprio/language inputs and
the true future-proprio target remain paired and fixed.

| Wrong action | Proprio error | Relative increase | Correct wins | Delta CI lower | Win CI lower |
|---|---:|---:|---:|---:|---:|
| Physical zero | 0.03795950 | +1,924.8% | 98.73% | 0.034879 | 98.42% |
| Within-task shuffle | 0.05497844 | +2,832.5% | 98.85% | 0.051279 | 98.64% |
| Cross-task shuffle | 0.05403125 | +2,782.0% | 99.15% | 0.050560 | 98.95% |

The preregistered auxiliary action gate requires at least +5% within-shuffle
error, 55% paired accuracy, positive delta CI, and accuracy CI above 50%.
All four checks pass by large margins. The auxiliary head does not merely
extrapolate time from `p0`; it uses the causal action chunk.

## Visual global action conditioning is retained

| Wrong action | Visual error | Relative increase | Correct wins | Delta CI lower | Win CI lower |
|---|---:|---:|---:|---:|---:|
| Physical zero | 0.01207419 | +25.42% | 77.35% | 0.002288 | 75.76% |
| Within-task shuffle | 0.01225129 | +27.25% | 78.83% | 0.002495 | 77.85% |
| Cross-task shuffle | 0.01229027 | +27.66% | 79.93% | 0.002526 | 78.96% |
| Reverse | 0.01017886 | +5.73% | 65.48% | 0.000497 | 64.14% |
| Fixed permutation | 0.00972651 | +1.03% | 56.62% | 0.0000819 | 55.46% |
| Repeat first | 0.00979957 | +1.79% | 47.96% | 0.000148 | 46.90% |

The frozen within-task visual gate passes all four checks. Magnitude response
also remains structured: errors at 0/0.25/0.5/1/1.5x are
`0.0120742/0.0111338/0.0101605/0.0096274/0.0098441`, and separation from 1x
tracks absolute scale distance with Spearman `0.9747`.

## Nearby candidates are clear in proprio but remain weak in visual latent

The C1-v2.1 A0/A1/A2 actions are unchanged: mean pairwise raw/normalized
chunk L2 is `0.22893/1.75294`.

| Horizon | Visual cosine separation | Visual sep./error | Proprio RMSE separation | Proprio sep./error |
|---:|---:|---:|---:|---:|
| h1 | 2.8242e-5 | 0.3971% | 0.007661 | 37.17% |
| h2 | 3.5743e-5 | 0.4332% | 0.012478 | 41.56% |
| h3 | 4.1769e-5 | 0.4612% | 0.016700 | 43.54% |
| h4 | 4.7720e-5 | 0.4946% | 0.020482 | 45.16% |
| h5 | 5.3356e-5 | 0.5279% | 0.023879 | 46.36% |
| h6 | 5.8382e-5 | 0.5548% | 0.026897 | 47.12% |
| h7 | 6.2749e-5 | 0.5742% | 0.029584 | 47.38% |
| h8 | 6.9240e-5 | 0.6077% | 0.032375 | 47.36% |
| **Mean** | **4.9650e-5** | **0.5157%** | **0.021257** | **45.50%** |

Physical separation is strong at every horizon and increases smoothly. Group
mean standardized-trajectory separations are `0.013521` for ee position,
`0.008048` for gripper, and `0.025249` for joints. The physical ratio passes
its 5% floor on every task; task ratios range from 37.42% to 54.18%.

Visual separation does not improve enough:

| Model | Visual separation/error | Multiple vs WM-v1 |
|---|---:|---:|
| WM-v1 | 0.00423536 | 1.000x |
| WM-v2A | 0.00530693 | 1.253x |
| WM-v2B | 0.00515718 | 1.218x |
| Gate | >=0.01000000 | >=2.000x |

WM-v2B is 2.82% below WM-v2A's ratio and no horizon or task reaches 0.01.
The shared GRU contains local physical information, but the normalized visual
head does not express it at a scale above ordinary visual prediction noise.

## Directional sanity checks support the same interpretation

The oracle is imitation-oriented: lower raw-action MSE to the recorded expert
chunk should correspond to lower predicted-future error. It is not a
closed-loop optimality or counterfactual-success claim.

| Predicted target used for ranking | Pairwise accuracy | 95% CI | Spearman | 95% CI | Gate pass |
|---|---:|---:|---:|---:|---|
| Visual future latent | 54.89% | [53.84%,55.97%] | 0.1016 | [0.0791,0.1245] | yes |
| Proprio future delta | 62.12% | [61.13%,63.11%] | 0.2531 | [0.2327,0.2736] | yes |

The proprio relation is stronger and positive for every task (pairwise
58.45--64.65%, Spearman 0.181--0.311). Visual direction passes in aggregate,
but tasks 1 and 4 are weak. This is not arbitrary separation: nearby physical
predictions move in a direction related to expert-action quality, while the
visual head carries a smaller but still positive ranking signal.

## Per-task evidence

| Task | n | Visual error | Proprio error | Visual sep./err | Proprio sep./err | Visual rank | Proprio rank |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 415 | 0.009216 | 0.001307 | 0.5609% | 49.54% | 53.49% | 62.89% |
| 1 | 628 | 0.007099 | 0.001347 | 0.6532% | 52.82% | 52.65% | 64.65% |
| 2 | 594 | 0.009336 | 0.001620 | 0.6810% | 51.72% | 57.74% | 59.76% |
| 3 | 442 | 0.010123 | 0.001056 | 0.5374% | 53.79% | 53.70% | 58.45% |
| 4 | 698 | 0.011319 | 0.001640 | 0.3580% | 37.42% | 52.15% | 60.55% |
| 5 | 507 | 0.008535 | 0.001366 | 0.6712% | 54.18% | 57.66% | 62.98% |
| 6 | 666 | 0.011609 | 0.002029 | 0.4146% | 37.57% | 54.75% | 62.26% |
| 7 | 696 | 0.009274 | 0.004138 | 0.5468% | 38.25% | 53.30% | 63.46% |
| 8 | 603 | 0.009816 | 0.002198 | 0.5126% | 47.55% | 56.11% | 63.02% |
| 9 | 620 | 0.009463 | 0.001278 | 0.4091% | 46.43% | 57.74% | 62.42% |

The physical/visual transfer gap is present across all ten tasks rather than
being driven by one task.

## Frozen gate decision

| Gate group | Result | Pass |
|---|---|---|
| Visual prediction quality | +2.73% vs D14; beats copy-current | yes |
| Visual global action signal | within-shuffle +27.25%; 78.83% wins | yes |
| Visual local sensitivity | 0.005157; 1.218x WM-v1 | **no** |
| Visual local direction | 54.89%; Spearman 0.1016; positive CIs | yes |
| Proprio prediction | 0.001875 vs zero 0.032502 | yes |
| Proprio action signal | within-shuffle +2,832.5%; 98.85% wins | yes |
| Proprio local separation | ratio 0.4550 vs 0.05 floor | yes |
| Proprio local direction | 62.12%; Spearman 0.2531; positive CIs | yes |

The preregistered decision order assigns
`PHYSICAL_SIGNAL_LEARNED_VISUAL_TRANSFER_INSUFFICIENT` because proprio
prediction, action conditioning, local physical separation, and proprio
direction all pass while visual local sensitivity fails.

This is not `PROPRIO_AUX_ACTION_SIGNAL_FAIL`: physical learning is strong. It
is not `PREDICTION_QUALITY_REGRESSION`: visual degradation is below 5%. It is
not generic `WM_V2B_AUXILIARY_INSUFFICIENT`: the more specific physical-
signal/visual-transfer condition is satisfied. It is not
`WM_V2B_LOCAL_SENSITIVITY_PASS`: both visual-local magnitude checks fail.

Seed23 and seed47 remain untrained. A future experiment would need to address
how the physical hidden-state signal is coupled into the visual prediction,
but this audit does not authorize an architecture change, new loss, ensemble,
candidate/reranker modification, or controller rollout.

## Integrity, NaN/Inf, and artifacts

| Artifact | SHA-256 |
|---|---|
| Gradient calibration result | `854f706caae98169136b7c7fd1f9dfca9d4b452801d336106647048c87cebef9` |
| Calibration freeze | `66e0934efa2ad39c47251f3f3b6ceff81f2e27bfe3dfe85f3ca2a1e040911166` |
| Best checkpoint, epoch 14 | `eb13775493a13d845e93db195c25d513b2fa8f679ac8578456fb8c64b2573848` |
| Last checkpoint, epoch 15 | `42096304ed3d5d8eda40bada64c5189713199075e56a8639c52dcda1309afffd` |
| Training metrics JSONL | `bf2a51ec72ace3650efad76d2569edf6e9c032b436b6d87df6e62069393a4947` |
| Run metadata | `c28ddea04bfafbc722f67eb3ace863d235740118a08dbea77347546b010afc0e` |
| Best-validation record | `d36b1d5e0a1c898f12fe88c3582bd81104b7ac85848b87382e74ea2da5cfd67a` |
| Aggregate diagnosis | `adec3dbd49295f19fb7cdacdfed260e4ec8afb928912c9e2657389f6131abf78` |
| Ignored diagnostic sidecar | `1553d59b734f65499e7e2e0fdec6c7174710cf0923c49a4ee88fe2d0e7a91a2a` |

All calibration losses/gradients, training gradients, checkpoint tensors,
visual/proprio predictions, ablation metrics, candidate separations, and
bootstrap inputs are finite. The ignored sidecar preserves all 5,869 unique
validation identities and the arrays needed for independent metric
recalculation. Checkpoints, JSONL, and the sidecar remain ignored; compact
metadata, diagnosis, config, tests, and this audit are tracked.

Forbidden-activity audit: C0-v2, candidate generation, production noise,
reranker score, CLIP, camera, D13 contract/split/H, and old checkpoints were
not modified. Future proprio was target-only. No simulator, controller,
success-rate selection, init-state access, D20-v2, C2-v2, seed23/47 training,
or fake counterfactual supervision occurred.
