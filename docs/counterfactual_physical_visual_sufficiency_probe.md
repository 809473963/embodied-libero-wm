# Neither EE+joint motion nor pre-action context clears the visual sufficiency gate

Date: 2026-09-25

Final mechanism verdict: `CONTEXT_DOMINATES_PHYSICAL_SIGNAL_NOT_ADDITIVE`

Authorization: `BLOCK_CONTEXT_CONDITIONED_WM_DESIGN`

## Executive answer

The evidence does **not** support the hypothesis that WM-v2C-EJ failed merely
because its physical-to-visual map lacked scene conditioning. Physical-only
P1 and context-conditioned P3 both fail; P3 is worse than the zero baseline,
and neither physical nor context shuffling causes a significant degradation.
Under the preregistered decision order the formal label is
`CONTEXT_DOMINATES_PHYSICAL_SIGNAL_NOT_ADDITIVE`, because P3 is not better
than P2 and physical shuffling is non-significant. That label must not be read
as evidence that context is sufficient: context-only P2 is also worse than
zero, and context shuffling is non-significant. The scientifically supported
answer is that the tested EE+joint plus pre-action-context representation is
still insufficient for the real local visual counterfactual.

The primary P3 probe has visual-differential RMSE `0.004252`,
versus `0.004227` for zero, `0.004227` for
physical-only, and `0.004247` for context-only. Its relative
improvements are -0.60%, -0.61%, and -0.14%, respectively.
Aggregate direction is `0.0105` with branch-block
95% CI [0.0014, 0.0199].

Replacing real EE+joint differences with the frozen calibrated WM-v2B.1
prediction gives RMSE `0.004241`, direction
`0.0105`, and -0.33% improvement
over zero. This second stage uses the exact same frozen P3 parameters.

This is a matched-branch supervised mechanism probe, not a deployable World
Model and not controller evidence.

## Frozen provenance

- Upstream freeze tag/commit: `wm-v2c-ej-transfer-insufficient` /
  `b39af288e9bea09e94b04ef43483f8c5217d0412`.
- Protocol commit: `b59cef72635c629a2af4f770d3ce0b69b6714a9b`; protocol
  SHA-256 `5d0679a2e9ad805d23b7f407d7960cd1154e18f0ebd155b7ba9f5e1748606cb7`.
- Calibration freeze commit: `4882245058f5ada089fff95cbeb4d5c69c2233f0`;
  freeze SHA-256 `19d00538e5f0273c630a02a519df00897753a96a0079e72a23d1b26a0bc24b36`.
- Frozen reports/results tree: 142
  files, aggregate SHA-256 `2101b083bee714eeb70d3181a04df952716696734b66c069239bea6d78df511d`.
- No existing report, result, checkpoint, or sidecar was overwritten.

## Exact branch split and isolation

Probe calibration uses init states 31--34 (40
branch points); the one-shot held-out evaluation uses 36--39
(40 branch points). Every branch contributes
all three candidate pairs and all eight horizons to one partition. Historical
30/35 is reference-only and was not fitted; 40--49 was not accessed. Existing
sidecars already contained real RGB/CLIP futures, proprio futures, common
initial observations, and candidate identities, so no simulator was rerun.

## Sample construction

For every branch, pair, and horizon, the physical input is
`standardized(p_real_i)-standardized(p_real_j)` restricted to
`ee_pos[0:3]+joints[5:12]` (10-D). Gripper dimensions 3/4 are excluded. The
target is `z_real_i-z_real_j` (512-D) from frozen L2-normalized CLIP futures.
The matching real visual separation remains `1-cos(z_i,z_j)`; predicted
separation is `0.5*||pred_delta_z||^2`, the unit-latent identity.

## Frozen pre-action context

The 256-D context is the frozen WM-v2B initial hidden representation produced
only from `z0`, train-normalized `proprio0`, and language. Actions and all future
values are absent. Repeating each exact common branch input three times gives
maximum context difference `0.000e+00`.

## P0--P3 definitions

- P0 predicts zero visual difference.
- P1 is no-bias 10-to-512 ridge regression and is exactly antisymmetric.
- P2 is a 256-to-512 ridge baseline with an unpenalized intercept; it sees only
  candidate-independent context.
- P3 is `W_o((W_p delta_p) * sigmoid(W_c c0))`, with 64 hidden units and no
  bias in `W_p` or `W_o`. It has 49,856
  parameters and no context-to-output shortcut.

## Inner calibration and immutable held-out evaluation

P1/P2 ridge alpha was selected only by leave-one-init-out CV on 31--34 from
`[0.0001, 0.001, 0.01, 0.1, 1.0]`. Selected alpha is
`0.01` for P1 and
`1.0` for P2. P3 used fixed
AdamW settings and inner-fold early stopping; the round-half-up median best
epoch was 76. The final calibration
weights were committed and hash-frozen before 36--39 was opened.

## Held-out RMSE and direction

| Probe | RMSE | vs zero | Direction | Direction 95% CI | Norm ratio | Sep. Pearson | Sep. Spearman |
|---|---:|---:|---:|---|---:|---:|---:|
| P0 zero | 0.004227 | — | 0.0000 | [0.0000, 0.0000] | 0.000 | 0.0000 | 0.0000 |
| P1 physical | 0.004227 | 0.02% | 0.0167 | [-0.0005, 0.0337] | 0.009 | 0.1046 | 0.2675 |
| P2 context | 0.004247 | -0.46% | 0.0027 | [-0.0168, 0.0234] | 0.104 | 0.2790 | 0.3295 |
| P3 real physical + context | 0.004252 | -0.60% | 0.0105 | [0.0014, 0.0199] | 0.103 | 0.1977 | 0.3473 |

P3 improvement CIs are [-0.0078, -0.0041]
versus zero, [-0.0078, -0.0044]
versus P1, and [-0.0053, 0.0023]
versus P2.

## Norm and separation-ranking fidelity

P3 norm-ratio median/p25/p75 are
`0.084` /
`0.051` /
`0.141`. Its within-context separation
Spearman is `0.3516` and
candidate-pair ordering accuracy is
`65.94%`.

## Physical and context shuffle controls

Both controls use fixed within-task branch-point derangements, preserving pair
and horizon. Physical shuffle changes P3 RMSE to
`0.004260` (0.17% degradation)
and direction to `-0.0016`. Context
shuffle changes RMSE to `0.004254`
(0.02% degradation) and direction to
`0.0101`. The corresponding relative
RMSE CIs are
[-0.0020, 0.0054]
and [-0.0002, 0.0007];
direction-degradation CIs are
[-0.0026, 0.0266]
and [-0.0003, 0.0012].

## Real versus predicted physical mediator

| Physical input to frozen P3 | RMSE | Direction | Norm ratio | Sep. Spearman | Within rank | Ordering |
|---|---:|---:|---:|---:|---:|---:|
| Real EE+joint delta | 0.004252 | 0.0105 | 0.103 | 0.3473 | 0.3516 | 65.94% |
| Frozen calibrated WM-v2B.1 delta | 0.004241 | 0.0105 | 0.083 | 0.2388 | 0.3391 | 65.00% |

## Comparison with frozen WM-v2C-EJ

WM-v2C-EJ matched transfer was `0.019030` and visual-delta direction was
`0.0203`. The supervised P1/P3 directions here are
`0.0167` / `0.0105`.
This comparison diagnoses representational sufficiency only: P1/P3 were fitted
on matched counterfactual branches and cannot be presented as deployable-WM
performance.

## Per-horizon evidence

| Cut | Probe | RMSE | Direction | Norm ratio | Sep. Spearman |
|---|---|---:|---:|---:|---:|
| h1 | p0 | 0.003247 | 0.0000 | 0.000 | 0.0000 |
| h1 | p1 | 0.003246 | 0.0317 | 0.003 | 0.4361 |
| h1 | p2 | 0.003264 | 0.0186 | 0.139 | 0.4283 |
| h1 | p3_real | 0.003247 | 0.0070 | 0.033 | 0.4503 |
| h2 | p0 | 0.003744 | 0.0000 | 0.000 | 0.0000 |
| h2 | p1 | 0.003744 | -0.0099 | 0.005 | 0.2557 |
| h2 | p2 | 0.003768 | 0.0137 | 0.117 | 0.4279 |
| h2 | p3_real | 0.003753 | -0.0031 | 0.061 | 0.4204 |
| h3 | p0 | 0.004200 | 0.0000 | 0.000 | 0.0000 |
| h3 | p1 | 0.004198 | 0.0407 | 0.007 | 0.1207 |
| h3 | p2 | 0.004208 | 0.0155 | 0.105 | 0.3822 |
| h3 | p3_real | 0.004206 | 0.0177 | 0.080 | 0.2595 |
| h4 | p0 | 0.004257 | 0.0000 | 0.000 | 0.0000 |
| h4 | p1 | 0.004255 | 0.0425 | 0.008 | 0.0514 |
| h4 | p2 | 0.004269 | 0.0070 | 0.102 | 0.3107 |
| h4 | p3_real | 0.004269 | 0.0253 | 0.100 | 0.1442 |
| h5 | p0 | 0.004680 | 0.0000 | 0.000 | 0.0000 |
| h5 | p1 | 0.004681 | -0.0080 | 0.009 | 0.1178 |
| h5 | p2 | 0.004704 | -0.0172 | 0.093 | 0.2972 |
| h5 | p3_real | 0.004714 | 0.0064 | 0.108 | 0.1658 |
| h6 | p0 | 0.004271 | 0.0000 | 0.000 | 0.0000 |
| h6 | p1 | 0.004268 | 0.0412 | 0.011 | 0.1057 |
| h6 | p2 | 0.004278 | 0.0307 | 0.101 | 0.3410 |
| h6 | p3_real | 0.004298 | 0.0273 | 0.131 | 0.1876 |
| h7 | p0 | 0.004650 | 0.0000 | 0.000 | 0.0000 |
| h7 | p1 | 0.004648 | 0.0235 | 0.012 | 0.0510 |
| h7 | p2 | 0.004699 | -0.0577 | 0.093 | 0.3056 |
| h7 | p3_real | 0.004690 | 0.0106 | 0.131 | 0.1431 |
| h8 | p0 | 0.004569 | 0.0000 | 0.000 | 0.0000 |
| h8 | p1 | 0.004571 | -0.0282 | 0.013 | -0.0898 |
| h8 | p2 | 0.004580 | 0.0109 | 0.095 | 0.3349 |
| h8 | p3_real | 0.004629 | -0.0071 | 0.145 | 0.0193 |

## Per-task evidence

| Cut | Probe | RMSE | Direction | Norm ratio | Sep. Spearman |
|---|---|---:|---:|---:|---:|
| 0 | p0 | 0.004283 | 0.0000 | 0.000 | 0.0000 |
| 0 | p1 | 0.004284 | -0.0034 | 0.006 | 0.2265 |
| 0 | p2 | 0.004315 | -0.0375 | 0.084 | 0.0303 |
| 0 | p3_real | 0.004296 | 0.0001 | 0.067 | 0.4382 |
| 1 | p0 | 0.002532 | 0.0000 | 0.000 | 0.0000 |
| 1 | p1 | 0.002533 | -0.0081 | 0.013 | 0.6003 |
| 1 | p2 | 0.002539 | 0.0220 | 0.096 | 0.0521 |
| 1 | p3_real | 0.002572 | 0.0014 | 0.159 | 0.6410 |
| 2 | p0 | 0.004142 | 0.0000 | 0.000 | 0.0000 |
| 2 | p1 | 0.004144 | 0.0117 | 0.010 | 0.3696 |
| 2 | p2 | 0.004165 | -0.0087 | 0.098 | -0.0730 |
| 2 | p3_real | 0.004182 | 0.0037 | 0.102 | 0.4447 |
| 3 | p0 | 0.004495 | 0.0000 | 0.000 | 0.0000 |
| 3 | p1 | 0.004495 | 0.0190 | 0.007 | 0.3897 |
| 3 | p2 | 0.004499 | 0.0286 | 0.081 | 0.2000 |
| 3 | p3_real | 0.004515 | 0.0037 | 0.086 | 0.3472 |
| 4 | p0 | 0.004931 | 0.0000 | 0.000 | 0.0000 |
| 4 | p1 | 0.004931 | -0.0146 | 0.005 | 0.4348 |
| 4 | p2 | 0.004953 | -0.0005 | 0.100 | 0.2569 |
| 4 | p3_real | 0.004997 | -0.0086 | 0.127 | 0.6008 |
| 5 | p0 | 0.003789 | 0.0000 | 0.000 | 0.0000 |
| 5 | p1 | 0.003787 | 0.0354 | 0.016 | 0.0644 |
| 5 | p2 | 0.003809 | 0.0395 | 0.150 | -0.4448 |
| 5 | p3_real | 0.003831 | 0.0173 | 0.152 | 0.2345 |
| 6 | p0 | 0.005599 | 0.0000 | 0.000 | 0.0000 |
| 6 | p1 | 0.005596 | 0.0800 | 0.006 | 0.3257 |
| 6 | p2 | 0.005609 | 0.0381 | 0.116 | -0.3638 |
| 6 | p3_real | 0.005590 | 0.0510 | 0.067 | 0.2871 |
| 7 | p0 | 0.004451 | 0.0000 | 0.000 | 0.0000 |
| 7 | p1 | 0.004450 | 0.0423 | 0.009 | 0.6678 |
| 7 | p2 | 0.004482 | -0.0199 | 0.100 | 0.2058 |
| 7 | p3_real | 0.004467 | 0.0135 | 0.081 | 0.6357 |
| 8 | p0 | 0.003362 | 0.0000 | 0.000 | 0.0000 |
| 8 | p1 | 0.003360 | 0.0165 | 0.014 | 0.1288 |
| 8 | p2 | 0.003388 | -0.0220 | 0.117 | 0.3739 |
| 8 | p3_real | 0.003399 | 0.0175 | 0.153 | 0.2878 |
| 9 | p0 | 0.003929 | 0.0000 | 0.000 | 0.0000 |
| 9 | p1 | 0.003928 | -0.0120 | 0.008 | 0.6000 |
| 9 | p2 | 0.003952 | -0.0126 | 0.101 | 0.1190 |
| 9 | p3_real | 0.003934 | 0.0053 | 0.078 | 0.6275 |

## Per-pair evidence

| Cut | Probe | RMSE | Direction | Norm ratio | Sep. Spearman |
|---|---|---:|---:|---:|---:|
| A0-A1 | p0 | 0.004028 | 0.0000 | 0.000 | 0.0000 |
| A0-A1 | p1 | 0.004027 | 0.0205 | 0.007 | 0.2323 |
| A0-A1 | p2 | 0.004038 | 0.0149 | 0.108 | 0.3151 |
| A0-A1 | p3_real | 0.004042 | 0.0147 | 0.081 | 0.2907 |
| A0-A2 | p0 | 0.003977 | 0.0000 | 0.000 | 0.0000 |
| A0-A2 | p1 | 0.003977 | 0.0063 | 0.007 | 0.2153 |
| A0-A2 | p2 | 0.003996 | 0.0039 | 0.111 | 0.3149 |
| A0-A2 | p3_real | 0.003990 | 0.0020 | 0.081 | 0.3116 |
| A1-A2 | p0 | 0.004644 | 0.0000 | 0.000 | 0.0000 |
| A1-A2 | p1 | 0.004643 | 0.0232 | 0.012 | 0.2091 |
| A1-A2 | p2 | 0.004671 | -0.0107 | 0.094 | 0.3785 |
| A1-A2 | p3_real | 0.004690 | 0.0148 | 0.139 | 0.2600 |

## Antisymmetry, finite values, and mechanical decision

P1 and P3 maximum `|f(c,-delta_p)+f(c,delta_p)|` are
`0.000e+00` and
`0.000e+00`. All inputs, fitted
parameters, predictions, residuals, and bootstrap statistics are finite.

P1 checks: `{"all_pair_direction_positive": true, "better_than_zero": false, "norm_not_catastrophic": false, "positive_direction": false}`.

P3 real-physical checks: `{"all_pair_direction_positive": true, "better_than_context_by_5pct": false, "better_than_physical_by_5pct": false, "better_than_zero": false, "context_shuffle_worse": false, "norm_not_catastrophic": false, "physical_shuffle_worse": false, "positive_direction": true}`.

P3 predicted-physical checks:
`{"all_pair_direction_positive": true, "better_than_context_by_5pct": false, "better_than_zero": false, "norm_not_catastrophic": false, "positive_direction": true}`.

Final mechanism verdict: `CONTEXT_DOMINATES_PHYSICAL_SIGNAL_NOT_ADDITIVE`. Authorization:
`BLOCK_CONTEXT_CONDITIONED_WM_DESIGN`. This authorization concerns only a separately
preregistered future World-Model design; it never authorizes a controller run.
The verdict name is the exact preregistered Case-C label; the failed P2 and
context-shuffle checks above prevent interpreting it as a positive context
mechanism.

## Artifact hashes

- Calibration weights: `621f42100526b8809d1db9a2249d2cb4197134df1c3a646ee52d21ff628aea4b`.
- Calibration source sidecar: `5e7c39e268b1696257107e2c8f068e2c4b40aa44f88ca8f6f92bc99d93d204d6`.
- Evaluation source sidecar: `76ff1ee009b686c5d71cd7968c37aeec02afcce9485bd3f1bd8ea232faca60fe`.
- Derived evaluation sidecar: `9d8fd55cb6cd71b07eaddcb06ced95f877d144162bca01e19323088ee26d1af6`.
- Compact result: `b2ea73ae18c96dd2c3fc3c990750705b51195b2f4982bbc747a32eed9696407d`.

## Forbidden-activity audit

No World Model, CLIP encoder, actor, candidate generator, scale, reranker, or
controller was changed or trained. No simulator ran; no success, reward, done,
or task outcome was read. There was no C1/C2/D20-v2 work, seed23/47 run,
future context, action identity, pair embedding, expert-action error, or init
40--49 access. Gripper remained excluded. Frozen upstream artifacts were
hash-checked and not overwritten.

After the immutable diagnosis and sidecar were written, this report's opening
interpretation was clarified to distinguish the mechanical Case-C label from
the stronger unsupported claim that context itself is sufficient. No metric,
bootstrap interval, gate, verdict, diagnosis JSON, or sidecar was changed.
