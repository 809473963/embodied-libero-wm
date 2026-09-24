# The physical mediator does not pass the counterfactual fidelity gate

Date: 2026-09-24

Authorization verdict: `BLOCK_WM_V2C_PHYSICAL_MEDIATOR`

Mechanism label(s): `PHYSICAL_MEDIATOR_COUNTERFACTUAL_FAIL`

## Executive summary

The frozen WM-v2B future-proprio head is evaluated without new inference or
simulator access. The audit reconstructs same-metric absolute errors from all
5,869 D13 validation samples and compares them with 40 matched branch points,
then directly tests predicted versus real A0/A1/A2 physical differences.

Differential RMSE is `0.020564` versus
`0.020133` for predicting no candidate difference,
a `-2.14%` relative improvement
(branch-block 95% CI [-0.1519, 0.1002]). Mean
direction cosine is `0.7734` (95% CI
[0.7303, 0.8144]), and predicted/real physical
magnitude ratio-of-means is `1.5817`.

The direction and ranking signal is real but insufficient for a physical
mediator. Within-context separation Spearman is `0.7562` and ordering
accuracy is `83.75%`, yet the predicted candidate differences are
systematically too large. That over-scaling makes the full vector prediction
slightly worse than predicting no candidate difference at all. Two of three
candidate pairs have negative improvement, and the aggregate confidence
interval includes zero.

Absolute branch prediction is also shifted: pooled branch RMSE is `3.365x`
validation RMSE. The gripper subgroup is especially weak, with direction
cosine `0.0138` and sign accuracy `49.57%`. The result therefore does not
receive either valid-relative-signal mechanism label.

The preregistered authorization decision is `BLOCK_WM_V2C_PHYSICAL_MEDIATOR`. This report
does not train WM-v2C or change WM-v2B, candidates, reranking, or controllers.

## Frozen evidence and coordinate contract

- Local counterfactual freeze commit/tag: `598a486b5cfea8390e7b72c41a7bd8a410b4b280` / `local-counterfactual-visual-transfer-confirmed`.
- Protocol source commit: `f2df2ba5d5a80042fe068c607510d3f7be9fe15f`.
- Only the hash-verified D13-validation and matched-branch sidecars were read.
- Branch scope remains task 0--9, init states 30/35, planning indices 0/8; no simulator or 40--49 access occurred.
- All physical quantities below use the frozen train-standardized 12-D coordinate order: ee position (3), gripper (2), joints (7).
- Validation elementwise SmoothL1(beta=1) was inverted to absolute residual magnitude; the reconstructed per-sample RMSE reproduces the frozen sidecar within `2e-7`.

## Absolute proprio error uses one metric family everywhere

| Source | RMSE | MAE | SmoothL1 beta=1 |
|---|---|---|---|
| D13 validation | 0.061368 | 0.034542 | 0.001875 |
| A0 | 0.206684 | 0.099969 | 0.021290 |
| A1 | 0.206444 | 0.100451 | 0.021241 |
| A2 | 0.206434 | 0.099953 | 0.021246 |
| A0/A1/A2 pooled | 0.206520 | 0.100124 | 0.021259 |

The pooled branch-to-validation ratios are RMSE
`3.365x`, MAE
`2.899x`, and SmoothL1
`11.340x`. This comparison
replaces the invalid cross-metric comparison between validation SmoothL1 and
branch RMSE.

The A0/A1/A2 absolute errors are similar to one another, so there is no single
candidate-specific absolute collapse. Their pooled shift is nevertheless
large and material: equality across candidates does not establish that their
*differences* are represented faithfully.

Per horizon (`R/M/S` = RMSE/MAE/SmoothL1):

| h | val R | val M | val S | A0 R | A0 M | A0 S | A1 R | A1 M | A1 S | A2 R | A2 M | A2 S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.02470 | 0.01562 | 0.00030 | 0.04292 | 0.03026 | 0.00092 | 0.04317 | 0.03080 | 0.00093 | 0.04305 | 0.03019 | 0.00093 |
| 2 | 0.03776 | 0.02198 | 0.00071 | 0.07611 | 0.04683 | 0.00290 | 0.07628 | 0.04702 | 0.00291 | 0.07627 | 0.04696 | 0.00291 |
| 3 | 0.04876 | 0.02791 | 0.00119 | 0.12409 | 0.06996 | 0.00770 | 0.12394 | 0.07023 | 0.00768 | 0.12447 | 0.07036 | 0.00775 |
| 4 | 0.05761 | 0.03311 | 0.00166 | 0.18355 | 0.09661 | 0.01685 | 0.18327 | 0.09700 | 0.01679 | 0.18365 | 0.09680 | 0.01686 |
| 5 | 0.06501 | 0.03784 | 0.00211 | 0.23501 | 0.12070 | 0.02761 | 0.23463 | 0.12117 | 0.02753 | 0.23488 | 0.12102 | 0.02758 |
| 6 | 0.07146 | 0.04221 | 0.00254 | 0.26441 | 0.13637 | 0.03492 | 0.26415 | 0.13690 | 0.03486 | 0.26381 | 0.13638 | 0.03478 |
| 7 | 0.07751 | 0.04647 | 0.00298 | 0.27915 | 0.14646 | 0.03879 | 0.27884 | 0.14721 | 0.03871 | 0.27844 | 0.14595 | 0.03861 |
| 8 | 0.08400 | 0.05118 | 0.00350 | 0.28628 | 0.15256 | 0.04062 | 0.28590 | 0.15327 | 0.04052 | 0.28589 | 0.15196 | 0.04055 |

Per proprio group:

| Group | val R | val M | val S | A0 R | A0 M | A0 S | A1 R | A1 M | A1 S | A2 R | A2 M | A2 S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ee_pos | 0.03229 | 0.02227 | 0.00052 | 0.06110 | 0.04092 | 0.00187 | 0.06090 | 0.04095 | 0.00185 | 0.06177 | 0.04108 | 0.00191 |
| gripper | 0.09299 | 0.05208 | 0.00432 | 0.45835 | 0.32889 | 0.10473 | 0.45760 | 0.32947 | 0.10440 | 0.45675 | 0.32589 | 0.10403 |
| joints | 0.05949 | 0.03479 | 0.00176 | 0.10773 | 0.05987 | 0.00577 | 0.10791 | 0.06051 | 0.00579 | 0.10869 | 0.06063 | 0.00588 |

Per task:

| Task | val R | val M | val S | A0 R | A0 M | A0 S | A1 R | A1 M | A1 S | A2 R | A2 M | A2 S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.05113 | 0.02997 | 0.00131 | 0.27659 | 0.13696 | 0.03808 | 0.27778 | 0.13836 | 0.03840 | 0.27456 | 0.13564 | 0.03754 |
| 1 | 0.05190 | 0.03187 | 0.00135 | 0.16200 | 0.07486 | 0.01312 | 0.16467 | 0.07805 | 0.01356 | 0.15955 | 0.07133 | 0.01273 |
| 2 | 0.05692 | 0.03295 | 0.00162 | 0.18920 | 0.10331 | 0.01790 | 0.18677 | 0.10033 | 0.01744 | 0.19218 | 0.10702 | 0.01847 |
| 3 | 0.04595 | 0.02703 | 0.00106 | 0.24434 | 0.11646 | 0.02984 | 0.24427 | 0.11699 | 0.02982 | 0.24362 | 0.11602 | 0.02967 |
| 4 | 0.05726 | 0.03503 | 0.00164 | 0.16257 | 0.09128 | 0.01321 | 0.15902 | 0.08913 | 0.01264 | 0.16407 | 0.09383 | 0.01346 |
| 5 | 0.05227 | 0.03267 | 0.00137 | 0.16673 | 0.07676 | 0.01390 | 0.16500 | 0.07381 | 0.01361 | 0.16925 | 0.08145 | 0.01432 |
| 6 | 0.06370 | 0.03583 | 0.00203 | 0.16218 | 0.08200 | 0.01315 | 0.16654 | 0.08495 | 0.01387 | 0.16076 | 0.08091 | 0.01292 |
| 7 | 0.09173 | 0.04442 | 0.00414 | 0.23641 | 0.10673 | 0.02784 | 0.23288 | 0.10666 | 0.02704 | 0.23762 | 0.10681 | 0.02812 |
| 8 | 0.06630 | 0.03765 | 0.00220 | 0.17704 | 0.10893 | 0.01567 | 0.17601 | 0.10946 | 0.01549 | 0.17811 | 0.10854 | 0.01586 |
| 9 | 0.05056 | 0.03268 | 0.00128 | 0.24734 | 0.10239 | 0.03018 | 0.24881 | 0.10678 | 0.03055 | 0.24378 | 0.09797 | 0.02937 |

## Counterfactual differential fidelity

For every matched branch point, pair, and horizon, the audit defines
`delta_p_real = p_real_i - p_real_j` and
`delta_p_pred = p_pred_i - p_pred_j` in standardized coordinates. Direction
uses cosine alignment; magnitude uses L2; differential error is RMSE against
the real delta. Ratios/directions require real L2 at least `1e-4`.

Overall:

| Cut | Direction | Norm ratio | Median ratio | p25 | p75 | Diff RMSE | Zero RMSE | Improvement | Improvement CI | Direction CI |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.7734 | 1.5817 | 1.6623 | 1.3317 | 2.2147 | 0.020564 | 0.020133 | -2.14% | [-0.1519, 0.1002] | [0.7303, 0.8144] |

By candidate pair:

| Pair | Direction | Norm ratio | Median ratio | p25 | p75 | Diff RMSE | Zero RMSE | Improvement | Improvement CI | Direction CI |
|---|---|---|---|---|---|---|---|---|---|---|
| A0-A1 | 0.7754 | 1.6464 | 1.6975 | 1.3289 | 2.1936 | 0.014237 | 0.013322 | -6.87% | [-0.2169, 0.0803] | [0.7289, 0.8182] |
| A0-A2 | 0.7682 | 1.4973 | 1.6423 | 1.3347 | 2.2266 | 0.016303 | 0.016383 | 0.48% | [-0.1139, 0.1047] | [0.7243, 0.8085] |
| A1-A2 | 0.7766 | 1.5950 | 1.6561 | 1.3440 | 2.2125 | 0.028287 | 0.027751 | -1.93% | [-0.1535, 0.1042] | [0.7325, 0.8175] |

By horizon:

| h | Direction | Norm ratio | Median ratio | p25 | p75 | Diff RMSE | Zero RMSE | Improvement | Improvement CI | Direction CI |
|---|---|---|---|---|---|---|---|---|---|---|
| h1 | 0.7624 | 2.5682 | 2.6266 | 2.1599 | 3.4259 | 0.008910 | 0.004616 | -93.03% | [-1.1751, -0.7318] | [0.7148, 0.8056] |
| h2 | 0.7978 | 1.7901 | 1.7466 | 1.5052 | 2.2490 | 0.011939 | 0.009907 | -20.51% | [-0.3528, -0.0655] | [0.7563, 0.8394] |
| h3 | 0.8090 | 1.6897 | 1.6258 | 1.4342 | 2.0921 | 0.014456 | 0.013312 | -8.59% | [-0.2277, 0.0596] | [0.7656, 0.8486] |
| h4 | 0.8021 | 1.6883 | 1.6122 | 1.4117 | 1.9893 | 0.016755 | 0.015898 | -5.40% | [-0.2196, 0.1025] | [0.7440, 0.8495] |
| h5 | 0.8005 | 1.6234 | 1.5298 | 1.3231 | 2.0162 | 0.019594 | 0.019175 | -2.18% | [-0.1731, 0.1226] | [0.7510, 0.8457] |
| h6 | 0.7717 | 1.5287 | 1.4793 | 1.2393 | 2.0667 | 0.023825 | 0.023804 | -0.09% | [-0.1537, 0.1309] | [0.7108, 0.8252] |
| h7 | 0.7353 | 1.4481 | 1.4343 | 1.2105 | 1.9001 | 0.027507 | 0.027883 | 1.35% | [-0.1393, 0.1427] | [0.6600, 0.8018] |
| h8 | 0.7084 | 1.3891 | 1.3762 | 1.1927 | 1.8067 | 0.031036 | 0.031327 | 0.93% | [-0.1296, 0.1271] | [0.6276, 0.7808] |

By proprio group:

| Group | Direction | Norm ratio | Median ratio | p25 | p75 | Diff RMSE | Zero RMSE | Improvement | Improvement CI | Direction CI |
|---|---|---|---|---|---|---|---|---|---|---|
| ee_pos | 0.7453 | 1.3330 | 1.3507 | 1.0803 | 1.7894 | 0.010698 | 0.013714 | 21.99% | — | — |
| gripper | 0.0138 | 1.8321 | 12.9555 | 3.6341 | 32.1452 | 0.025797 | 0.023984 | -7.56% | — | — |
| joints | 0.8196 | 1.7160 | 1.7683 | 1.3955 | 2.3132 | 0.022608 | 0.021772 | -3.84% | — | — |

By task:

| Task | Direction | Norm ratio | Median ratio | p25 | p75 | Diff RMSE | Zero RMSE | Improvement | Improvement CI | Direction CI |
|---|---|---|---|---|---|---|---|---|---|---|
| task 0 | 0.8813 | 1.5850 | 1.6757 | 1.3024 | 2.5364 | 0.014267 | 0.017911 | 20.34% | — | — |
| task 1 | 0.7648 | 1.3007 | 1.3859 | 1.1674 | 1.7778 | 0.025607 | 0.026822 | 4.53% | — | — |
| task 2 | 0.7755 | 1.8290 | 1.9582 | 1.5057 | 2.4050 | 0.022314 | 0.022163 | -0.68% | — | — |
| task 3 | 0.8386 | 1.8994 | 1.9172 | 1.6764 | 2.2079 | 0.016763 | 0.015059 | -11.32% | — | — |
| task 4 | 0.6917 | 1.3683 | 1.8098 | 1.4684 | 2.2808 | 0.023742 | 0.022055 | -7.65% | — | — |
| task 5 | 0.7674 | 1.7171 | 1.6825 | 1.4198 | 2.2557 | 0.018711 | 0.015096 | -23.94% | — | — |
| task 6 | 0.7810 | 1.5383 | 1.4004 | 1.2093 | 1.8882 | 0.023843 | 0.022560 | -5.69% | — | — |
| task 7 | 0.7505 | 1.3958 | 1.4916 | 1.2192 | 1.7508 | 0.018553 | 0.021713 | 14.55% | — | — |
| task 8 | 0.7002 | 1.6711 | 1.5010 | 1.3208 | 2.3666 | 0.020724 | 0.017067 | -21.43% | — | — |
| task 9 | 0.7833 | 1.6882 | 1.8189 | 1.3767 | 2.0520 | 0.018292 | 0.017580 | -4.05% | — | — |

## Why positive direction is not enough

The head usually points candidate differences in the right direction, but it
does not reproduce their size accurately. Aggregate predicted norm is
`1.582x` real norm. At h1 the ratio is `2.568x` and differential error is
`93.0%` worse than zero-difference. Only h7 and h8 have small positive point
estimates, and both confidence intervals cross zero.

The failure is not uniform across coordinates. EE-position differences
improve over zero by `21.99%`, but joints are `3.84%` worse and gripper is
`7.56%` worse, with gripper direction near zero. Only three of ten tasks have
a positive differential-improvement point estimate. A future mediator would
therefore require differential calibration and coordinate-level validation;
the current head cannot be coupled directly into a visual predictor on this
evidence.

## Separation ranking fidelity

Overall mean within-context Spearman is
`0.7562` (95% CI
[0.7078, 0.8031]); three-way pairwise
ordering accuracy is `83.75%`
(95% CI [0.8052, 0.8688]). Pooled
predicted-vs-real separation Spearman/Pearson are
`0.8885` / `0.8137`.

This passes the preregistered ordinal gate: the head can often tell which of
the three pairwise physical separations is larger. Ranking discards absolute
scale and vector residual, however, so it cannot rescue the failed
zero-difference and pair-consistency gates.

| h | Within Spearman | Ordering accuracy | Pooled Spearman | Pooled Pearson |
|---|---|---|---|---|
| 1 | 0.7625 | 84.17% | 0.7574 | 0.7953 |
| 2 | 0.7875 | 85.83% | 0.7986 | 0.8185 |
| 3 | 0.8000 | 86.67% | 0.8411 | 0.8493 |
| 4 | 0.7875 | 85.83% | 0.8449 | 0.8350 |
| 5 | 0.7500 | 83.33% | 0.8238 | 0.8212 |
| 6 | 0.7500 | 83.33% | 0.7891 | 0.7558 |
| 7 | 0.7250 | 81.67% | 0.7970 | 0.7317 |
| 8 | 0.6875 | 79.17% | 0.8152 | 0.7108 |

| Task | Within Spearman | Ordering accuracy | Pooled Spearman | Pooled Pearson |
|---|---|---|---|---|
| 0 | 0.7187 | 81.25% | 0.8843 | 0.8598 |
| 1 | 0.7812 | 85.42% | 0.8797 | 0.7901 |
| 2 | 0.6719 | 78.12% | 0.8924 | 0.9469 |
| 3 | 0.6250 | 75.00% | 0.9505 | 0.9560 |
| 4 | 0.7969 | 86.46% | 0.9367 | 0.7065 |
| 5 | 0.7344 | 82.29% | 0.8598 | 0.8521 |
| 6 | 0.7656 | 84.38% | 0.7982 | 0.7784 |
| 7 | 0.8125 | 87.50% | 0.9460 | 0.9478 |
| 8 | 0.8437 | 89.58% | 0.8483 | 0.8428 |
| 9 | 0.8125 | 87.50% | 0.9167 | 0.8935 |

## Dimension-wise sign fidelity

Only real standardized deltas with absolute magnitude at least `1e-3` enter
the sign comparison. Overall accuracy is
`82.73%` over `8966` eligible
dimension values.

| Dimension | Eligible | Coverage | Sign accuracy |
|---|---|---|---|
| ee_x | 846 | 88.12% | 79.79% |
| ee_y | 843 | 87.81% | 82.56% |
| ee_z | 859 | 89.48% | 87.08% |
| gripper_0 | 257 | 26.77% | 52.92% |
| gripper_1 | 205 | 21.35% | 45.37% |
| joint_0 | 840 | 87.50% | 84.52% |
| joint_1 | 820 | 85.42% | 82.80% |
| joint_2 | 904 | 94.17% | 85.40% |
| joint_3 | 735 | 76.56% | 85.71% |
| joint_4 | 926 | 96.46% | 88.55% |
| joint_5 | 902 | 93.96% | 83.70% |
| joint_6 | 829 | 86.35% | 84.92% |

Group accuracies are ee position `83.16%`,
gripper `49.57%`, and joints
`85.12%`. Horizon/task sign cuts remain in
the compact JSON.

## Visual/physical coupling remains diagnostic

Real physical versus real visual separation has Pearson/Spearman
`0.1900` /
`0.3109`.
Predicted physical versus predicted visual separation has Pearson/Spearman
`0.6965` /
`0.7070`.
These associations do not establish a causal mediation path. In particular,
the stronger predicted-predicted correlation measures internal co-response of
two outputs from the same model; it must not be interpreted as mediator
accuracy against the matched real futures.

| h | Real Pearson | Real Spearman | Pred Pearson | Pred Spearman |
|---|---|---|---|---|
| 1 | 0.3474 | 0.4038 | 0.7892 | 0.8288 |
| 2 | 0.3066 | 0.3604 | 0.7791 | 0.8139 |
| 3 | 0.0070 | 0.0765 | 0.7692 | 0.8133 |
| 4 | 0.2359 | 0.2117 | 0.7475 | 0.8040 |
| 5 | 0.1414 | 0.1652 | 0.7449 | 0.7895 |
| 6 | 0.0758 | 0.0664 | 0.7693 | 0.7809 |
| 7 | 0.0795 | 0.1145 | 0.7627 | 0.7599 |
| 8 | 0.1372 | 0.1770 | 0.7330 | 0.7295 |

## Preregistered authorization decision

| Gate | Result |
|---|---|
| differential_improvement | FAIL |
| direction_alignment | pass |
| ranking_relation | pass |
| no_candidate_specific_collapse | FAIL |

Candidate-pair collapse checks:

```json
{
  "A0-A1": {
    "relative_improvement_positive": false,
    "direction_positive": true,
    "magnitude_ratio_at_least_0_05": true
  },
  "A0-A2": {
    "relative_improvement_positive": true,
    "direction_positive": true,
    "magnitude_ratio_at_least_0_05": true
  },
  "A1-A2": {
    "relative_improvement_positive": false,
    "direction_positive": true,
    "magnitude_ratio_at_least_0_05": true
  }
}
```

Final authorization verdict: `BLOCK_WM_V2C_PHYSICAL_MEDIATOR`.

Mechanism label(s): `PHYSICAL_MEDIATOR_COUNTERFACTUAL_FAIL`.

`PHYSICAL_MEDIATOR_RELATIVE_SIGNAL_VALID` and
`PHYSICAL_MEDIATOR_VALID_VISUAL_TRANSFER_BLOCK_CONFIRMED` are not assigned.
Although direction and ranking pass, the preregistered definition requires all
four gate groups, including positive differential improvement and no
candidate-pair collapse. Direct WM-v2C design from this frozen head remains
blocked. Any future experiment must be separately preregistered; this audit
does not authorize a new objective, architecture, training run, or controller.

## Integrity and forbidden-activity audit

- Result: `results/world_model_v2b_physical_mediator/diagnosis.json`, SHA-256 `07bc7dba47a71b917c98c2be541527641157c66742f0314b29cf8db37aebd39d`.
- Derived sidecar: `artifacts/stage3/wm_v2b_physical_mediator/metrics.npz`, SHA-256 `c5e974895d881ef8884a845a95d2bc50ce4d0f82f24b6c158629b5e567934917`.
- Frozen matched-branch sidecar SHA-256: `6c81a692bac418a9defe3a8abc0086408b909ce1c99b9241c49361a7c298e12d`.
- Frozen validation sidecar SHA-256: `1553d59b734f65499e7e2e0fdec6c7174710cf0923c49a4ee88fe2d0e7a91a2a`.
- All source/derived numeric arrays and reported values are finite; NaN/Inf audit: `pass`.
- No simulator, inference, training, controller, success/reward/done signal, fake target, or init-state access occurred.
