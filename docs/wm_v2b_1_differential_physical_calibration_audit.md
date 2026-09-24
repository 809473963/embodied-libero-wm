# Partial Physical Mediator Pass

Date: 2026-09-24

Final verdict: `PARTIAL_PHYSICAL_MEDIATOR_PASS`

WM-v2C authorization: `BLOCK_WM_V2C_PHYSICAL_MEDIATOR`

Mechanism label(s): `EE_JOINT_MEDIATOR_VALID_GRIPPER_INVALID`

## Executive summary

The frozen WM-v2B physical head was calibrated on init states 31--34 and
evaluated once on disjoint init states 36--39. The primary form was fixed
before evaluation as eight positive/negative-unconstrained per-horizon
closed-form scalars; no neural model was trained and no evaluation result was
used to choose a form.

On held-out evaluation, primary calibrated RMSE is
`0.013184` versus zero-difference
`0.020335` and raw frozen prediction
`0.018707`. Improvement over zero is
`35.17%` with branch-block 95% CI
`[0.236, 0.454]`;
improvement over raw is `29.52%`.

Direction remains `0.754` and separation
ranking remains Spearman `0.761` /
ordering accuracy `84.06%`. The
mechanical gate therefore assigns `PARTIAL_PHYSICAL_MEDIATOR_PASS` and keeps
`BLOCK_WM_V2C_PHYSICAL_MEDIATOR`.

Policy-induced absolute physical-model shift replicates beyond historical
30/35: branch RMSE ratios to D13 validation are `3.365x`
(historical), `1.739x` (31--34), and `2.014x`
(36--39). Historical includes plan 0 and 8, while both new partitions use
plan 8 only; the new calibration/evaluation comparison is directly matched.

## Preregistration and isolation

- Protocol commit: `097e0571d18ce5995fda6903adf49f497449de81`; SHA-256 `3f76587593c84d418ba2de92e834f9f4286bd6b3a639d481f3cffb0497957cb0`.
- Calibration alpha freeze commit: `2392a42f8d5cd4d7161efd2ee230cbb1e6126b4f`; freeze SHA-256 `9fa482a5ec3efe1d05142b27a373b3dc38633a2ce696b5c3ef92c82acdce39a8`.
- Calibration: init 31--34, 40 branch points, planning index 8, 64-step prefixes.
- Evaluation: init 36--39, 40 branch points, planning index 8, 64-step prefixes.
- Historical init 30/35 is reference-only. Init 40--49 was not accessed.
- Every branch used a fresh environment and exact prefix replay; no mid-episode restore, reward, done, success, or task completion signal was used.

All 80 new branch points passed bitwise RGB/observable/prefix identity,
zero-tolerance proprio identity, simulator-state tolerance `1e-10`, and CLIP
cosine `>=0.999999` before candidate execution.

## Frozen coefficients

Global alpha: `0.532477451`.

Per-horizon alphas:

```text
[0.40681922142640414, 0.5922941688472336, 0.5940356526370816, 0.5694837515893315, 0.5326402927412073, 0.5160091728198142, 0.5137498993535594, 0.5233804369411192]
```

Groupwise alphas:

```json
{
  "ee_pos": 0.6864590530750556,
  "gripper": 0.4191461809363306,
  "joints": 0.5263412312661285
}
```

No alpha was clipped, sign-corrected, or iteratively tuned. Horizon×group
coefficients were also frozen from calibration data but are eligible only for
the preregistered secondary diagnostic and cannot authorize GO.

## Absolute error and distribution shift

| Source | RMSE | MAE | SmoothL1 | RMSE / validation |
|---|---|---|---|---|
| validation | 0.061368 | 0.034542 | 0.001875 | 1.000x |
| historical_30_35 | 0.206520 | 0.100124 | 0.021259 | 3.365x |
| calibration_31_34 | 0.106715 | 0.055846 | 0.005689 | 1.739x |
| evaluation_36_39 | 0.123605 | 0.059680 | 0.007619 | 2.014x |

## Held-out calibration-form comparison

| Form | RMSE | Zero | Raw | vs zero | zero CI | vs raw | norm ratio | direction | rank rho | ordering |
|---|---|---|---|---|---|---|---|---|---|---|
| global_scalar | 0.013148 | 0.020335 | 0.018707 | 35.34% | [0.236,0.458] | 29.72% | 0.785 | 0.754 | 0.761 | 84.06% |
| per_horizon_scalar | 0.013184 | 0.020335 | 0.018707 | 35.17% | [0.236,0.454] | 29.52% | 0.788 | 0.754 | 0.761 | 84.06% |
| groupwise_scalar | 0.012899 | 0.020335 | 0.018707 | 36.57% | [0.251,0.468] | 31.05% | 0.790 | 0.767 | 0.766 | 84.38% |

`per_horizon_scalar` is the only full-authorization form. The other rows are
diagnostics and were not used to select a winner from held-out data.

## Primary per-horizon behavior

| h | alpha | RMSE | zero | vs zero | 95% CI | norm ratio | direction |
|---|---|---|---|---|---|---|---|
| 1 | 0.406819 | 0.002973 | 0.004505 | 34.00% | [0.227,0.444] | 0.987 | 0.773 |
| 2 | 0.592294 | 0.006328 | 0.009925 | 36.24% | [0.254,0.466] | 1.019 | 0.798 |
| 3 | 0.594036 | 0.009360 | 0.014446 | 35.20% | [0.240,0.468] | 0.917 | 0.779 |
| 4 | 0.569484 | 0.011460 | 0.017825 | 35.71% | [0.245,0.466] | 0.849 | 0.765 |
| 5 | 0.532640 | 0.013179 | 0.021014 | 37.29% | [0.256,0.471] | 0.774 | 0.747 |
| 6 | 0.516009 | 0.015540 | 0.024424 | 36.37% | [0.235,0.473] | 0.737 | 0.738 |
| 7 | 0.513750 | 0.017893 | 0.027383 | 34.66% | [0.222,0.461] | 0.708 | 0.728 |
| 8 | 0.523380 | 0.019684 | 0.029581 | 33.46% | [0.217,0.444] | 0.705 | 0.706 |

## Candidate-pair consistency

| Pair | RMSE | zero | vs zero | 95% CI | direction | norm ratio | gate |
|---|---|---|---|---|---|---|---|
| A0-A1 | 0.010713 | 0.015218 | 29.61% | [0.161,0.438] | 0.746 | 0.767 | pass |
| A0-A2 | 0.008660 | 0.013928 | 37.83% | [0.296,0.448] | 0.757 | 0.806 | pass |
| A1-A2 | 0.018213 | 0.028548 | 36.20% | [0.247,0.463] | 0.760 | 0.789 | pass |

## Coordinate groups and gripper

The table uses the preregistered groupwise form for the partial-mediator gate.
Gripper is never hidden inside the aggregate.

| Group | alpha | vs zero | 95% CI | direction | norm ratio | sign | sign coverage | partial gate |
|---|---|---|---|---|---|---|---|---|
| ee_pos | 0.686459 | 41.28% | [0.352,0.462] | 0.741 | 0.779 | 85.07% | 90.00% | pass |
| gripper | 0.419146 | -6.17% | [-0.159,0.005] | 0.142 | 0.672 | 54.09% | 24.17% | FAIL |
| joints | 0.526341 | 47.87% | [0.393,0.549] | 0.811 | 0.836 | 85.11% | 89.23% | pass |

Primary-form gripper gate details:

```json
{
  "zero_improvement": false,
  "direction": true,
  "sign": false,
  "coverage": true
}
```

## Per-task held-out primary results

| Task | RMSE | zero | vs zero | norm ratio | direction |
|---|---|---|---|---|---|
| 0 | 0.008891 | 0.013038 | 31.80% | 0.861 | 0.736 |
| 1 | 0.009181 | 0.016482 | 44.30% | 1.001 | 0.828 |
| 2 | 0.011795 | 0.023024 | 48.77% | 0.762 | 0.853 |
| 3 | 0.010552 | 0.015675 | 32.68% | 0.874 | 0.758 |
| 4 | 0.011981 | 0.015848 | 24.40% | 0.569 | 0.744 |
| 5 | 0.012314 | 0.027443 | 55.13% | 0.862 | 0.928 |
| 6 | 0.012975 | 0.017317 | 25.07% | 0.947 | 0.597 |
| 7 | 0.012901 | 0.020002 | 35.50% | 0.682 | 0.679 |
| 8 | 0.024251 | 0.031150 | 22.15% | 0.704 | 0.655 |
| 9 | 0.010311 | 0.015596 | 33.88% | 0.656 | 0.763 |

## Ranking fidelity

Primary-form within-context Spearman is
`0.7609` with 95% CI
`[0.7125, 0.8094]`.
Pair-of-pair ordering accuracy is
`84.06%` with 95% CI
`[0.8083, 0.8729]`.
Pooled predicted/real separation Pearson/Spearman are
`0.8272` / `0.8650`.

## Mechanical decision

| Primary gate | Result |
|---|---|
| aggregate_zero_improvement | pass |
| aggregate_raw_improvement | pass |
| direction | pass |
| ranking | pass |
| candidate_pair_consistency | pass |
| full_gripper_validity | FAIL |
| positive_alpha | pass |

Candidate-pair checks:

```json
{
  "A0-A1": {
    "improvement_zero_positive": true,
    "direction_positive": true,
    "magnitude_ratio_in_0_5_1_5": true
  },
  "A0-A2": {
    "improvement_zero_positive": true,
    "direction_positive": true,
    "magnitude_ratio_in_0_5_1_5": true
  },
  "A1-A2": {
    "improvement_zero_positive": true,
    "direction_positive": true,
    "magnitude_ratio_in_0_5_1_5": true
  }
}
```

Partial group checks:

```json
{
  "ee_pos": {
    "zero_improvement": true,
    "direction": true,
    "sign": true
  },
  "joints": {
    "zero_improvement": true,
    "direction": true,
    "sign": true
  },
  "gripper": {
    "zero_improvement": false,
    "direction": true,
    "sign": false
  }
}
```

Final verdict: `PARTIAL_PHYSICAL_MEDIATOR_PASS`. WM-v2C authorization remains
`BLOCK_WM_V2C_PHYSICAL_MEDIATOR`. This experiment does not design or train WM-v2C.

## Integrity and forbidden-activity audit

- Result: `results/world_model_v2b_1/diagnosis.json`, SHA-256 `5690d3b30642143d605c7adddcb6f945c230ac2134a861a854bbae5b531f2c23`.
- Derived sidecar: `artifacts/stage3/wm_v2b_1_differential_calibration/analysis_metrics.npz`, SHA-256 `bc74c881324970bdde82a44de92a7f48162de17e9faa320d61061736a87dea6c`.
- Calibration sidecar SHA-256: `5e7c39e268b1696257107e2c8f068e2c4b40aa44f88ca8f6f92bc99d93d204d6`.
- Evaluation sidecar SHA-256: `76ff1ee009b686c5d71cd7968c37aeec02afcce9485bd3f1bd8ea232faca60fe`.
- NaN/Inf audit: `pass`.
- No WM-v2B retraining/modification, neural calibration, evaluation fitting,
  controller/candidate/reranker change, C1/C2 success run, D20-v2, reward/done,
  task-success selection, fake target, WM-v2C training, or init 40--49 access occurred.
