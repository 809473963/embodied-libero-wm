# D20 Validation-only Threshold Selection Audit

## Verdict

**D20 PASS; D21 implementation is authorized.** The thresholds are frozen and
must not be changed in response to fixed simulator test behavior.

## Selection protocol

- Implementation commit: `65bd19898ec88a2d53737a6383200a1e75d5e4b4`
- Source: D13 validation demos 45–49 through the immutable D16 sample record
- Source samples: 5,869
- Check horizons: h2, h4, h6, h8
- Check points: 23,476
- Tasks / episodes: 10 / 50
- Grid: 6 mismatch quantiles × 6 uncertainty quantiles = 36 pairs
- Target: nominal `(mismatch OR uncertainty)` trigger rate ≤1%
- Selection: highest nonzero eligible rate, then lowest quantile sum
- Fixed evaluation episodes, task success, bootstrap, and D19 control outcome:
  not used for selection

Validation demonstrations are treated as nominal expert trajectories. There is
no failure/deviation label, so this is false-trigger calibration rather than an
AUROC or failure-detection optimization.

## Selected thresholds

| Field | Value |
|---|---:|
| mismatch quantile | 0.995 |
| mismatch threshold | **0.0290635973** |
| uncertainty quantile | 0.9975 |
| uncertainty threshold | **0.0044514909** |
| combined trigger count | 177 / 23,476 |
| combined trigger rate | **0.00753961** |
| mismatch-only rate | 0.00502641 |
| uncertainty-only rate | 0.00251320 |
| both rate | 0.0 |

Trigger rule:

```text
mismatch > 0.0290635973 OR uncertainty > 0.0044514909
```

Eight grid points satisfy the ≤1% target. The stored selected row exactly
matches a fresh application of the frozen deterministic selection rule.

## Validation behavior

- Any-trigger episode fraction: 60%
- Mean triggers per validation episode: 3.54
- Per-horizon trigger rates:
  - h2: 0.290%
  - h4: 0.426%
  - h6: 0.852%
  - h8: 1.448%
- Per-task rates range from 0.040% (task 1) to 1.599% (task 2).

Even a low pointwise rate produces triggers in many episodes because each
episode supplies many overlapping checks. D21/D24 must therefore audit replan
frequency and loops rather than assuming a 0.75% check rate is operationally
rare.

## Post-selection distribution-shift warning

Only after threshold selection, the frozen D19 task-0 log was inspected as an
excluded diagnostic. It stores mean h1–h8 uncertainty for each selected
candidate, not horizon-specific D21 check values. Of 760 plans, 59.47% have
mean uncertainty above the selected threshold.

This signal was **not** used to alter the threshold. It indicates that C1
selected candidates occupy a more uncertain distribution than expert
validation actions and predicts a meaningful risk of over-replanning. D21 may
be implemented to measure the actual behavior; D24 must block later expansion
if a high-frequency replan loop appears.

## Artifact integrity

- Result: `results/threshold_selection.json`
- Size: 28,048 bytes
- SHA-256: `3d07d136d9a9ae493ea04ad9bd08ec5f4099a993e301de0906c47996b46c23a1`
- Strict JSON / NaN / Inf audit: pass
- Grid row count: 36
- Eligible row count: 8
- Full per-task, per-episode, per-horizon, and grid metrics are retained.

The result is the one-time frozen D20 selection. Future C2 runs must record
this exact result hash.
