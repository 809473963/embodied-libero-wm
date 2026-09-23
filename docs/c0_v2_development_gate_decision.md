# C0-v2 development gate decision

Decision time: 2026-09-23, before any confirmatory-holdout rollout.

## Decision

`GO_CONFIRMATORY_HOLDOUT`

The preregistered controller-development split uses official init-state indices 30–39 and
contains 10 episodes for each of all 10 LIBERO-Spatial tasks. Init-state indices 40–49 were
not run, inspected, or used when making this decision.

| Metric | Frozen C0-v1 | C0-v2 causal | Delta |
|---|---:|---:|---:|
| Successes / 100 | 11 | 45 | +34 |
| Macro success rate | 11% | 45% | +34 pp |
| Tasks with non-zero success | 3 / 10 | 7 / 10 | +4 tasks |
| Mean episode steps | 281.88 | 210.58 | -71.30 |
| Planned-action clipping ratio | 2.49% | 4.00% | +1.51 pp |

Per-task successes:

| Task | C0-v1 | C0-v2 |
|---:|---:|---:|
| 0 | 0/10 | 9/10 |
| 1 | 0/10 | 7/10 |
| 2 | 0/10 | 7/10 |
| 3 | 0/10 | 8/10 |
| 4 | 3/10 | 0/10 |
| 5 | 0/10 | 5/10 |
| 6 | 4/10 | 5/10 |
| 7 | 0/10 | 0/10 |
| 8 | 4/10 | 4/10 |
| 9 | 0/10 | 0/10 |

All preregistered conditions pass:

- C0-v2 successes >= 20: 45 (`PASS`)
- C0-v2 non-zero tasks >= 5: 7 (`PASS`)
- success improvement over C0-v1 >= 10: +34 (`PASS`)
- non-zero-task improvement over C0-v1 >= 2: +4 (`PASS`)
- finite actions and metrics: `PASS`
- C0-v2 clipping ratio <= 10%: 4.002% (`PASS`)

The single first-pass checkpoint is now frozen for the one-shot holdout:

- path: `results/c0_v2/seed11/best.pt`
- SHA-256: `2346603c8123e0a1cdee408cb635a2ffd12d6bc4e0cc5d91eacbc1330f5faac0`
- selection: minimum demo-validation raw action MSE only; no controller-development outcome
  participated in checkpoint or hyperparameter selection

This GO does not claim uniform improvement: tasks 4, 7, and 9 remain at 0/10 for C0-v2,
and task 4 regresses from 3/10 to 0/10. It says only that the preregistered development
evidence is strong and broad enough to authorize one untouched evaluation on init-state
indices 40–49. Those holdout outcomes may not be used to tune or retrain this checkpoint.
