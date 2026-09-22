# D21 C2 Prediction Verification and Replanning Audit

## Verdict

**D21 functional PASS, with a high-frequency replanning warning.** The frozen
D20 thresholds are used exactly, prediction verification occurs after every
two executed actions, and all early replans discard a non-empty suffix before
planning again on the next control step. A one-episode production smoke test
found no zero-action or same-step loop. It did, however, trigger 90 early
replans in 300 actions, so only the narrow D22 task-0 comparison is authorized;
all-task expansion must wait for the planned behavior review.

## Frozen contracts

- Implementation commit: `43d6594f0eded9373833ce434a17aeaccfa12cc9`
- C2 contract: `c2-v1`
- C1 candidate/reranking contract: unchanged `c1-v1`
- D20 result SHA-256:
  `3d07d136d9a9ae493ea04ad9bd08ec5f4099a993e301de0906c47996b46c23a1`
- Mismatch threshold: `0.029063597321510315`
- Uncertainty threshold: `0.004451490938663483`
- Trigger: mismatch above threshold **OR** uncertainty above threshold
- Check horizons: h2, h4, h6, h8
- h8 exceedance is recorded but cannot cause an early replan because no action
  remains in the selected chunk.

No threshold was selected or changed using simulator evaluation behavior. C0,
C1 candidate generation, the actor checkpoint, ensemble checkpoints, action
scale, H=8, fixed episode seeds, and fixed initial states remain unchanged.

## Implementation behavior

For each C1-selected candidate, C2 retains the D15 ensemble mean prediction
and horizon-wise ensemble uncertainty. After actions 2, 4, 6, and 8, the live
upright `agentview` RGB is encoded by the same frozen CLIP image encoder. C2
computes `1 - cosine(predicted_z, observed_z)` and applies the frozen OR rule.
If a threshold is exceeded before h8, the remaining chunk suffix is discarded
and a fresh RGB/proprioception-based C1 plan is created at the next control
step.

`replan_events.jsonl` contains only actual early-replan events. The tracked
summary embeds the complete plan and verification traces plus the path, count,
and SHA-256 of the ignored JSONL artifact.

## Tests

The following checks pass:

- synthetic no-trigger case;
- uncertainty-only early trigger;
- h8 threshold exceedance without an empty early replan;
- one real D13 validation observation through C1 planning and h2 verification;
- finite selected actions, mismatch, uncertainty, gradients/model regressions;
- D13 dataset, D14 model, D15 ensemble, C1, and D20 threshold regression tests.

The real validation interface check produced mismatch `0.0034036636` and
uncertainty `0.0014861822`; neither exceeded its threshold.

## Production smoke result

Task 0, fixed seed 10000 and fixed initial state 0, was run for the standard
300-step limit with the frozen all-task seed-11 C0 actor.

| Metric | Value |
|---|---:|
| episodes / successes | 1 / 0 |
| environment actions | 300 |
| plans | 103 |
| verification events | 150 |
| early replans | 90 |
| mean executed actions per plan | 2.9126 |
| mismatch-only replans | 3 |
| uncertainty-only replans | 86 |
| both-threshold replans | 1 |

Verification counts by horizon are h2/h4/h6/h8 = `103/19/15/13`. Early
replan counts are `84/4/2/0`. Mismatch ranges from `0.00448364` to
`0.04179299`; uncertainty ranges from `0.00170036` to `0.01168497`.

## Loop and integrity audit

- Plan indices are contiguous `0..102`.
- Every plan executes 2, 4, 6, or 8 actions; no plan executes zero actions.
- Executed-action counts sum exactly to 300.
- Every verification horizon is in `{2,4,6,8}` and
  `remaining_actions == 8 - horizon`.
- `early_replan` is exactly `threshold_exceeded AND remaining_actions > 0`.
- All 150 checks occur after an even number of environment actions.
- All numerical values are finite.
- The 90 summary events exactly match the 90 JSONL lines and its stored hash.
- Resume reconstructs the JSONL deterministically from the tracked episode
  trace and refreshes its hash-addressed metadata.

Artifacts:

- Tracked summary: `results/c2/seed11/task0_d21.json`
- Summary SHA-256:
  `00fd12074df5440b731fad3430b2867994bc9bfb1bbc21fb3cde8f3d50e094e9`
- Ignored event log: `results/c2/seed11/replan_events_d21.jsonl`
- Event-log SHA-256:
  `7cbee1ac2439bb1f6eb3e83d9c8fc256ec0fe32d330ce4f6af0e5e7b7862810e`

## Risk and next-day decision

This is not a zero-action dead loop: the simulator always advances and 13
chunks execute fully. It is nevertheless clear over-replanning relative to the
D20 nominal expert distribution: 87.4% of plans end early, mostly because of
uncertainty at h2. The result confirms D20's predeclared distribution-shift
warning and must not be used to tune the frozen thresholds.

D22 may run the prescribed task-0 C0/C1/C2 comparison to measure the control
effect under identical episodes. This D21 smoke does **not** authorize D23
all-task expansion on its own. If the full task-0 run confirms a high-frequency
loop, progression must stop for behavior diagnosis rather than silently
changing validation-frozen thresholds.
