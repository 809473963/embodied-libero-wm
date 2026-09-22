# D22 Task-0 C0/C1/C2 Ablation Audit

## Verdict

**D22 is complete; D23 is blocked by high-frequency C2 replanning.** C0, C1,
and C2 all score 0/20 on the same task-0 episodes, so this ablation provides no
evidence of a success-rate improvement or regression. C2 nevertheless ends
2,018 of 2,242 plans early (90.01%), averages 100.9 early replans per episode,
and obtains no success. This meets the execution plan's explicit
stop-and-diagnose condition; the evaluation must not be expanded to all ten
tasks until the behavior is understood under a protocol that does not tune on
fixed test episodes.

## Reproducible implementation

- D21 implementation commit: `43d6594f0eded9373833ce434a17aeaccfa12cc9`
- D21 validation commit: `3aa63db2ade081df20970d7b45ba7afb5ac2339d`
- D22 comparison implementation commit:
  `28baf85cb78ee8622fc7b4581a930c13503e9a93`
- Comparison command: `.venv/bin/python scripts/compare_d22_task0.py`
- Test command: `.venv/bin/python tests/test_d22_task0_comparison.py`

The comparison script rejects incomplete results, any difference in episode
identity, actor checkpoint, step limit, or environment action clipping, any
zero-action C2 plan, incomplete action accounting, invalid early-replan
semantics, and a mismatched replan JSONL hash/count.

## Protocol equality

| Check | Result |
|---|---|
| suite/task | `libero_spatial`, task 0 for all methods |
| actor training seed | 11 for all methods |
| actor checkpoint SHA-256 | `8666019d8572f0334965a560c41a3a3442b37692a4b65847375caf3f2f24bbb4` |
| episodes | identical indices 0–19 |
| seeds | identical 10000–10019 |
| initial states | identical fixed indices 0–19 |
| maximum steps | 300 for all methods |
| environment action scale | all evaluators apply `clip(action, -1, 1)` |
| success API | official `env.check_success()` |

C0 is the existing all-task actor result, not the historical task-0-only model.
C1 and C2 both wrap that exact actor. The only intended method differences are
C1's N=3 pre-execution reranking and C2's additional frozen-threshold
verification/replanning.

## Single-task result

| Method | Successes | Rate | Plans | Verification checks | Early replans |
|---|---:|---:|---:|---:|---:|
| C0 | 0/20 | 0.0% | not logged | 0 | 0 |
| C1 | 0/20 | 0.0% | 760 | 0 | 0 |
| C2 | 0/20 | 0.0% | 2,242 | 3,000 | 2,018 |

All 20 paired success outcomes are false for all three methods. Consequently,
`C1-C0`, `C2-C0`, and `C2-C1` are all exactly 0.0 on this task. With a 0/20
floor for every method, no comparative efficacy claim is identifiable.

## C2 behavior

- Every episode reaches 300 actions and has exactly 150 checks.
- All 6,000 C2 environment actions are accounted for by a non-empty plan.
- All 20 episodes contain early replans; the per-episode range is 66–127.
- Mean executed actions per plan: 2.6762.
- Early-plan fraction: 2,018 / 2,242 = 90.0089%.
- Trigger reasons: 37 mismatch-only, 1,859 uncertainty-only, 122 both.
- Check counts h2/h4/h6/h8: `2242/281/255/222`.
- Early replans h2/h4/h6/h8: `1960/25/33/0`.
- There are 2,046 threshold exceedances; 28 occur at h8 and are correctly
  logged without an empty replan.
- C2 creates 2.95 times as many plans as C1. Mean plan inference is 62.93 ms,
  excluding environment rendering and verification encoding.

The loop is not an implementation-level infinite loop: every plan advances at
least two simulator actions. It is an operational high-frequency replan loop,
dominated by uncertainty at h2 and producing substantial extra inference with
no observed task success.

## Artifact integrity

| Artifact | SHA-256 |
|---|---|
| `results/c0/seed11/summary.json` | `9d7b76f11a39baf9d1e1d543e2c528b3abe270a63c11eca0fbe219b27988dfc2` |
| `results/c1/seed11/task0_d19.json` | `24bb1aadb32cf222f469fec890e082c8c0d5b235b4dc272ee1ef82581e45f120` |
| `results/c2/seed11/task0_d22.json` | `a89189cbbdd09464d51e38002abac43cabc0cb8596de254fa1f678fc090f0c2c` |
| ignored `results/c2/seed11/replan_events_d22.jsonl` | `d76a05b80a1d8756eb5671acb3e51c0bf41fe3540a44750a3782a9c284855107` |
| `results/d22_task0_ablation.json` | `55ba8f6f7491d90cdd922fb8365a5473c646d042453ecfc05bfc8ce8917a65ab` |

The C2 raw JSON retains every episode, plan, and verification event. The JSONL
contains exactly 2,018 early-replan records and is deliberately ignored as a
derived large event stream, but is hash-addressed by both the raw summary and
the D22 comparison.

## Stop decision

`results/d22_task0_ablation.json` freezes status
`BLOCK_D23_HIGH_FREQUENCY_REPLANNING` and `d23_authorized=false`. D20 thresholds
remain unchanged; the fixed test episodes are not threshold-selection data.
The next valid action is diagnosis, not a ten-task run or post-hoc threshold
tuning. A future correction must derive any changed control rule from training
or validation evidence, receive a new version, and rerun task 0 under a new run
ID while preserving these negative results.
