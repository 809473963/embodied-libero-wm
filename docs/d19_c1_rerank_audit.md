# D19 C1 Pre-execution Reranking Audit

## Verdict

**D19 PASS. Final project G2 is now PASS; proceed to D20 only.**

C1 generates three action chunks, scores them with the frozen D15 ensemble,
selects one before execution, and executes the complete selected chunk. It does
not compare predictions with future observations, interrupt a chunk, or replan
early. C2 remains unauthorized until D20 threshold selection is complete.

## Frozen C1 contract

- Implementation commit: `9ccbae77d39ad72a1976071b0a8eb68fd01967f0`
- Actor: unchanged all-task C0 seed-11 checkpoint
- Actor SHA-256: `8666019d8572f0334965a560c41a3a3442b37692a4b65847375caf3f2f24bbb4`
- World model: D15 members `[11,23,47]`
- Candidate count: 3
- Candidate 0: exact clipped C0 action chunk
- Candidates 1/2: deterministic antithetic temporally smoothed noise
- Noise std: `[0.05,0.05,0.05,0.02,0.02,0.02,0.0]`
- Gripper perturbation: zero
- Candidate range: clipped to `[-1,1]`
- Selection score:
  `goal_progress - 0.25*mean_uncertainty - 0.001*normalized_action_deviation`
- Execution: full selected 8-step chunk; no execution-time verification

The candidate generator uses a rollout-seed/plan-index-derived CPU RNG. The
same observation and rollout seed reproduces exactly the same selected chunk,
candidate index, and scores. The policy input remains agentview RGB, language,
and the declared 12-D proprioception only.

## Tests

- synthetic candidates have shape `[3,8,7]`
- candidate 0 equals the base chunk exactly
- candidates 1/2 are antithetic when clipping is inactive
- gripper actions are unchanged across candidates
- all candidate score components have shape `[3]` and are finite
- one real D13 observation passes frozen CLIP, C0, all three WM members, and
  reranking
- when candidate 0 is selected, its executed chunk is bitwise equal to C0
- fixed rollout seed reproduces the real planning result
- C0 adapter and D15 ensemble regressions still pass

## Single-task production log

- Task: Spatial task 0
- Fixed episodes: the same 20 seeds/init-state indices as C0
- Max steps: 300
- Completed episodes: 20/20
- Successes: 0/20
- C0 result on the same task/actor/protocol: 0/20
- Planning events: 760 (38 per episode)
- Candidate selection counts: base `20`, positive perturbation `366`, negative
  perturbation `374`
- Non-base selection fraction: 97.37%
- Mean / p50 / p95 planning latency: 72.37 / 70.62 / 83.94 ms
- Maximum latency: 697.39 ms (cold/outlier plan)
- Mean top-two score margin: 0.0003386
- Mean uncertainty of selected candidates: 0.0047222
- Base raw-action clipping ratio: 0.9445%

Every event satisfies:

- selected candidate equals score argmax;
- all scores, goal progress, uncertainty, deviation, and similarity values are
  finite;
- `executed_full_chunk=true` and `early_replan=false`;
- episode plan indices are contiguous and planning steps are `0,8,…,296`;
- environment actions are finite 7-D values clipped to `[-1,1]`.

The output's resume path was tested both after a one-episode smoke run and on
the completed 20-episode result.

## Interpretation and risks

This day establishes a real C1 control output, not a success improvement. C1
and C0 both score 0/20 on task 0, so no efficacy claim is made.

The reranker is nonconstant, but candidate 0 is selected only 2.6% of the time.
The selected online uncertainty mean (`0.00472`) is also above the D15 held-out
uncertainty p99 (`0.00354`). This indicates action/rollout distribution shift
and makes the small score margins important. The frozen D19 config must not be
tuned on these fixed test episodes. D20 should select only the mismatch
threshold on validation episodes, and D22 must report this C1 behavior honestly.

## Artifacts and G2 completion

- Result: `results/c1/seed11/task0_d19.json`
- Result size: 1,024,211 bytes
- Result SHA-256: `24bb1aadb32cf222f469fec890e082c8c0d5b235b4dc272ee1ef82581e45f120`
- Machine-readable G2 completion: `configs/d19_g2_completion.json`

The stage-level G2 definition requires a trained world model, held-out
transition evidence, and a C1 output. D15, D16, and this D19 result now satisfy
those three items. Final project G2 is therefore PASS, while C2 remains pending
D20/D21.
