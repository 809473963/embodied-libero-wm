# Autonomous proposal-to-value research stops at the candidate-value gate

Date: 2026-10-02

Highest completed stage: **B**

Final formal labels:

```text
EXPANDED_CANDIDATE_INTERFACE_VALID
STRUCTURED_GRIPPER_PROPOSALS_VALID
N64_REQUIRED_FOR_HEADROOM
STRONG_EXPANDED_CANDIDATE_HEADROOM

PAIRWISE_PROGRESS_SIGNAL_VALID
ABSOLUTE_PROGRESS_BASELINE_VALID
PAIRWISE_PROGRESS_OUTPERFORMS_ABSOLUTE
PROGRESS_MODEL_LANGUAGE_INSENSITIVE
PROGRESS_MODEL_ROBOT_PHASE_DOMINATED
CANDIDATE_VALUE_AUDIT_BLOCKED
```

Current authorization:

```text
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
STOP
```

```text
Selected Candidate Interface: yes — Selected Expanded Candidate Interface v1 (C48_G15)
Selected Candidate Value Model: no
Selected World Model Interface: no
Controller authorized: NO
```

## Executive conclusion

The autonomous gated study resolves the proposal layer but not the value
layer. Stage A freezes a compliant one-shot N64 interface: one frozen C0-v2
default, 48 continuous alternatives in 24 exact antithetic pairs, and 15
structured gripper alternatives including G3, both constants, and twelve
single switches. It retains very strong imitation-oracle headroom: `68.096%`
H8 gain, CI `[66.213%,69.821%]`, `80.255%` H1 gain, and positive results on all
ten tasks.

Stage B then asks the logically prior value question using real expert
demonstration observations, not predicted or candidate futures. A small
antisymmetric visual+language pairwise model does learn temporal progress:
`74.245%` local `Delta t<=8` accuracy and a `9.141`-point advantage over the
absolute baseline. But it fails the semantics needed for candidate value.
Wrong language reduces accuracy by only `3.802` points rather than the required
5, while proprio-only reaches `96.224%`, beating the visual model by `21.979`
points. The learned preference is therefore best described as a strong
demonstration/robot-phase ordering signal, not a validated language-conditioned
goal evaluator.

The frozen decision tree requires an immediate stop. Stage C was not started,
because designing a World Model around an evaluator that has not established
goal dependence would reverse the intended causal order.

## Stage outcomes

| Stage | Question | Outcome | Authorization |
|---|---|---|---|
| A | Is there one exact finite expanded candidate interface? | **pass** | `GO_CANDIDATE_VALUE_AUDIT` consumed by Stage B |
| B | Is there a natural-input, task-language-conditioned progress preference? | **fail** | `BLOCK_WORLD_MODEL_INTERFACE_AUDIT` |
| C | Can action-conditioned prediction preserve the selected preference? | not run | prohibited by Stage B |

## Selected Candidate Interface v1

| Field | Frozen value |
|---|---|
| Horizon | 8 |
| Total candidates | 64 |
| A0 | frozen clipped C0-v2 chunk, exactly once |
| Continuous alternatives | 48 |
| Antithetic pairs | 24 complete pairs, zero unmatched |
| Dimensions | translation + rotation; gripper unchanged |
| Scale | 2.5x frozen production sigma |
| Temporal smoothing | `[0.25,0.5,0.25]` |
| Gripper alternatives | 15 structured forms |
| Mandatory forms | G3 `c0_binarized`, all-negative, all-positive |
| Additional forms | 12 calibration-selected single switches |
| Bounds | `[-1,1]` |
| Seed base | 1901 |
| Contract hash | `5cd3311d847c410a75cca6970e7acf4e5ec3af811c739946df7ec18532a9f14a` |

The interface is selected as a deterministic proposal contract. It is not an
authorized deployed planner and has no real candidate futures in this study.

## Value audit outcome

| Evaluator | Local <=8 accuracy | 95% CI | Role/result |
|---|---:|---|---|
| V0 task terminal prototype | 55.234% | [54.089%,56.432%] | terminal similarity is insufficient locally |
| V1 absolute visual+language | 65.104% | [64.062%,66.146%] | valid weak baseline |
| V2 visual+language | 74.245% | [73.073%,75.443%] | strong pairwise temporal signal |
| V2 proprio+language | 96.224% | [95.365%,97.031%] | severe robot-phase shortcut |
| V2 visual+proprio+language | 80.990% | [79.427%,82.604%] | combined input remains phase-heavy |

The visual pairwise model passes accuracy, CI, task robustness, symmetry,
locality, and V1 comparison. It fails two independent mandatory safeguards:

- wrong-language degradation is `3.802` points, below 5;
- proprio-only is `21.979` points better than visual+language.

No candidate-value model is selected.

## Final scientific answers

### Q1 — Is there an exact, deterministic, approximately N64 candidate interface with antithetic symmetry and G3?

Yes. The frozen interface has exactly N64, 24 complete continuous antithetic
pairs, zero unmatched directions, one A0, and mandatory G3 plus both constants.

### Q2 — Does the corrected structured-gripper plus continuous pool retain strong oracle headroom?

Yes under the explicitly offline imitation oracle. H8 gain is `68.096%`, H1 is
`80.255%`, and all ten tasks improve. This is not reward, success, or deployed
value.

### Q3 — What roles do constants, switches, and G3 play in the final gripper proposals?

All contribute distinct corrections. In the complete pool, G3 wins `25.09%`
of rows, single switches `10.04%`, and constants `4.51%`. G3 supplies frequent
trajectory-wide sign discretization, switches model within-chunk state changes,
and constants cover persistent negative/positive states.

### Q4 — Is there a natural-input, language-conditioned evaluator that learns progress beyond terminal similarity and robot phase?

No validated evaluator. Visual V2 learns local temporal order beyond V0, but
language dependence is below the frozen materiality threshold and proprio-only
is dramatically stronger. The complete claim therefore fails.

### Q5 — Is pairwise progress clearly better than absolute scalar progress, especially for `Delta t<=8`?

Yes. Pairwise V2 reaches `74.245%` versus V1's `65.104%`, a `9.141`-point
paired advantage with CI `[7.604,10.651]`.

### Q6 — Does wrong language significantly damage the evaluator and prove real task-goal dependence?

It causes a statistically positive but mechanically insufficient `3.802`-point
drop. That does not clear the frozen 5-point goal-conditioning gate.

### Q7 — If the value audit had passed, what should a World Model predict for the evaluator?

The proper target would have been the smallest future representation that
preserved the frozen V2 preference ordering, rather than pixels or latent MSE
for their own sake. Because no V2 was selected, no representation is now
authorized or specified.

### Q8 — Do action-conditioned predicted futures preserve the evaluator's preference ordering?

Unknown and deliberately untested. Stage C was prohibited after Stage B failed;
no action-conditioned World Model was trained or evaluated.

### Q9 — Where is the current bottleneck: candidate proposal, value/progress, or World Model prediction?

The current demonstrated bottleneck is **value/progress semantics**. Candidate
proposal is resolved and strong under the imitation oracle. World Model
prediction was never reached, so it cannot be blamed or validated.

### Q10 — What is the only legal next authorization?

There is no GO authorization within this task. The formal action is `STOP` with
`BLOCK_WORLD_MODEL_INTERFACE_AUDIT` and `BLOCK_SPATIAL_WM_PROTOTYPE`. Any future
attempt to separate goal semantics from robot/demo phase would require a new,
separately preregistered study; this report does not authorize it automatically.

## Integrity and stopping boundary

All Stage A and Stage B numerical artifacts, model weights, sidecars, protocols,
and reports have SHA-256 ledgers in their stage reports. Calibration used only
31--34 and frozen development evaluation only 36--39. Demo/init40--49, reward,
done, success, task outcomes, online candidate execution, simulator rollouts,
new encoders, World Models, rerankers, and controllers were never accessed or
run. The final complete repository suite passes `213` tests with 21 dependency
deprecation warnings and no failures.

The final causal chain is therefore:

```text
proposal space: validated
        ↓
goal-conditioned value: not validated
        ↓
World Model interface: not authorized
        ↓
controller: not authorized
```
