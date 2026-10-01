# Pairwise temporal progress is real, but it is not a validated goal-conditioned value

Date: 2026-10-02

Final mechanism labels:

```text
PAIRWISE_PROGRESS_SIGNAL_VALID
ABSOLUTE_PROGRESS_BASELINE_VALID
PAIRWISE_PROGRESS_OUTPERFORMS_ABSOLUTE
PROGRESS_MODEL_LANGUAGE_INSENSITIVE
PROGRESS_MODEL_ROBOT_PHASE_DOMINATED
CANDIDATE_VALUE_AUDIT_BLOCKED
```

Next-stage authorization: `BLOCK_WORLD_MODEL_INTERFACE_AUDIT`

Spatial World Model authorization: `BLOCK_SPATIAL_WM_PROTOTYPE`

Selected candidate value model: `none`

Controller authorization: `false`

## Executive answer

The frozen natural-image pairwise model learns a stable local temporal-order
signal, but it does not clear the goal-conditioning or robot-shortcut gates.
On held-out demos36--39, V2 visual+language reaches `74.245%` accuracy for
non-terminal pairs with `Delta t<=8` (task-stratified demo-block 95% CI
`[73.073%,75.443%]`) and is positive on all ten tasks. It exceeds the absolute
V1 baseline by `9.141` percentage points, CI `[7.604,10.651]`, and is not a
terminal-only classifier. This assigns `PAIRWISE_PROGRESS_SIGNAL_VALID` and
`PAIRWISE_PROGRESS_OUTPERFORMS_ABSOLUTE`.

The signal is not sufficiently goal-conditioned. Replacing the correct task
language by the deterministic wrong-task language reduces local V2 accuracy by
only `3.802` points, CI `[2.656,4.974]`. The interval is positive, but the point
estimate misses the preregistered 5-point materiality requirement. Therefore
`GOAL_CONDITIONED_PROGRESS_VALID` is not assigned and
`PROGRESS_MODEL_LANGUAGE_INSENSITIVE` is assigned.

More decisively, proprio+language reaches `96.224%` local accuracy, exceeding
the visual+language primary by `21.979` points. Visual+proprio+language reaches
`80.990%`, also above visual alone. The audit therefore assigns
`PROGRESS_MODEL_ROBOT_PHASE_DOMINATED`: expert-trajectory order is largely
recoverable from robot phase without establishing that the evaluator judges
task-goal progress in natural images.

Cross-demo controls reinforce that limit. On same-task, different-demo states
at nominally matched normalized progress, V2's mean absolute probability
deviation from indifference is `0.459`; its p90 is effectively `0.5`. The model
is often highly confident even when the construction supplies no meaningful
progress separation. It can order cross-demo states separated by 0.25 progress
with `91.25%` accuracy, but that does not repair the matched-stage
overconfidence or the stronger proprio shortcut.

The formal result is `CANDIDATE_VALUE_AUDIT_BLOCKED`. Stage C was not run. No
value representation, World Model interface, reranker, or controller is
selected or authorized.

## Frozen provenance and stage boundary

Stage A is frozen at commit `5d38525` under annotated tag
`expanded-candidate-interface-valid`. Its exact C48_G15 N64 contract and
`GO_CANDIDATE_VALUE_AUDIT` authorization remain valid.

The Stage B protocol, pair construction, architectures, training schedule,
controls, 10,000-draw bootstrap, thresholds, and decision mapping were
committed at `2c942fd` before training. Calibration weights and diagnostics
were frozen at `a47575e` before demos36--39 were opened. The one-shot numerical
evaluation was frozen at `bd24b68`.

No protocol, threshold, language mapping, model form, or result was amended
after evaluation. Because Stage B failed, the mandated decision tree blocks
Stage C rather than changing the model or weakening a gate.

## Data boundary and label meaning

| Partition | Demo IDs | Trajectories | States | Pair rows | Use |
|---|---:|---:|---:|---:|---|
| Calibration | 31--34 | 40 | 5,044 | 15,360 | fit features/models/temperatures and freeze |
| Frozen development evaluation | 36--39 | 40 | 4,883 | 15,360 | one-shot metrics and gates; not blind confirmation |
| Historical reference | 30/35 | — | — | — | not opened |
| Forbidden | 40--49 | 0 | 0 | 0 | never accessed |

Every label is derived from temporal order or normalized position in an
existing expert demonstration. No action, reward, done, success, simulator
state, object pose, or outcome enters any evaluator. Consequently all positive
results mean only **demonstration-progress preference**. They are not
task-optimal value, candidate value, or control success.

## Pair construction

Each trajectory contributes 32 forward pairs and their exact reverses to every
frozen bin:

| Gap bin | Evaluation rows | Primary local gate |
|---|---:|---|
| 1--2 | 2,560 | yes |
| 3--4 | 2,560 | yes |
| 5--8 | 2,560 | yes |
| 9--16 | 2,560 | no |
| >16 (17--32 support) | 2,560 | no |
| terminal-anchor | 2,560 | no |

Equal bin counts prevent far and terminal pairs from dominating. Pair indices
are deterministic evenly spaced samples from each finite population. The local
gate contains 7,680 rows and explicitly excludes terminal anchors.

The cross-demo controls use the same task but different cyclically paired demos:
640 matched-progress pairs and 1,280 direction-balanced pairs separated by
0.25 normalized progress. No image is synthesized or modified.

## Frozen evaluator families

| Model | Input and form | Trainable parameters | Role |
|---|---|---:|---|
| V0 | cosine to calibration-only task terminal prototype | 0 | terminal/prototype baseline |
| V1 | visual + language, 64-unit scalar MLP | 65,665 | absolute normalized-progress baseline |
| V2 visual | `(za,zb,zb-za,language)`, antisymmetric 64-unit MLP | 131,201 | primary |
| V2 proprio | same pair form with 12-D proprio | 35,201 | robot-phase ablation |
| V2 visual+proprio | same pair form with 524-D state | 133,505 | combined ablation |

V2 uses `raw(a,b,l)-raw(b,a,l)`, so reverse probabilities are complementary by
construction. The held-out maximum complement error is `2.22e-16`. All frozen
encoders remain unchanged.

## Aggregate and local results

| Evaluator | Overall acc. | Local <=8 acc. | Local 95% CI | Local AUROC | Log loss | Gap rho | Positive tasks |
|---|---:|---:|---|---:|---:|---:|---:|
| V0 prototype | 63.815% | 55.234% | [54.089%,56.432%] | 0.570 | 0.702 | 0.128 | 9/10 |
| V1 absolute | 77.461% | 65.104% | [64.062%,66.146%] | 0.728 | 0.824 | 0.436 | 10/10 |
| **V2 visual+language** | **82.305%** | **74.245%** | **[73.073%,75.443%]** | **0.836** | 1.064 | **0.611** | **10/10** |
| V2 proprio+language | 96.107% | 96.224% | [95.365%,97.031%] | 0.991 | 0.128 | 0.900 | 10/10 |
| V2 visual+proprio+language | 88.320% | 80.990% | [79.427%,82.604%] | 0.904 | 0.667 | 0.738 | 10/10 |

The visual V2 point accuracy exceeds V1 by `9.141` points with a wholly
positive paired interval. Its held-out log loss is nevertheless worse because
the model is overconfident on some errors; accuracy and ordering, not calibrated
probability quality, carry the positive pairwise label.

V1 narrowly clears the exact 65% local gate, so
`ABSOLUTE_PROGRESS_BASELINE_VALID` is assigned. V0 remains near chance locally
despite reaching `82.58%` on terminal-anchor pairs, showing why the audit did
not use an overall or terminal-only criterion.

## Gap behavior and monotonicity

| Gap | V0 | V1 | V2 visual | V2 proprio | V2 visual+proprio |
|---|---:|---:|---:|---:|---:|
| 1--2 | 54.61% | 57.58% | 66.09% | 95.39% | 72.19% |
| 3--4 | 54.06% | 63.98% | 73.75% | 96.41% | 81.64% |
| 5--8 | 57.03% | 73.75% | 82.89% | 96.88% | 89.14% |
| 9--16 | 64.38% | 86.09% | 88.28% | 96.88% | 94.92% |
| >16 | 70.23% | 93.59% | 92.97% | 93.98% | 97.73% |
| terminal-anchor | 82.58% | 89.77% | 89.84% | 97.11% | 94.30% |

For visual V2, full-trajectory forward-pair agreement is `61.45%`, `75.52%`,
`83.59%`, and `90.29%` at exact gaps 1/4/8/16. The signal strengthens with
temporal distance but is already above chance at adjacent states; therefore
`TERMINAL_PROGRESS_ONLY` is not assigned.

## Language sensitivity

Wrong language is a fixed derangement `task -> (task+1) mod 10`; it is not
chosen from evaluation results.

| Evaluator | Correct local | Wrong-language local | Drop | 95% CI |
|---|---:|---:|---:|---|
| V0 | 55.234% | 53.438% | 1.797 pp | [0.312,3.255] pp |
| V1 | 65.104% | 63.385% | 1.719 pp | [0.417,2.995] pp |
| **V2 visual** | **74.245%** | **70.443%** | **3.802 pp** | **[2.656,4.974] pp** |
| V2 proprio | 96.224% | 82.188% | 14.036 pp | [12.552,15.469] pp |
| V2 visual+proprio | 80.990% | 78.359% | 2.630 pp | [1.172,4.063] pp |

The primary model reacts to language statistically, but not by the frozen
materiality threshold. A positive interval is insufficient when the required
point drop is 5 points. `GOAL_CONDITIONED_PROGRESS_VALID` is therefore withheld.

## Robot-phase shortcut

The proprio-only model exceeds the primary visual model by `21.979` local
accuracy points; the paired primary-minus-proprio CI is
`[-23.438,-20.495]` points. This is far beyond the frozen severe-shortcut rule,
which triggers when proprio reaches at least 65% and visual fails to lead by 2
points.

The result does not say visual features contain no progress information: V2
visual clearly beats V1 and V0. It says temporal-order supervision in these
expert trajectories is even more directly encoded by robot configuration and
phase, so it cannot validate a natural-image goal/value representation.

## Cross-demo shortcut controls

| Evaluator | Matched-progress mean | Matched p90 | Mismatched acc. |
|---|---:|---:|---:|
| V0 | 0.139 | 0.288 | 71.88% |
| V1 | 0.405 | 0.500 | 92.66% |
| V2 visual | 0.459 | 0.500 | 91.25% |
| V2 proprio | 0.462 | 0.500 | 94.38% |
| V2 visual+proprio | 0.466 | 0.500 | 97.66% |

The matched columns are absolute deviation from probability 0.5, so lower is
better. The learned models are severely overconfident across different demos at
the same nominal progress. Strong mismatched ordering is compatible with
learning demo/scene/robot phase; it is not sufficient evidence of task-goal
comparison.

## Mechanical decision

| Required gate | Result |
|---|---|
| V2 local accuracy >=65% | pass: 74.245% |
| Local CI lower >55% | pass: 73.073% |
| At least 7 positive tasks | pass: 10/10 |
| Wrong-language degradation >=5 pp, positive CI | **fail: 3.802 pp** |
| No severe proprio shortcut | **fail: proprio is 21.979 pp better** |
| V2 non-inferior to V1 | pass |
| V2 materially outperforms V1 | pass: +9.141 pp |
| Terminal-only pattern absent | pass |
| Complete Stage B gate | **FAIL** |

Final mechanism labels:

```text
PAIRWISE_PROGRESS_SIGNAL_VALID
ABSOLUTE_PROGRESS_BASELINE_VALID
PAIRWISE_PROGRESS_OUTPERFORMS_ABSOLUTE
PROGRESS_MODEL_LANGUAGE_INSENSITIVE
PROGRESS_MODEL_ROBOT_PHASE_DOMINATED
CANDIDATE_VALUE_AUDIT_BLOCKED
```

Final authorization:

```text
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

No value model is selected. Stage C is prohibited by the frozen decision tree.

## Required answers

**Q1 — Does natural visual input contain a stable local progress signal?**

Yes descriptively. V2 visual+language reaches `74.245%` on the primary local
set with a strong CI and 10/10 positive tasks.

**Q2 — Is the result only terminal similarity?**

No. Adjacent and 3--4-step accuracies are `66.09%` and `73.75%`; the complete
local gate passes. V0 terminal similarity does not pass locally.

**Q3 — Does pairwise modeling outperform absolute progress?**

Yes. V2 exceeds V1 by `9.141` local points with positive CI. Both receive
positive mechanism labels, while pairwise is clearly stronger.

**Q4 — Is V2 truly goal-conditioned by task language?**

Not under the frozen gate. Wrong language lowers accuracy by only `3.802`
points, below the required 5.

**Q5 — Is the learned preference mainly visual task progress rather than robot phase?**

No. Proprio-only achieves `96.224%` and decisively exceeds the visual model.
This triggers the robot-phase-dominated label.

**Q6 — Do cross-demo controls rule out scene/time shortcuts?**

No. V2 is highly confident on many matched-progress cross-demo pairs. The
mismatched-pair success therefore cannot uniquely establish task-goal value.

**Q7 — Can this evaluator rank the frozen N64 candidates?**

Not established. Stage A candidates have no real futures here, and Stage B uses
only expert demonstration observations. No candidate-value bridge was run.

**Q8 — May Stage C design a value-aligned World Model interface?**

No. Stage B fails two mandatory gates, so
`BLOCK_WORLD_MODEL_INTERFACE_AUDIT` is final for this task.

## Artifacts, finite values, and reproducibility

| Artifact | SHA-256 |
|---|---|
| Protocol | `c16e54be49847e573860e974ba50a4fbdec696fda405b16f7b72f382ea048563` |
| Calibration freeze | `09c394b5fef43e81f98dcce0b162c88e536d65678ee21ebf1688c0820171bd5f` |
| Calibration weights | `dd6ea952d7fd6b607a9c7cc601cef41560a7bb6c00a476c7de1a7ff7fe7aeb8d` |
| Calibration sidecar | `1f012337ad7b9ce71dabcb7d87c5fa92f62c213d03c0440ce7471f80fa8d3600` |
| Evaluation diagnosis | `200d4ba14a8668f142f3ac41c56b71e1bbb81e1521af989ecd2b1af4ee69e01b` |
| Evaluation sidecar | `fda27082dffc7739291c77609b1ab862bad462d9246da0dc2484ccdc688b16ec` |
| Probe implementation | `4fa5b4d7f9800e8209827c8673749dbf687063d66b313a6db52f9583df39fbfe` |
| Audit runner | `e10b3b27abd9f000a50534daa0a238730d51d559a2fa7fbcb3f84b8d2a294db8` |

The ignored sidecars retain all pair identities, labels, gaps, probabilities,
wrong-language probabilities, and calibration artifacts needed for independent
recalculation. All admitted inputs, parameters, predictions, bootstrap arrays,
and reportable metrics are finite.

Six protocol tests and six result tests independently verify frozen hashes,
partition and input boundaries, balanced pair construction, cross-demo
controls, model capacity and antisymmetry, calibration artifacts, finite
sidecars, recomputed local accuracies, the pairwise advantage, failed language
materiality, decisive proprio shortcut, and final authorization mapping. They
pass `12/12`. The complete repository suite passes `213` tests; the remaining
21 messages are dependency deprecation warnings, not failures.

## Forbidden-activity audit

No simulator, candidate execution, branch future, reward, done, success, task
outcome, expert-action input, demo/init40--49, synthetic image, modified RGB,
new encoder, encoder fine-tuning, World Model, learned proposal, reranker,
controller, CEM/MPPI/beam search, C1/C2, D20-v2, or seed23/47 experiment was
run or modified. Stage C was not started. No upstream artifact was overwritten.
