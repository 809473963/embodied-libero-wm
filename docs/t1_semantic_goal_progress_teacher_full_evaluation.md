# Robust grounding validates a sparse-but-dense-enough semantic goal-progress teacher

Date: 2026-10-02

Final mechanism labels:

```text
ROBUST_SAME_SCENE_GROUNDING_VALID
PERSISTENT_REFERENT_CONTRACT_VALID
OFFICIAL_GOAL_MARGIN_VALID
SEMANTIC_LOCAL_PROGRESS_SIGNAL_VALID
SEMANTIC_LANGUAGE_COUNTERFACTUAL_VALID
SEMANTIC_ROBOT_PHASE_INVARIANCE_SUPPORTED
SEMANTIC_GOAL_PROGRESS_TEACHER_VALID
```

Next-stage authorization: `GO_VALUE_DISTILLATION`

Continuing blocks:

```text
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

Controller authorization: `false`

Selected semantic teacher: `T1-M2`

## Executive answer

The fully robust-grounded clean evaluation validates T1 as a semantic
goal-progress teacher. Its persistent referent is selected once in the initial
scene by the frozen G2 resolver. The state is lexicographic: first the exact
official `On(referent, plate)` Boolean, then M2, the minimum normalized clearance
of the official horizontal-distance and vertical-order inequalities. Contact
remains exclusively in the official Boolean. No robot pose, time, action,
reward, success, or grounding score is mixed into the teacher.

On frozen development demos36--39, local `Delta t<=8` non-tie density is
`57.91%`, with all ten tasks above the `30%` floor. Progress, regression, and
ties occupy `44.95%`, `12.97%`, and `42.09%` of local pairs. Time agrees with
only `77.61%` of non-ties: the teacher is correlated with demonstration order
but permits substantial ties and regression.

Language has material same-scene effects. Across 32,757 eligible state-level
counterfactuals, the persistent referent and semantic state change in `53.29%`;
across 252,300 eligible local pair counterfactuals, preference changes in
`29.59%`. Every task exceeds the frozen 20% state-change threshold. The two
robot/semantic controls pass across all ten tasks. A calibration-fitted
ridge-linear proprio-only pair probe reaches `80.01%` held out, below the
preregistered `90%` severe-shortcut boundary.

Every gate passes and demo40--49 receive zero numeric and zero metadata access.
The result authorizes `GO_VALUE_DISTILLATION` only. Distillation was not run;
World Model, interface, reranker, controller, and task-success work remain
blocked.

## Frozen construction

```text
initial natural physical scene + real instruction source relation
    -> frozen G2 UNIQUE referent
    -> persist MuJoCo object identity for the trajectory
    -> T1 = [official On Boolean, M2 continuous margin]
```

The grounding confidence is a gate, never a term in T1. T2 was not activated:
T1 already cleared calibration density (`58.31%`) on 10/10 tasks.

## Margin revalidation

The three variants were selected without temporal or pair-density information.

| Margin | Calibration goal AUC | Positive median | Negative median | Separation |
|---|---:|---:|---:|---:|
| M0 old surface candidate | 0.8846 | -3.152 | -8.254 | 5.102 |
| M1 softened surface scale | 0.9038 | -1.576 | -8.246 | 6.670 |
| **M2 official inequalities** | **0.9533** | **0.176** | **-8.246** | **8.422** |

M2 is exactly:

```text
min(
  (0.03 - horizontal_center_distance) / 0.03,
  (bowl_center_z - plate_center_z) / 0.03
)
```

These are the two installed numerical `check_ontop` conditions. The installed
contact test remains the Boolean first coordinate and is not approximated by
M2. Held-out goal-positive and goal-negative medians are `0.161` and `-8.341`;
all values are finite in `[-16.285,0.966]`. This assigns
`OFFICIAL_GOAL_MARGIN_VALID` without claiming that M2 replaces official success.

## Calibration and held-out summary

| Metric | Calibration | Frozen development evaluation |
|---|---:|---:|
| Correct-instruction grounding | 40/40 | 40/40 |
| T0 local non-tie | 5.55% | 5.14% |
| T1 local non-tie | 58.31% | 57.91% |
| T1 usable tasks | 10/10 | 10/10 |
| Progress | 44.05% | 44.95% |
| Regression | 14.26% | 12.97% |
| Tie | 41.69% | 42.09% |
| Time agreement among non-ties | 75.55% | 77.61% |
| Semantic-state language change | 51.92% | 53.29% |
| Pair-preference language change | 28.41% | 29.59% |
| Material-change tasks | 10/10 | 10/10 |
| Proprio-only accuracy | 78.54% | 80.01% |

T0 remains too sparse and has zero usable tasks, demonstrating why official
success alone is not an adequate local teacher.

## Per-gap evidence

| Gap | Rows | Non-tie | Progress | Regression | Tie | Time agreement |
|---|---:|---:|---:|---:|---:|---:|
| 1--2 | 9,646 | 53.46% | 41.16% | 12.31% | 46.54% | 76.98% |
| 3--4 | 9,486 | 57.03% | 43.71% | 13.32% | 42.97% | 76.64% |
| 5--8 | 18,492 | 60.69% | 47.56% | 13.12% | 39.31% | 78.37% |
| 9--16 | 35,064 | 65.63% | 54.92% | 10.71% | 34.37% | 83.69% |

The primary local result is not carried only by later gaps: even gaps 1--2
have `53.46%` non-ties and `12.31%` regressions.

## Per-task held-out evidence

| Task | Rows | Non-tie | Progress | Regression | Language state change |
|---:|---:|---:|---:|---:|---:|
| 0 | 3,592 | 53.76% | 40.03% | 13.72% | 25.7% |
| 1 | 3,984 | 54.47% | 38.45% | 16.01% | 70.4% |
| 2 | 3,688 | 59.63% | 43.79% | 15.84% | 47.0% |
| 3 | 2,880 | 49.34% | 41.70% | 7.64% | 46.2% |
| 4 | 4,736 | 62.77% | 48.82% | 13.96% | 33.3% |
| 5 | 3,024 | 67.76% | 52.71% | 15.05% | 62.5% |
| 6 | 3,880 | 53.40% | 41.13% | 12.27% | 66.7% |
| 7 | 4,152 | 65.05% | 53.52% | 11.54% | 60.0% |
| 8 | 3,528 | 49.09% | 33.62% | 15.48% | 63.9% |
| 9 | 4,160 | 61.08% | 53.25% | 7.84% | 60.0% |

All tasks clear both the 30% density and 20% language-change floors.

## Robot controls

Calibration freezes proprio distance thresholds at the 10th and 75th
percentiles. Held out, there are 3,691 semantic-tie/strong-robot-difference
pairs across 10 tasks and 2,063 robot-matched/semantic-non-tie pairs across 10
tasks. By explicit construction their teacher tie and non-tie rates are 100%.
Together with the `80.01%` low-capacity proprio probe, this supports semantic
information beyond raw robot phase, while not claiming complete statistical
independence from demonstration phase.

## Mechanical decision

| Gate | Held-out result |
|---|---|
| Robust same-scene grounding | pass: 40/40 correct, 10/10 tasks |
| Goal margin physically validated | pass: official inequalities, AUC 0.9533 calibration |
| Local non-tie >=30% | pass: 57.91% |
| >=7 usable tasks | pass: 10/10 |
| Material language dependence | pass: 53.29%, 10/10 tasks |
| Counterfactual set in >=7 tasks | pass: 10/10 |
| Semantic-tie / robot-different control | pass: 3,691 rows, 10 tasks |
| Robot-match / semantic-different control | pass: 2,063 rows, 10 tasks |
| Proprio-only below 90% | pass: 80.01% |
| Finite and forbidden access zero | pass |

Final authorization:

```text
GO_VALUE_DISTILLATION
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

Value distillation is not executed in this task.

## Required answers

**TQ1 — What is the final continuous margin?** M2, the minimum normalized
horizontal-clearance and vertical-order clearance shown above.

**TQ2 — How does it relate to `check_ontop`?** It exactly mirrors the two
numeric inequalities. Official contact stays in the preceding Boolean and is
not replaced by the margin.

**TQ3 — Is H<=8 signal dense enough?** Yes: `57.91%` held-out non-ties and
10/10 usable tasks.

**TQ4 — Do ties and regressions remain common?** Yes: `42.09%` ties and
`12.97%` regressions locally.

**TQ5 — Is it merely later-is-better?** No. Time agreement is `77.61%`, not
deterministic, and explicit regressions plus ties are substantial.

**TQ6 — Does same-state language change semantics?** Yes. It changes the
persistent referent/semantic state in `53.29%` and local preference in `29.59%`.

**TQ7 — Do robot-different semantic equivalents remain tied?** Yes: 3,691
held-out rows across all ten tasks satisfy this control.

**TQ8 — Do robot-matched semantic differences remain non-ties?** Yes: 2,063
held-out rows across all ten tasks satisfy the opposite control.

**TQ9 — What does proprio-only achieve?** `80.01%`, below the frozen 90%
severe-shortcut threshold.

**TQ10 — Are the final label and GO legal?** Yes. All Stage G and Stage T
gates pass, so `SEMANTIC_GOAL_PROGRESS_TEACHER_VALID` and
`GO_VALUE_DISTILLATION` are assigned; no distillation is run.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| Protocol | `69238119e8302f7c1063a8c5588a5f9ad09755a80ed7c7622fd74c1168d1dbb0` |
| Frozen teacher definition | `249f423f5e7be05ea16bac6498dd01a486493cc036dd097ad4f3d5919a99ef09` |
| Calibration result | `4dafeec3be6a09f28c1e150192803a163c98da9b13fd91d0334f0e7ae609bba3` |
| Evaluation result | `a22a4b3a1f6be65bb94b814aabfd65cb9aaf72c6cbcc87c976bf75bfe22af3ab` |
| Calibration sidecar | `cf72c30686b14eff5015a0a2a06f9b3c6e70812586a65b0fa7a9fea9026c0fe5` |
| Evaluation sidecar | `d7a7068fe9d114eba28359c750217723efb85ed179a6f04b86b57b6c5ff4f833` |

## Forbidden-activity audit

No demo40--49 numeric data or metadata, historical demo30/35, reward, done,
stored success, task outcome, RGB, action, value distillation, V2 training,
candidate future, World Model, learned proposal, reranker, score, controller,
CEM, MPPI, beam search, C1/C2, D20-v2, or seed23/47 experiment was accessed or
run. Earlier blocked artifacts remain unchanged.
