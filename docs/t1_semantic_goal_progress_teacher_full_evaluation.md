# T1 is a valid language-conditioned semantic goal-progress teacher

Date: 2026-10-02

Final mechanism labels:

```text
ROBUST_SAME_SCENE_GROUNDING_VALID
ROBUST_REFERENT_MARGIN_VALID
COUNTERFACTUAL_GROUNDING_SET_VALID
OFFICIAL_GOAL_MARGIN_VALID
SEMANTIC_LOCAL_PROGRESS_SIGNAL_VALID
SEMANTIC_LANGUAGE_COUNTERFACTUAL_VALID
SEMANTIC_ROBOT_PHASE_INVARIANCE_SUPPORTED
SEMANTIC_GOAL_PROGRESS_TEACHER_VALID
```

Next-stage authorization: `GO_VALUE_DISTILLATION`

Continuing authorizations:

```text
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

Selected privileged teacher: `T1-M2`

Value distillation executed: `false`

Controller authorization: `false`

## Executive answer

The independently preregistered full evaluation validates T1 as a privileged
semantic goal-progress teacher. Frozen G2 resolves the instruction-selected
black bowl once in the initial scene and preserves that physical identity.
T1 then compares states lexicographically: exact official
`On(referent, plate)` first and M2 continuous goal-relation margin second.
M2 is the minimum normalized signed slack of the two numeric inequalities in
the installed `check_ontop`; contact remains exclusively in the official
Boolean. No time, action, reward, success log, robot state, or grounding score
enters the teacher.

On frozen development demos36--39, T1 has `57.91%` local `Delta t<=8`
non-ties (task-stratified demo-block 95% CI `[56.80%,59.04%]`) and all ten
tasks exceed the frozen `30%` usability floor. Local pairs contain `44.95%`
progress, `12.97%` regression, and `42.09%` ties. Later time agrees with the
teacher on `77.61%` of non-ties, so the signal is related to demonstration
progress but is not a deterministic clock. The T0 official-Boolean baseline
has only `5.14%` local non-ties and zero usable tasks.

Language changes the same physical state materially. Across `32,757` eligible
state counterfactuals, G2 chooses a different persistent bowl in `53.29%` and
the T1 semantic state changes in exactly those cases. Across `252,300`
eligible local pair counterfactuals, preference changes in `29.59%` (CI
`[28.47%,30.62%]`) and 9/10 source tasks reach the frozen `20%` threshold.
Full better/worse reversal is rarer at `1.29%` (CI `[0.97%,1.63%]`); most
language effects turn an original non-tie into a tie. This still satisfies the
preregistered material-preference-change definition, and the taxonomy is
reported without upgrading ties into reversals.

Both robot controls pass across all ten tasks. There are `3,691` held-out
semantic-tie pairs with large robot differences and `2,063` robot-matched
pairs with clear semantic differences. The calibration-fitted ridge-linear
proprio-only probe reaches `80.01%` accuracy (CI `[78.89%,81.19%]`, AUROC
`0.832`), below the frozen `90%` severe-shortcut threshold and `16.21`
percentage points below the historical temporal-supervision result of
`96.224%`. This supports semantic information beyond raw robot phase; it does
not claim statistical independence from robot motion.

All eleven final gates pass. The access ledgers contain only allowlisted
`states` datasets from demos31--34 and demos36--39; demo40--49 received zero
numeric and zero metadata access. `GO_VALUE_DISTILLATION` is therefore
authorized, but no value model is trained here. World Model, candidate-value,
reranking, controller, and task-success work remain blocked.

## 1. Frozen G2 grounding provenance

Stage G is frozen at commit `d070f2382f7df15698f747c90813306f1db3c47c`
under annotated tag `robust-same-scene-grounding-final`. Its resolver SHA-256
is `337ebe063792402369e779bcec4800cb0ce542ea10fd8992eb8f66796dd56881`.
The current protocol verifies that hash before either partition is collected.

G2, its normalized source geometry, strict `second_minus_first > 0` confidence
rule, two-bowl ordering, eligibility rule, and persistent-referent contract are
unchanged. Correct-instruction grounding remains 40/40 on calibration and
40/40 on frozen development, with 100% precision, coverage, task coverage,
and agreement on the exact-predicate-accepted subset.

This independent task began at annotated tag
`t1-semantic-teacher-full-evaluation-start` (`0cf3d26`). The protocol and
implementation were committed at `c8beb45` before new calibration collection;
the M2/T1 calibration definition was frozen at `33a0f73` before demos36--39
were opened. The one-shot numerical evaluation was frozen at `0fe78e0`.

## 2. Data boundary

| Partition | Demo IDs | Episodes | Frames | Pair rows | Use |
|---|---|---:|---:|---:|---|
| Calibration | 31--34 | 40 | 5,044 | 75,264 | margin selection, epsilon, T1/T2 choice, controls, ridge fit |
| Frozen development evaluation | 36--39 | 40 | 4,883 | 72,688 | one-shot gates and diagnosis; not blind confirmation |
| Historical only | 30/35 | 0 | 0 | 0 | not opened |
| Forbidden | 40--49 | 0 | 0 | 0 | no numeric or metadata access |

Only exact allowlisted `states` arrays were opened. No RGB, action, reward,
done, stored success, or final task-outcome field was read.

## 3. Official `On` implementation

The installed predicate is frozen from source, not reconstructed from memory:

```text
On(subject, support) -> support.check_ontop(subject)

support.check_ontop(subject) is true iff:
  support_center_z <= subject_center_z
  AND support.check_contact(subject)
  AND ||support_center_xy - subject_center_xy||_2 < 0.03 m
```

The source files have SHA-256 values
`319137e95330dbb22a266a8dceb8110f18413c2bf17461550222228c4bc310a1`
for the predicate dispatch and
`ca6f53123336b6487bdb4feb02534ddb3a0627a4b847fe89da4460520ef01366`
for object state logic. Contact is a simulator geometry-contact test. The two
remaining checks use body centers; they do not use oriented bounding boxes or
scanned-object surface offsets.

## 4. Limited margin candidate family

The preregistered family contains exactly three candidates:

| Family | Definition | Role |
|---|---|---|
| M0 | minimum of horizontal slack, z-order slack, and unrotated surface-gap penalty | historical reference |
| M1 | M0 with `body_xmat @ model_offset` world bottom/top surfaces | orientation-aware surface surrogate |
| M2 | minimum normalized horizontal and z-order slack | direct numeric `On` surrogate |

Margin selection used only official-geometry consistency, official-positive
versus negative separation, finite stability, task robustness, local numeric
smoothness, interpretability, and simplicity. Temporal ordering, non-tie
density, language effects, the proprio probe, and downstream model behavior
were unavailable to the selection.

## 5. Orientation-aware geometry audit

M0's raw z offsets are not generally rotation-correct. Replacing them by
orientation-aware world offsets changes the absolute surface gap on calibration
by mean `0.540 mm`, p90 `1.961 mm`, and maximum `11.361 mm`; held out the
corresponding values are `0.502 mm`, `1.857 mm`, and `11.591 mm`. M1 therefore
implements the required corrected surface calculation.

The audit does not force a surface term into M2. The installed numeric
`check_ontop` inequalities use body centers, so orientation and model offsets
are mathematically absent from those two slacks. Contact, which does depend on
actual geometry, remains the official Boolean coordinate. This separation
avoids pretending that an oriented surface heuristic is part of the installed
numeric acceptance rule.

## 6. Selected margin M2

Calibration mechanically selects M2:

```text
horizontal_slack = (0.03 - ||bowl_xy - plate_xy||_2) / 0.03
vertical_slack   = (bowl_z - plate_z) / 0.03
M2               = min(horizontal_slack, vertical_slack)
```

M2 is a signed minimum slack to the numeric portion of the official `On`
acceptance region. It is not a continuous replacement for contact or official
success. A state can have positive M2 and remain official-negative because
contact is false; those cases are expected and are why the official Boolean is
the first T1 coordinate.

| Margin | Calibration AUC | Held-out AUC | Cal p99 adjacent jump | Held-out p99 jump |
|---|---:|---:|---:|---:|
| M0 | 0.8846 | 0.8925 | 0.4050 | 0.4000 |
| M1 | 0.8844 | 0.8929 | 0.4060 | 0.3995 |
| **M2** | **0.9533** | **0.9621** | **0.3886** | **0.3897** |

All candidates are finite and admissible, but M2 is the only direct expression
of the installed numeric tests and also has the strongest separation.

## 7. Goal-positive and goal-negative distributions

| Partition / official state | n | Mean | Median | p10 | p90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibration positive | 602 | 0.200 | 0.176 | 0.123 | 0.330 | 0.0068 | 0.529 |
| Calibration negative | 9,486 | -8.647 | -8.246 | -15.359 | -1.516 | -16.587 | 0.980 |
| Held-out positive | 583 | 0.188 | 0.161 | 0.125 | 0.308 | 0.0156 | 0.479 |
| Held-out negative | 9,183 | -8.746 | -8.341 | -15.042 | -1.908 | -16.285 | 0.966 |

Calibration and held-out median separations are `8.422` and `8.502`; overlap
coefficients are `0.0463` and `0.0411`. Every task has held-out AUC above
`0.933`. Goal-positive states are therefore systematically higher, while the
positive tail among official negatives correctly exposes contact-only failures.

## 8. Tie-threshold calibration

Only after M2 was frozen, calibration set:

```text
epsilon_goal = max(
    1e-4,
    10th percentile of positive adjacent-frame |Delta M2| above 1e-4
) = 0.004065508647683736
```

The rule does not maximize temporal agreement or non-tie density. If official
goal Booleans match and `|M2_b-M2_a| <= epsilon_goal`, T1 returns `TIE`.

## 9. T0 binary baseline

| Partition | Local rows | Non-tie | 95% CI | Progress | Regression | Tie | Tasks >=30% |
|---|---:|---:|---|---:|---:|---:|---:|
| Calibration | 38,912 | 5.55% | [4.84%,6.28%] | 4.59% | 0.95% | 94.45% | 0/10 |
| Held out | 37,624 | 5.14% | [4.69%,5.61%] | 4.46% | 0.68% | 94.86% | 0/10 |

Held-out task-level T0 non-tie density ranges from `3.46%` to `7.80%`.
Official success alone is therefore far too sparse for the local teacher gate.

## 10. T1 definition

For language `l`, G2 grounds one physical referent in the initial scene and
retains it. At state `s`:

```text
T1(s,l) = [official_On(persistent_referent, plate), M2(s,l)]
```

Pair preference is lexicographic. A true official goal beats a false one;
otherwise M2 determines better/worse outside the frozen epsilon and returns a
tie within it. There is no weighted sum and no robot, time, action, reward,
success log, or G2 confidence term.

## 11. T1 local-density calibration

On calibration, local `Delta t<=8` density is `58.31%`, CI
`[56.97%,59.63%]`. Progress, regression, and ties are `44.05%`, `14.26%`, and
`41.69%`. All ten tasks individually exceed `30%`. T1 therefore passes before
the frozen development partition is opened.

## 12. T1/T2 mechanical selection

The preregistered conditional T2 could activate only if calibration T1 had
less than `30%` local non-ties or fewer than seven usable tasks. T1 reaches
`58.31%` and 10/10 tasks. T2 is therefore not activated, and no grasp or reach
stage enters the selected teacher.

## 13. Calibration freeze

Commit `33a0f73` freezes the G2 hash, persistent identity rule, M2 equation,
`epsilon_goal`, T1 selection, counterfactual rule, robot-control thresholds,
ridge probe weights, all gates, bootstrap seed, and 10,000-draw shape. Only
after that commit was demo36--39 opened.

The bootstrap unit is the complete demo block within each task. Each draw
resamples the four admitted demos independently inside all ten tasks, retaining
the correlated state pairs and counterfactuals from each demo.

## 14. Frozen development evaluation

The held-out run applies frozen M2, epsilon, T1, controls, and ridge weights
once. It does not reselect a margin, drop a task, retune task usability, alter
counterfactual eligibility, or activate T2. Its complete gate is `true`, all
numeric sidecar arrays are finite, and the numerical artifacts are frozen at
commit `0fe78e0`.

## 15. Per-gap density

Held-out results:

| Gap | Rows | Non-tie | 95% CI | Progress | Regression | Tie | Time agreement |
|---|---:|---:|---|---:|---:|---:|---:|
| 1--2 | 9,646 | 53.46% | [52.38%,54.55%] | 41.16% | 12.31% | 46.54% | 76.98% |
| 3--4 | 9,486 | 57.03% | [55.90%,58.17%] | 43.71% | 13.32% | 42.97% | 76.64% |
| 5--8 | 18,492 | 60.69% | [59.51%,61.87%] | 47.56% | 13.12% | 39.31% | 78.37% |
| 9--16 | 35,064 | 65.63% | [64.34%,66.91%] | 54.92% | 10.71% | 34.37% | 83.69% |

The signal is already dense at gap 1--2 and is not created solely by long
temporal separation.

## 16. Progress, regression, and ties

For the primary local set, held-out T1 is `44.95%` progress (CI
`[43.80%,46.13%]`), `12.97%` regression (CI `[11.70%,14.17%]`), and `42.09%`
ties (CI `[40.96%,43.20%]`). These outcomes are not repaired or forced toward
later-is-better. Regression is a first-class semantic label; ties remain
substantial.

## 17. Per-task density

| Task | Local rows | T0 non-tie | T1 non-tie | Progress | Regression | Time agreement |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,592 | 7.80% | 53.76% | 40.03% | 13.72% | 74.47% |
| 1 | 3,984 | 6.28% | 54.47% | 38.45% | 16.01% | 70.60% |
| 2 | 3,688 | 5.42% | 59.63% | 43.79% | 15.84% | 73.44% |
| 3 | 2,880 | 5.31% | 49.34% | 41.70% | 7.64% | 84.52% |
| 4 | 4,736 | 4.60% | 62.77% | 48.82% | 13.96% | 77.77% |
| 5 | 3,024 | 4.76% | 67.76% | 52.71% | 15.05% | 77.79% |
| 6 | 3,880 | 5.62% | 53.40% | 41.13% | 12.27% | 77.03% |
| 7 | 4,152 | 3.81% | 65.05% | 53.52% | 11.54% | 82.27% |
| 8 | 3,528 | 4.79% | 49.09% | 33.62% | 15.48% | 68.48% |
| 9 | 4,160 | 3.46% | 61.08% | 53.25% | 7.84% | 87.17% |

The weakest T1 task still reaches `49.09%`, comfortably above the required
`30%` floor.

## 18. Teacher/time agreement

Later time agrees with held-out T1 on `77.61%` of local non-ties; calibration
agreement is `75.55%`. The relationship strengthens with gap but is never
defined by time. The `42.09%` tie rate and `12.97%` regression rate provide
direct counterexamples to an automatic later-is-better rule.

## 19. Same-state language counterfactuals

Each other real task instruction is grounded once by frozen G2 in the exact
same initial physical scene, and its chosen bowl identity is preserved through
the trajectory. Only G2 `UNIQUE` alternatives are admitted.

| Metric | Calibration | Held out |
|---|---:|---:|
| Eligible state cases | 33,765 | 32,757 |
| Referent switch | 51.92% | 53.29% |
| Official goal-Boolean change | 6.25% | 6.39% |
| T1 semantic-state change | 51.92% | 53.29% |
| Median absolute M2 change | 0.418 | 0.525 |
| p90 absolute M2 change | 9.628 | 10.036 |

Because the physical state and robot configuration are identical, these are
direct language-conditioned changes rather than time or robot-motion effects.

## 20. Pair-preference language flips

| Metric | Calibration | Held out |
|---|---:|---:|
| Eligible local pair counterfactuals | 260,400 | 252,300 |
| Material preference change | 28.41% | 29.59% |
| 95% CI | [27.18%,29.64%] | [28.47%,30.62%] |
| Tasks >=20% | 7/10 | 9/10 |
| Full better/worse reversal | 1.36% | 1.29% |
| Reversal 95% CI | [1.02%,1.70%] | [0.97%,1.63%] |

Held-out taxonomy over all 252,300 eligible pairs:

| Transition | Count | Rate |
|---|---:|---:|
| Better -> worse | 2,176 | 0.862% |
| Worse -> better | 1,082 | 0.429% |
| Non-tie -> tie | 71,083 | 28.174% |
| Tie -> non-tie | 304 | 0.120% |
| Unchanged | 177,655 | 70.414% |

The language gate passes on material preference change, not on reversal alone.
The stronger reversal evidence is positive but modest and is not conflated
with the larger non-tie-to-tie effect.

## 21. Referent-switch stratification

Held out, `15,302` same-referent language applications have `0%` semantic
state change, while all `17,455` different-referent applications change the
semantic state (`100%`). Calibration shows the same `0%/100%` separation.
Thus the same-state effect is entirely mediated by which physical bowl the
real instruction selects, rather than by injecting arbitrary language tokens
into a fixed object value.

Held-out pair-change rates by source task are `11.79%`, `36.75%`, `27.95%`,
`22.23%`, `20.19%`, `41.23%`, `34.31%`, `38.06%`, `29.55%`, and `36.33%` for
tasks0--9. Nine tasks clear `20%`; task0 does not. The frozen requirement is at
least seven.

## 22. Semantic-match / robot-difference control

Calibration freezes a large-robot-difference threshold at the 75th percentile
of standardized 12-D proprio pair distance (`1.0219`). Held out, `3,691`
same-task, same-language pairs across 10/10 tasks exceed it while T1 remains a
tie. Their proprio distance has median `1.236`, p90 `1.688`, and maximum
`2.824`; their semantic-margin absolute difference has median approximately
zero and p90 `0.000616`, below `epsilon_goal`.

## 23. Robot-match / semantic-difference control

The calibration-frozen robot-match threshold is the 10th percentile,
`0.17597`. Held out, `2,063` pairs across 10/10 tasks fall below it while T1 is
a clear non-tie. Their proprio distance median is `0.138`; semantic-margin
separation has median `0.104`, p90 `0.347`, and maximum `0.470`.

Together Sections 22 and 23 show that T1 is not a restatement of raw robot
configuration. They do not assert that robot and semantic state are
statistically independent along expert demonstrations.

## 24. Proprio-only shortcut probe

The only learned diagnostic is the preregistered ridge-linear probe over
`[p_a,p_b,p_b-p_a]`, trained on calibration non-tie local pairs with ridge
`0.01`. It is not part of T1.

| Task | Held-out rows | Accuracy |
|---:|---:|---:|
| 0 | 1,931 | 74.78% |
| 1 | 2,170 | 69.82% |
| 2 | 2,199 | 78.17% |
| 3 | 1,421 | 80.65% |
| 4 | 2,973 | 85.54% |
| 5 | 2,049 | 82.92% |
| 6 | 2,072 | 83.88% |
| 7 | 2,701 | 86.52% |
| 8 | 1,732 | 66.17% |
| 9 | 2,541 | 84.49% |

Aggregate held-out accuracy is `80.01%`, CI `[78.89%,81.19%]`, on `21,789`
rows; AUROC is `0.8317`. Its upper CI remains far below the `90%` severe
shortcut boundary.

## 25. Comparison with the old temporal shortcut

The frozen historical temporal-progress proprio+language result is `96.224%`.
The T1 proprio-only probe is `16.21` percentage points lower held out
(`17.68` points lower on calibration). Semantic supervision therefore
substantially reduces the previous robot-phase shortcut. Residual predictability
at `80.01%` remains real, so the conclusion is reduced domination rather than
eliminated correlation.

## 26. Mechanical gates

| Required gate | Frozen development result |
|---|---|
| G2 grounding hash unchanged | pass |
| Goal-margin geometry audit | pass: M2, held-out AUC 0.9621 |
| All teacher values finite | pass |
| Local non-tie >=30% | pass: 57.91%, CI lower 56.80% |
| At least 7 usable tasks | pass: 10/10 |
| Language pair change >=20% | pass: 29.59%, CI lower 28.47% |
| Language effect in at least 7 tasks | pass: 9/10 |
| Semantic-match/robot-difference control | pass: 3,691 rows, 10 tasks |
| Robot-match/semantic-difference control | pass: 2,063 rows, 10 tasks |
| Proprio-only accuracy <90% | pass: 80.01%, CI upper 81.19% |
| Forbidden access count zero | pass |

All gates pass simultaneously.

## 27. Final labels

The frozen decision mapping assigns:

```text
ROBUST_SAME_SCENE_GROUNDING_VALID
ROBUST_REFERENT_MARGIN_VALID
COUNTERFACTUAL_GROUNDING_SET_VALID
OFFICIAL_GOAL_MARGIN_VALID
SEMANTIC_LOCAL_PROGRESS_SIGNAL_VALID
SEMANTIC_LANGUAGE_COUNTERFACTUAL_VALID
SEMANTIC_ROBOT_PHASE_INVARIANCE_SUPPORTED
SEMANTIC_GOAL_PROGRESS_TEACHER_VALID
```

`SEMANTIC_ROBOT_PHASE_INVARIANCE_SUPPORTED` means the two cross-controls pass
and the low-capacity proprio probe stays below the preregistered severe line.
It is not a claim of zero robot information.

## 28. Authorization

The only new authorization is:

```text
GO_VALUE_DISTILLATION
```

This permits a separately preregistered attempt to distill the frozen
privileged T1-M2 teacher into an allowed natural-RGB-plus-language value model.
It does not select a student architecture, train V1/V2, open candidate futures,
run a World Model, construct a reranker, or authorize control. The following
remain in force:

```text
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
controller authorization = false
```

## 29. Artifacts, hashes, and automated verification

| Artifact | SHA-256 |
|---|---|
| Protocol | `c47748302715abdad675341ba08da3d2345d0c2ef9e08029466c4b9daf505261` |
| Official margin contract | `82eef389548472471e4ad00e2d34aad67650d06ec9eb632f46a4df0f891185b7` |
| Frozen T1 definition | `9be0804395af0aae0f5bea3305f95b5121642311f18e9f58277493775512e31f` |
| Calibration result | `432d414f831632c8f5aefaaab5860a8f4a53d62f4da7b6eaac03b3e890e22f24` |
| Evaluation result | `063037d1dc8361485ff394997f31a545cddffb438cd871f6a0a608a7ae351ffa` |
| Language counterfactual result | `a64d430b731f0c56f34250c69f05bd66462ba5feb91fb2014b98a45124aa1473` |
| Robot-control result | `73e7cba2d25fd6e511a457f78b547710be577e1bf69f623cb5f48ffdec44b61c` |
| Calibration sidecar | `fbfd700d8972f6be8ed1f43455784b9098a7aebd07b0cba49bc108651f963604` |
| Evaluation sidecar | `cd6ce906f83753561f7464f4fb58fadf02576e891dd70ad2aee403197a39b06a` |
| Calibration access ledger | `39b0f612af4f35e8bd5de4526953d9a7800e41c05f0e46ad878481e93cd1768e` |
| Evaluation access ledger | `07caed1db60b0d015817ec72ae305bd12819779c85a33a77caeb8ea41589442d` |
| Probe implementation | `5a9ef3a2bfe374423d886d9891d7e0d643000396e11247c1510c67fef07cd780` |
| Audit runner | `ab60e7767b8b6a13b981a7d0e566e7f2260a828ca8d051a65c9ef975c175e6e2` |
| Final decision | `1df89db9e7cd5b60805cf321fdb5e90eb45c88eb5f7487deb8662cd734d3dbb0` |

Sixteen focused independent protocol/calibration/evaluation tests pass. The
complete repository suite passes `278` tests; the remaining 21 messages are
dependency deprecation warnings, not failures. Historical manifests whose
required output paths were reused by this explicitly independent study are
verified against their immutable annotated-tag Git objects.

## 30. Access and forbidden-activity audit

The calibration ledger contains only task0--9 demo31--34 `states`; the
evaluation ledger contains only task0--9 demo36--39 `states`. Both record zero
metadata operations, zero forbidden attempts, and zero forbidden access.
Demo30/35 and demo40--49 were not opened.

No RGB, action, reward, done, stored success, final task outcome, semantic
student, V1/V2 value model, CLIP evaluator, candidate future, candidate
reranker, World Model, learned proposal, score, controller, CEM, MPPI, beam
search, C1/C2, D20-v2, seed23/47, or task-success experiment was accessed or
run. The frozen G2 equations and artifacts were not modified.

## Required answers

**Q1 — What is the final continuous `On` margin?**

M2: the minimum of normalized horizontal-distance clearance and vertical-order
clearance, using the exact `0.03 m` scale from installed `check_ontop`.

**Q2 — How is it derived from official `check_ontop` geometry?**

It is the signed minimum slack of the two installed numeric inequalities.
Contact remains the official Boolean first coordinate and is not approximated.

**Q3 — Does it correctly handle bowl/plate orientation and geometry offsets?**

Yes in the precise sense required by the source. M1 audits rotated model
offsets and shows that unrotated surfaces can differ by up to `11.59 mm`.
Final M2 does not use those offsets because the installed numeric checks use
body centers; geometry-dependent contact stays in the Boolean.

**Q4 — Do official goal-positive states receive systematically better margins?**

Yes. Held-out positive/negative medians are `0.161/-8.341`, goal AUC is
`0.9621`, and every task has AUC above `0.933`.

**Q5 — How sparse is the T0 binary goal locally?**

Held-out local non-ties are `5.14%`, CI `[4.69%,5.61%]`, and no task reaches
the `30%` usability floor.

**Q6 — What is T1 local non-tie density at `Delta t<=8`?**

`57.91%`, CI `[56.80%,59.04%]`.

**Q7 — How much local signal does each task have?**

Tasks0--9 have `53.76, 54.47, 59.63, 49.34, 62.77, 67.76, 53.40, 65.05,
49.09, 61.08%` non-ties. All ten pass.

**Q8 — Does T1 allow substantial ties and regressions?**

Yes. Held-out local pairs are `42.09%` ties and `12.97%` regressions.

**Q9 — What is agreement with later-is-better?**

`77.61%` among held-out local non-ties. It is a correlation, not the label
definition, and is far from deterministic.

**Q10 — How often does the semantic state change under a real same-state counterfactual instruction?**

`53.29%` of `32,757` eligible held-out cases.

**Q11 — How often does pairwise preference change?**

`29.59%`, CI `[28.47%,30.62%]`, across `252,300` eligible local
counterfactuals; 9/10 source tasks exceed `20%`.

**Q12 — How much is true preference reversal rather than tie/non-tie change?**

`1.29%`, CI `[0.97%,1.63%]`. Most material changes are non-tie to tie and are
reported separately.

**Q13 — Is the different-language same-state effect mainly due to a different physical referent?**

Yes. Same-referent cases change `0%`; different-referent cases change `100%`.

**Q14 — Does the semantic-match/robot-difference control hold?**

Yes: `3,691` held-out pairs across all ten tasks remain semantic ties despite
large proprio differences.

**Q15 — Does the robot-match/semantic-difference control hold?**

Yes: `2,063` held-out pairs across all ten tasks are clear semantic non-ties
despite similar proprioception.

**Q16 — What is proprio-only probe accuracy?**

`80.01%`, CI `[78.89%,81.19%]`, with AUROC `0.8317`.

**Q17 — Is the robot-phase shortcut substantially lower than the old temporal result?**

Yes. It is `16.21` percentage points below `96.224%` held out and remains
below the severe `90%` line.

**Q18 — Is T2 needed?**

No. T1 already passed calibration density at `58.31%` on 10/10 tasks, so the
conditional activation rule prohibited T2.

**Q19 — Does the teacher satisfy language-conditioned, locally dense, non-temporal, and non-proprio-dominated requirements?**

Yes under the frozen operational gates: material same-state language effects,
dense short-gap signal, substantial ties/regressions, both robot controls, and
a proprio probe below `90%`. “Non-temporal” and “invariance supported” do not
mean zero correlation; they mean time and raw proprio do not define the
teacher or cross the preregistered domination tests.

**Q20 — May the project assign the final valid-teacher label and authorize distillation?**

Yes:

```text
SEMANTIC_GOAL_PROGRESS_TEACHER_VALID
GO_VALUE_DISTILLATION
```

The authorization stops here. No value distillation is executed in this task.
