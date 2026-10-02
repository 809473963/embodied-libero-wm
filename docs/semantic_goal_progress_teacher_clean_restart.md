# Exact same-scene grounding is too brittle to authorize a semantic progress teacher

Date: 2026-10-02

Final mechanism labels:

```text
SEMANTIC_GOAL_PROGRESS_TEACHER_UNRESOLVED
```

Next-stage authorizations:

```text
BLOCK_VALUE_DISTILLATION
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

Selected semantic teacher: `none`

Controller authorization: `false`

## Executive answer

The clean restart succeeds as an integrity-controlled experiment but stops at
the calibration grounding gate. The exact-path access guard passes, its union
ledger contains only demo31--34 `states` arrays, and demo40--49 have zero
numeric and zero metadata access. The frozen development partition demo36--39
was never opened.

The preregistered grounding function evaluated each real instruction's BDDL
source predicate against both physical black bowls in the restored episode
initial state. It returned a unique correct referent in only `23/40` calibration
episodes (`57.5%`). The other `17` returned `NOT_FOUND`; none was ambiguous.
Strict source sites are only about one centimeter wide, and restored bowls can
sit a few millimeters beyond them after initialization. Object-support cases
can also lose the exact contact required by `On`. For example, task1's bowl is
visually and geometrically nearest the next-to-ramekin site but lies roughly
`3 mm` beyond its x boundary, so the exact site predicate rejects both bowls.

The available subset contains promising but non-authorizing evidence. The T1
calibration candidate has local `Delta t<=8` non-tie density `59.49%`, meets the
density rule on exactly 7/10 tasks, includes both progress (`41.55%`) and
regression (`17.94%`), and its proprio-only linear probe reaches `75.97%`, below
the severe `90%` shortcut line. Among eligible real-instruction
counterfactuals, referents switch in `100%`, semantic state changes in `100%`,
and local pair preference changes in `57.57%`.

Those subset results cannot repair missing correct-language grounding. A
post-calibration replacement such as nearest-region grounding would change the
frozen semantic contract after seeing data. The decision tree therefore stops
before evaluation and assigns `SEMANTIC_GOAL_PROGRESS_TEACHER_UNRESOLVED`.
No value distillation, candidate bridge, World Model, reranker, or controller
was run.

## Prior blocked provenance and clean restart boundary

The prior blocked audit remains frozen at commit `121ec18` and annotated tag
`semantic-goal-progress-teacher-blocked`. Its boundary incident and decision
amendment were not edited. A new annotated start tag,
`semantic-goal-progress-teacher-clean-restart`, points to that preserved
boundary. The exact-path guard was committed at `981ddd9`; the new semantic
protocol was preregistered at `75b6ca3`.

The first calibration preflight exposed only a predicate-dispatch case mismatch:
the source contract preserved `On/In`, while the installed registry accepts
lowercase keys. It stopped after one allowed task0/demo31 state array, wrote no
result, and was corrected at `f863bfa` by lowercasing only the dispatch key.
No semantic rule, threshold, gate, or partition changed.

## Hard access guard and tests

`AllowedDemoGuard` validates task, partition, demo ID, and dataset before it
opens an HDF5 file. Its public API does not expose files or groups. Calls that
attempt group enumeration, `keys`, `visit`, `visititems`, recursive walks, or
non-allowlisted existence/metadata/numeric access raise immediately.

Fourteen synthetic/source tests passed before the complete calibration run:
nine access tests and five semantic-protocol tests. They cover allowed exact
access; forbidden numeric, metadata, existence, and enumeration calls;
partition disjointness; exact reconstruction of all ten BDDL source contracts;
and T1/T2 lexicographic behavior.

The union access audit records three calibration-only invocations: the one-row
case preflight, the complete calibration collection, and a calibration-only
grounding diagnosis. Their union is still exactly all task0--9 demo31--34
`states` datasets. No HDF5 metadata operation was used outside those exact
paths.

## Exact task semantics

| Task | Instruction source | Frozen source atom | Final goal |
|---:|---|---|---|
| 0 | between plate and ramekin | `On(bowl, between_region)` | `On(bowl, plate)` |
| 1 | next to ramekin | `On(bowl, next_to_ramekin_region)` | `On(bowl, plate)` |
| 2 | table center | `On(bowl, table_center)` | `On(bowl, plate)` |
| 3 | on cookie box | `On(bowl, cookies)` | `On(bowl, plate)` |
| 4 | cabinet top drawer | `In(bowl, cabinet_top_region)` | `On(bowl, plate)` |
| 5 | on ramekin | `On(bowl, ramekin)` | `On(bowl, plate)` |
| 6 | next to cookie box | `On(bowl, next_to_box_region)` | `On(bowl, plate)` |
| 7 | on stove | `On(bowl, stove_cook_region)` | `On(bowl, plate)` |
| 8 | next to plate | `On(bowl, next_to_plate_region)` | `On(bowl, plate)` |
| 9 | on wooden cabinet | `On(bowl, cabinet_top_side)` | `On(bowl, plate)` |

The installed terminal predicate remains exact: `On(subject, plate)` calls
`plate.check_ontop(subject)`, which requires contact, subject center z not below
plate center z, and horizontal center distance below `0.03 m`.

## Grounding function and persistent referent contract

For instruction contract `l` and restored initial physical scene `s`, the
frozen function substitutes each of `akita_black_bowl_1` and
`akita_black_bowl_2` into `l`'s source atom and invokes the installed runtime
predicate. One match is `UNIQUE`, two are `AMBIGUOUS`, and zero is
`NOT_FOUND`. Only `UNIQUE` is admitted.

Grounding occurs once at the initial scene. The returned MuJoCo object name is
retained through every later frame. This is source-supported: the BDDL source
relation initializes a named physical object, and the goal keeps that object
identity after motion invalidates the initial spatial description. Alternate
real instructions follow the same initial grounding and persistence rule.

## Grounding result and failure examples

Across 40 episodes and ten real instructions per episode, exact grounding was
unique in `53/400` attempts (`13.25%`), ambiguous in `0`, and not found in
`347`. For the episode's correct instruction, it was unique in `23/40`.

| Task | Correct UNIQUE episodes | Rate |
|---:|---:|---:|
| 0 | 2/4 | 50% |
| 1 | 0/4 | 0% |
| 2 | 3/4 | 75% |
| 3 | 4/4 | 100% |
| 4 | 4/4 | 100% |
| 5 | 0/4 | 0% |
| 6 | 2/4 | 50% |
| 7 | 4/4 | 100% |
| 8 | 0/4 | 0% |
| 9 | 4/4 | 100% |

Task1/demo31 illustrates the strict-site failure. The next-to-ramekin site is
centered at `(-0.18, 0.32, 0.895)`, while bowl1 is at approximately
`(-0.1932, 0.3264, 0.8982)`. The site half-width is about `0.01 m`; bowl1 is
the obvious nearest referent but is just outside the exact bound. Both bowls
therefore evaluate false. Task5 similarly loses the exact object contact needed
by its initialization `On` atom after state restoration.

Nearest-source-region, enlarged-site, and task-local-name fallbacks were not
preregistered and were not used.

## Counterfactual instruction set

The set contains only the other nine real task source atoms. A correct/alternate
pair is eligible only when both ground `UNIQUE` in the same initial scene.
Within the limited eligible set, the physical referent changes in `100%` of
same-state cases. Seven source tasks have at least one eligible alternate;
tasks1, 5, and 8 have none because the correct instruction never grounds.

## T0, T1, and conditional T2

T0 is only the official binary terminal predicate. On the correctly grounded
subset it is too sparse: local non-tie density is `4.76%`, with no task reaching
the `30%` usability threshold.

T1 is lexicographic `[goal_satisfied, goal_relation_margin]`. Its calibration
epsilon is `0.0049524`, fixed as the 10th percentile of positive adjacent-frame
changes above the numerical floor. T1 reaches `59.49%` local non-tie density
and exactly seven usable tasks, so the preregistered calibration family rule
does not activate T2. T2 remains fully defined but unused: official goal first,
then exact two-fingerpad grasp; controlled states compare goal margin and
uncontrolled states compare task-relative EE-to-referent readiness. No weighted
sum exists.

T1 is only a calibration candidate, not a selected semantic teacher, because
the upstream grounding gate fails.

## Official goal geometry and margin definition

The frozen candidate margin is:

```text
min(
  (0.03 - horizontal_center_distance) / 0.03,
  (bowl_center_z - plate_center_z) / 0.03,
  -abs(bowl_bottom - plate_top) / 0.03
)
```

The first two terms directly mirror the installed numeric `check_ontop`
inequalities. The third uses model offsets to reject a bowl vertically remote
from the plate. Contact remains exclusively in the official Boolean.

All candidate margins are finite, ranging from `-16.587` to `-0.00475`.
Goal-positive and goal-negative medians are `-3.152` and `-8.254`. These values
show coarse separation, but the positive distribution is not centered near
the surface optimum. Since grounding already stops the experiment, the report
does not assign `OFFICIAL_GOAL_MARGIN_VALID`; rotated scanned-object offsets
would require an independently frozen geometry refinement before such a claim.

## Local signal density and teacher/time agreement

The following values are calibration-only and restricted to episodes whose
correct instruction grounded uniquely.

| Gap | Rows | Non-tie | Progress | Regression | Tie | Time agreement among non-ties |
|---|---:|---:|---:|---:|---:|---:|
| 1--2 | 5,949 | 55.39% | 37.75% | 17.63% | 44.61% | 68.16% |
| 3--4 | 5,857 | 58.73% | 40.60% | 18.13% | 41.27% | 69.13% |
| 5--8 | 11,438 | 62.02% | 44.02% | 18.00% | 37.98% | 70.98% |
| 9--16 | 21,772 | 67.18% | 51.30% | 15.88% | 32.82% | 76.36% |

Local aggregate time agreement is `69.85%`, far from a deterministic
later-is-better rule. Explicit ties and regressions are common.

## Language counterfactual results

Among `2,916` eligible same-state calibration cases, referents switch in
`100%` and the T1 semantic state changes in `100%`. Across `22,536` eligible
local pair counterfactuals, preference changes in `57.57%`. Seven tasks reach
the frozen 20% material-change threshold. These numbers demonstrate that the
correct same-scene idea works where grounding is admitted, but they do not
meet the preceding complete-grounding requirement.

## Robot controls and proprio shortcut probe

Calibration-frozen standardized proprio thresholds find `2,037` semantic-tie
but strong-robot-difference rows across seven tasks, and `1,237` robot-matched
but semantic-non-tie rows across seven tasks. The construction confirms that
the explicit teacher does not directly include robot pose and that semantic
differences can occur under similar proprioception.

The single ridge-linear 36-D pair probe fits `[p_a,p_b,p_b-p_a]`. Its
calibration accuracy is `75.97%` on `13,829` non-tie local rows, below the
severe `90%` line. Because evaluation was blocked, this remains a calibration
diagnostic rather than a passed robot-phase-invariance result.

## Per-task calibration diagnostics

| Task | Local rows | Non-tie | Progress | Regression | Eligible language cases | Language change |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,008 | 49.45% | 30.58% | 18.87% | 167 | 100% |
| 1 | 0 | — | — | — | 0 | — |
| 2 | 2,884 | 54.23% | 35.09% | 19.14% | 374 | 100% |
| 3 | 3,040 | 49.41% | 32.14% | 17.27% | 398 | 100% |
| 4 | 4,848 | 63.72% | 45.34% | 18.38% | 624 | 100% |
| 5 | 0 | — | — | — | 0 | — |
| 6 | 1,824 | 53.29% | 35.64% | 17.65% | 237 | 100% |
| 7 | 4,432 | 71.10% | 49.64% | 21.46% | 572 | 100% |
| 8 | 0 | — | — | — | 0 | — |
| 9 | 4,208 | 60.79% | 47.72% | 13.07% | 544 | 100% |

Zero-row tasks are grounding failures, not evidence of tied semantic dynamics.

## Mechanical gates

| Required gate | Calibration result |
|---|---|
| Exact access guard | pass |
| Forbidden access count | pass: 0 |
| Same-scene grounding contract valid | **FAIL: correct UNIQUE 57.5%** |
| Persistent referent rule source-validated | source pass, empirical use incomplete |
| Valid counterfactual set | subset exists |
| Material language dependence | subset pass: 100%, 7 tasks |
| Local `Delta t<=8` non-tie >=30% | subset pass: 59.49% |
| At least 7 usable tasks | subset pass: exactly 7 |
| Semantic-match/robot-difference control | calibration pass |
| Robot-match/semantic-difference control | calibration pass |
| No severe raw-proprio shortcut | calibration pass: 75.97% |
| All values finite | pass |
| Complete Stage B1 gate | **FAIL** |

The first semantic gate fails, so the frozen decision tree prohibits opening
demo36--39. The evaluation result file explicitly records `NOT_OPENED` and
contains no fabricated evaluation metric.

## Final decision and authorization

Final mechanism labels:

```text
SEMANTIC_GOAL_PROGRESS_TEACHER_UNRESOLVED
```

Final authorizations:

```text
BLOCK_VALUE_DISTILLATION
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

No `SAME_SCENE_LANGUAGE_GROUNDING_VALID`,
`OFFICIAL_GOAL_MARGIN_VALID`, `SEMANTIC_LOCAL_PROGRESS_SIGNAL_VALID`,
`SEMANTIC_LANGUAGE_COUNTERFACTUAL_VALID`, or
`SEMANTIC_GOAL_PROGRESS_TEACHER_VALID` label is assigned. The subset results
are not promoted into full-stage claims.

## Required answers

**Q1 — How is each instruction grounded in the same scene?**

Its frozen BDDL source atom is evaluated twice in the restored episode initial
state, once per physical black bowl. Exactly one true result is `UNIQUE`; zero
is `NOT_FOUND`; two is `AMBIGUOUS`. Only `UNIQUE` is admitted.

**Q2 — Is the referent regrounded every frame?**

No. It is grounded once initially and the physical MuJoCo object name persists.
BDDL uses the source relation for initialization and keeps that named object in
the final goal even after motion destroys the source relation.

**Q3 — How often does another real instruction select a different bowl?**

In eligible correct/alternate cases, `100%`. Eligibility is limited because
exact grounding succeeds in only `23/40` correct episodes.

**Q4 — Does the teacher change with language?**

On the eligible subset, yes: state changes in `100%` and local preference in
`57.57%`. The complete language-conditioned teacher is unresolved because
grounding coverage fails first.

**Q5 — Is there a valid continuous official-goal margin?**

A finite, interpretable calibration candidate was implemented from official
numeric inequalities and surface geometry. It is not validated as the final
margin: goal-positive values remain far from its nominal surface optimum, and
evaluation was stopped.

**Q6 — Is official goal plus margin locally dense enough?**

On correctly grounded calibration episodes, yes: `59.49%` non-tie at
`Delta t<=8`, with seven usable tasks. This cannot override failed grounding.

**Q7 — Was T2 added?**

No. T1 passed its calibration density rule, so the conditional exact-grasp and
reach-readiness expansion remained disabled.

**Q8 — Are ties and regressions allowed?**

Yes. Local ties are `40.51%` and regressions `17.94%`; the teacher is not
defined as later-is-better.

**Q9 — Do semantic-matched, robot-different states remain tied?**

The calibration construction finds `2,037` such rows across seven tasks, all
tied by the explicit semantic teacher. Evaluation confirmation was not run.

**Q10 — Are robot-matched, semantic-different states distinguished?**

The calibration construction finds `1,237` such rows across seven tasks, all
non-tied by construction. Evaluation confirmation was not run.

**Q11 — How well does raw proprio recover preference?**

The frozen lightweight calibration probe reaches `75.97%`, below the `90%`
severe-shortcut threshold. There is no held-out result.

**Q12 — How strongly does the teacher agree with time?**

It agrees with later time on `69.85%` of local non-ties. Language changes many
states and `57.57%` of eligible local pair preferences, so the subset is not a
pure clock; the full claim remains blocked by grounding.

**Q13 — Did demo40--49 receive genuine zero access, including metadata?**

Yes. The guard and union ledger report zero forbidden demo IDs, zero forbidden
numeric reads, and zero forbidden metadata operations. Demo36--39 were also
left unopened after the calibration stop.

**Q14 — May the project output a valid teacher and `GO_VALUE_DISTILLATION`?**

No. The correct-instruction grounding gate fails, so the only valid result is
`SEMANTIC_GOAL_PROGRESS_TEACHER_UNRESOLVED` and
`BLOCK_VALUE_DISTILLATION`.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| Clean protocol | `2bdae11f9a41d3986d86c368e1f0574244446b9bf00538a6c28334a6b0e0f8fc` |
| Access allowlist | `a2045de035846fa5b9aa092d3b0455323228f108b49d9b45f537ba4ce5c4924c` |
| Task grounding contract | `2a5ce1ef29ece3118587d2bb48a1bccf748c2d5dbf4128cb90f6d90bd0bd90bb` |
| Blocked teacher definition | `5940ed3a5b2a0de3ec75a0445646411426f3299d3e91e6d26d40109fc7cf5b24` |
| Calibration result | `2f097b6969a353c91e8f6b2b8b984678b808e4ef66d45f434a88b3264d2bdd49` |
| Evaluation-not-run record | `1852e4b0f50a3085304002dbaaf490659462ec27e3e21e8d28bb2eb64157c40f` |
| Calibration sidecar | `1ea7cce61c0d4cb753d9e1165f38042efad405f0a7db5ac055d907563565ca7d` |
| Calibration ledger | `39b0f612af4f35e8bd5de4526953d9a7800e41c05f0e46ad878481e93cd1768e` |
| Union access audit | `ea21ce8815025e2b4e0d1d46e1bf0b4ec2d05acb5f1441488eb44bf9350de0bb` |

## Forbidden-activity audit

No demo36--49 numeric array or metadata was opened. No historical demo30/35
was opened. No reward, done, stored success, task outcome, image, action, or
candidate future was read. No value distillation, pairwise V2 retraining,
candidate bridge, World Model, learned proposal, reranker, value/cost model,
controller, CEM, MPPI, beam search, C1/C2, D20-v2, seed23/47, encoder change,
or task-success experiment was run. The prior blocked artifacts were preserved.
