# Semantic goal-progress teacher audit stops at an integrity boundary

Date: 2026-10-02

Final mechanism labels:

```text
SEMANTIC_GOAL_PROGRESS_TEACHER_UNRESOLVED
DATA_OR_IMPLEMENTATION_BLOCK
```

Next-stage authorization: `BLOCK_VALUE_DISTILLATION`

World-Model interface authorization: `BLOCK_WORLD_MODEL_INTERFACE_AUDIT`

Spatial World Model authorization: `BLOCK_SPATIAL_WM_PROTOTYPE`

Controller authorization: `false`

Selected semantic teacher: `none`

## Executive answer

Stage B1 did not produce a semantic progress teacher. Source inspection did
establish the exact official terminal relation: in all ten tasks, the
instruction-selected black bowl must satisfy `On(referent, plate)`, where the
installed predicate calls `plate.check_ontop(referent)`. The tasks differ in
the spatial referring expression used to select the bowl, not in the target
plate or terminal relation.

The first preregistered source inventory incorrectly compared BDDL-local object
symbols across files. Every task calls its selected bowl
`akita_black_bowl_1`, so that inventory reported one goal signature and zero
goal-changing counterfactuals. That conclusion is withdrawn. Each task also
contains `akita_black_bowl_2`, often placed in a relation named by another real
task instruction. On task0's scene, for example, the correct instruction
selects the bowl between the plate and ramekin, while task1's real instruction
can select the other bowl next to the ramekin. Correct same-state
counterfactual evaluation must ground the source-relation phrase against the
current scene and may change `On(bowl1, plate)` into `On(bowl2, plate)`.

Before a corrected grounding protocol was frozen, an exploratory HDF5 schema
walk used to locate an existing state sidecar enumerated task0 demo40--49 group
metadata. It loaded no numeric array value, reward, done, or success field, but
it exposed group names, trajectory lengths through shapes, and dtypes. The
task's boundary forbids any demo40--49 access, so metadata-only enumeration is
treated as a formal violation. Work stopped immediately: no semantic state,
tie threshold, local-density analysis, proprio probe, B2 distillation, B3
candidate bridge, World Model, or controller was run.

The mechanical result is therefore a data/implementation block, not evidence
that semantic language conditioning is absent. The preliminary
`SEMANTIC_TEACHER_LANGUAGE_INSUFFICIENT` label is explicitly withdrawn.

## Frozen provenance and correction history

The upstream temporal-progress audit is frozen at commit
`8e784928a86394d23d7ab05c21f02602d3332728` under annotated tag
`candidate-value-audit-blocked`. Its conclusions remain unchanged:

```text
PAIRWISE_PROGRESS_SIGNAL_VALID
ABSOLUTE_PROGRESS_BASELINE_VALID
PAIRWISE_PROGRESS_OUTPERFORMS_ABSOLUTE
PROGRESS_MODEL_LANGUAGE_INSENSITIVE
PROGRESS_MODEL_ROBOT_PHASE_DOMINATED
CANDIDATE_VALUE_AUDIT_BLOCKED
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

The new research start was tagged `semantic-candidate-value-research-start`.
The first Stage-B1 protocol and source-only implementation were committed as
`7c94067` before the automated BDDL inventory. That protocol correctly froze
the data partitions and stop rules but made one semantic mistake: it defined a
counterfactual by inequality of BDDL-local goal symbols, rather than by
grounding each language's referring expression in the same physical scene.

The source diagnosis is preserved for provenance, but its negative language
label is superseded by
`configs/semantic_goal_progress_teacher_decision_amendment.json`. The boundary
incident is recorded separately and forces the final block. No threshold or
result was changed to manufacture a pass.

## Data boundary and actual access

| Partition or source | Intended use | Actual access |
|---|---|---|
| Ten official BDDL files | task-semantic inventory | all ten read |
| demo31--34 | teacher construction/calibration | no rows read |
| demo36--39 | frozen development evaluation | no rows read |
| demo30/35 | historical only | not read |
| task0 demo40--49 | forbidden | metadata enumerated; no array values read |
| tasks1--9 demo40--49 | forbidden | not accessed |

The incident exposed ten forbidden demo group schemas for task0. Numeric
values read from those groups: `0`. This narrower fact does not restore
authorization because the contract prohibited access, not merely training on
the values.

## Exact task-semantic contracts

All task IDs and BDDL paths follow the frozen suite ordering. `Relevant source`
is the relation that grounds the instruction-selected bowl in that task's
initial scene. The exact goal is identical after local role assignment.

| Task | Instruction source phrase | Relevant source atom | Exact BDDL goal |
|---:|---|---|---|
| 0 | between plate and ramekin | `On(bowl1, between_plate_ramekin_region)` | `On(bowl1, plate)` |
| 1 | next to ramekin | `On(bowl1, next_to_ramekin_region)` | `On(bowl1, plate)` |
| 2 | table center | `On(bowl1, table_center)` | `On(bowl1, plate)` |
| 3 | on cookie box | `On(bowl1, cookies)` | `On(bowl1, plate)` |
| 4 | in cabinet top drawer | `In(bowl1, cabinet_top_region)` | `On(bowl1, plate)` |
| 5 | on ramekin | `On(bowl1, ramekin)` | `On(bowl1, plate)` |
| 6 | next to cookie box | `On(bowl1, next_to_box_region)` | `On(bowl1, plate)` |
| 7 | on stove | `On(bowl1, stove_cook_region)` | `On(bowl1, plate)` |
| 8 | next to plate | `On(bowl1, next_to_plate_region)` | `On(bowl1, plate)` |
| 9 | on wooden cabinet | `On(bowl1, cabinet_top_side)` | `On(bowl1, plate)` |

Every `:obj_of_interest` section names `akita_black_bowl_1` and `plate_1`.
That naming is local to each task file. It is not a proof that two different
instructions ground the same physical bowl when applied to one scene.

## Exact official predicates and success condition

The installed parser packages the BDDL `:goal` conjunction into `goal_state`.
`LiberoTabletopManipulation._check_success` evaluates every goal atom and
conjoins the Boolean results. For these ten tasks there is one atom:

```text
On(instruction-selected black bowl, plate_1)
```

The installed `On` implementation is exactly:

```text
target.check_ontop(subject)
```

No official task goal atom specifies grasp, reach, transport, or release.
Those could be explanatory stage predicates in a future teacher, but their
definitions and tie tolerances would need a clean calibration-only protocol.
They were not defined or evaluated here.

## Why the first language gate was invalid

The source inventory found one literal local goal string and ten distinct
relevant-object initial relations. It also found 90 directed swaps among real
task instructions. Its claim of zero semantic counterfactuals followed from
this invalid comparison:

```text
task A local bowl1 symbol == task B local bowl1 symbol
```

The correct question is instead:

```text
resolve instruction A's source relation in state s -> physical referent rA
resolve instruction B's source relation in the same state s -> referent rB
compare On(rA, plate) with On(rB, plate)
```

The BDDL initial states show why this matters. Each scene contains a second
black bowl whose relation matches another admitted task source:

| State task | Second bowl relation matches instruction from task |
|---:|---:|
| 0 | 1 |
| 1 | 6 |
| 2 | 8 |
| 3 | 9 |
| 4 | 9 |
| 5 | 3 |
| 6 | 7 |
| 7 | 9 |
| 8 | 1 |
| 9 | 7 |

This table is source-level evidence that a corrected counterfactual route is
plausible. It is not an evaluated language-dependence result. Dynamic
uniqueness, persistence of grounding, semantic stage, and value flips were not
measured.

## Teacher state vector, predicates, and ties

No teacher state was frozen. A future clean restart would need to decide, on
calibration data only, whether the minimal lexicographic state can validly use:

```text
goal relation satisfied
object controlled
reach readiness
within-stage geometric margin
```

Only the goal relation has an exact official definition in this report. No
grasp/contact threshold, reach threshold, transport margin, release rule, or
continuous tie tolerance is selected. Treating any of them as frozen would be
post-block invention.

## Withheld quantitative analyses

The following required B1 analyses were not run after the integrity block:

- fraction of states whose semantic value changes under a valid grounded
  counterfactual instruction;
- pair-preference flip rate, stage-change rate, and margin-change magnitude;
- semantic-matched/robot-different invariance controls;
- robot-matched/semantic-different opposite controls;
- non-tie, progress, regression, and tie density for gaps 1--2, 3--4, 5--8,
  and 9--16;
- per-task usability and the 7/10 rule;
- teacher/time agreement;
- the lightweight proprio-only preference probe.

Consequently no claim is made about local signal density, time equivalence,
robot-phase dominance, finite teacher margins, or teacher usefulness.

## Mechanical decision

| Gate | Result |
|---|---|
| Exact BDDL task inventory | complete |
| Exact terminal predicate identified | complete |
| Correct same-scene object grounding frozen | no; v1 rule defective |
| Valid language-counterfactual set | unresolved |
| Calibration teacher constructed | not run |
| Frozen evaluation | not opened |
| Forbidden partition untouched | **FAIL: task0 demo40--49 metadata enumerated** |
| B1 complete gate | **FAIL / blocked** |

Final mechanism labels:

```text
SEMANTIC_GOAL_PROGRESS_TEACHER_UNRESOLVED
DATA_OR_IMPLEMENTATION_BLOCK
```

Final authorizations:

```text
BLOCK_VALUE_DISTILLATION
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

The blocked result does not authorize a quiet retry, B2 distillation, B3
candidate execution, a World Model, reranking, or control. Any future work
would require a separately authorized clean restart with source-relation
grounding preregistered before touching admitted data and with an access guard
that prevents even metadata enumeration outside the named groups.

## Required B1 answers

**Is a semantic teacher selected?** No. Only the official terminal predicate
was established; the stage-aware teacher was not constructed.

**Does the teacher materially depend on language?** Unresolved. The initial
negative answer was invalid because it compared task-local symbols rather than
same-scene grounded referents.

**Is local semantic signal dense enough for short-horizon planning?** Not
tested.

**Is it distinct from time and raw robot phase?** Not tested.

**May value distillation begin?** No. B1 is blocked and the frozen decision
tree requires `BLOCK_VALUE_DISTILLATION`.

## Artifacts and hashes

| Artifact | SHA-256 |
|---|---|
| V1 protocol | `6377a71907243fccb6f9e7425cb4aeda29330d60c9692b4a8dce2db3828ee4ec` |
| Source inventory | `68a8b0a06f540ff9f733938c442f48a0955ec03078c18a898fb1a19dd4560083` |
| Superseded source diagnosis | `f490b33900fc6e4c4cd4a44e633f5e3e1246b0c9aa98ade21f6cba1b9df2def3` |
| Boundary incident | `a114c1c1d26f25b04c5ef33fb11291a2d8f03eedeb2a2789e85e5ede5f4b1b16` |
| Decision amendment | `da8a8f6d40fb75ec08acf9e659cc0469ce744bb8eb7403fa28edc581e324e41c` |
| Corrected diagnosis | `9ebf0fdd5bf9f2458587f933d2e55322ab5f4cbbf97e0d6f6d88fa18ed9599fa` |

The source inventory and diagnosis are retained rather than rewritten. The
amendment explicitly withdraws their invalid language-insufficiency mapping.

Ten focused protocol/result tests verify the frozen source hashes, partition
rules, exact task ordering, semantic-versus-textual counterfactual distinction,
withdrawn preliminary label, incident scope, zero forbidden numeric values,
zero downstream training/execution, report completeness, and final block. The
complete repository suite passes `223` tests; the remaining 21 messages are
dependency deprecation warnings, not failures.

## Forbidden-activity audit

No numeric demo40--49 array, reward, done, success, or task outcome value was
loaded. Nevertheless, task0 demo40--49 metadata was enumerated and is the
formal blocking violation. No simulator, semantic teacher, proprio probe,
value distillation model, candidate future, World Model, encoder, learned
proposal, reranker, score, controller, CEM, MPPI, beam search, C1/C2, D20-v2,
or seed23/47 experiment was run or modified. B2 and B3 were not started.
