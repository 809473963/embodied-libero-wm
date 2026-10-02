# Semantic candidate-value research stops at Stage B1 preflight

Date: 2026-10-02

Highest completed stage: `B1`

Semantic teacher selected: `no`

Deployable value evaluator selected: `no`

Real candidate-value bridge validated: `no`

World Model authorized: `no`

Controller authorized: `NO`

Final state:

```text
SEMANTIC_GOAL_PROGRESS_TEACHER_UNRESOLVED
DATA_OR_IMPLEMENTATION_BLOCK
BLOCK_VALUE_DISTILLATION
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

## Outcome

The official task definitions show a common terminal relation—place the
instruction-selected black bowl on the plate—but ten different source
relations that identify the intended bowl. A first source-only gate mistakenly
treated BDDL-local `bowl1` names as globally identical and its negative
language conclusion was withdrawn. The presence and placement of the second
black bowl show that same-scene counterfactual instructions may select a
different physical object.

Before that corrected grounding could be preregistered, an exploratory schema
walk enumerated task0 demo40--49 metadata. No values were loaded, but any
forbidden-partition access violates the task contract. B1 therefore stops as a
data/implementation block. B2 distillation and B3 candidate bridging were not
run.

## Q1 — Why did the old temporal-progress supervision fail?

It learned expert-trajectory order more than task-goal value. Visual pairwise
accuracy was `74.245%`, but wrong-language degradation was only `3.802`
points and proprio+language reached `96.224%`, exposing weak goal dependence
and a dominant robot-phase shortcut.

## Q2 — Which task predicates define the new semantic teacher's value?

No full teacher was selected. The only frozen exact predicate is the official
terminal condition `On(instruction-selected black bowl, plate)`, implemented
as `plate.check_ontop(bowl)` and conjoined by the environment success check.
Potential control, reach, transport, release, and within-stage predicates were
not defined after the block.

## Q3 — Does teacher value truly change when language changes on the same state?

Unresolved. The initial “no” was invalid because task-local role names were
compared without grounding the referring expression. Source layouts show a
plausible corrected test in which another real instruction selects the second
black bowl, but no state-level teacher comparison was run.

## Q4 — Is the teacher independent of trajectory index and normalized time?

No teacher was constructed, so this is not established. The intended protocol
forbids time/index inputs, but a design constraint is not empirical evidence.

## Q5 — Can raw proprioception or robot phase easily predict the teacher?

Not tested. The required proprio-only diagnostic was withheld after the B1
block.

## Q6 — Is semantic progress dense enough within Delta t<=8 for a short-horizon planner?

Not tested. No semantic pair labels or non-tie-density statistics were
constructed.

## Q7 — Can natural RGB plus language distill teacher preference?

Not tested. B2 was prohibited because B1 did not pass.

## Q8 — Does the learned evaluator change correctly under counterfactual language on the same image?

Not tested. No deployable evaluator exists.

## Q9 — Does pairwise semantic evaluation outperform an absolute semantic-value predictor?

Not tested. Neither semantic model was trained. The old temporal pairwise
advantage does not answer this semantic question.

## Q10 — Is the model near indifference on semantic-matched cross-demo states?

Not tested. The old temporal model was not near indifference, but it is not the
new semantic evaluator requested here.

## Q11 — Can the evaluator recover teacher ranking on real C48_G15 N64 futures?

Not tested. B3 was never entered and no expanded candidate future was executed.

## Q12 — Does the value bridge hold separately for structured gripper candidates?

Not tested. Structured gripper proposals retain their Stage-A imitation-oracle
validity, but semantic candidate value was not evaluated.

## Q13 — Is there a high-precision conservative override signal relative to A0?

Not established. No candidate ranking, regret, precision, or override analysis
was run.

## Q14 — Where is the current bottleneck: semantic supervision, value distillation, or candidate transfer?

The earliest unresolved layer is semantic supervision. More precisely, the
clean object-grounding teacher audit was not completed because its first
protocol had a grounding defect and the subsequent forbidden-metadata incident
invalidated continuation. Distillation and transfer remain downstream unknowns.

## Q15 — Is the project qualified to enter a Value-Aligned World Model Audit?

No. The mechanical decision is `BLOCK_WORLD_MODEL_INTERFACE_AUDIT`. A semantic
teacher, deployable evaluator, and real candidate-value bridge are all absent.

## Preserved independent facts

The exact C48_G15 N64 candidate interface remains frozen and valid. The old
temporal-progress findings remain frozen and negative for goal-conditioned
value. The spatial World Model remains independently blocked. None of those
facts are overwritten by this preflight incident.

