# Spatial value long-run summary

Date: 2026-10-03

Highest completed stage: `R`

```text
Selected visual representation: none
Selected semantic value student: none
Real candidate-value bridge: not run
World Model authorized: no
Controller authorized: NO
```

## Final outcome

The conditional research chain stopped correctly at Stage R. Exact frozen CLIP
ViT-B/32 patch tokens were deterministic and perfectly reproduced the existing
global visual path, but none of R1--R3 passed the preregistered material
improvement gate over R0. R3 was the strongest diagnostic patch family: it
improved teacher-change agreement from `61.63%` to `63.07%`, only `+1.44`
points rather than the required `+4`. Its cross-demo semantic-match deviation
worsened from `0.2829` to `0.4165`, and cross-demo semantic-difference accuracy
fell from `41.88%` to `33.75%`.

Final mechanism labels:

```text
CLIP_PATCH_REPRESENTATION_INSUFFICIENT
FROZEN_CLIP_VISUAL_INFORMATION_LIMIT_SUPPORTED
```

Final authorization state:

```text
BLOCK_SEMANTIC_VALUE_REDISTILLATION
BLOCK_REAL_CANDIDATE_VALUE_BRIDGE
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
controller authorization = false
```

The information-limit label is bounded to the exact frozen ViT-B/32 global and
projected 7x7 patch representations plus the preregistered small heads. It does
not assert that every CLIP layer, every capacity, or every other visual encoder
is intrinsically insufficient. The present task does not authorize an encoder
switch.

## Required answers

### Q1 — Does global CLIP mainly fail because of global pooling, or do patch features also lack semantic information?

The tested patch-level route also fails. R3 adds only `1.44` language points
and substantially worsens both cross-demo controls. This supports a bounded
frozen-CLIP visual information limit under the registered projected tokens and
small heads; it does not prove a universal encoder limit.

### Q2 — Is the R1 patch grid clearly better than the global feature?

No. R1 reaches `64.93%` ordinary accuracy and `61.35%` teacher-change
agreement versus `67.37%` and `61.63%` for R0. Its cross-demo difference
accuracy is also lower: `32.50%` versus `41.88%`.

### Q3 — Does language-conditioned patch selection bring a material improvement?

No. R2 reaches only `59.57%` teacher-change agreement; R3 reaches `63.07%`,
still below the required `+4`-point increment over R0. Neither improves a
required cross-demo condition.

### Q4 — Does the dual-slot relational representation further improve the selected-object/plate relation?

Not under T1. R3 is the best patch family on language agreement and ordinary
accuracy, but cross-match deviation rises to `0.4165`, cross-difference
accuracy falls to `33.75%`, and same-referent false flips rise to `41.17%`.

### Q5 — When the same RGB receives different language, does the selected representation make a task-relevant change?

There is no selected representation. R2/R3 attention changes strongly, but
the changes are weakly or incorrectly aligned with the teacher. Attention
movement alone is not task-relevant transfer.

### Q6 — Is the cross-demo semantic-match shortcut reduced?

No. Mean `|P-0.5|` is `0.2829` for R0, `0.3104` for R1, `0.3376` for R2, and
`0.4165` for R3. Every patch family is more overconfident on the strict matched
control.

### Q7 — Is cross-demo semantic-difference discrimination improved?

No. R0 is `41.88%`; R1/R2/R3 are `32.50%`, `40.62%`, and `33.75%`.

### Q8 — What is the final selected visual representation?

`none`. R3 is only the diagnostic lexicographic leader and fails the material
gate.

### Q9 — Under the new representation, can a scalar potential catch the direct pairwise model?

Not tested. Stage V was not authorized after Stage R failed.

### Q10 — Does ordinary local semantic accuracy exceed 70%?

No Stage R family reaches 70% on demo34. R0 reaches `67.37%`; the best patch
family, R3, reaches `66.20%`.

### Q11 — Does teacher-changing counterfactual agreement exceed 70%?

No. R3 is highest among patch families at `63.07%`; R0 is `61.63%`.

### Q12 — Does true better/worse reversal improve clearly?

It improves descriptively: R0 is `22.50%`, R1 `27.28%`, R2 `34.46%`, and R3
`30.76%`. That secondary gain is insufficient because the primary language,
ordinary, and scene-control gates fail.

### Q13 — Are same-referent language swaps stable?

No patch family improves stability. Mean probability change increases from
`0.1015` at R0 to `0.1134/0.1486/0.1844` for R1/R2/R3; false flips increase
from `25.82%` to `31.90/39.58/41.17%`.

### Q14 — Do semantic ties remain near 0.5?

Relatively yes as a secondary diagnostic. Mean deviation is
`0.1052/0.0717/0.0777/0.0965` for R0--R3. Tie behavior does not rescue the
failed material gate.

### Q15 — Does a scene/demo shortcut remain?

Yes. Every patch candidate is more overconfident than R0 on semantic-match
cross-demo pairs, and none can reliably order clear cross-demo semantic
differences.

### Q16 — What is the final selected semantic value student?

`none`. Re-distillation was blocked and not run.

### Q17 — If Stage B ran, could the student recover T1 ranking on real N64 candidate futures?

Not established. Stage B did not run because no representation and no student
passed the preceding gates.

### Q18 — What is top-3 oracle recall?

Not available; no real N64 candidate futures were collected or ranked.

### Q19 — How much true semantic improvement over A0 is recovered?

Not measured; Stage B was prohibited.

### Q20 — What is conservative override precision?

Not measured; no candidate override threshold or bridge was evaluated.

### Q21 — Can structured gripper candidates be evaluated correctly?

Not established by this run. The frozen C48_G15 contract was not executed.

### Q22 — Where is the final demonstrated bottleneck?

At the `visual representation` layer. The privileged T1 teacher remains valid,
but neither the global projection nor the tested projected patch representations
provide a sufficiently semantic, language-conditioned, scene-stable bridge.
Value re-distillation and real candidate transfer were not reached.

### Q23 — Is `GO_VALUE_ALIGNED_WORLD_MODEL_AUDIT` authorized?

No. It requires Stage R, V, and B all to pass. Stage R failed, so World Model
authorization remains blocked and controller authorization is `NO`.

## Integrity and stopping boundary

Only demo31--34 natural RGB and frozen features were opened. Demo36--39 were
not opened; demo40--49 received zero numeric and metadata access. Twelve
focused tests recompute the result and pass. The complete repository suite
passes `309` tests with 21 dependency deprecation warnings. No Stage V or B
artifact was fabricated, and no World Model or controller work was performed.
