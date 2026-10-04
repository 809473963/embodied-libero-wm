# Frozen visual backbone research stops after the alternate-encoder audit

Date: 2026-10-04

```text
Highest completed stage: E

CLIP internal layer valid: no

Alternate encoder audit executed: yes

Selected visual encoder: none

Selected visual layer: none

Selected semantic value student: none

Real candidate-value bridge authorized: no

Real candidate-value bridge executed: NO

World Model authorization: BLOCKED

Controller authorization: false
```

Final mechanism labels:

```text
CLIP_INTERNAL_SPATIAL_REPRESENTATION_INSUFFICIENT
FROZEN_CLIP_BACKBONE_LIMIT_SUPPORTED
FROZEN_VISUAL_BACKBONE_AUDIT_BLOCKED
NATURAL_VISUAL_SEMANTIC_INFORMATION_UNRESOLVED
```

Final authorizations:

```text
BLOCK_SEMANTIC_VALUE_REDISTILLATION
BLOCK_REAL_CANDIDATE_VALUE_BRIDGE
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

## Executive answer

The gated chain completes Stage I and Stage E, then stops before Stage V.
Stage I shows that the previous projected CLIP patch failure is not explained
solely by the final visual projection or token normalization. The fixed L9
representation improves teacher-changing language agreement to `66.98%`,
`+4.96` points over the same-reader L12-proj baseline, and reaches `68.85%`
ordinary calibration accuracy. It still worsens cross-demo semantic-match
deviation and cross-demo semantic-difference accuracy, so no CLIP internal
layer clears the joint material gate.

Stage E tests exactly two frozen alternate spatial backbones: DINOv2-S/14 and
SigLIP-B/16-224. Neither improves the formal global-CLIP baseline. DINOv2
reaches `61.73%` ordinary accuracy and `57.97%` teacher-change agreement;
SigLIP reaches `64.04%` and `60.30%`. Their M2 ordering probes collapse across
demos to `32.33%` and `30.08%`, even though referent and sparse terminal-goal
cues remain detectable.

The final supported diagnosis is a bounded frozen-visual-representation block:
the tested natural representations do not jointly preserve language-selected
object specificity, cross-demo relation geometry, and ordinary local T1 value
under the preregistered small readers. This is not proof that every visual
encoder or trainable representation must fail. It is also not a student-
distillation result, because the decision tree prohibited Stage V.

No demo36--39 evaluation was opened after the Stage E calibration failure. No
candidate future, N64 ranking, World Model, reranker, controller, or task-
success experiment was run.

## Completed decision chain

```text
Stage I: fixed CLIP ViT-B/32 internals
  -> no internal layer passes
  -> CLIP_INTERNAL_SPATIAL_REPRESENTATION_INSUFFICIENT
  -> FROZEN_CLIP_BACKBONE_LIMIT_SUPPORTED
  -> GO_ALTERNATE_FROZEN_VISUAL_ENCODER_AUDIT

Stage E: frozen DINOv2-S/14 and SigLIP-B/16-224
  -> neither representation passes
  -> FROZEN_VISUAL_BACKBONE_AUDIT_BLOCKED
  -> NATURAL_VISUAL_SEMANTIC_INFORMATION_UNRESOLVED
  -> BLOCK_SEMANTIC_VALUE_REDISTILLATION
  -> BLOCK_REAL_CANDIDATE_VALUE_BRIDGE
  -> STOP

Stage V: not authorized, not executed
```

## Stage I summary

Every extraction uses the same frozen `openai/clip-vit-base-patch32` revision,
the same natural RGB, the same 49-patch grid, and the same bounded P1 reader.

| Layer | Ordinary | Teacher change | Reversal | Same-ref flip | Tie dev. | Cross-match dev. | Cross-diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0 | 60.25% | 53.91% | 22.17% | 41.45% | 0.0735 | **0.1537** | 30.63% |
| L3 | 61.62% | 57.84% | **42.07%** | 39.56% | 0.0923 | 0.4257 | 36.88% |
| L6 | 67.46% | 64.08% | 28.37% | **37.82%** | 0.0896 | 0.3337 | 36.25% |
| L9 | **68.85%** | **66.98%** | 35.65% | 39.12% | 0.0664 | 0.3314 | 31.25% |
| L12-pre | 62.46% | 59.73% | 24.02% | 40.76% | **0.0646** | 0.2964 | 36.88% |
| L12-postLN | 66.57% | 62.36% | 23.48% | 41.69% | 0.0788 | 0.2759 | 36.88% |
| L12-proj | 65.61% | 62.02% | 26.20% | 37.34% | 0.0789 | 0.2304 | **38.75%** |

L9 is the strongest primary T1 layer, but relative to L12-proj it combines a
teacher-change improvement of `+4.96` points with a cross-match regression of
`0.1010` and a cross-difference regression of `7.50` points. It therefore
fails the required joint gate. L0 has the best cross-match neutrality but poor
ordinary and language-counterfactual value; no post-hoc single-metric choice is
allowed.

The independent probes peak at L6:

| Probe | Best layer | Calibration result |
|---|---|---:|
| Same-scene referent change | L6 | 81.48% accuracy |
| M2 cross-demo ordering | L6 | 52.63% accuracy |
| Official-goal cross-demo | L6/L9/L12-postLN | 70.37% accuracy |

Referent recovery is substantial, but cross-demo M2 ordering remains barely
above chance. The representative L9 diagnostic is therefore Case B—referent
cues without stable relation geometry—not Case A readout/fusion sufficiency.

Full details are in
[clip_internal_spatial_layer_audit.md](/home/luo/embodied-libero-wm/docs/clip_internal_spatial_layer_audit.md).

## Stage E summary

| Representation | Ordinary | Teacher change | Reversal | Same-ref flip | Tie dev. | Cross-match dev. | Cross-diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen global CLIP baseline | 67.37% | 61.63% | 22.50% | 25.82% | 0.1052 | 0.2829 | 41.88% |
| DINOv2-S14 P1 | 61.73% | 57.97% | 25.00% | 38.03% | 0.0504 | 0.2117 | 44.38% |
| SigLIP-B16-224 P1 | 64.04% | 60.30% | 22.50% | 40.29% | 0.0702 | 0.4034 | 30.63% |

| Family | Teacher-change Δ | Cross-diff Δ | Cross-match improvement | Ordinary Δ | Complete |
|---|---:|---:|---:|---:|---|
| DINOv2-S14 | -3.66 pp | +2.50 pp | +0.0712 | -5.65 pp | **FAIL** |
| SigLIP-B16-224 | -1.33 pp | -11.25 pp | -0.1205 | -3.33 pp | **FAIL** |

Neither family reaches the mandatory `+5 pp` teacher-change improvement or the
ordinary non-regression gate. DINOv2 clears only the match-neutrality branch;
SigLIP clears neither cross-demo branch.

The probes reinforce the relation-generalization diagnosis:

| Family | Referent acc. | M2 ordinary | M2 cross-demo | Goal state AUC | Goal cross-demo |
|---|---:|---:|---:|---:|---:|
| DINOv2-S14 | 73.58% | 71.10% | 32.33% | 0.948 | 66.67% |
| SigLIP-B16-224 | 73.99% | 66.76% | 30.08% | 0.958 | 62.96% |

Full details are in
[alternate_frozen_visual_encoder_audit.md](/home/luo/embodied-libero-wm/docs/alternate_frozen_visual_encoder_audit.md).

## Required answers

**Q1 — Is projected CLIP failure caused by late-layer/projection compression?**

Not solely. L9 recovers `+4.96` teacher-change points relative to L12-proj, so
late processing does weaken one useful signal. But L9's cross-demo match and
difference are worse, and all fixed internal points fail the complete gate.

**Q2 — Which internal layer most retains T1 semantic information?**

L9 by the frozen primary ordering: `66.98%` teacher-change agreement and
`68.85%` ordinary accuracy. It is a diagnostic representative, not a selected
valid layer.

**Q3 — Which layer has the highest teacher-changing language agreement?**

L9 at `66.98%`.

**Q4 — Which layer has the best cross-demo semantic invariance?**

L0 has the lowest semantic-match deviation, `0.1537`, but its ordinary and
teacher-change results are only `60.25%/53.91%`. Invariance alone cannot select
it.

**Q5 — Which layer best discriminates cross-demo semantic differences?**

Among non-projected internal layers, L3, L12-pre, and L12-postLN tie at
`36.88%`. L12-proj is `38.75%`, still below the global-CLIP reference. No layer
provides strong discrimination.

**Q6 — Where is selected-referent diagnostic performance best?**

L6 at `81.48%` accuracy.

**Q7 — Where is M2 ordering diagnostic performance best?**

L6, but its cross-demo accuracy is only `52.63%`; this is not strong relational
generalization.

**Q8 — Do good probes plus poor value imply fusion/readout is the bottleneck?**

No. The referent probe is good, but the decisive M2 cross-demo probe is weak.
The evidence fits missing or unstable relational geometry (Case B), not Case A
where both referent and M2 information are clearly present and only fusion
fails.

**Q9 — Can the conclusion be upgraded to `FROZEN_CLIP_BACKBONE_LIMIT_SUPPORTED`?**

Yes, within the explicit scope of this ViT-B/32 checkpoint, fixed layers,
bounded readers, and current natural RGB data. It is not a claim about all CLIP
models or trainable CLIP.

**Q10 — Which alternate frozen encoders were tested?**

DINOv2-S/14 final 16x16 patch tokens and SigLIP-B/16-224 final 14x14 patch
tokens, with frozen weights and unchanged CLIP text features.

**Q11 — Did a new encoder improve language CF plus cross-demo behavior, rather than ordinary accuracy alone?**

No. DINOv2 improves cross-match neutrality but regresses ordinary and language
CF; SigLIP regresses all primary and cross-demo quantities relative to the
formal baseline.

**Q12 — Did an alternate encoder reduce same-referent false language flips?**

No. The baseline is `25.82%`; DINOv2 and SigLIP rise to `38.03%` and `40.29%`.

**Q13 — Which encoder is selected?**

None.

**Q14 — What is ordinary local semantic accuracy after redistillation?**

Not applicable: Stage V was blocked and no redistilled student exists. The old
global-CLIP S2 evaluation remains `69.81%`; it was not rerun or replaced.

**Q15 — Does redistilled teacher-change agreement exceed 70%?**

Not tested because redistillation was not authorized. No positive claim is
made.

**Q16 — How much did true better/worse reversal improve?**

There is no Stage V reversal result. Stage E representation diagnostics are
calibration-only and cannot substitute for a frozen student evaluation.

**Q17 — Did same-referent language invariance improve after redistillation?**

Not tested. At Stage E, both alternate representations worsen false flips.

**Q18 — Does the Stage V cross-demo semantic-match gate pass?**

Not evaluated; Stage V did not run.

**Q19 — Does Stage V cross-demo semantic-difference accuracy exceed 60%?**

Not evaluated. Stage E reaches only `44.38%` for DINOv2 and `30.63%` for
SigLIP on calibration cross-demo pairs.

**Q20 — Can the scalar potential match the direct pairwise student?**

Unresolved in this representation chain because Stage V was blocked before
either student was trained.

**Q21 — What is the final student?**

`none`.

**Q22 — Where is the final bottleneck?**

The chain stops at `alternate visual representation`: late CLIP compression is
not the sole cause, the tested whole frozen CLIP backbone fails, and the two
tested alternate frozen spatial representations also fail to supply stable
language-conditioned cross-demo relation information. Student distillation is
not the final diagnosed stage because it was never authorized. The wording
remains bounded; natural visual semantic information in all possible encoders
is unresolved, not disproved.

**Q23 — Is `GO_REAL_CANDIDATE_VALUE_BRIDGE` authorized?**

No. The final authorization is `BLOCK_REAL_CANDIDATE_VALUE_BRIDGE`.

## Integrity and forbidden-activity audit

Stage I and E use only demo31--34. Stage E never opened demo36--39, and all
guards report zero demo40--49 numeric and metadata access. T1-M2, G2,
persistent referents, official `On`, M2, epsilon, labels, counterfactuals, and
splits remain unchanged.

No encoder was fine-tuned. No Stage V model, real candidate future, C48_G15
N64 tournament, candidate ranking, World Model, reranker, learned proposal,
controller, reward, done, success, or task outcome was executed. The maximum
authorization requested by the task was not reached.

Seventeen focused Stage I/E/final-summary tests and the complete `326`-test
repository suite pass. The 21 remaining messages are dependency deprecation
warnings, not test failures.
