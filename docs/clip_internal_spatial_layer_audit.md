# CLIP internal layers improve within-demo semantics but not cross-demo relational value

Date: 2026-10-04

Final mechanism labels:

```text
CLIP_INTERNAL_SPATIAL_REPRESENTATION_INSUFFICIENT
FROZEN_CLIP_BACKBONE_LIMIT_SUPPORTED
```

Next-stage authorization: `GO_ALTERNATE_FROZEN_VISUAL_ENCODER_AUDIT`

Semantic value re-distillation authorization: `false`

Selected CLIP internal layer: `none`

## Executive answer

The failure of projected CLIP patch tokens is not explained solely by the
768-to-512 projection or final token normalization. Under one fixed
language-conditioned single-slot reader, L9 is the strongest internal layer:
ordinary local T1 accuracy is `68.85%` and teacher-changing language agreement
is `66.98%`. Relative to the exact L12-proj baseline (`65.61%` and `62.02%`),
these are gains of `+3.24` and `+4.96` percentage points.

That improvement does not transfer across demonstrations. L9's semantic-match
deviation is `0.3314`, worse than L12-proj's `0.2304` by `0.1010`, while its
semantic-difference accuracy is `31.25%`, `7.50` points below the projected
baseline. The frozen material rule required the language gain plus either a
`0.03` match improvement or a `7`-point difference improvement. No layer
passes.

The information probes localize the break more precisely. L6 best detects
whether two real instructions select the same physical bowl (`81.48%` on
unseen demo34 scenes) and separates the sparse official goal state (state AUC
`0.962`, cross-demo goal-change accuracy `70.37%`). It also orders M2 within
demo at `74.40%`, but the same M2 probe falls to `52.63%` across demos. For the
primary L9 layer the cross-demo M2 result is only `35.34%`. The resulting
diagnostic is Case B: object-specific language selection is accessible, but
stable relational geometry is not.

Stage I therefore supports `FROZEN_CLIP_BACKBONE_LIMIT_SUPPORTED` only within
the frozen scope: this ViT-B/32 checkpoint, the seven preregistered spatial
states, the bounded readers, and the current natural RGB data. The frozen
decision tree now requires Stage E; demo36--39 remain unopened.

## Exact implementation and extraction integrity

The installed `transformers==4.21.1` implementation was inspected directly.
Its encoder `hidden_states[0]` is the pre-layer-normalized embedding entering
block 1, and indices 3/6/9/12 are outputs after those exact block counts. The
full tensor contract is in `docs/clip_internal_layer_contract.md`.

Extraction covered all 5,044 demo31--34 frames and retained 49 patch tokens.
The pre-layernorm reconstruction and hidden state 0 differed by exactly zero.
Re-encoded global CLIP differed from the frozen store by exactly zero, and
L12-proj was float16-identical to the previous Stage-R patch cache. The access
ledger contains only demo31--34; demo36--49 and demo40--49 metadata were not
opened.

## Primary P1 layer comparison

All rows use identical reader architecture, hidden size, training pairs,
optimizer, epochs, temperature procedure, and demo31--33 fit/demo34 selection.

| Layer | Ordinary | Teacher change | True reversal | Tie deviation | Same-ref ΔP | Same-ref flip | Cross-match | Cross-diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 60.25% | 53.91% | 22.17% | 0.0735 | 0.1504 | 41.45% | 0.1537 | 30.63% |
| L3 | 61.62% | 57.84% | 42.07% | 0.0923 | 0.1719 | 39.56% | 0.4257 | 36.88% |
| L6 | 67.46% | 64.08% | 28.37% | 0.0896 | 0.1630 | 37.82% | 0.3337 | 36.25% |
| **L9** | **68.85%** | **66.98%** | 35.65% | 0.0664 | 0.1569 | 39.12% | 0.3314 | 31.25% |
| L12-pre | 62.46% | 59.73% | 24.02% | 0.0646 | 0.1669 | 40.76% | 0.2964 | 36.88% |
| L12-postLN | 66.57% | 62.36% | 23.48% | 0.0788 | 0.1636 | 41.69% | 0.2759 | 36.88% |
| L12-proj | 65.61% | 62.02% | 26.20% | 0.0789 | 0.1472 | 37.34% | **0.2304** | **38.75%** |

L3 has the highest true-reversal score (`42.07%`) but fails ordinary accuracy,
language gain, and both cross-demo criteria. True reversal alone is not a
selection criterion.

## Mechanical material gate

| Layer | Teacher-change Δ | Cross-match improvement | Cross-diff Δ | Ordinary Δ | Complete |
|---|---:|---:|---:|---:|---|
| L0 | -8.11 pp | +0.0767 | -8.13 pp | -5.36 pp | fail |
| L3 | -4.18 pp | -0.1953 | -1.88 pp | -3.99 pp | fail |
| L6 | +2.06 pp | -0.1033 | -2.50 pp | +1.85 pp | fail |
| L9 | +4.96 pp | -0.1010 | -7.50 pp | +3.24 pp | fail |
| L12-pre | -2.29 pp | -0.0661 | -1.88 pp | -3.15 pp | fail |
| L12-postLN | +0.35 pp | -0.0455 | -1.88 pp | +0.96 pp | fail |
| L12-proj | baseline | baseline | baseline | baseline | fail |

L0's low cross-match deviation is not a pass: language agreement and ordinary
accuracy regress strongly and cross-demo difference falls. L9 clears only the
language and ordinary pieces.

## Diagnostic information probes

These probes use privileged labels only during diagnostic training and
measurement. No privileged value is a deployment input.

| Layer | Referent accuracy | Referent AUC | M2 ordinary | M2 cross-demo | Goal AUC | Goal cross-demo |
|---|---:|---:|---:|---:|---:|---:|
| L0 | 59.42% | 0.658 | 64.57% | 36.09% | 0.841 | 55.56% |
| L3 | 75.00% | 0.798 | 74.05% | 34.59% | 0.953 | 66.67% |
| **L6** | **81.48%** | **0.884** | **74.40%** | **52.63%** | 0.962 | **70.37%** |
| L9 | 66.91% | 0.749 | 71.15% | 35.34% | **0.967** | **70.37%** |
| L12-pre | 74.13% | 0.818 | 69.19% | 34.59% | 0.932 | 59.26% |
| L12-postLN | 72.47% | 0.794 | 69.76% | 36.84% | 0.934 | **70.37%** |
| L12-proj | 76.88% | 0.843 | 69.74% | 41.35% | 0.924 | 62.96% |

The selected-referent diagnostic is label-permutation invariant: it classifies
whether two real instructions in the same physical scene keep or change the
persistent referent from symmetric representation differences. It never asks
the model to memorize bowl1 versus bowl2.

The probe matrix resolves to Case B for the primary L9 diagnostic: referent
selection passes, but cross-demo M2 ordering fails. L6 is the strongest probe
layer overall, yet its cross-demo M2 score remains below the frozen 65% line.
This is evidence against blaming only the final projection and against claiming
that a different fusion head has already solved the representation problem.

## Attention is not grounding evidence

Every P1 layer changes attention strongly under language swaps. L9 mean total
variation is `0.867` and mean center displacement is `0.633`. Nevertheless its
cross-demo relational metrics fail. Attention motion is therefore reported as
a mechanism diagnostic, not promoted to object grounding or value validity.

## Decision

```text
CLIP_INTERNAL_SPATIAL_REPRESENTATION_INSUFFICIENT
FROZEN_CLIP_BACKBONE_LIMIT_SUPPORTED
GO_ALTERNATE_FROZEN_VISUAL_ENCODER_AUDIT
```

No CLIP internal layer is selected and semantic value re-distillation is not
authorized from Stage I. Stage E must compare only the small preregistered set
of alternate frozen encoders. Real candidate futures, a World Model, reranker,
controller, and task-success evaluation remain prohibited.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| Protocol | `5ce527fa55dbfe96b49b2fd7cc24d224ec0e94ed8b0be61fff1ea8e0299e6be2` |
| Extraction contract | `a7cf5a2f78f5648cb44f0d15ea8480f89d679220a88d11802156368f62d7d40e` |
| Feature manifest | `62ce8e62ffb7454b4bb7d7e2cc3074180f38ad030ce395127f3be1cf73725376` |
| Calibration result | `b75cd7776d71d108b5baa516457a1e080e7b94134f4f5560b9c4ed14a8b19d85` |
| Information probes | `9e86960a8dc490b05bb8165e4b2928f3cdecdd1c4c0a25e73d322fe4144f1b78` |
| Final decision | `46194f520488fe1416cf8e0d5c11361364c82eec8efbecd7e483f6856312eede` |

The required Stage-I outputs are finite. Evaluation demo36--39 was not opened.
No alternate encoder, Stage V student, candidate future, World Model, reranker,
controller, or task-success experiment was run in Stage I.
