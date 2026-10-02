# Frozen CLIP patch pooling does not recover the semantic value bridge

Date: 2026-10-03

Final mechanism labels:

```text
CLIP_PATCH_REPRESENTATION_INSUFFICIENT
FROZEN_CLIP_VISUAL_INFORMATION_LIMIT_SUPPORTED
```

Next-stage authorization: `BLOCK_SEMANTIC_VALUE_REDISTILLATION`

Continuing authorizations:

```text
BLOCK_REAL_CANDIDATE_VALUE_BRIDGE
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

Selected visual representation: `none`

Stage V executed: `false`

Stage B executed: `false`

Controller authorization: `false`

## Executive answer

Exposing the exact frozen CLIP ViT-B/32 patch grid does not materially repair
the language-conditioned semantic value representation on the calibration
selection split. R0, the global-feature reproduction, reaches `67.37%`
ordinary local non-tie accuracy and `61.63%` teacher-change agreement on
demo34. R3, the strongest patch candidate by the frozen lexicographic rule,
reaches `66.20%` and `63.07%`. Its language agreement improves by only `1.44`
percentage points, below the required `4` points.

The remaining required evidence moves in the wrong direction. R3's cross-demo
semantic-match mean `|P-0.5|` rises from `0.2829` to `0.4165`, rather than
falling by at least `0.02`, and its semantic-difference accuracy falls from
`41.88%` to `33.75%`, rather than increasing by at least 5 points. Its ordinary
accuracy is within the allowed two-point regression, but the complete material
gate fails.

R2 and R3 do change their attention strongly when language changes. R2's mean
attention total variation is `0.837` and center displacement is `0.615`; R3's
two slots have total variation `0.826/0.774`. Those changes are not
task-relevant under T1: R2's counterfactual agreement is only `59.57%`, and
both language-conditioned families are less stable on same-referent swaps and
more overconfident on cross-demo semantic matches than R0. Attention movement
is therefore not promoted into an object-selection or localization claim.

The formal result is
`CLIP_PATCH_REPRESENTATION_INSUFFICIENT` and the protocol-defined bounded label
`FROZEN_CLIP_VISUAL_INFORMATION_LIMIT_SUPPORTED`. The latter applies only to
this exact frozen ViT-B/32 checkpoint, its projected 7x7 patch tokens, and the
four preregistered heads under the present data and capacity limits. It is not
a universal claim that all CLIP layers or all visual encoders lack semantic
information.

Stage R failed before demo36--39 could be opened. The authorization tree
therefore stops: no semantic value re-distillation, real N64 candidate future,
World Model, reranker, controller, or task-success experiment was run.

## Frozen provenance and stage boundary

The upstream semantic value audit is frozen at commit `87a62fb` and annotated
tag `semantic-value-distillation-blocked`. This long run began at annotated tag
`spatial-value-long-run-start`. The Stage R protocol and exact CLIP spatial
contract were committed at `58ed9bf`; the implementation was committed at
`c8a995d` before patch extraction or training.

Patch integrity was frozen at `84f1c98`. The first calibration preflight then
stopped before its first optimizer step because PyTorch deterministic CUDA
execution required `CUBLAS_WORKSPACE_CONFIG`. Commit `66bb47c` set the required
workspace before importing PyTorch. It changed no feature, pair, model,
optimizer, epoch, temperature, selection rule, gate, partition, or
authorization. The numerical calibration diagnosis was frozen at `0e6125c`,
and the authorization boundary at `45b40db`.

No demo36--39 evaluation result exists. Stage R was deliberately defined as a
calibration representation screen: train on demo31--33, select on demo34, and
open no later partition unless the material-improvement gate passes.

## Exact CLIP spatial contract

The frozen checkpoint is `openai/clip-vit-base-patch32` revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Its image input is the same
upright, unmodified 128x128 natural RGB used by D13 and the same pinned
`CLIPProcessor` output at 224x224. The encoder path is:

```text
224x224 RGB
  -> Conv2d patch embedding, kernel/stride 32
  -> 7x7 = 49 spatial tokens, hidden width 768
  -> prepend CLS and add 50 learned positional embeddings
  -> pre-layernorm and 12 frozen transformer layers
```

The global feature is the encoded CLS token after vision post-layernorm,
shared 768-to-512 visual projection, and L2 normalization. The spatial grid is
encoded positions 1--49 through the same post-layernorm, projection, and
per-token L2 normalization. CLS is excluded from the patch grid. Fixed 2-D
coordinates are image-grid cell centers from `-6/7` to `+6/7`; they contain no
camera or simulator geometry. The full source contract is recorded in
`docs/clip_spatial_feature_contract.md`.

## Patch integrity

| Integrity check | Result |
|---|---:|
| Frames | 5,044 |
| Patch tensor | `[5044,49,512]`, float16 |
| Finite | pass |
| Repeated extraction | exact float16 equality |
| Minimum global parity cosine | `0.99999994` |
| Maximum global parity absolute error | `0.0` |
| Trainable CLIP parameter tensors | `0` |
| CLIP gradients present | `false` |
| Forbidden access | `0` |

The exact global parity shows that patch exposure used the existing visual
path rather than a subtly different preprocessing or checkpoint.

## Data boundary

| Use | Demo IDs | Opened in Stage R |
|---|---|---|
| Probe fit | 31--33 | yes |
| Representation validation | 34 | yes |
| Frozen development evaluation | 36--39 | no |
| Historical | 30/35 | no |
| Forbidden | 40--49 | no |

The source-RGB ledger contains 40 exact
`data/demo_{id}/obs/agentview_rgb` reads and no metadata operation. The D13
ledger contains the corresponding 40 exact `vision` reads plus one global
language matrix read. Both contain only demo31--34 and zero forbidden attempt.

## Frozen representation families

| Family | Representation | Parameters | Language in pooling |
|---|---|---:|---|
| R0 | frozen global projection `[512]` | 131,201 | no |
| R1 | one task-independent content slot, mean value, center, entropy | 78,753 | no |
| R2 | one language-query slot, mean value, center, entropy | 95,137 | yes |
| R3 | two language-query slots and registered relational features | 124,801 | yes |

Every family uses the same 64-hidden-unit direct antisymmetric comparison form
and stays below 150,000 trainable parameters. The probe consumes only frozen
natural visual features, fixed image coordinates, and frozen language. T1 and
G2 enter labels and strata only.

The fit contains `62,868` frozen balanced rows from demo31--33. Validation uses
all `19,248` demo34 ordinary pairs, `67,112` language-counterfactual
applications, 32 strict semantic-match cross-demo directions, and 320 clear
semantic-difference directions. Cross-demo pairs use a demo34 anchor and the
frozen cyclic demo31 reference. Because this is a single-demo-per-task
selection split, no pseudo-independent bootstrap interval is claimed.

## Primary calibration comparison

| Family | Ordinary acc. | AUROC | Teacher-change | True reversal | Tie mean | Cross-match mean | Cross-diff acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 global | 67.37% | 0.692 | 61.63% | 22.50% | 0.1052 | 0.2829 | 41.88% |
| R1 patch | 64.93% | 0.640 | 61.35% | 27.28% | 0.0717 | 0.3104 | 32.50% |
| R2 language slot | 64.83% | 0.638 | 59.57% | 34.46% | 0.0777 | 0.3376 | 40.62% |
| R3 dual slot | 66.20% | 0.676 | 63.07% | 30.76% | 0.0965 | 0.4165 | 33.75% |

Here `Tie mean` and `Cross-match mean` are mean `|P-0.5|`, so lower is better.
True-reversal performance improves for every patch family, especially R2, but
it was a secondary metric and does not offset lower ordinary and
counterfactual transfer plus worse scene controls.

## Material improvement relative to R0

| Family | Language increment | Cross-match improvement | Cross-diff increment | Ordinary increment | Complete gate |
|---|---:|---:|---:|---:|---|
| R1 | -0.28 pp | -0.0274 | -9.38 pp | -2.44 pp | FAIL |
| R2 | -2.06 pp | -0.0547 | -1.25 pp | -2.55 pp | FAIL |
| R3 | +1.44 pp | -0.1335 | -8.13 pp | -1.18 pp | FAIL |

The frozen gate requires at least `+4` language points, either at least `0.02`
lower cross-match deviation or at least `+5` cross-difference points, and no
more than a `2`-point ordinary regression. No family satisfies the first or
second requirement. R1/R2 also violate the ordinary non-regression bound.

The lexicographic diagnostic ordering is R3, R1, R2. It does not select R3:
selection additionally requires the complete material gate.

## Language-swap attention diagnostics

| Family / slot | Entropy, original/alternate | Mean attention TV | Mean center change |
|---|---:|---:|---:|
| R1 | 0.800 / 0.800 | 0.000 | 0.000 |
| R2 | 0.452 / 0.430 | 0.837 | 0.615 |
| R3 slot A | 0.363 / 0.377 | 0.826 | 0.658 |
| R3 slot B | 0.590 / 0.627 | 0.774 | 0.479 |

R1 is exactly language-independent by construction. R2/R3 learn highly
language-sensitive attention, but task relevance is contradicted by their
counterfactual and cross-demo metrics. No slot is identified as a bowl,
plate, or object location.

## Same-referent stability

| Family | Mean `|delta P|` | False preference flips |
|---|---:|---:|
| R0 | 0.1015 | 25.82% |
| R1 | 0.1134 | 31.90% |
| R2 | 0.1486 | 39.58% |
| R3 | 0.1844 | 41.17% |

The language-conditioned patch pools respond even when the alternate real
instruction retains the same physical referent. This is evidence of weak
semantic invariance, not successful object selection.

## Mechanical decision

| Gate | Result |
|---|---|
| Frozen checkpoint and preprocessing identity | pass |
| Patch finite, deterministic, correct shape | pass |
| R0 exact global reproduction | pass |
| Any R1--R3 language gain at least +4 pp | **no** |
| Any R1--R3 cross-demo improvement condition | **no** |
| Any family passes complete material gate | **no** |
| demo36--39 unopened | pass |
| demo40--49 numeric/metadata access zero | pass |

Final labels:

```text
CLIP_PATCH_REPRESENTATION_INSUFFICIENT
FROZEN_CLIP_VISUAL_INFORMATION_LIMIT_SUPPORTED
```

Final authorization:

```text
BLOCK_SEMANTIC_VALUE_REDISTILLATION
BLOCK_REAL_CANDIDATE_VALUE_BRIDGE
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

No visual representation or semantic value student is selected.

## Required Stage R answers

**Did R1 clearly outperform global CLIP?** No. It loses `2.44` ordinary points,
`0.28` language points, and `9.38` cross-demo difference points.

**Did language-conditioned patch selection materially help?** No. R2 is worse
than R0 on every material-gate increment. R3 adds only `1.44` language points
and regresses both cross-demo controls.

**Did the dual-slot relation isolate the selected-object/plate relation?** No.
Its attention changes with language, but same-referent false flips rise to
`41.17%`, cross-match overconfidence rises to `0.4165`, and cross-difference
accuracy falls to `33.75%`.

**What is selected?** Nothing. R3 is only the strongest diagnostic patch
family under the lexicographic order.

**May semantic value re-distillation or the real candidate bridge begin?** No.
The frozen decision tree stops at Stage R.

## Artifacts, hashes, and verification

| Artifact | SHA-256 |
|---|---|
| Protocol | `3827bfaa94e478659d44f57be75fb42eea1c1788b2ca320fbed1c6680d0c5eca` |
| Architectures | `beb0a394928e1fdb8b760acc95c5d836a9baa997286e9a29f123f4b4ab9b2db1` |
| CLIP spatial contract | `d250a2d5d44a0292cae7b17e4f27f61fb162950bdb3e3723e8629ca5d45cb9ce` |
| Probe implementation | `13f18e877d27e6d2e9b71f3c3da72968cc37f71efc8e3e4d1b2f61e7cb59d091` |
| Audit runner | `8e15f9bfb9ccd30978447923312be8b8098deb6dd67278e8b671510082d284f1` |
| Patch integrity | `6fdae2bcf0a06c8c3890974cf0bc29cf826d7a1ccba8e0d6d22fc30688884cad` |
| Calibration diagnosis | `739f8591fb8512919144ae02084c729093f05ee9a2a7cba72d4f31a8550fe8bf` |
| Final decision | `578396bdcff2ad884e7ddcd1171b4e8d881fcb8aeceb71652d32be6d3d6df2eb` |
| Calibration freeze | `d99f5e41d82e81799ab53e71fc43b9afee8868fa43cb3c801f1901dc42e03826` |
| Patch features | `1667a3204671b0a855ca8f8126267b50cf6f8f606981b71ee89070887f8fe7c8` |
| Weights | `3d5bce311b933f5f593a6ca7bca9b4ad77216af8901565046d392346383f3da5` |
| Calibration sidecar | `81b74660e92eeaf842f08b2ba4dc01d54af28cea4c60b2c665d32c8db940c871` |
| Access ledger | `3b47580ba1f9fe4d71567cae6d9cfe9e765ee9cf6ee9558b11f1cecbd0c7c43d` |

Twelve focused tests independently verify source hashes, partitions, token
coordinates and shapes, family parameter ceilings, antisymmetry, R1 language
invariance, patch integrity, recomputed ordinary/counterfactual/cross-demo
metrics, material gates, attention normalization, frozen hashes, and the stop
mapping. All reportable arrays are finite. The complete repository suite passes
`309` tests; its 21 messages are dependency deprecation warnings, not failures.

## Forbidden-activity audit

No demo36--49 numeric array or metadata was opened. No historical demo30/35
was opened. No segmentation, object mask, box, G2 identity, plate location,
simulator projection, object pose, proprioception, time, action, reward, done,
success, or task outcome entered a representation. No alternate encoder,
encoder update, candidate future, N64 tournament, semantic re-distillation,
World Model, learned proposal, reranker, controller, CEM, MPPI, beam search, or
task-success experiment was run. Frozen T1-M2, G2, C48_G15, and C0-v2 were not
modified.
