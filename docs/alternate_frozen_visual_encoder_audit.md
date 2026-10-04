# Alternate frozen spatial encoders do not clear the semantic value gate

Date: 2026-10-04

Final mechanism labels:

```text
FROZEN_VISUAL_BACKBONE_AUDIT_BLOCKED
NATURAL_VISUAL_SEMANTIC_INFORMATION_UNRESOLVED
```

Next-stage authorizations:

```text
BLOCK_SEMANTIC_VALUE_REDISTILLATION
BLOCK_REAL_CANDIDATE_VALUE_BRIDGE
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

Selected visual encoder: `none`

Selected visual layer: `none`

Stage V executed: `false`

Controller authorization: `false`

## Executive answer

The bounded alternate-backbone audit does not find a frozen natural-image
representation that clears the T1-M2 semantic value gate. DINOv2-S/14 and
SigLIP-B/16-224 were evaluated with their frozen final spatial tokens, the
same frozen CLIP task-language vector, and the same small language-conditioned
P1 reader. Both were fit on demo31--33 and selected only on demo34. Neither
opened demo36--39.

DINOv2 reaches `61.73%` ordinary local T1 accuracy and `57.97%`
teacher-changing language agreement. Relative to the frozen global-CLIP
baseline, these are `-5.65` and `-3.66` percentage points. Its cross-demo
semantic-match deviation improves by `0.0712`, but cross-demo semantic
difference improves by only `2.50` points and the two mandatory primary
conditions fail.

SigLIP is descriptively stronger on the two primary quantities at `64.04%`
ordinary accuracy and `60.30%` teacher-change agreement, but these remain
`-3.33` and `-1.33` points below global CLIP. It also regresses substantially
on both cross-demo controls: semantic-match deviation rises from `0.2829` to
`0.4034`, and semantic-difference accuracy falls from `41.88%` to `30.63%`.

The privileged diagnostics show why ordinary spatial resolution is not enough.
Referent-change accuracy is `73.58%/73.99%` for DINOv2/SigLIP and official-goal
state AUC is `0.948/0.958`; however, cross-demo M2 ordering is only
`32.33%/30.08%`, both below chance. The representations retain object and
terminal-state cues, but the tested bounded readout does not recover stable
language-selected bowl-to-plate relation geometry across scenes.

The mechanical result is therefore a block, not a new encoder selection.
Stage V is prohibited, demo36--39 remains unopened, and the project cannot
authorize the real candidate-value bridge from this task.

## Provenance and authorization boundary

Stage I is frozen at commit `7efb7bb` and established:

```text
CLIP_INTERNAL_SPATIAL_REPRESENTATION_INSUFFICIENT
FROZEN_CLIP_BACKBONE_LIMIT_SUPPORTED
GO_ALTERNATE_FROZEN_VISUAL_ENCODER_AUDIT
```

The Stage E protocol was committed at `e06cb8b` before either alternate
checkpoint was downloaded, any alternate feature was extracted, or any reader
was trained. A synthetic-input loader preflight then made the library's slow
image processor explicit with `use_fast=false`; this implementation freeze was
committed at `367cc93` before any demo RGB was opened. It did not change a
checkpoint, preprocessing value, token source, data row, training setting,
metric, threshold, or decision rule.

Calibration feature extraction and its exact access ledger were frozen at
`cbe532f`. The one-shot calibration diagnosis was frozen at `bead98c`. No
evaluation-time amendment occurred because Stage E never authorized opening
the evaluation partition.

## Resource preflight and selected families

Before download, the repository contained only the frozen CLIP checkpoint;
there was no DINOv2 or SigLIP checkpoint and no installed `timm`, `open_clip`,
or `safetensors`. The system had an 8 GB RTX 4070 Laptop GPU and about 193 GiB
free disk. To avoid changing the project environment, Transformers `4.57.1`,
Tokenizers `0.22.1`, and Safetensors `0.6.2` were installed only under the
ignored `.cache/stage_e_python` directory.

Exactly two families were admitted. A third was disabled rather than adding an
unjustified model or package stack.

| Family | Frozen repository / revision | Official preprocessing | Spatial token source | Cache shape |
|---|---|---|---|---:|
| DINOv2-S14 | `facebook/dinov2-small` / `ed25f3a...` | bicubic short-edge 256, center crop 224, ImageNet normalization | final normalized patch tokens, CLS excluded | `[5044,256,384]` |
| SigLIP-B16-224 | `google/siglip-base-patch16-224` / `7fd15f0...` | bicubic direct resize 224, mean/std 0.5, no crop | final patch-only vision tokens | `[5044,196,768]` |

The architecture and checkpoint choices follow the official
[DINOv2 repository](https://github.com/facebookresearch/dinov2/blob/main/README.md),
[DINOv2-S/14 model card](https://huggingface.co/facebook/dinov2-small), and
[SigLIP-B/16-224 checkpoint](https://huggingface.co/google/siglip-base-patch16-224/tree/main).
All encoder parameters were frozen. No adapter, LoRA, fine-tuning, privileged
crop, mask, object box, or simulator coordinate entered either representation.

## Data and access boundary

| Partition | Demo IDs | Stage E use | Actual access |
|---|---|---|---|
| Fit | 31--33 | P1 and diagnostic-probe fitting | opened exact RGB and frozen feature paths |
| Selection | 34 | metrics, temperatures, gates | opened exact RGB and frozen feature paths |
| Frozen development | 36--39 | prohibited until Stage V | not opened |
| Historical | 30/35 | disabled | not opened |
| Forbidden | 40--49 | forbidden including metadata | zero access |

The extraction guard read only `obs/agentview_rgb` from each admitted source
demo. The inherited frozen feature guard read the task-language matrix and the
already admitted global-CLIP arrays used to preserve exact T1 row identity.
Both alternate encoders processed the same upright, unmodified 128x128 RGB
frames. Their encoder-specific processors performed only the official resize,
crop, rescale, and normalization.

The two float16 feature caches contain 5,044 frames and total about 2.4 GB.
All values are finite; repeated extraction of the frozen sample is bitwise
identical after float16 serialization; trainable encoder parameters are zero.
The union ledger records demo31--34 only, no metadata operation, no forbidden
attempt, and zero demo40--49 numeric or metadata access.

## Shared reader and calibration protocol

Every family uses one P1 reader:

```text
native spatial tokens + fixed normalized x/y coordinates
  -> 24-D key and value
frozen CLIP task language [512]
  -> 24-D query
single attention-weighted slot + mean + center + entropy = 51-D state
two states + difference + language
  -> 64-unit shared raw head
  -> antisymmetric pair logit
```

The DINOv2 and SigLIP readers contain `73,577` and `92,009` trainable
parameters, below the frozen 150k ceiling. Both use the same 30 epochs, AdamW,
learning rate `0.001`, weight decay `1e-4`, deterministic seed 41, pair sampler,
temperature grid, and all admitted ordinary, tie, counterfactual, robot-control,
and cross-demo training rows. Native grid size changes only the number of
frozen tokens; reader structure and state dimension are identical.

## Frozen baseline and material gate

The formal baseline is the immutable previous global-CLIP calibration result:

| Metric | Global CLIP baseline |
|---|---:|
| Ordinary local non-tie accuracy | 67.37% |
| Teacher-change agreement | 61.63% |
| True-reversal both-correct | 22.50% |
| Same-referent mean probability change | 0.1015 |
| Same-referent false flip | 25.82% |
| Tie mean deviation | 0.1052 |
| Cross-demo match deviation | 0.2829 |
| Cross-demo difference accuracy | 41.88% |

An alternate encoder must simultaneously achieve:

```text
teacher-change increment >= +5 percentage points
AND (cross-demo difference increment >= +7 points
     OR cross-demo match deviation improvement >= 0.03)
AND ordinary accuracy regression <= 2 points
```

Raw privileged probes cannot select an encoder.

## Primary calibration results

| Representation | Ordinary | Teacher change | Reversal | Tie dev. | Same-ref ΔP | Same-ref flip | Cross-match dev. | Cross-diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen global CLIP | 67.37% | 61.63% | 22.50% | 0.1052 | 0.1015 | 25.82% | 0.2829 | 41.88% |
| DINOv2-S14 P1 | 61.73% | 57.97% | 25.00% | 0.0504 | 0.1160 | 38.03% | 0.2117 | 44.38% |
| SigLIP-B16-224 P1 | 64.04% | 60.30% | 22.50% | 0.0702 | 0.1557 | 40.29% | 0.4034 | 30.63% |

DINOv2 is better calibrated on ordinary T1 ties and cross-demo semantic
matches, but its ordinary and teacher-change results regress and its
cross-demo difference increment is only `+2.50` points. SigLIP is closer on
ordinary and counterfactual transfer, but it makes both same-referent language
invariance and cross-demo behavior materially worse.

The fixed selection ordering ranks SigLIP ahead of DINOv2 only because it first
sorts teacher-change agreement. That ordering is descriptive: the selector may
choose only among complete-gate passes, and there are none.

## Gate deltas

| Family | Teacher-change Δ | Cross-diff Δ | Cross-match improvement | Ordinary Δ | Complete |
|---|---:|---:|---:|---:|---|
| DINOv2-S14 | -3.66 pp | +2.50 pp | +0.0712 | -5.65 pp | **FAIL** |
| SigLIP-B16-224 | -1.33 pp | -11.25 pp | -0.1205 | -3.33 pp | **FAIL** |

DINOv2 passes one alternative cross-demo clause but fails both mandatory
primary clauses. SigLIP fails every material clause. No rounding or probe-based
override is permitted.

## Privileged information probes

These diagnostics use G2 identity, M2, and official `On` only as labels. Their
inputs remain frozen visual tokens plus frozen task language, and they have no
selection authority.

| Encoder | Referent acc. | Referent AUC | M2 ordinary | M2 cross-demo | Goal state AUC | Goal cross-demo |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2-S14 | 73.58% | 0.814 | 71.10% | 32.33% | 0.948 | 66.67% |
| SigLIP-B16-224 | 73.99% | 0.829 | 66.76% | 30.08% | 0.958 | 62.96% |

Both encoders permit above-chance within-scene referent discrimination and
strong sparse official-goal classification. The M2 result does not transfer:
cross-demo accuracy is below chance and substantially below the corresponding
ordinary diagnostic. This is consistent with a scene-specific appearance or
geometry shortcut, not a stable goal relation.

Attention changes are large under language changes: mean attention total
variation is `0.910` for DINOv2 and `0.925` for SigLIP. This is not treated as
grounding evidence. The counterfactual and cross-demo failures show that
changing attention does not imply selecting the correct physical referent and
recovering its relation to the plate.

## Mechanism interpretation

The audit narrows the current limit but does not prove that natural images lack
semantic information in principle. It supports only the following bounded
statement:

```text
For the frozen DINOv2-S/14 and SigLIP-B/16-224 final spatial tokens,
the fixed CLIP text representation, the preregistered P1 reader,
and the current natural RGB data, no representation clears the
language-counterfactual plus cross-demo semantic value gate.
```

The diagnostic pattern is closest to “referent cues present, relational
geometry not stable.” It does not authorize encoder fine-tuning, masked
supervision, object crops, a different text encoder, or a larger reader. Those
would be separate experiments.

## Mechanical decision

| Gate | DINOv2-S14 | SigLIP-B16-224 |
|---|---|---|
| Teacher change +5 pp | fail | fail |
| Cross-demo difference +7 pp | fail | fail |
| Cross-demo match improvement 0.03 | pass | fail |
| Ordinary no worse than -2 pp | fail | fail |
| Complete representation gate | **FAIL** | **FAIL** |

Final labels:

```text
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

Stage V is not run. No visual encoder, visual layer, reader, or semantic value
student is selected.

## Required Stage E answers

**Which alternate encoders were tested?** DINOv2-S/14 final 16x16 patch tokens
and SigLIP-B/16-224 final 14x14 patch tokens, both frozen and both paired with
the unchanged frozen CLIP text vector.

**Did either improve language counterfactuals plus cross-demo behavior?** No.
DINOv2 improves cross-demo match neutrality but regresses counterfactual and
ordinary accuracy; SigLIP regresses counterfactual, ordinary, cross-match, and
cross-difference metrics.

**Did either reduce same-referent false language flips?** No. The baseline is
`25.82%`; DINOv2 reaches `38.03%` and SigLIP `40.29%`.

**Which encoder is selected?** None. SigLIP leads the descriptive ordering on
teacher-change agreement, but selection is restricted to complete-gate passes.

**May semantic value redistillation begin?** No. The decision tree requires
`BLOCK_SEMANTIC_VALUE_REDISTILLATION` and an immediate stop before Stage V.

## Artifacts, finite values, and automated verification

| Artifact | SHA-256 |
|---|---|
| Protocol | `e75e760411690b2d33c6c0ed3476860a04d0b4c3c752ac1056c75496368855f0` |
| Encoder contracts | `4d43e0dcdc551381dfed82e7a1f659504bf6438f2f061cb36f73a58cd06d5427` |
| Feature manifest | `1efcad03b7c456d24513a35895947cfa2529909937ca1f089a64d1fcaef7a7b1` |
| Access ledger | `2176c08d139e89407b531cca9d17fcfb828c2a312bee6e15b91d147b3e4465ef` |
| Calibration diagnosis | `78212ea0d833074560585d8855cbd6b75d9d36586802f74bf2a4d75f08ef828d` |
| Information probes | `885c76f82ac6570a9c1678c3a400a4d9112356e639b8e0b1fa72c78012d91bef` |
| Final decision | `d66884b7f0212646a36ca2203e9ee2ad9738cf57c3be7c4670c3cc2a8347627b` |
| Calibration weights | `8fa3cfd6270ad7293065f472bdfe1c6dc21c406c022e84b1a850f16b91eb5a1a` |
| Calibration sidecar | `c8c46eeb05eaa6ddad491df7fb60958292878765bae400825ec6404c79ae83cb` |
| Audit runner | `e57282430732c0b31babffdfa6e60e972a9551d858b79d005abad30367808aa1` |

Seventeen focused Stage I/E/final-summary tests independently verify frozen hashes, partitions,
native-grid shapes, equal reader capacity, antisymmetry, feature determinism,
finite caches, exact access, recomputed ordinary and counterfactual metrics,
material gates, diagnostic logits, artifact hashes, and the stop decision.
The complete repository suite passes `326` tests; the remaining 21 messages
are dependency deprecation warnings, not failures.

## Forbidden-activity audit

No demo36--39 or demo40--49 numeric array or metadata was opened in Stage E.
No encoder weight was trained. No privileged crop, mask, object box, object
pose, simulator coordinate, reward, done, success, task outcome, candidate
future, N64 tournament, semantic redistillation, World Model, learned proposal,
reranker, score, controller, CEM, MPPI, beam search, C1/C2, D20-v2, seed23/47,
or task-success experiment was run. The frozen T1-M2 teacher, G2 resolver,
counterfactuals, labels, epsilon, and split were unchanged.
