# The oracle footprint target is valid, but natural extraction blocks a spatial World Model prototype

Date: 2026-10-01

Final mechanism labels:

```text
OBJECT_FOOTPRINT_TARGET_VALID
ORACLE_SPATIAL_TARGET_VALID
NATURAL_SPATIAL_EXTRACTION_BLOCK
OBJECT_AUXILIARY_TARGET_VALID
OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED
SPATIAL_TARGET_DESIGN_UNRESOLVED
```

Next-stage authorization: `BLOCK_SPATIAL_WM_PROTOTYPE`

Selected design: `none`

Controller authorization: `false`

## Executive answer

The study finds a compact target worth keeping as a teacher, but no target is
ready for a deployable World Model prototype. The calibration-frozen oracle
target is a 7x7 relevant-object occupancy map plus one visible-area scalar:
50 float values instead of the full 49x512 patch tensor. On held-out init36--39,
adding this footprint summary to global CLIP improves object-mask-difference
RMSE by `30.45%` (branch-bootstrap 95% CI `[20.15%,40.94%]`) and
visible-area-difference RMSE by `62.64%` (CI `[55.81%,67.64%]`). Its spatial
shuffle worsens the two errors by `72.09%` and `243.54%`, and it improves
9/10 and 10/10 tasks, all eight horizons, and all three candidate pairs.

The footprint contains object information beyond correlated robot motion.
After fitting `ObjMask ~ EE` and `Area ~ EE` only on calibration states, the
same held-out footprint improves residual RMSE by `18.49%` (CI
`[10.72%,25.35%]`) and `55.66%` (CI `[48.45%,61.02%]`). This assigns
`OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED`; the footprint result is not
only an EE-motion proxy.

The natural-image proxy fails the required bridge from teacher to deployment.
Its binary localization is useful—IoU `0.650`, balanced accuracy `0.862`, and
positive-cell recall `0.763`—but its soft occupancy is badly miscalibrated:
MSE is `0.04396`, `20.32x` the calibration-mean-map baseline
(`-1932.02%` relative improvement). Visible-area MAE improves over its constant
baseline by only `0.62%`. More importantly, global plus proxy improves held-out
ObjMask/Area RMSE by only `4.72%` / `4.19%`; both are below 10%, and the Area
interval crosses zero. Its EE-conditioned residual gains are only `2.11%` and
`2.45%`, with both intervals crossing zero.

The generic 49-D rank-1 spatial map also fails: its ObjMask/Area gains are
`5.84%` / `2.13%`, and it clears neither residual gate. Therefore the full
patch tensor is not required to define a strong oracle object target, but the
current frozen natural features plus small decoder do not recover that target
with sufficient calibrated fidelity. No D0--D5 candidate satisfies relevance,
increment, shuffle, residual-object, robustness, and deployment feasibility
together. The result is `BLOCK_SPATIAL_WM_PROTOTYPE`; no World Model was
trained.

## Frozen provenance and authorization boundary

The upstream image-space restart is frozen at commit
`b5963a67718bf0420ebf139125033412a6a2b01f` under annotated tag
`natural-image-space-target-design-go`. Its positive object-family result and
`GO_SPATIAL_TARGET_DESIGN` authorization remain unchanged.

The new protocol and implementation were first committed at `0ad2864` before
any new calibration result. Two calibration invocations stopped before an
artifact or metric was written: the first required the deterministic CuBLAS
workspace variable, and the second found that CUDA adaptive area-pooling has
no deterministic backward. The exact 7-to-4 area operation was replaced by
its fixed separable overlap matrix and frozen at `d27a15b`. No target,
candidate, loss, threshold, gate, or evaluation value changed. Calibration
was frozen at `0ae521b` before init36--39 was opened; the numerical diagnosis
was frozen at `efe758f`.

This round was authorized to design and probe targets only. It did not
authorize or perform World Model training, action-conditioned dynamics,
reranking, scoring, controller rollout, C1/C2, D20-v2, seed23/47, success
evaluation, or init40--49 access.

## Data split and exposure

| Partition | Init states | Branch points | Pair/horizon rows | Use |
|---|---|---:|---:|---|
| Calibration | 31--34 | 40 | 960 | all form/rank/resolution/decoder/threshold/ridge choices |
| Frozen development evaluation | 36--39 | 40 | 960 | one-shot metrics, bootstrap, gates, verdict |
| Historical reference | 30/35 | — | — | not fitted or re-evaluated |
| Forbidden | 40--49 | 0 | 0 | never accessed |

Every target is a per-image state `S(I_t)`. Candidate-pair differences are
used only for evaluation. The study reuses already frozen natural RGB-derived
global/patch features, per-object masks, and image-space targets. No simulator
ran. The development evaluation is explicitly not a blind confirmation.

## D0--D5 exact target definitions

| Design | Per-image state | Role | Deployment-compatible here |
|---|---|---|---|
| D0 | 512-D frozen global CLIP | immutable baseline | yes |
| D1 | selected 7x7x1 generic patch-PCA map | low-dimensional generic spatial state | yes |
| D2 | 7x7 relevant-object occupancy + visible area | oracle footprint teacher | no |
| D3 | small natural-patch decoder's 7x7 occupancy + area | spatial proxy only | only if recovery passes; it does not |
| D4 | 512-D global CLIP + D3 | hybrid semantic/spatial target | only if recovery passes; it does not |
| D5 | 512-D global CLIP + D2 auxiliary target | teacher-supervised auxiliary design | training labels only; not enough to authorize GO here |

T1, the full `49x512` natural patch tensor, is an upper-information reference
only and cannot be selected. The result field named `full_patch_reference` in
the probe block refers to the previously frozen 10-D patch summary; the actual
25,088-D patch tensor is used directly only for the separation reference.

## Oracle versus deployment distinction

D2/T3 uses simulator segmentation to define a training and mechanism teacher.
It is never a runtime input. A future model could in principle predict this
head from current observations and actions, but this study did not train such
a dynamics model. D5 therefore establishes an auxiliary-target design worth
retaining, not deployability.

D3/T4 is the required natural-image bridge: frozen patch features enter a
task-agnostic `Linear(512,32)-GELU-Linear(32,1)` per-patch decoder, plus a
pooled area head. Neither object IDs, masks, poses, future state, reward, nor
success enter the decoder. Because this bridge fails the frozen recovery and
primary gates, oracle validity cannot authorize a prototype.

## Calibration choices were frozen before held-out access

### Generic T2 map

Entries are leave-one-init-out G+map improvement over G on calibration data.

| Candidate | Per-image dim. | ObjMask | Area | Worst target |
|---|---:|---:|---:|---:|
| PCA rank 1 | 49 | 7.51% | 5.32% | 5.32% |
| PCA rank 2 | 98 | 7.98% | 5.42% | 5.42% |
| PCA rank 4 | 196 | 5.37% | 5.05% | 5.05% |
| Patch-to-global residual | 49 | 0.88% | 1.48% | 0.88% |

Rank 1 lies within the frozen one-percentage-point tolerance of the best worst
target and is smaller, so it was selected mechanically.

### Footprint resolution

| Resolution | Dim. with area | ObjMask | Area | Worst target |
|---|---:|---:|---:|---:|
| 4x4 | 17 | 33.53% | 75.42% | 33.53% |
| 7x7 | 50 | 48.52% | 75.84% | 48.52% |

Both resolutions show strong oracle value, but 4x4 is not within the frozen
proxy-selection tolerance and loses substantial ObjMask fidelity. The frozen
form is 7x7 plus area.

### Natural proxy form

| Form | Normalized CV loss | Occupancy IoU | Balanced acc. | Positive recall | Area-MAE improvement |
|---|---:|---:|---:|---:|---:|
| 4x4 linear | 121.22 | 0.067 | 0.534 | 0.067 | -3076.05% |
| 4x4 width 32 | 73.75 | 0.387 | 0.692 | 0.392 | -27.06% |
| 7x7 linear | 66.83 | 0.313 | 0.658 | 0.327 | -3076.05% |
| 7x7 width 32 | 21.39 | 0.461 | 0.739 | 0.497 | -27.06% |

The selected form is 7x7, width 32, with an OOF IoU-selected threshold of
`0.20`. The calibration results already warned that localization and soft
calibration were separating; the held-out evaluation applies the frozen form
without repair or recalibration.

## Dimensionality and cost

| Target | Dimensions | float32 memory | Pair-distance work |
|---|---:|---:|---:|
| T0 global CLIP | 512 | 2,048 B | 512 components |
| T1 full patches | 25,088 | 100,352 B | 49 patch cosines |
| T2 selected generic | 49 | 196 B | 49 components |
| T3 oracle footprint | 50 | 200 B | 50 components |
| T4 natural proxy | 50 | 200 B | 50 components |
| T5 global + proxy | 562 | 2,248 B | 512 + 50 components |
| T6 global + footprint auxiliary | 562 | 2,248 B | 512 + 50 components |

T3 uses `0.20%` of T1's dimensions and exceeds T1's raw separation
association with ObjMask. Low dimensionality is therefore not the oracle
bottleneck; natural recoverability is.

## ObjMask and Area fidelity beyond global CLIP

The table uses the frozen global-plus-spatial ridge and branch-block CIs.
D4 is numerically the same probe as D3, and D5 the same as D2; their target
structures differ, not their pair-summary evidence.

| Design | ObjMask gain | 95% CI | ObjMask shuffle | Area gain | 95% CI | Area shuffle |
|---|---:|---|---:|---:|---|---:|
| D1 generic | 5.84% | [2.44%,8.58%] | 10.40% | 2.13% | [-5.36%,7.16%] | 11.99% |
| D2/D5 footprint | 30.45% | [20.15%,40.94%] | 72.09% | 62.64% | [55.81%,67.64%] | 243.54% |
| D3/D4 natural proxy | 4.72% | [1.28%,8.10%] | 7.77% | 4.19% | [-3.45%,10.07%] | 5.60% |

All D2/D5 shuffle interval lower bounds are positive (`48.55%` for ObjMask
and `210.31%` for Area). The oracle target passes the 10% increment, shuffle,
task, horizon, and pair gate for both targets. D1 and D3/D4 do not reach 10%.

## Candidate separation and ordinary-motion normalization

Absolute distances across target families are not compared directly. Each
ratio divides candidate-pair separation by adjacent h1--h8 movement in the
same real candidate trajectories. Hybrid distances normalize global and
spatial components separately using calibration temporal movement.

| Target/design | Candidate mean | Temporal mean | Ratio | ObjMask rho | Area rho | EE rho |
|---|---:|---:|---:|---:|---:|---:|
| D0 global | 0.004575 | 0.006135 | 0.746 | 0.560 | 0.418 | 0.482 |
| T1 full patches | 0.019308 | 0.026038 | 0.742 | 0.697 | 0.504 | 0.652 |
| D1 generic | 0.017644 | 0.022757 | 0.775 | 0.485 | 0.391 | 0.502 |
| D2 footprint | 0.002081 | 0.003677 | 0.566 | 0.785 | 0.724 | 0.642 |
| D3 natural proxy | 0.037481 | 0.042513 | 0.882 | 0.452 | 0.340 | 0.415 |
| D4 global + proxy | 0.850 normalized | 1.038 normalized | 0.819 | 0.554 | 0.410 | 0.495 |
| D5 global + footprint | 0.699 normalized | 1.063 normalized | 0.658 | 0.731 | 0.594 | 0.611 |

Every candidate passes the preregistered scale-control check, so the failure
is not caused by zero candidate separation. D2 is the most object-aligned
target despite having a smaller candidate/temporal ratio than global CLIP.
The full-patch reference is not needed to obtain the strongest object
association.

## Robot leakage and object information beyond EE

| Design | EE gain | ObjMask gain | Area gain | ObjMask residual | 95% CI | Area residual | 95% CI |
|---|---:|---:|---:|---:|---|---:|---|
| D1 generic | 7.60% | 5.84% | 2.13% | 1.15% | [-1.16%,3.03%] | -0.17% | [-6.49%,4.23%] |
| D2/D5 footprint | 15.11% | 30.45% | 62.64% | 18.49% | [10.72%,25.35%] | 55.66% | [48.45%,61.02%] |
| D3/D4 proxy | 3.78% | 4.72% | 4.19% | 2.11% | [-1.23%,4.80%] | 2.45% | [-4.46%,8.00%] |

The oracle footprint remains useful after controlling for EE, so it is not
assigned `ROBOT_COUPLING_LIMIT`. The natural proxy loses precisely this
conditional object value; its residual CIs cross zero.

## Natural-image recovery quality

| Metric | Held-out value | Frozen requirement | Result |
|---|---:|---:|---|
| Occupancy BCE | 0.19565 | descriptive | — |
| Occupancy MSE | 0.04396 | descriptive | — |
| MSE improvement vs calibration-mean map | -1932.02% | >=25% | FAIL |
| IoU at threshold 0.20 | 0.650 | >=0.25 | pass |
| Balanced accuracy | 0.862 | >=0.70 | pass |
| Positive-cell recall | 0.763 | >=0.50 | pass |
| Visible-area MAE | 0.003677 | descriptive | — |
| Area-MAE improvement vs constant | 0.62% | >=20% | FAIL |

The decoder localizes which cells contain an object, but its probabilities do
not reproduce soft occupancy fractions and its area head adds essentially no
held-out value. Binary localization alone cannot redefine the target after
evaluation. `NATURAL_SPATIAL_EXTRACTION_BLOCK` is therefore the correct label.

## Per-horizon robustness

Entries are G+spatial improvement over G in percent.

| h | D2 ObjMask | D2 Area | D3 ObjMask | D3 Area |
|---:|---:|---:|---:|---:|
| 1 | 57.7 | 67.7 | -4.3 | -2.6 |
| 2 | 24.6 | 52.9 | -17.6 | -28.6 |
| 3 | 31.3 | 64.3 | 9.3 | 14.2 |
| 4 | 33.6 | 63.0 | 3.0 | 3.6 |
| 5 | 22.7 | 66.0 | 7.0 | 8.1 |
| 6 | 38.0 | 66.3 | 6.6 | 3.0 |
| 7 | 24.7 | 59.0 | 9.8 | 1.1 |
| 8 | 28.7 | 57.0 | 4.1 | 7.5 |

D2 is positive on all eight horizons for both object targets. D3 regresses at
h1/h2 and never approaches the aggregate 10% requirement consistently.

## Per-task robustness

| Task | D2 ObjMask | D2 Area | D3 ObjMask | D3 Area |
|---:|---:|---:|---:|---:|
| 0 | 32.2 | 58.8 | 4.3 | -4.0 |
| 1 | 40.2 | 75.4 | -18.5 | -10.2 |
| 2 | 35.9 | 62.4 | 9.7 | -28.6 |
| 3 | 38.8 | 44.0 | 9.1 | 1.1 |
| 4 | 22.5 | 63.9 | 6.5 | 3.0 |
| 5 | 23.1 | 51.6 | -4.5 | -23.6 |
| 6 | 16.0 | 40.6 | -2.2 | -2.2 |
| 7 | 60.6 | 71.7 | 11.0 | 18.0 |
| 8 | -7.0 | 42.9 | 5.8 | 15.8 |
| 9 | 61.0 | 80.5 | -7.3 | 7.8 |

D2 ObjMask improves 9/10 tasks and D2 Area 10/10. D3 improves only 6/10 and
5/10, respectively.

## Candidate-pair robustness

| Pair | D2 ObjMask | D2 Area | D3 ObjMask | D3 Area |
|---|---:|---:|---:|---:|
| A0-A1 | 31.1% | 52.7% | 3.4% | -3.9% |
| A0-A2 | 29.8% | 61.6% | 4.0% | 3.9% |
| A1-A2 | 30.4% | 67.2% | 5.5% | 7.3% |

The oracle effect is not an A1-A2-only artifact. The proxy fails the all-pair
Area condition and remains far below 10% even where positive.

## Full-patch/PCA reference and dimensional efficiency

The raw full-patch tensor has ObjMask/Area separation Spearman `0.697/0.504`.
The 50-D oracle footprint reaches `0.785/0.724`, so 25,088 dimensions are not
required for the confirmed object-footprint relation. The selected 49-D PCA
rank-1 map reaches only `0.485/0.391` and improves global ObjMask/Area probes
by `5.84%/2.13%`; generic compression alone is insufficient. The useful
compression is target-specific, not generic patch dimensionality reduction.

## Mechanical decision

| Design | Primary object pass | Residual object pass | Separation | Deployment | Final pass |
|---|---|---|---|---|---|
| D1 generic | no | no | pass | yes | no |
| D2 oracle footprint | ObjMask + Area | both | pass | no | no |
| D3 natural proxy | no | no | pass | no: recovery | no |
| D4 global + proxy | no | no | pass | no: recovery | no |
| D5 global + footprint auxiliary | ObjMask + Area | both | pass | teacher-only | no |

D2 and D5 establish that the 7x7 footprint-plus-area state is a valid and
efficient teacher/auxiliary target. They do not satisfy the deployment rule.
D1 is deployable but lacks object and residual value. D3/D4 are structurally
deployable but fail both natural recovery and target relevance. No candidate
passes all required gates, so no target specification is frozen for World
Model prediction.

Final mechanism labels:

```text
OBJECT_FOOTPRINT_TARGET_VALID
ORACLE_SPATIAL_TARGET_VALID
NATURAL_SPATIAL_EXTRACTION_BLOCK
OBJECT_AUXILIARY_TARGET_VALID
OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED
SPATIAL_TARGET_DESIGN_UNRESOLVED
```

Final authorization:

```text
BLOCK_SPATIAL_WM_PROTOTYPE
```

## Five required answers

**Q1 — Do we need the complete 49-patch high-dimensional tensor?**

No for the oracle object target. A 50-D 7x7 footprint-plus-area state is more
strongly associated with ObjMask/Area than the 25,088-D full-patch separation
and clears every oracle target gate. But a generic 49-D patch-PCA map is not
enough, so this is target-specific compression rather than proof that any
small patch map works.

**Q2 — Can object-footprint and visible-area change be compressed into a
small state?**

Yes as an oracle teacher. The frozen 50-D target produces `30.45%` and
`62.64%` held-out increments, strong shuffles, and broad task/horizon/pair
robustness. The 4x4 form is also informative, but the calibration rule selects
7x7 because it preserves materially more ObjMask information.

**Q3 — Is the state about the object, or mainly coupled robot motion?**

The oracle target contains object information beyond EE motion. Its
EE-conditioned residual gains are `18.49%` and `55.66%` with positive CIs.
The natural proxy does not preserve that conditional value.

**Q4 — Can natural RGB reliably provide the state without runtime simulator
segmentation?**

Not with the frozen small decoder. It localizes object-positive cells well,
but soft occupancy is severely miscalibrated, area recovery is negligible,
and downstream object/residual increments fail. Runtime segmentation remains
forbidden, so this is a real deployment block.

**Q5 — Which target has enough evidence for the next World Model to predict?**

None. D2/D5 are valid teacher designs but not independently deployable; D1 is
deployable but insufficient; D3/D4 fail recovery and relevance. The only
authorized conclusion is `BLOCK_SPATIAL_WM_PROTOTYPE`.

## Candidate headroom remains limited

The frozen C1-v2.1 best-of-three expert-action MSE oracle gain remains about
`1.648%`. It does not enter any target-design gate, but it means that even a
future successful spatial World Model may have limited exploitable action
headroom under the current A0/A1/A2 candidate set.

## Artifacts and finite-value audit

| Artifact | SHA-256 |
|---|---|
| Protocol | `3f12d72ac8f5ab2fcadd9b6a96ad6fea2f9f6ae2bd65ee6de39b3525072498c1` |
| Calibration freeze | `622fe6dd473fccb16f943ed78e3d6f592183181d111bef9869d5d80dad7ea3d0` |
| Calibration weights | `791ce70b3f1313a7a5c6c626e20e0f656cde098b039989c8a698e58bb4acc0c3` |
| Calibration derived | `2baee7dc428a564670046465cfa9851a3443c26c02b8000cd63e81d7cb30222e` |
| Evaluation derived | `7cab9829b8c5b25bb9116acc4216382206d346b7caf7755e207edb3e14670754` |
| Compact diagnosis | `9d788cc7a6d64a29deede070b8b756e48543a6c89b685d25f3151a049cf72e97` |

All admitted inputs, targets, representations, decoder parameters,
predictions, distances, residuals, bootstrap statistics, and reported numeric
arrays are finite. Undefined correlations are represented as `null`; no
missing value is converted to evidence. The ignored sidecars preserve all 960
identities, target matrices, predictions, proxy/teacher maps, and candidate
and temporal distances needed for independent recalculation.

Nine focused tests independently verify the protocol boundary and hashes,
frozen choices and gates, evaluation sidecar identity, finite arrays,
recomputed oracle increments, oracle/proxy distinction, final authorization,
and forbidden-activity mapping. The complete repository suite passes
`135` tests; the remaining 21 messages are dependency deprecation warnings,
not test failures.

## Forbidden-activity audit

No simulator, encoder, CLIP fine-tuning, World Model, GRU, Transformer,
action-conditioned dynamics model, actor, candidate generator, production
scale, reranker, score, or controller was run or modified. No C1/C2, D20-v2,
seed23/47, reward, done, success, task outcome, init40--49, or controller
success evaluation occurred. Simulator masks were used only as frozen offline
teacher labels; they were never natural-proxy inputs or proposed deployment
inputs. No upstream artifact was overwritten.
