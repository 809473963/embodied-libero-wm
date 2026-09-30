# Natural object-footprint targets clear the spatial-design gate, but patch attribution remains nonunique

Date: 2026-10-01

Final mechanism labels:

```text
SPATIAL_EE_SIGNAL_CONFIRMED
IMAGE_SPACE_TARGET_MISMATCH_SUPPORTED
SPATIAL_SIGNAL_REMAINS_NONUNIQUE
```

Next-stage authorization: `GO_SPATIAL_TARGET_DESIGN`

Controller authorization: `false`

## Executive answer

The camera-contract-backed restart clears the preregistered gate for a future
spatial-target design. Two natural image-space object targets pass the complete
held-out primary rule. Adding the frozen 10-D patch summary to the matched 10-D
global summary improves object-mask symmetric-difference RMSE by `16.79%`
(branch-bootstrap 95% CI `[12.19%,20.93%]`) and visible-area-difference RMSE by
`11.10%` (CI `[7.00%,14.23%]`). Both exceed the frozen 10% threshold, pass the
full-patch shuffle, improve at least 7/10 tasks and 4/8 horizons, and improve all
three candidate pairs. The object family therefore authorizes
`GO_SPATIAL_TARGET_DESIGN`.

That authorization is deliberately narrower than a unique semantic mechanism
claim. Shuffling only the relevant-object patch scalar does not significantly
worsen either passing object probe: object-mask degradation is `1.23%` with CI
`[-0.84%,3.37%]`, and visible-area degradation is `0.08%` with CI
`[-0.36%,0.51%]`. The eight image-space targets are also correlated; held-out
Spearman is `0.835` between EE displacement and object-mask difference, and
`0.904` between object-centroid and relative-vector displacement. The final
label is therefore `SPATIAL_SIGNAL_REMAINS_NONUNIQUE`, not
`SPATIAL_OBJECT_IMAGE_SIGNAL_CONFIRMED`.

The robot result is independently strong. EE image displacement improves by
`15.03%` (CI `[6.24%,22.68%]`) with a `39.65%` full-patch shuffle degradation.
Robot evidence alone could not authorize design, but it assigns
`SPATIAL_EE_SIGNAL_CONFIRMED`.

The prior 3D target was partly mismatched to the recoverable spatial signal.
The image-space object-mask increment (`16.79%`) and visible-area increment
(`11.10%`) exceed the frozen 3D object-translation increment (`6.90%`) by
`9.89` and `4.21` percentage points. This assigns
`IMAGE_SPACE_TARGET_MISMATCH_SUPPORTED`. The same rescue does not occur for
object-centroid displacement, relative geometry, proximity, or adjacency.

This is a representation and target-design authorization only. It does not
select a final target, train a World Model, change CLIP, modify candidates or
the reranker, or authorize a controller. The frozen candidate set still has
only about `1.648%` imitation-oriented best-of-three headroom.

## Frozen camera contract and preserved old block

The restart consumes only:

```text
artifacts/camera_projection_contract/frozen_contract.json
SHA-256 5eb187c5b3b7eab7a8f2503f9ee8e9dc4bb12e2e65cfb5d017b1913c2d614e0e
status CAMERA_PROJECTION_CONTRACT_VALIDATED
authorization GO_IMAGE_SPACE_TARGET_AUDIT_RESTART
```

The direct `agentview` RGB is unchanged. EE world coordinates are projected as
floating, unclipped coordinates under the frozen map:

```text
u_upright = u_raw
v_upright = H - 1 - v_raw
```

No convention was compared or reselected. The old audit remains permanently
frozen with its original result:

```text
DATA_OR_IMPLEMENTATION_BLOCK
CAMERA_PROJECTION_BLOCK
```

Its 4-pixel broad-robot-mask margin, result, report, and artifacts were not
changed or recomputed. The restart is an independent experiment enabled by the
tight-landmark contract.

## Preregistration, corrections, and freeze order

The restart protocol, targets, 10,000-replicate bootstrap, shuffles, split,
thresholds, and decision mapping were preregistered at commit `8eb1d7f` before
new per-object target collection or result viewing. Implementation-only stops
were retained in the protocol history:

- the exact site marker was occluded after the plan-8 prefix, so the already
  frozen sanity check was moved to the official zero-step init31 observation;
- a combined replay guard was split into named booleans and then bounded
  first-mismatch diagnostics;
- closing a short-lived sanity environment after persistent branch renderers
  had been created invalidated their shared offscreen context, so sanity was
  completed before the branch renderers were created;
- the first analysis command stopped at local-module import, before protocol or
  data loading, and was corrected by adding the repository root to `sys.path`.

No target, representation, threshold, bootstrap draw, model form, or held-out
result was changed by these corrections. The final corrected protocol was
committed as `c9ae65d`, SHA-256
`796a57e50726f5119d1ae3e3bce76b2338d6fe9208c4d5357bc4ef6f54658d95`.

The execution order was:

1. collect and freeze calibration per-object masks;
2. fit all preprocessing, PCA, ridge alphas, and weights on init31--34;
3. commit calibration freeze `21eecbd`;
4. open and collect init36--39 per-object masks;
5. commit the evaluation mask manifest;
6. run the one-shot frozen development evaluation;
7. freeze the immutable diagnosis at commit `acacc81`.

## Development-data boundary

This is a mechanism-development audit, not a blind confirmation.

| Partition | Init states | Branch points | Rows | Use |
|---|---|---:|---:|---|
| Calibration | 31--34 | 40 | 960 | target construction, leave-one-init-out CV, fit, freeze |
| Frozen development evaluation | 36--39 | 40 | 960 | one-shot metrics, bootstrap, gates, verdict |
| Historical reference only | 30/35 | — | — | not fitted or re-evaluated |
| Forbidden | 40--49 | 0 | 0 | never accessed |

Each branch point retains all A0-A1/A0-A2/A1-A2 pairs and h1--h8. Split and
bootstrap unit is the matched branch point within task, retaining its 24
correlated rows. Evaluation is explicitly recorded as “frozen development evaluation, not blind confirmation.”

The only new replay filled data absent from the old sidecars: per-object
natural future masks. Every one of the 80 calibration/evaluation branch points
used three fresh environments, the same official init state, exact 64-step
prefix replay, and copied frozen candidate tensors. All 1,920 candidate/horizon
frames passed exact frozen RGB, merged semantic-mask, EE-state, and camera-matrix
checks before a per-object sidecar was admitted. No reward, done, success, or
task outcome was read.

## Natural-image-only representation contract

All representation features come from the same unmodified natural RGB:

```text
natural RGB
  -> frozen CLIP ViT-B/32 global projection
  -> frozen natural patch-token features
```

No robot-only, object-only, removed-region, flat-color, or masked-CLIP input is
used. Native MuJoCo segmentation is used only to define offline natural-image
targets and patch coverage; it is not an encoder input.

The frozen representation comparison is unchanged:

| Representation | Dimensions | Definition |
|---|---:|---|
| G | 10 | global cosine/L2 plus eight calibration-fitted global-differential PCA scores |
| P | 10 | mean/max/top-5/top-10, robot/relevant/broad-interaction means, centroid x/y, spread |
| G+P | 20 | exact concatenation of G and P |
| G/P PCA | 16 each | equal-rank signed differential PCA, secondary |
| G+P+strict | 21 | G+P plus the preregistered strict interaction-patch scalar, secondary |

All ridge models have intercepts. Alpha was selected only by leave-one-init-out
CV on init31--34 from `[1e-4,1e-3,1e-2,1e-1,1,10]`.

| Target | G α | P α | G+P α | G+P+strict α | G PCA16 α | P PCA16 α | Valid rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| EE | 0.1 | 0.0001 | 0.0001 | 0.0001 | 10 | 10 | 960 |
| ObjCtr | 1 | 1 | 1 | 1 | 10 | 10 | 960 |
| ObjMask | 0.1 | 0.0001 | 0.0001 | 0.1 | 10 | 10 | 960 |
| Area | 1 | 1 | 1 | 1 | 10 | 10 | 960 |
| RelVec | 1 | 1 | 1 | 1 | 10 | 10 | 960 |
| RelDist | 1 | 1 | 1 | 1 | 10 | 10 | 960 |
| Prox | 10 | 10 | 10 | 10 | 10 | 10 | 960 |
| Adj | 1 | 0.001 | 0.01 | 0.01 | 10 | 10 | 960 |

## Image-space target construction

Projected EE coordinates are normalized by image width/height. Relevant objects
come only from runtime `parsed_problem["obj_of_interest"]`; both object slots
are retained. For scalar targets, the calibration-frozen aggregation is the
maximum valid object-wise magnitude.

| Family | Target | Definition |
|---|---|---|
| Robot | EE | candidate-pair projected EE coordinate displacement |
| Object | ObjCtr | maximum centroid displacement for objects visible in both candidates |
| Object | ObjMask | maximum full-image natural per-object mask symmetric-difference fraction |
| Object | Area | maximum absolute visible-area-fraction difference |
| Relative | RelVec | maximum change in `(object centroid - projected EE)` |
| Relative | RelDist | maximum change in EE-to-object-centroid distance |
| Interaction | Prox | maximum change in EE-to-natural-object-mask distance, normalized by image diagonal |
| Interaction | Adj | maximum change in object-boundary pixels adjacent to natural robot mask |

If an object is invisible, visibility is zero; mask and area metrics remain
defined, while centroid and relative targets are missing and never imputed.
Per-object `1-IoU`, bounding boxes, centroids, areas, and visibility are retained
diagnostically. The frozen proximity threshold is 8 pixels, adjacency dilation
is 4 pixels, and the strict interaction patch uses a 16-pixel Gaussian around
the projected EE weighted by relevant-object patch coverage.

Future masks and EE coordinates are offline diagnostic oracles. They are not
deployment inputs for a World Model or controller.

## Camera-contract implementation validity

This check only validates use of the frozen contract; it is not new camera
science.

| Partition | Points | Finite | Positive depth | Inside image | Depth range |
|---|---:|---:|---:|---:|---|
| Calibration | 960 | 100% | 100% | 100% | 0.7055--1.0931 |
| Evaluation | 960 | 100% | 100% | 100% | 0.7195--1.0921 |

The ten calibration zero-step exact-site sanity errors range from `0.0851` to
`0.6258 px`, below the frozen 2-pixel implementation limit.

## Target redundancy matrix

Held-out Spearman correlations show that the target families are not
independent. Abbreviations follow the target table above.

| | EE | ObjCtr | ObjMask | Area | RelVec | RelDist | Prox | Adj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EE | 1.000 | 0.555 | 0.835 | 0.472 | 0.600 | 0.485 | 0.396 | 0.216 |
| ObjCtr | 0.555 | 1.000 | 0.588 | 0.542 | 0.904 | 0.720 | 0.286 | 0.392 |
| ObjMask | 0.835 | 0.588 | 1.000 | 0.591 | 0.585 | 0.462 | 0.338 | 0.333 |
| Area | 0.472 | 0.542 | 0.591 | 1.000 | 0.562 | 0.467 | 0.255 | 0.444 |
| RelVec | 0.600 | 0.904 | 0.585 | 0.562 | 1.000 | 0.820 | 0.270 | 0.427 |
| RelDist | 0.485 | 0.720 | 0.462 | 0.467 | 0.820 | 1.000 | 0.266 | 0.426 |
| Prox | 0.396 | 0.286 | 0.338 | 0.255 | 0.270 | 0.266 | 1.000 | 0.158 |
| Adj | 0.216 | 0.392 | 0.333 | 0.444 | 0.427 | 0.426 | 0.158 | 1.000 |

The high EE--ObjMask and ObjCtr--RelVec correlations prevent interpreting a
family pass as proof of a unique causal or semantic source.

## Raw global-versus-patch associations

Patch mean has higher held-out Spearman correlation than global CLIP distance
for all eight targets. The patch-minus-global interval is positive for seven;
adjacency is the exception.

| Target | Global ρ | Patch ρ | Δρ | 95% CI |
|---|---:|---:|---:|---|
| EE | 0.482 | 0.652 | 0.170 | [0.127,0.213] |
| ObjCtr | 0.406 | 0.460 | 0.054 | [0.011,0.100] |
| ObjMask | 0.560 | 0.697 | 0.137 | [0.089,0.187] |
| Area | 0.418 | 0.504 | 0.086 | [0.034,0.138] |
| RelVec | 0.419 | 0.483 | 0.063 | [0.014,0.113] |
| RelDist | 0.331 | 0.406 | 0.076 | [0.026,0.126] |
| Prox | 0.198 | 0.252 | 0.053 | [0.021,0.087] |
| Adj | 0.371 | 0.416 | 0.045 | [-0.013,0.099] |

Raw association is descriptive. Authorization uses the stricter G+P
increment, bootstrap, full-patch shuffle, and task/horizon/pair robustness.

## Primary probe and shuffle results

| Target | G RMSE | P RMSE | G+P RMSE | G+P vs G | 95% CI | Full-patch shuffle | 95% CI | Primary pass |
|---|---:|---:|---:|---:|---|---:|---|---|
| EE | 0.00182880 | 0.00155575 | 0.00155398 | 15.03% | [6.24%,22.68%] | 39.65% | [24.88%,56.42%] | **yes** |
| ObjCtr | 0.00625980 | 0.00591452 | 0.00588896 | 5.92% | [4.61%,8.37%] | 6.13% | [1.16%,8.86%] | no: <10% |
| ObjMask | 0.00060711 | 0.00049582 | 0.00050517 | 16.79% | [12.19%,20.93%] | 44.42% | [31.64%,59.43%] | **yes** |
| Area | 0.00026858 | 0.00023839 | 0.00023876 | 11.10% | [7.00%,14.23%] | 11.03% | [7.22%,15.61%] | **yes** |
| RelVec | 0.00638316 | 0.00601814 | 0.00598863 | 6.18% | [4.71%,8.97%] | 6.75% | [1.86%,9.34%] | no: <10% |
| RelDist | 0.00542726 | 0.00519847 | 0.00518259 | 4.51% | [3.13%,7.74%] | 4.73% | [1.28%,5.91%] | no: <10% |
| Prox | 0.09334499 | 0.09314354 | 0.09319581 | 0.16% | [-0.09%,0.45%] | 0.06% | [-0.23%,0.39%] | no |
| Adj | 0.01272283 | 0.01117664 | 0.01122566 | 11.77% | [9.54%,13.92%] | 2.96% | [-0.25%,6.53%] | no: shuffle |

The object family passes because ObjMask and Area independently clear the full
primary rule. Relative targets have consistent positive increments and full
shuffles, but neither reaches 10%. Adjacency reaches 10% but fails the
full-patch shuffle.

## Semantic attribution controls

| Target | Relevant-only degradation | 95% CI | Strict-interaction degradation | 95% CI |
|---|---:|---|---:|---|
| EE | 0.03% | [-0.17%,0.22%] | 0.05% | [-0.33%,0.41%] |
| ObjCtr | -0.06% | [-0.27%,0.32%] | 0.20% | [-0.10%,0.53%] |
| ObjMask | 1.23% | [-0.84%,3.37%] | 7.99% | [4.06%,12.09%] |
| Area | 0.08% | [-0.36%,0.51%] | 0.84% | [-0.45%,1.90%] |
| RelVec | 0.03% | [-0.33%,0.73%] | 0.44% | [0.02%,1.03%] |
| RelDist | -0.06% | [-0.23%,0.48%] | 0.30% | [-0.14%,0.86%] |
| Prox | 0.04% | [-0.00%,0.10%] | 0.00% | [-0.00%,0.00%] |
| Adj | -0.24% | [-1.70%,1.26%] | 1.60% | [-1.46%,5.00%] |

The strict scalar carries incremental information for ObjMask and a small
amount for RelVec, but neither result identifies relevant-object patch content
uniquely. For the object family, the preregistered semantic attribution check
is the relevant-only shuffle, and it fails for both passing targets. This is
why the final mechanism label is `SPATIAL_SIGNAL_REMAINS_NONUNIQUE`.

## Equal-rank signed PCA remains negative

The hand-designed invariant/spatial summaries, not generic 16 linear patch
differential directions, carry the positive result.

| Target | Global PCA16 RMSE / ρ | Patch PCA16 RMSE / ρ |
|---|---:|---:|
| EE | 0.001992 / -0.067 | 0.001989 / -0.025 |
| ObjCtr | 0.006524 / 0.020 | 0.006529 / 0.010 |
| ObjMask | 0.000699 / -0.059 | 0.000698 / -0.063 |
| Area | 0.000289 / 0.011 | 0.000289 / -0.010 |
| RelVec | 0.006724 / 0.011 | 0.006729 / 0.009 |
| RelDist | 0.005536 / 0.038 | 0.005537 / 0.058 |
| Prox | 0.093305 / 0.017 | 0.093299 / -0.003 |
| Adj | 0.013049 / -0.006 | 0.013049 / -0.031 |

## Horizon robustness

Entries are G+P relative RMSE improvement over G in percent.

| h | EE | ObjCtr | ObjMask | Area | RelVec | RelDist | Prox | Adj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.7 | 7.3 | 18.7 | 24.1 | 11.7 | 9.5 | 0.5 | 7.7 |
| 2 | 4.9 | 5.4 | -2.7 | 10.1 | 5.7 | 3.3 | -0.2 | 6.3 |
| 3 | 2.9 | 6.0 | 13.9 | 13.6 | 6.0 | 3.9 | -0.1 | 8.3 |
| 4 | -0.5 | 6.0 | 8.4 | 11.9 | 5.3 | 5.7 | 0.1 | 14.0 |
| 5 | 13.3 | 6.5 | 21.3 | 9.7 | 6.4 | 5.6 | -0.1 | 12.4 |
| 6 | 21.7 | 5.0 | 23.4 | 12.1 | 5.4 | 3.2 | 0.3 | 12.2 |
| 7 | 21.2 | 6.5 | 21.2 | 5.2 | 6.2 | 5.4 | 0.4 | 13.8 |
| 8 | 15.7 | 6.3 | 15.2 | 10.2 | 6.9 | 5.6 | 0.1 | 12.1 |

| Horizon group | EE | ObjCtr | ObjMask | Area | RelVec | RelDist | Prox | Adj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h1--2 | 16.4 | 6.1 | 8.0 | 17.1 | 7.9 | 5.1 | 0.1 | 6.8 |
| h3--5 | 6.1 | 6.1 | 14.6 | 11.5 | 6.0 | 4.9 | -0.0 | 12.0 |
| h6--8 | 19.1 | 5.7 | 19.5 | 9.6 | 5.9 | 4.1 | 0.3 | 12.7 |

ObjMask is positive on seven horizons and Area on all eight. Their patterns are
not restricted to immediate h1 change.

## Per-task robustness

Entries are G+P relative RMSE improvement over G in percent.

| Task | EE | ObjCtr | ObjMask | Area | RelVec | RelDist | Prox | Adj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15.8 | 11.5 | 10.2 | -9.8 | 10.8 | 6.6 | -0.7 | 10.9 |
| 1 | -12.0 | 14.0 | -17.5 | 11.2 | 10.5 | 4.5 | -1.4 | 26.4 |
| 2 | 25.7 | 13.5 | 28.5 | -1.2 | 15.6 | 23.1 | -0.6 | -12.1 |
| 3 | 11.9 | 3.1 | 13.8 | 14.8 | 1.2 | -3.5 | 1.2 | -0.4 |
| 4 | 25.0 | 1.9 | 32.8 | 16.7 | 6.6 | 13.0 | -0.0 | 13.8 |
| 5 | 9.1 | 3.5 | -10.1 | 16.2 | 3.9 | 2.4 | 1.3 | -0.6 |
| 6 | 19.1 | 7.3 | 4.2 | 3.3 | 5.8 | -0.2 | 3.6 | -7.1 |
| 7 | 16.1 | 9.9 | 13.9 | 19.3 | 11.5 | 7.8 | -3.1 | 15.7 |
| 8 | 13.2 | 1.6 | 14.8 | -2.1 | 3.1 | -0.8 | -0.3 | 32.7 |
| 9 | 8.8 | -4.6 | 12.0 | 10.6 | 0.0 | 4.9 | 0.1 | 12.0 |

ObjMask is positive on 8/10 tasks and Area on 7/10, exactly clearing the
preregistered task-count rule. The two targets do not fail on the same tasks,
which reinforces the family-level result while also showing heterogeneity.

## Candidate-pair robustness

| Pair | EE | ObjCtr | ObjMask | Area | RelVec | RelDist | Prox | Adj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0-A1 | 3.9 | 6.0 | 15.1 | 9.8 | 6.4 | 4.3 | 0.3 | 11.0 |
| A0-A2 | 5.1 | 6.6 | 17.4 | 10.3 | 5.6 | 5.0 | 0.2 | 11.2 |
| A1-A2 | 20.4 | 5.5 | 17.2 | 11.9 | 6.3 | 4.4 | -0.0 | 12.9 |

Both passing object targets improve on all three pairs. ObjMask clears 10% on
each pair; Area is positive on all three and misses 10% only slightly for
A0-A1. The frozen gate requires positive pair improvement, not 10% in every
pair.

## Image-space versus frozen 3D targets

| Image-space target | Frozen 3D reference | Image-space increment | 3D increment | Difference | Supports mismatch |
|---|---|---:|---:|---:|---|
| ObjCtr | object translation | 5.92% | 6.90% | -0.97 pp | no |
| ObjMask | object translation | 16.79% | 6.90% | +9.89 pp | **yes** |
| Area | object translation | 11.10% | 6.90% | +4.21 pp | **yes** |
| RelVec | relative translation | 6.18% | 8.77% | -2.59 pp | no |
| RelDist | relative translation | 4.51% | 8.77% | -4.26 pp | no |
| Prox | relative translation | 0.16% | 8.77% | -8.61 pp | no |

The supported mismatch is specifically about natural image-space object
footprint and visibility change. It is not a blanket result that every 2D
target is superior to the 3D target.

## Sparse interaction diagnostics

The strict proximity regression is effectively negative: G+P improves only
`0.16%`, its CI crosses zero, and the full-patch shuffle fails. Adding the
strict interaction scalar to G+P changes RMSE from `0.09319581` to
`0.09319560`, which is negligible.

The near-contact classifier is also not an authorization path. The 8-pixel
definition is active on `950/960` rows, so the held-out label is extremely
imbalanced. Patch-only AUROC/balanced accuracy are `0.936/0.781`, but G+P is
only `0.802/0.538`; this secondary diagnostic cannot override the failed
interaction regression and shuffle gates.

## Mechanical decision

| Family | Passing targets | Family pass | Authorization role |
|---|---|---|---|
| Robot | EE | yes | mechanism label only; cannot authorize by itself |
| Object | ObjMask, Area | **yes** | authorizes spatial-target design |
| Relative | none | no | no authorization |
| Interaction | none | no | no authorization |

| Required decision component | Result |
|---|---|
| G+P increment ≥10% with positive CI | EE, ObjMask, Area, Adj pass magnitude/CI |
| Full-patch shuffle with positive CI | EE, ObjCtr, ObjMask, Area, RelVec, RelDist pass |
| ≥7 tasks, ≥4 horizons, all 3 pairs positive | EE, ObjMask, Area pass |
| Complete robot target pass | yes |
| Complete object/relative/interaction target pass | **yes: ObjMask and Area** |
| Relevant-only semantic attribution for passing object targets | fail |
| Strict interaction family pass | fail |

Final mechanism labels:

```text
SPATIAL_EE_SIGNAL_CONFIRMED
IMAGE_SPACE_TARGET_MISMATCH_SUPPORTED
SPATIAL_SIGNAL_REMAINS_NONUNIQUE
```

Next-stage authorization:

```text
GO_SPATIAL_TARGET_DESIGN
```

The GO permits only a separately preregistered design study for a spatial
visual target. Because semantic attribution remains nonunique, that design
must preserve the distinction between object-footprint prediction and generic
correlated spatial motion, and it must retain natural-image, capacity-matched,
shuffle, task/horizon/pair, and held-out controls. It does not authorize
patch-level World-Model training automatically.

`REPRESENTATION_AUDIT_STOP_RECOMMENDED` is not assigned because the frozen
non-robot authorization gate passed. Repeating another similar representation
probe is nevertheless not authorized by this report; the allowed next action
is the separately preregistered spatial-target design only.

## Four required answers

**Q1 — Is the stronger patch signal mainly immediate 2D robot motion?**

No. EE motion is strongly represented and receives its own confirmed label,
but object-mask and visible-area targets also clear the full primary gate over
multiple horizons, tasks, and all candidate pairs. Their high redundancy with
EE motion prevents claiming that the signal is independent of robot motion.

**Q2 — Does the patch representation stably encode relevant-object natural
image-space change?**

At the target level, yes: object mask and area changes pass the complete
frozen gate. At the semantic-attribution level, no unique relevant-object
patch source is established because relevant-only shuffles cross zero. The
correct conclusion is object-family target relevance with nonunique spatial
support.

**Q3 — Does it stably encode EE-object relative or interaction geometry?**

Not under the frozen gate. Relative targets improve by only `4.51--6.18%`, and
interaction targets fail either increment, full shuffle, or robustness. No
relative or interaction mechanism label is assigned.

**Q4 — Did a 2D-representation/3D-target mismatch partly explain the prior
weak relevance result?**

Yes, specifically for object footprint and visibility. Their image-space
increments exceed the frozen 3D object-translation result enough to satisfy
the preregistered mismatch rule. Centroid and relative targets do not support
the same claim.

## Artifacts, finite values, and automated verification

| Artifact | SHA-256 |
|---|---|
| Restart protocol | `796a57e50726f5119d1ae3e3bce76b2338d6fe9208c4d5357bc4ef6f54658d95` |
| Calibration object-mask manifest | `e0a9e802b05019c22bd6d85c1bb3e08e138a6865ab8343a5f2a5af45c7750b9e` |
| Calibration object masks | `1130aa7fdec1fd0cbe2a8458339bd4187252b729c7ce929388e8f50dd8ae321f` |
| Calibration freeze | `b521ca76329542af57e45aaecb45356a3c5fcce74fc5ec4bc85e3fa61c37142c` |
| Calibration weights | `b04e15c6cb6c9cfd97803d804bafb1e2641698d657d58850fa33d18e36e1eff3` |
| Calibration derived sidecar | `48dd842842d26d2a94d009c04b6bc1c526119743be2e9fd1c6bd32b9e8ef0989` |
| Evaluation object-mask manifest | `9519a75766c6e2abb833183017a6356e0cd4e14598ebc698da60db4325bacc48` |
| Evaluation object masks | `c3871bbce09fdf25eb3b3503fa5a24c098d662b600bb2e4da157757bd7bccd7c` |
| Evaluation derived sidecar | `ecd1941ad0b9fb692e0ff99d53710b4ee6e75f161780171d6f9177aa2a22ed8f` |
| Compact diagnosis | `1d9e90627491cb0dace90c7e03948d3d5588913dca8700f7be326272cf244ed9` |

All admitted inputs, targets, representations, weights, predictions,
residuals, bootstrap statistics, projected coordinates, and depths are finite.
The ignored sidecars retain all 960 identities, eight targets and validity
masks, object-wise geometry, G/P/G+P/PCA/strict representations, predictions,
projected EE coordinates, and interaction diagnostics needed for independent
recalculation.

The 14 required protocol tests pass. Seven additional result tests independently
verify artifact hashes, the init-state boundary, finite identities, RMSE and
increment recomputation from the frozen sidecar, the object-family gate,
nonunique semantic attribution, authorization, preserved old block, and
forbidden-activity mapping. Together they pass `21/21`. The complete repository
suite passes `126` tests; the remaining 21 messages are dependency deprecation
warnings, not test failures.

## Forbidden-activity audit

No RGB was modified and no masked-image CLIP inference was run. No encoder,
World Model, actor, candidate generator, production scale, reranker, or
controller was trained or modified. No C1/C2, D20-v2, seed23/47, reward, done,
success, final task outcome, init40--49, or controller-success run occurred.
No upstream artifact, old blocked audit, or frozen camera contract was
overwritten. The only simulator work was the preregistered replay required to
recover missing per-object natural masks, with exact frozen RGB/mask/state
identity checks.
