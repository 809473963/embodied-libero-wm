# Visible area is the minimum valid teacher, while the soft field retains additional object-shape information

Date: 2026-10-01

Final mechanism labels:

```text
VISIBLE_AREA_DOMINANT_TARGET
MINIMAL_CONTINUOUS_FOOTPRINT_VALID
BINARY_COMPONENT_REDUNDANT_GIVEN_AREA
SOFT_OCCUPANCY_ADDS_VALUE_BEYOND_AREA
OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED
```

Selected minimal oracle target: `F1 continuous visible area`

Dimension: `1`

Next-stage authorization: `GO_MINIMAL_SPATIAL_EXTRACTION_AUDIT`

Spatial World Model prototype authorization: `BLOCK_SPATIAL_WM_PROTOTYPE`

Alternative spatial-encoder audit authorization: `false`

Controller authorization: `false`

## Executive answer

The smallest preregistered continuous state is sufficient for one complete
target family: the single exact visible-area fraction. Adding F1 to the frozen
global summary lowers held-out Area RMSE by `59.13%` (task-stratified
branch-bootstrap 95% CI `[51.07%,65.09%]`), retains `94.40%` of F0's Area
gain, and lowers the EE-conditioned Area-residual RMSE by `53.81%` (CI
`[45.75%,59.63%]`). Its shuffle worsens Area RMSE by `214.86%`, CI
`[185.61%,252.85%]`; it improves all ten tasks, all eight horizons, and all
three candidate pairs. The frozen minimum-dimension rule therefore selects F1
and authorizes only `GO_MINIMAL_SPATIAL_EXTRACTION_AUDIT`.

That decision does not mean area replaces the full footprint for every
question. F1 improves ObjMask RMSE by `10.85%`, but retains only `35.63%` of
F0's `30.45%` ObjMask gain. F2, the 49-cell soft map without a separate area
scalar, retains `99.78%` of F0's ObjMask gain and `99.30%` of its Area gain.
Conditioned on area, adding the binary map changes ObjMask RMSE by only
`+0.083%`, CI `[-0.576%,0.683%]`; adding the soft map changes it by
`+21.98%`, CI `[12.08%,31.89%]`. Binary location is redundant once area is
known, while continuous cell occupancy still contains material ObjMask
information.

Centroid and simple moments do not recover that missing ObjMask value. F3--F7
retain only `30.87--32.68%` of F0's ObjMask gain. Adding centroid to area
slightly worsens ObjMask RMSE (`-1.63%`, CI `[-3.79%,0.65%]`), adding spread
changes it by `+0.09%`, and covariance by `-0.06%`. These candidates remain
excellent Area summaries because all contain exact area, but they are not
compact replacements for the complete soft field's ObjMask information.

The operational answer is deliberately narrow: the next experiment should
test natural-RGB recovery of one continuous visible-area scalar. It must not
claim to recover the full soft footprint, and no World Model, alternative
encoder, reranker, or controller is authorized.

## Frozen provenance and authorization boundary

The upstream extraction audit is frozen at commit
`3a22845cf3cd59bdf898a10daa61059e5fab1e93` under annotated tag
`natural-spatial-extraction-block`. Its labels and
`BLOCK_SPATIAL_WM_PROTOTYPE` remain unchanged.

The protocol and implementation were preregistered at commit `0bcd624` before
new calibration or decomposition results. The first calibration exposed a
decision-mapping omission before init36--39 was opened: the conditional soft
occupancy label referenced F1→F3 rather than the intended F1→F0 comparison.
The correction was recorded and committed as `1939eb0`; it changed no
candidate, formula, data, target, ridge model, threshold, bootstrap draw,
shuffle, numeric gate, split, or authorization. Calibration was rebuilt and
frozen at `eaa4521`. The one-shot numerical diagnosis was then frozen at
`565a2c6` before this narrative report was written.

This round decomposes already-frozen oracle labels only. It does not train or
evaluate a natural-image decoder, alternative encoder, or dynamics model.

## Data split and one-shot boundary

| Partition | Init states | Branch points | Pair/horizon rows | Use |
|---|---|---:|---:|---|
| Calibration | 31--34 | 40 | 960 | formula stability, ridge CV/fit, freeze |
| Frozen development evaluation | 36--39 | 40 | 960 | one-shot metrics, bootstrap, gates, verdict |
| Historical reference | 30/35 | — | — | not fitted or re-evaluated |
| Forbidden | 40--49 | 0 | 0 | never accessed |

The split and bootstrap unit is the matched branch point within task, retaining
all three candidate pairs and eight horizons. The evaluation is frozen
development evaluation, not blind confirmation. No simulator ran; the audit
reused frozen per-object masks, G features, ObjMask/Area targets, EE controls,
and residual targets.

## Exact F0--F7 states

All candidates are deterministic per-image states derived from the same 7x7
soft relevant-object occupancy map. Grid-cell centres are normalized to
`[0,1]`; moments are occupancy weighted.

| Candidate | Dim. | Exact state |
|---|---:|---|
| F0 | 50 | 49 soft occupancy fractions + exact visible area |
| F1 | 1 | exact continuous visible-area fraction |
| F2 | 49 | 49 soft occupancy fractions, without exact area |
| F3 | 3 | `[area,cx,cy]` |
| F4 | 5 | `[area,cx,cy,sigma_x,sigma_y]` |
| F5 | 6 | F4 + occupancy-weighted `cov_xy` |
| F6 | 7 | `[area,cx,cy,xmin,xmax,ymin,ymax]` on support `occupancy>0` |
| F7 | 8 | `[area,cx,cy,sigma_x,sigma_y,cov_xy,skew_x,skew_y]` |

For example,

```text
cx = sum_k(o_k x_k) / sum_k(o_k)
sigma_x = sqrt(sum_k(o_k (x_k-cx)^2) / sum_k(o_k))
cov_xy = sum_k(o_k (x_k-cx)(y_k-cy)) / sum_k(o_k)
skew_x = [sum_k(o_k (x_k-cx)^3) / sum_k(o_k)] / sigma_x^3
```

Calibration contained no invisible image and no zero occupancy mass. The
frozen missing rule nevertheless treats zero mass as
`DATA_OR_IMPLEMENTATION_BLOCK`; it never inserts zero, the image centre, or a
previous centroid. F7 was retained because every calibration value was finite;
its maximum absolute state value was `4.5712`.

## Pair summary and calibration

Every candidate uses the same six-slot pair summary: spatial RMSE, spatial
MAE, maximum absolute difference, top-quartile mean, centroid displacement
(zero only when the target has no centroid), and scale difference. Exact area
is used only when present; F2 uses its soft-map mean. Ridge regression with an
intercept standardizes features from calibration only. Alpha is selected by
leave-one-init-out CV on init31--34 from
`[1e-4,1e-3,1e-2,1e-1,1,10]`, with exact ties favoring larger alpha.

The frozen global baseline `G` is the same 10-D global CLIP summary used
upstream. Primary targets are ObjMask and Area; residual targets are the
already-frozen calibration-only `ObjMask~EE` and `Area~EE` residuals. No EE
definition or residual control was refitted here.

The primary gate requires at least 10% improvement with positive CI, a
positive-lower-bound shuffle degradation, at least 7/10 tasks, 4/8 horizons,
and all three pairs positive. The matched residual must improve at least 5%
with positive CI, and the raw gain must retain at least 80% of F0. The selected
candidate is the lowest-dimensional complete pass.

## Held-out G versus G+Fk accuracy

| Rep. | ObjMask RMSE | ObjMask MAE | R² | Spearman | Area RMSE | Area MAE | R² | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G | 0.00060711 | 0.00042368 | 0.227 | 0.546 | 0.00026858 | 0.00018458 | 0.138 | 0.399 |
| F0 | 0.00042227 | 0.00027524 | 0.626 | 0.811 | 0.00010034 | 0.00006808 | 0.880 | 0.875 |
| F1 | 0.00054125 | 0.00037113 | 0.386 | 0.642 | 0.00010977 | 0.00007007 | 0.856 | 0.862 |
| F2 | 0.00042267 | 0.00027550 | 0.625 | 0.811 | 0.00010151 | 0.00006827 | 0.877 | 0.873 |
| F3 | 0.00055005 | 0.00037561 | 0.366 | 0.659 | 0.00010416 | 0.00006841 | 0.870 | 0.875 |
| F4 | 0.00054958 | 0.00037380 | 0.367 | 0.662 | 0.00010315 | 0.00006837 | 0.873 | 0.872 |
| F5 | 0.00054991 | 0.00037411 | 0.366 | 0.661 | 0.00010308 | 0.00006835 | 0.873 | 0.872 |
| F6 | 0.00054671 | 0.00037225 | 0.373 | 0.660 | 0.00010452 | 0.00006864 | 0.869 | 0.874 |
| F7 | 0.00054737 | 0.00037297 | 0.372 | 0.659 | 0.00010387 | 0.00006861 | 0.871 | 0.866 |

## Primary, residual, and F0-retention results

| F | Dim. | ObjMask gain | Area gain | ObjMask residual | Area residual | ObjMask retention | Area retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 | 50 | 30.45% | 62.64% | 18.49% | 55.66% | 100.00% | 100.00% |
| F1 | 1 | 10.85% | 59.13% | 11.78% | 53.81% | 35.63% | 94.40% |
| F2 | 49 | 30.38% | 62.20% | 18.51% | 55.91% | 99.78% | 99.30% |
| F3 | 3 | 9.40% | 61.22% | 10.00% | 55.71% | 30.87% | 97.73% |
| F4 | 5 | 9.48% | 61.59% | 10.16% | 55.30% | 31.13% | 98.33% |
| F5 | 6 | 9.42% | 61.62% | 10.16% | 55.25% | 30.95% | 98.37% |
| F6 | 7 | 9.95% | 61.08% | 11.25% | 55.93% | 32.68% | 97.52% |
| F7 | 8 | 9.84% | 61.33% | 10.82% | 55.83% | 32.32% | 97.90% |

F0 independently reproduces the frozen upstream gains. All F1 confidence
intervals are positive: ObjMask `[4.69%,17.42%]`, Area
`[51.07%,65.09%]`, ObjMask residual `[5.67%,17.15%]`, and Area residual
`[45.75%,59.63%]`. F1 completes the Area gate, but not the ObjMask gate because
its ObjMask retention is far below 80%.

## Spatial shuffle controls

| F | ObjMask degradation | 95% CI | Area degradation | 95% CI |
|---|---:|---|---:|---|
| F0 | 75.78% | [51.85%,111.21%] | 263.99% | [228.77%,305.02%] |
| F1 | 19.73% | [11.87%,28.97%] | 214.86% | [185.61%,252.85%] |
| F2 | 73.62% | [48.07%,110.02%] | 240.06% | [203.47%,284.02%] |
| F3 | 21.51% | [14.74%,30.31%] | 232.39% | [198.71%,274.97%] |
| F4 | 21.81% | [13.90%,31.49%] | 246.10% | [207.08%,295.11%] |
| F5 | 21.15% | [13.30%,31.14%] | 233.20% | [193.95%,282.73%] |
| F6 | 21.33% | [13.72%,31.23%] | 228.38% | [187.58%,277.08%] |
| F7 | 21.04% | [13.81%,30.37%] | 234.46% | [199.21%,279.16%] |

Every primary candidate-sensitive effect has a positive lower bound. The
selection difference is therefore retention and dimensionality, not a failed
shuffle.

## Information-source decomposition

| Nested comparison | ObjMask | 95% CI | Area | 95% CI | ObjMask residual | Area residual |
|---|---:|---|---:|---|---:|---:|
| F1 area → F3 +centroid | -1.63% | [-3.79%,0.65%] | 5.11% | [1.86%,8.42%] | -2.01% | 4.12% |
| F3 → F4 +spread | 0.09% | [-0.55%,0.70%] | 0.97% | [-1.27%,3.06%] | 0.17% | -0.92% |
| F4 → F5 +covariance | -0.06% | [-0.10%,-0.03%] | 0.07% | [-0.02%,0.17%] | 0.00% | -0.12% |
| F1 area → F0 soft+area | 21.98% | [12.08%,31.89%] | 8.59% | [4.92%,11.43%] | 7.60% | 4.02% |
| F1 area → B2 binary+area | 0.08% | [-0.58%,0.68%] | -0.54% | [-1.12%,-0.02%] | -2.28% | -0.96% |
| F2 soft → F0 soft+area | 0.095% | [0.021%,0.184%] | 1.16% | [-0.52%,2.61%] | -0.022% | -0.57% |

### Area-only analysis

F1 is not merely a tautological Area regressor. It also improves ObjMask by
`10.85%`, survives shuffle, and improves ObjMask residual by `11.78%`, so
visible scale tracks object change beyond EE motion. Nevertheless, its ObjMask
retention is only `35.63%`; it cannot replace F0 for footprint-shape fidelity.

### Soft-only analysis

Removing the explicit area scalar from F0 barely changes the result. F2 gains
`30.38% / 62.20%` on ObjMask/Area and `18.51% / 55.91%` on their residuals.
Its raw retentions are `99.78% / 99.30%`. The soft occupancy values already
encode nearly all useful scale through their continuous magnitudes.

### Binary conditional on area

B2 improves ObjMask by `10.92%` in absolute terms, but relative to F1 its
conditional increment is only `0.083%` and the CI crosses zero. Its Area and
both residual comparisons are negative. This assigns
`BINARY_COMPONENT_REDUNDANT_GIVEN_AREA`.

### Shape necessity

Centroid, spread, covariance, support extent, and skew do not approach F0
ObjMask retention. F3--F7 all keep exact area and therefore pass the Area
family, but their ObjMask gains remain `9.40--9.95%`, below the 10% threshold
for most and below 80% retention for all. The audit therefore does not assign
`AREA_CENTROID_TARGET_VALID`, `CONTINUOUS_SHAPE_EXTENT_REQUIRED`,
`SHAPE_DETAIL_NOT_REQUIRED`, or `FULL_SOFT_FOOTPRINT_STRUCTURE_REQUIRED`.
The selected family is Area; within the distinct ObjMask question, the full
soft map still carries information not captured by these moments.

## Candidate separation, temporal normalization, and alignment

| F | Candidate mean | Temporal mean | Ratio | ObjMask ρ | Area ρ | EE ρ | ObjMask-residual ρ | Area-residual ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 | 0.000479 | 0.000848 | 0.565 | 0.799 | 0.700 | 0.644 | 0.581 | 0.493 |
| F1 | 0.000253 | 0.000386 | 0.656 | 0.492 | 0.866 | 0.388 | 0.386 | 0.794 |
| F2 | 0.000484 | 0.000857 | 0.564 | 0.799 | 0.695 | 0.644 | 0.581 | 0.488 |
| F3 | 0.001550 | 0.002239 | 0.692 | 0.576 | 0.793 | 0.469 | 0.409 | 0.684 |
| F4 | 0.001837 | 0.002611 | 0.704 | 0.607 | 0.717 | 0.489 | 0.435 | 0.585 |
| F5 | 0.001585 | 0.002252 | 0.704 | 0.604 | 0.718 | 0.487 | 0.431 | 0.586 |
| F6 | 0.010996 | 0.014614 | 0.752 | 0.447 | 0.550 | 0.381 | 0.300 | 0.441 |
| F7 | 0.027454 | 0.036733 | 0.747 | 0.498 | 0.501 | 0.396 | 0.375 | 0.367 |

All candidate/temporal ratios are material, so no candidate fails from zero
separation. F1 is more strongly aligned with Area than with EE and retains a
high Area-residual correlation, consistent with the residual gate. Absolute
distances are not compared across representations because their coordinates
differ.

## Per-horizon primary evidence

Entries are G+Fk versus G RMSE improvements in percent.

| h | F0 Obj | F1 Obj | F2 Obj | F3 Obj | F4 Obj | F5 Obj | F6 Obj | F7 Obj | F0 Area | F1 Area | F2 Area | F3 Area | F4 Area | F5 Area | F6 Area | F7 Area |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 57.7 | 29.1 | 57.8 | 30.1 | 30.9 | 30.8 | 31.0 | 30.5 | 67.7 | 64.8 | 67.9 | 66.7 | 67.7 | 67.8 | 67.1 | 66.9 |
| 2 | 24.6 | 19.2 | 24.5 | 17.4 | 17.9 | 17.9 | 18.2 | 18.0 | 52.9 | 52.8 | 54.3 | 53.1 | 55.5 | 55.5 | 52.7 | 54.3 |
| 3 | 31.3 | 7.8 | 31.0 | 3.3 | 4.4 | 4.3 | 6.3 | 5.8 | 64.3 | 57.3 | 62.6 | 63.7 | 61.1 | 61.1 | 62.4 | 62.1 |
| 4 | 33.6 | 13.1 | 33.5 | 12.6 | 13.3 | 13.2 | 13.4 | 13.3 | 63.0 | 58.6 | 61.6 | 60.3 | 59.1 | 59.1 | 60.0 | 59.2 |
| 5 | 22.7 | 16.2 | 22.7 | 10.5 | 10.4 | 10.5 | 11.2 | 11.5 | 66.0 | 65.0 | 66.6 | 63.4 | 64.5 | 64.4 | 64.4 | 65.0 |
| 6 | 38.0 | 15.9 | 38.0 | 15.1 | 15.4 | 15.3 | 15.4 | 15.1 | 66.3 | 62.5 | 65.3 | 64.5 | 64.7 | 64.7 | 64.3 | 64.7 |
| 7 | 24.7 | 2.8 | 24.5 | 2.9 | 2.5 | 2.4 | 3.2 | 2.9 | 59.0 | 55.2 | 59.4 | 58.2 | 60.1 | 60.1 | 57.7 | 58.9 |
| 8 | 28.7 | 5.5 | 28.7 | 4.8 | 4.4 | 4.4 | 4.7 | 4.8 | 57.0 | 54.4 | 57.0 | 57.1 | 59.5 | 59.6 | 57.3 | 57.6 |

## Per-horizon residual evidence

| h | F0 ObjR | F1 ObjR | F2 ObjR | F3 ObjR | F4 ObjR | F5 ObjR | F6 ObjR | F7 ObjR | F0 AreaR | F1 AreaR | F2 AreaR | F3 AreaR | F4 AreaR | F5 AreaR | F6 AreaR | F7 AreaR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.5 | 13.1 | 12.6 | 16.0 | 15.1 | 14.9 | 14.5 | 13.9 | 43.0 | 41.7 | 44.2 | 42.3 | 45.5 | 45.6 | 47.5 | 45.3 |
| 2 | 5.5 | 6.4 | 5.5 | 2.7 | 3.4 | 3.3 | 4.6 | 4.2 | 37.4 | 38.5 | 38.5 | 36.8 | 39.3 | 39.3 | 36.9 | 38.8 |
| 3 | 16.8 | 4.8 | 16.9 | 4.1 | 1.7 | 1.7 | 8.5 | 9.7 | 60.1 | 53.1 | 59.5 | 60.4 | 58.0 | 57.8 | 61.4 | 59.4 |
| 4 | 30.2 | 18.0 | 30.2 | 16.8 | 16.5 | 16.4 | 19.0 | 17.5 | 58.8 | 58.5 | 58.3 | 59.3 | 57.7 | 57.6 | 59.4 | 57.8 |
| 5 | 19.5 | 18.3 | 19.4 | 11.0 | 12.5 | 12.6 | 14.9 | 12.8 | 58.3 | 59.9 | 59.4 | 58.8 | 59.6 | 59.6 | 59.0 | 60.4 |
| 6 | 28.5 | 17.5 | 28.5 | 17.6 | 18.8 | 18.7 | 15.4 | 16.1 | 59.0 | 56.9 | 58.6 | 59.2 | 58.1 | 58.0 | 59.3 | 58.9 |
| 7 | 12.5 | 3.3 | 12.6 | 4.0 | 3.4 | 3.4 | 4.7 | 4.7 | 51.9 | 48.7 | 52.4 | 51.6 | 51.3 | 51.2 | 50.9 | 52.0 |
| 8 | 12.9 | 9.6 | 12.9 | 7.1 | 7.5 | 7.6 | 8.0 | 7.6 | 51.5 | 48.9 | 52.4 | 50.9 | 51.3 | 51.3 | 50.8 | 51.4 |

F1 Area and Area-residual improvements are positive at every horizon. The
ObjMask loss from simplifying F0 is most visible at h3 and h7--h8.

## Per-task primary evidence

| Task | F0 Obj | F1 Obj | F2 Obj | F3 Obj | F4 Obj | F5 Obj | F6 Obj | F7 Obj | F0 Area | F1 Area | F2 Area | F3 Area | F4 Area | F5 Area | F6 Area | F7 Area |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32.2 | 10.1 | 31.9 | 13.5 | 14.1 | 14.0 | 15.7 | 14.7 | 58.8 | 60.0 | 58.7 | 61.5 | 63.5 | 63.6 | 61.7 | 59.3 |
| 1 | 40.2 | 17.2 | 40.5 | 16.0 | 15.8 | 15.9 | 16.2 | 16.0 | 75.4 | 76.5 | 72.3 | 76.0 | 75.9 | 75.9 | 76.2 | 76.3 |
| 2 | 35.9 | -5.7 | 35.9 | 0.2 | 0.3 | 0.1 | -0.5 | -1.1 | 62.4 | 70.0 | 61.6 | 69.9 | 69.9 | 69.8 | 70.3 | 70.8 |
| 3 | 38.8 | 3.7 | 38.8 | 0.4 | 0.9 | 0.8 | 0.3 | 0.7 | 44.0 | 35.4 | 41.7 | 33.9 | 34.1 | 34.1 | 34.6 | 34.3 |
| 4 | 22.5 | 9.9 | 22.6 | 7.5 | 7.3 | 7.3 | 7.7 | 7.7 | 63.9 | 61.1 | 64.1 | 63.8 | 65.2 | 65.4 | 63.5 | 64.1 |
| 5 | 23.1 | -3.4 | 23.0 | 1.5 | 1.1 | 1.0 | 1.2 | 0.2 | 51.6 | 49.8 | 51.9 | 52.6 | 50.8 | 50.8 | 51.8 | 45.7 |
| 6 | 16.0 | 5.0 | 15.3 | -2.8 | -5.9 | -5.9 | -0.5 | -0.9 | 40.6 | 28.7 | 39.9 | 39.0 | 41.9 | 41.9 | 37.8 | 41.8 |
| 7 | 60.6 | 39.7 | 60.6 | 47.5 | 47.0 | 46.8 | 46.5 | 46.4 | 71.7 | 70.1 | 72.0 | 70.8 | 69.7 | 69.7 | 70.7 | 70.0 |
| 8 | -7.0 | 2.6 | -7.0 | 1.8 | 1.8 | 1.8 | 1.8 | 1.9 | 42.9 | 42.2 | 43.1 | 43.9 | 43.6 | 43.5 | 43.1 | 43.8 |
| 9 | 61.0 | 43.2 | 61.1 | 27.9 | 34.6 | 34.5 | 34.3 | 34.1 | 80.5 | 81.3 | 80.4 | 78.7 | 78.4 | 78.4 | 79.4 | 78.9 |

F1 Area is positive on 10/10 tasks. Its ObjMask effect is heterogeneous and
regresses tasks 2 and 5, reinforcing why only the Area gate selects it.

## Per-task residual evidence

| Task | F0 ObjR | F1 ObjR | F2 ObjR | F3 ObjR | F4 ObjR | F5 ObjR | F6 ObjR | F7 ObjR | F0 AreaR | F1 AreaR | F2 AreaR | F3 AreaR | F4 AreaR | F5 AreaR | F6 AreaR | F7 AreaR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13.4 | 4.4 | 13.5 | 4.7 | 4.0 | 4.1 | 6.2 | 6.3 | 51.2 | 51.3 | 52.1 | 49.9 | 53.7 | 53.6 | 54.2 | 53.0 |
| 1 | 7.8 | 3.2 | 8.1 | 6.8 | 4.9 | 5.2 | 3.4 | 5.8 | 58.0 | 57.4 | 60.9 | 58.4 | 57.2 | 57.1 | 55.4 | 56.7 |
| 2 | -4.0 | 9.0 | -4.1 | 15.7 | 13.1 | 13.0 | 12.0 | 14.3 | 51.9 | 53.8 | 50.6 | 49.9 | 49.6 | 49.4 | 49.7 | 51.0 |
| 3 | 30.0 | 2.5 | 29.9 | -5.1 | -4.5 | -4.6 | 0.3 | -2.0 | 30.8 | 32.3 | 30.0 | 31.9 | 31.5 | 31.5 | 32.0 | 31.4 |
| 4 | 13.8 | 7.6 | 13.8 | 5.3 | 6.2 | 6.2 | 5.6 | 5.0 | 45.7 | 40.6 | 46.8 | 44.9 | 44.6 | 44.6 | 44.9 | 44.9 |
| 5 | 5.8 | 12.9 | 5.8 | 11.0 | 12.5 | 12.6 | 11.0 | 9.7 | 41.8 | 42.1 | 41.6 | 40.9 | 39.4 | 39.3 | 40.6 | 35.5 |
| 6 | 13.2 | 6.9 | 13.3 | 4.1 | 3.5 | 3.5 | 9.3 | 7.7 | 40.9 | 30.5 | 41.9 | 42.1 | 43.8 | 43.6 | 43.8 | 46.9 |
| 7 | 39.1 | 30.7 | 39.0 | 33.8 | 33.6 | 33.5 | 30.7 | 31.3 | 65.5 | 67.8 | 65.5 | 66.0 | 63.3 | 63.4 | 65.6 | 64.3 |
| 8 | -5.5 | -6.6 | -5.4 | 4.2 | -0.7 | -0.6 | 2.5 | 1.6 | 38.3 | 33.9 | 39.2 | 38.6 | 38.5 | 38.4 | 36.2 | 37.8 |
| 9 | 37.7 | 36.6 | 37.7 | 29.1 | 31.0 | 30.8 | 33.8 | 35.0 | 79.5 | 79.8 | 79.6 | 79.3 | 78.5 | 78.6 | 79.6 | 79.2 |

## Candidate-pair robustness

| Pair | F0 Obj | F1 Obj | F2 Obj | F3 Obj | F4 Obj | F5 Obj | F6 Obj | F7 Obj | F0 Area | F1 Area | F2 Area | F3 Area | F4 Area | F5 Area | F6 Area | F7 Area |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0-A1 | 31.1 | 8.1 | 31.0 | 6.2 | 6.0 | 6.0 | 6.7 | 6.8 | 52.7 | 48.0 | 53.1 | 50.3 | 51.1 | 51.1 | 49.8 | 50.1 |
| A0-A2 | 29.8 | 12.7 | 29.8 | 9.1 | 9.3 | 9.2 | 10.4 | 10.1 | 61.6 | 59.2 | 61.6 | 59.8 | 60.9 | 61.0 | 60.2 | 60.7 |
| A1-A2 | 30.4 | 11.2 | 30.4 | 10.8 | 10.9 | 10.8 | 11.0 | 10.9 | 67.2 | 63.7 | 66.2 | 66.5 | 66.2 | 66.3 | 66.2 | 66.3 |

| Pair | F0 ObjR | F1 ObjR | F2 ObjR | F3 ObjR | F4 ObjR | F5 ObjR | F6 ObjR | F7 ObjR | F0 AreaR | F1 AreaR | F2 AreaR | F3 AreaR | F4 AreaR | F5 AreaR | F6 AreaR | F7 AreaR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0-A1 | 15.1 | 5.4 | 15.1 | 3.2 | 2.9 | 2.9 | 5.6 | 5.0 | 46.1 | 44.8 | 47.1 | 46.6 | 46.7 | 46.7 | 46.5 | 46.2 |
| A0-A2 | 22.3 | 14.2 | 22.3 | 11.1 | 11.5 | 11.5 | 12.3 | 11.9 | 57.2 | 56.0 | 57.6 | 57.2 | 57.4 | 57.4 | 58.0 | 57.9 |
| A1-A2 | 18.3 | 14.0 | 18.3 | 13.1 | 13.4 | 13.4 | 13.8 | 13.4 | 58.7 | 56.3 | 58.5 | 58.5 | 57.6 | 57.6 | 58.7 | 58.7 |

F1 is positive on all three raw target pairs and all three residual pairs.

## Dimensional efficiency

Descriptively, ObjMask/Area gain per target dimension is `10.85/59.13`
percentage points for F1, `10.13/20.41` for F3, `1.90/12.32` for F4,
`1.57/10.27` for F5, `1.42/8.73` for F6, and `1.23/7.67` for F7. F2's
values are `0.62/1.27` and F0's `0.61/1.25`. These ratios do not enter the
authorization; they only show why F1 is the efficient Area teacher.

## Mechanical decision

| Candidate | Complete ObjMask gate | Complete Area gate | Dimension | Selection status |
|---|---|---|---:|---|
| F0 | reference only | reference only | 50 | not eligible by `<F0` rule |
| F1 | no: 35.63% retention | **yes** | 1 | **selected** |
| F2 | yes | yes | 49 | passes but not minimal |
| F3 | no | yes | 3 | passes Area but not minimal |
| F4 | no | yes | 5 | passes Area but not minimal |
| F5 | no | yes | 6 | passes Area but not minimal |
| F6 | no | yes | 7 | passes Area but not minimal |
| F7 | no | yes | 8 | passes Area but not minimal |

Selected minimal oracle target:

```text
Name: F1 continuous visible area
Dimension: 1
Exact formula: union relevant-object visible pixels / (128*128)
ObjMask gain: 10.85% [4.69%,17.42%]
Area gain: 59.13% [51.07%,65.09%]
ObjMask residual: 11.78% [5.67%,17.15%]
Area residual: 53.81% [45.75%,59.63%]
Retention vs F0: ObjMask 35.63%; Area 94.40%
Task robustness: Area 10/10 positive
Horizon robustness: Area 8/8 positive
Pair robustness: Area 3/3 positive
```

The selection is unique because no representation can have fewer than one
dimension. It is selected through the Area target only. F2 remains the
diagnostic answer for preserving full ObjMask information, but is not the
minimum candidate under the preregistered “one complete target” rule.

## Seven required answers

**Q1 — How much of F0's value comes from continuous visible area?**

Area alone retains `94.40%` of F0's Area gain and `96.66%` of its Area-residual
gain. It retains only `35.63%` of F0's ObjMask gain and `63.73%` of its
ObjMask-residual gain. Area dominates scale prediction, not the entire
footprint-shape signal.

**Q2 — What remains after deleting visible area from the 49-D soft map?**

Almost everything: F2 gains `30.38% / 62.20%` on ObjMask/Area and
`18.51% / 55.91%` on their residuals. It retains `99.78% / 99.30%` of F0's
raw gains. Continuous occupancy already carries scale.

**Q3 — Once area is known, how much does binary location add?**

Essentially none. B2 over F1 changes ObjMask by `+0.083%`, with a CI crossing
zero, and changes Area by `-0.54%`. The binary component is redundant given
area.

**Q4 — Does adding centroid to area recover F0's ObjMask value?**

No. F3 reaches `9.40%` ObjMask gain and `30.87%` retention, slightly below F1;
the conditional F1→F3 ObjMask change is `-1.63%`.

**Q5 — Are spread, extent, or covariance needed?**

Not for the selected Area teacher, and they do not recover the missing ObjMask
value. F4--F7 retain only `30.95--32.68%` of F0's ObjMask gain. Spread and
covariance add approximately zero after area+centroid.

**Q6 — What is the minimum continuous state that retains at least 80% of one
F0 target and clears every original gate?**

One dimension: exact continuous visible area. It retains `94.40%` of F0's Area
gain and passes primary, residual, shuffle, task, horizon, and pair gates.

**Q7 — What should the next natural-extraction audit target?**

The 1-D continuous visible-area scalar. The future audit must separately state
that this does not recover the soft field's additional ObjMask information.

## Final decision and authorization

Final mechanism labels:

```text
VISIBLE_AREA_DOMINANT_TARGET
MINIMAL_CONTINUOUS_FOOTPRINT_VALID
BINARY_COMPONENT_REDUNDANT_GIVEN_AREA
SOFT_OCCUPANCY_ADDS_VALUE_BEYOND_AREA
OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED
```

Next-stage authorization:

```text
GO_MINIMAL_SPATIAL_EXTRACTION_AUDIT
```

Persistent block:

```text
BLOCK_SPATIAL_WM_PROTOTYPE
```

The GO authorizes only a separately preregistered natural-RGB extraction audit
for the frozen 1-D F1 target. It does not authorize a new encoder, natural
decoder choice in this round, World Model, reranker, or controller.

## Artifacts, finite values, and automated verification

| Artifact | SHA-256 |
|---|---|
| Protocol | `dde748f90ebda8c052e3880bcaa4d3151e0a87e7c6f8c3742251d67e0dc5acf4` |
| Calibration freeze | `f8516574faa3a730b566d2c8b51f4199fa8d074faad84c3cc889803c10924bf4` |
| Calibration weights | `f0877cf62985a2b35c441e5cfbfd7086811661c42654ec0d3013e608a8a11e9c` |
| Calibration derived | `ce05c63d0a50a8125500469831e6af4c4b99d0745e4299f1e90230e0a122b52d` |
| Evaluation derived | `3bf4882e311df0e2aca9482c46c998203f8d0ca0b5fe72230c0b10024d936602` |
| Compact diagnosis | `e56a515c5929cf8b3dfdd908602f16f732e5cb2dafb9b887604e4a567abc53c9` |

All admitted states, summaries, targets, predictions, residuals, distances,
correlations, bootstrap statistics, and sidecar arrays are finite. Undefined
correlations would remain explicit `null`; none was converted into evidence.
The ignored sidecars preserve all 960 identities, F0--F7 summaries and states,
targets, and frozen predictions required for independent recomputation.

Six protocol tests and seven result tests independently verify hashes,
partition boundaries, exact dimensions/formulas, fixed six-slot summaries,
finite arrays, every raw/residual increment, upstream F0 reproduction, F1's
unique minimum selection, conditional binary/soft conclusions, authorization,
and forbidden activity. The complete repository suite passes `162` tests; the
remaining 21 messages are dependency deprecation warnings, not test failures.

## Forbidden-activity audit

No simulator, modified RGB, masked-image inference, natural decoder, SAM,
DINO, SigLIP, CLIP variant, new encoder, CLIP fine-tuning, PCA target,
autoencoder, learned latent, World Model, GRU, Transformer, CNN,
action-conditioned dynamics model, actor, candidate generator, production
scale, reranker, score, or controller was run or modified. No C1/C2, D20-v2,
seed23/47, reward, done, success, final task outcome, init40--49, or controller
success evaluation occurred. Frozen masks were offline oracle labels only and
were never proposed as deployment inputs. No upstream artifact was overwritten.

The frozen A0/A1/A2 best-of-three expert-action MSE oracle gain remains about
`1.648%`. It does not enter any target gate; candidate headroom remains a
separate limitation.
