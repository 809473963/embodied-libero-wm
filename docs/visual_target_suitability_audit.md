# Global CLIP compresses spatially localized counterfactual change, but the nuisance source is not uniquely attributable

Date: 2026-09-30

Final mechanism verdict:

```text
GLOBAL_CLIP_SPATIAL_COMPRESSION
```

Next-stage authorization: `BLOCK_VISUAL_TARGET_REDESIGN`

Controller authorization: `false`

## Executive answer

The current global 512-D CLIP target is strongly associated with how the image
changes, but the held-out evidence does not establish that those changes are a
stable task-relevant object signal. Patch features explain substantially more
of the real A0/A1/A2 global CLIP separation than full-image pixel summaries:
V4 reduces held-out RMSE from V3's `0.002731` to `0.002051`, a
`24.90%` improvement with branch-bootstrap 95% CI
`[19.17%,29.80%]`. Shuffling only the patch group within task, horizon, and
candidate pair worsens RMSE by `70.66%`, CI `[57.12%,84.94%]`. Mean patch
distance has Spearman `0.826` with global `D_clip`, and the median global
distance is only `0.0574` times the median top-10-patch distance. These frozen
checks mechanically assign `GLOBAL_CLIP_SPATIAL_COMPRESSION`.

The raw image-difference decomposition is visually suggestive but not a valid
unique attribution by itself. Robot-region pixels contain `86.26%` of total
absolute RGB difference on average, versus `9.52%` for relevant objects and
`2.91%` for background. The ten most changed patches are classified as robot
on `54.40%` of rows, background on `44.54%`, and relevant object on only
`1.06%`. Translation- and rotation-action differences also associate more
strongly with robot pixel change (Spearman `0.778` / `0.719`) than with
relevant-object pixel change (`0.433` / `0.369`).

Those facts do **not** clear the preregistered robot-dominance gate. Adding
robot pixel features to V0 improves RMSE by only `1.14%`, below the frozen 5%
minimum, and the robot-group shuffle interval crosses zero. Relevant-object
and visibility groups fail in the same way. Masked-image CLIP supports the
same qualitative tendency—robot removal has median drop `0.329`, whereas
relevant-object removal has median drop `-0.0195`—but masking is an OOD
attribution-style diagnostic and cannot replace the nested/shuffle evidence.

The report therefore cannot claim any of the stronger conclusions
`ROBOT_APPEARANCE_DOMINATES_CLIP_COUNTERFACTUAL`,
`RELEVANT_OBJECT_VISUAL_CHANGE_DOMINATES`,
`OCCLUSION_DRIVES_CLIP_COUNTERFACTUAL`, or
`GLOBAL_CLIP_TARGET_UNSUITABLE_FOR_LOCAL_DYNAMICS`. It also cannot positively
claim that global CLIP aligns with task-relevant visual state. The evidence is
insufficient to keep global CLIP as the core local-dynamics target **by
default**, but it is also insufficient to authorize a replacement. Model,
representation, reranker, and controller changes remain blocked.

## Frozen provenance

The preceding structured-state result is frozen at commit
`776eee731703e3b5350a3a8ce935451ded08aea9`, annotated tags
`local-visual-missing-state-negative` and
`structured-physical-route-blocked`. Its labels remain:

```text
ROBOT_KINEMATICS_ONLY_INSUFFICIENT
NO_TESTED_STATE_EXPLAINS_VISUAL_COUNTERFACTUAL
BLOCK_STRUCTURED_PHYSICAL_WM_ROUTE
```

Protocol `visual-target-suitability-audit-v1` was first committed as
`87a3e66a2470d30e4e774293c66649ecc7b49337` before branch-mask collection or
feature results. Its scientific split, features, 10,000-replicate bootstrap,
thresholds, gates, and decision mapping never changed. Implementation-only
preflights are retained rather than hidden:

- v1 stopped at RGB orientation because an unnecessary vertical flip broke
  direct-render/observation identity;
- v1.1's first calibration branch stopped before artifact output because
  MuJoCo type-6 site IDs were compared with type-5 geom IDs;
- v1.2 restricted semantic matching to type-5 geoms and passed the inventory;
- v1.3 added the repository root to `sys.path` after the analysis command
  stopped before protocol/data/model loading.

The corrected mask protocol was frozen at `f35959c`, calibration masks at
`c72e9ca`, calibration features/models at `9dc99cc`, and held-out masks at
`751e112`. C0-v2, C1-v2.1 candidate generation and production scale,
D13/D14/D15, WM-v2A/B/B.1/C-EJ, matched RGB/CLIP futures, and all upstream
reports/results/checkpoints were hash-checked and not overwritten.

## Data split and one-shot boundary

| Partition | Init states | Branch points | Pair/horizon rows | Use |
|---|---|---:|---:|---|
| Calibration | 31--34 | 40 | 960 | replacement RGB, ridge CV/fit, montage identities |
| Held-out evaluation | 36--39 | 40 | 960 | one-shot metrics, bootstrap, gates, verdict |
| Historical | 30/35 | — | — | reference only; not fitted or evaluated here |
| Forbidden | 40--49 | 0 | 0 | never accessed |

Each branch point retains all A0-A1/A0-A2/A1-A2 pairs and h1--h8. The
bootstrap resamples matched branch points within each task, retaining all 24
correlated rows. No pair/horizon row was randomly split.

## Image preprocessing verification

Every frozen branch sidecar already contained upright `uint8 [128,128,3]`
future RGB and its frozen CLIP latent. Direct `sim.render` RGB was bitwise
identical to the observation without a vertical flip. Both calibration and
held-out feature extraction used frozen
`openai/clip-vit-base-patch32`, revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, the D13 processor, L2
normalization, and an exact B=3 A0/A1/A2 batch for every branch/horizon/variant.

On held-out data, minimum cosine between re-encoded full images and frozen
`real_z` is `1.0`, maximum elementwise difference is `0.0`, and maximum
difference between recomputed and frozen `D_clip` is `9.39e-6`, below the
precommitted `2e-5` gate. The masked replacement color was selected only from
calibration futures and frozen as RGB `[126,119,111]`.

## Mask-generation inventory and semantics

The installed renderer provides native MuJoCo element segmentation with
`(object_type, object_id)`. Only type 5 (geom) pixels enter semantic geom-ID
classes; type 6 (site) pixels fall into background because site and geom IDs
share overlapping integer namespaces. The final per-frame masks form an exact
four-way partition:

```text
robot | relevant object | other object | background
```

Robot IDs come from robot-model and gripper visual/contact geoms. Relevant
objects come only from runtime `parsed_problem["obj_of_interest"]`; across all
ten tasks these resolve to `akita_black_bowl_1` and `plate_1`, with relevant
geom IDs `83--123` and `193--203`. Other object geoms are the remaining
`objects_dict` models; table, fixtures, sites, and empty pixels are background.
No fixed bounding box or external detector is used.

For occlusion, robot geoms are temporarily assigned to otherwise-unused
renderer group 5, that group is hidden for one segmentation render, and all
geom groups and renderer flags are restored. Every calibration and evaluation
branch passes bitwise future RGB replay, exact partition, absence of type-5
robot pixels in the hidden render, non-decreasing relevant-object area,
zero simulator-state change, and exact metadata restoration.

## Real global CLIP baseline

Held-out `D_clip = 1-cos(z_i,z_j)` over 960 rows is:

| n | Mean | Median | p25 | p75 | p90 | p95 | Min | Max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 960 | 0.004575 | 0.003702 | 0.002198 | 0.006012 | 0.009045 | 0.011218 | 0.000223 | 0.021249 |

## Pixel-difference baseline

All pixel metrics use the same upright RGB frames and are computed before any
masking. Associations with `D_clip` are held-out pooled values with
task-stratified branch-bootstrap Spearman intervals.

| Pixel diagnostic | Pearson | Spearman | Spearman 95% CI |
|---|---:|---:|---:|
| Full-image MAE | 0.529 | 0.612 | [0.548,0.669] |
| Full-image RMSE | 0.515 | 0.596 | [0.527,0.659] |
| Unequal-pixel fraction | 0.568 | 0.636 | [0.573,0.690] |
| Robot-region mean | 0.471 | 0.530 | [0.458,0.595] |
| Relevant-object mean | 0.348 | 0.382 | [0.265,0.495] |
| Other-object mean | 0.148 | 0.455 | [0.332,0.562] |
| Background-region mean | 0.508 | 0.595 | [0.523,0.662] |

Global CLIP separation therefore rises with overall image movement, but this
correlation alone does not distinguish task-relevant change from arm edges,
shadows, occlusion boundaries, or mixed patches.

## Region-specific pixel change and energy decomposition

For per-region intensity, masks are candidate unions so change at either
candidate location is counted. For total energy only, pixels are assigned
exclusively with fixed priority robot > relevant > other > background so
contributions sum to one.

| Region | Mean change per union pixel | Mean energy contribution | Median contribution | p95 contribution |
|---|---:|---:|---:|---:|
| Robot | 0.02132 | 86.26% | 88.06% | 97.17% |
| Relevant object | 0.01544 | 9.52% | 8.43% | 23.93% |
| Other object | — | 1.30% | 0.35% | 2.64% |
| Background | — | 2.91% | 2.66% | 5.95% |

Robot pixels dominate raw difference energy in every task under the simple
1.5x descriptive ratio, but the causal/unique attribution gate additionally
requires V1 incremental value and a significant robot-group shuffle. Those
requirements fail below, so no robot-dominance mechanism label is assigned.

## Masked-image CLIP diagnostics

The calibration-frozen color replaces excluded pixels. `Retained` is
`D_variant/D_full`; `drop` is `(D_full-D_variant)/D_full`. All 960 rows clear
the `1e-7` denominator floor.

| Variant | Distance mean | Distance median | Retained median | Drop median |
|---|---:|---:|---:|---:|
| Full | 0.004575 | 0.003703 | 1.000 | 0.000 |
| Robot only | 0.008566 | 0.006857 | 1.923 | -0.923 |
| Relevant object only | 0.008253 | 0.006486 | 1.682 | -0.682 |
| Robot removed | 0.003208 | 0.002599 | 0.671 | 0.329 |
| Relevant object removed | 0.004805 | 0.003779 | 1.019 | -0.019 |
| Background only | 0.003916 | 0.003132 | 0.794 | 0.206 |

The removal pair is directionally consistent with robot appearance mattering
more: removing the robot lowers median distance by 32.9%, while removing
relevant objects does not lower it. The isolation variants often have
retained fractions above one, demonstrating the OOD effect of placing a small
region on a flat calibration color. Consequently absolute masked distances
are not treated as natural-image counterfactuals or sufficient attribution.

| h | Full mean | Robot-only retained | Relevant-only retained | Robot-removed drop | Relevant-removed drop | Background-only retained |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.002699 | 1.810 | 1.439 | 0.423 | 0.089 | 0.644 |
| 2 | 0.003588 | 1.865 | 1.403 | 0.429 | 0.008 | 0.723 |
| 3 | 0.004515 | 2.091 | 1.730 | 0.440 | -0.013 | 0.705 |
| 4 | 0.004638 | 2.209 | 1.814 | 0.256 | -0.043 | 0.944 |
| 5 | 0.005607 | 1.619 | 1.415 | 0.318 | 0.022 | 0.883 |
| 6 | 0.004670 | 2.064 | 1.886 | 0.192 | -0.089 | 0.866 |
| 7 | 0.005535 | 1.722 | 1.768 | 0.325 | -0.038 | 0.811 |
| 8 | 0.005345 | 1.768 | 1.830 | 0.263 | -0.068 | 0.969 |

## Visibility and occlusion audit

Normal relevant-object visible area is compared with a state-preserving
robot-hidden render. Pairwise occlusion-fraction difference has mean
`0.00877`, median `0.00515`, and Spearman `0.266` with `D_clip`, CI
`[0.170,0.356]`. This association is real, but V3 adds only `0.43%` over V2
and its CI crosses zero; visibility-group shuffle also crosses zero. The
complete preregistered occlusion gate therefore fails.

## Calibration-only representative montage

The diagnostic montage was selected before held-out access using nearest
calibration `D_clip` to p10/p50/p90 and first-row tie breaking:

| Selection | Row | Task/init | Pair/horizon | Calibration D_clip |
|---|---:|---|---|---:|
| low p10 | 187 | task1/init34 | A1-A2/h4 | 0.001374 |
| median p50 | 16 | task0/init31 | A1-A2/h1 | 0.003599 |
| high p90 | 622 | task6/init32 | A1-A2/h7 | 0.007490 |

The ignored montage contains candidate i, candidate j, absolute-difference
heatmap, and robot/relevant overlay. No held-out example was hand-picked.

## Patch/token interface availability

The pinned encoder exposes `last_hidden_state[:,1:]`. Diagnostic patch tokens
use:

```text
49x768 vision tokens
  -> frozen post_layernorm
  -> frozen visual_projection
  -> per-token L2 normalization
  -> [49,512] on a 7x7 grid
```

Masks are resized from 128x128 to 224x224 by nearest neighbor and mean-pooled
over exact 32x32 patch cells. This is representation-change analysis, not an
attention attribution claim.

## Patch-region sensitivity

| Patch diagnostic | Mean | Median | Spearman with D_clip | 95% CI |
|---|---:|---:|---:|---:|
| All-patch mean | 0.01931 | 0.01821 | 0.826 | [0.790,0.858] |
| Maximum patch | — | — | 0.535 | [0.474,0.594] |
| Top-10 mean | 0.06726 | 0.06455 | 0.785 | [0.744,0.824] |
| Robot-region mean | 0.04826 | 0.04606 | 0.764 | [0.717,0.810] |
| Relevant-region mean | 0.03351 | 0.02866 | 0.614 | [0.536,0.689] |
| Other-region mean | — | — | 0.536 | [0.451,0.620] |
| Background-region mean | 0.01279 | 0.01170 | 0.787 | [0.743,0.826] |

Among the ten largest patch changes, mean dominant-region fractions are
`54.40%` robot, `1.06%` relevant object, `0%` other object, and `44.54%`
background. A “background” patch may straddle a robot edge because the label
uses the largest mask coverage in a coarse 7x7 cell; this table must not be
read as proof of renderer noise.

## Global-pooling compression audit

The median full global distance is `0.003703`, while median top-10-patch
distance is `0.06455`; their ratio is `0.05736`, below the frozen `0.25`
compression threshold. V4 and patch shuffle both pass, and top-10-patch
association has CI lower `0.744`, above the frozen `0.10` floor. These three
facts assign `GLOBAL_CLIP_SPATIAL_COMPRESSION`.

This label means that spatially resolved changes carry large and independently
predictive signal which the pooled scalar distance mixes/compresses. It does
not mean that global CLIP is independent of patch change—their correlation is
strong—or that a patch target is automatically more task relevant.

## Candidate action geometry association

| Candidate action distance | Outcome | Spearman | 95% CI |
|---|---|---:|---:|
| Translation | Robot pixel mean | 0.778 | [0.737,0.820] |
| Translation | Relevant pixel mean | 0.433 | [0.356,0.514] |
| Translation | Robot patch mean | 0.540 | [0.496,0.582] |
| Translation | Relevant patch mean | 0.333 | [0.258,0.410] |
| Translation | D_clip | 0.385 | [0.324,0.442] |
| Rotation | Robot pixel mean | 0.719 | [0.668,0.769] |
| Rotation | Relevant pixel mean | 0.369 | [0.258,0.474] |
| Rotation | Robot patch mean | 0.481 | [0.390,0.571] |
| Rotation | Relevant patch mean | 0.280 | [0.158,0.399] |
| Rotation | D_clip | 0.359 | [0.277,0.445] |

Production perturbations first map more strongly into robot appearance than
relevant-object appearance. That chain is descriptive and does not establish
that robot appearance uniquely causes `D_clip`; correlated arm, shadow,
occlusion, and neighboring patch changes remain mixed.

## V0--V4 low-capacity probes

All preprocessing and ridge alphas were fitted only on init31--34. Selected
alphas were `1/1/1/1/0.001` for V0--V4. The table is the one-shot init36--39
evaluation.

| Probe | Dim. | RMSE | MAE | R² | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Calibration-mean constant | 1 | 0.003311 | 0.002438 | -0.007 | undefined | undefined |
| V0 full pixel | 4 | 0.002815 | 0.001998 | 0.272 | 0.566 | 0.640 |
| V1 + robot pixel | 10 | 0.002783 | 0.001961 | 0.289 | 0.558 | 0.619 |
| V2 + relevant pixel | 16 | 0.002743 | 0.001929 | 0.309 | 0.573 | 0.628 |
| V3 + visibility/occlusion | 20 | 0.002731 | 0.001927 | 0.315 | 0.578 | 0.628 |
| V4 + patch regional features | 31 | 0.002051 | 0.001450 | 0.613 | 0.788 | 0.845 |

## Nested incremental value

| Child vs parent | Relative RMSE improvement | 95% CI | Frozen gate |
|---|---:|---:|---|
| V0 vs constant | 14.98% | [12.58%,17.41%] | pass |
| V1 vs V0 | 1.14% | [0.22%,1.98%] | fail: below 5% |
| V2 vs V1 | 1.43% | [0.16%,2.62%] | fail: below 5% |
| V3 vs V2 | 0.43% | [-0.11%,0.90%] | fail |
| V4 vs V3 | 24.90% | [19.17%,29.80%] | **pass** |

Robot and relevant-object features correlate with the target but add little
after global pixel change. Patch features add a large held-out increment.

## Group shuffle controls

Each new group is deranged only across branch points with identical task,
horizon, and candidate pair. Parent features and target remain fixed.

| Shuffled group | Original RMSE | Shuffled RMSE | Degradation | 95% CI | Pass |
|---|---:|---:|---:|---:|---|
| Robot pixel in V1 | 0.002783 | 0.002828 | 1.63% | [-0.73%,3.90%] | no |
| Relevant pixel in V2 | 0.002743 | 0.002773 | 1.10% | [-0.69%,2.79%] | no |
| Visibility in V3 | 0.002731 | 0.002743 | 0.42% | [-0.17%,0.99%] | no |
| Patch in V4 | 0.002051 | 0.003500 | 70.66% | [57.12%,84.94%] | **yes** |

Only patch features satisfy both incremental and shuffle requirements.

## Horizon behavior

| h | D_clip mean | Robot energy | Relevant energy | Background energy | Occlusion diff. | Top-10 patch | Top-10 robot | Top-10 relevant | Top-10 background | Pixel rho | Patch rho |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.002699 | 82.1% | 12.1% | 4.3% | 0.0035 | 0.03871 | 57.1% | 1.0% | 41.9% | 0.762 | 0.863 |
| 2 | 0.003588 | 85.3% | 9.9% | 3.4% | 0.0049 | 0.05517 | 56.0% | 0.7% | 43.3% | 0.694 | 0.846 |
| 3 | 0.004515 | 87.0% | 8.9% | 2.8% | 0.0092 | 0.06444 | 55.3% | 0.8% | 43.9% | 0.619 | 0.818 |
| 4 | 0.004638 | 86.9% | 9.1% | 2.8% | 0.0096 | 0.06918 | 55.0% | 0.9% | 44.1% | 0.540 | 0.807 |
| 5 | 0.005607 | 87.1% | 9.0% | 2.7% | 0.0108 | 0.07325 | 54.4% | 0.9% | 44.7% | 0.562 | 0.859 |
| 6 | 0.004670 | 87.5% | 8.9% | 2.4% | 0.0105 | 0.07702 | 53.2% | 1.2% | 45.6% | 0.532 | 0.783 |
| 7 | 0.005535 | 87.1% | 9.2% | 2.4% | 0.0111 | 0.08080 | 52.3% | 1.6% | 46.1% | 0.525 | 0.782 |
| 8 | 0.005345 | 87.2% | 9.1% | 2.5% | 0.0106 | 0.07953 | 51.7% | 1.5% | 46.8% | 0.365 | 0.727 |

Robot energy dominance is already present at h1/h2; later horizons do not
show a transition to relevant-object dominance. Relevant top-patch coverage
remains at or below 1.6%. Patch/global association weakens modestly by h8 but
remains high.

## Per-task behavior

| Task | D_clip mean | Robot energy | Relevant energy | Background energy | Occlusion diff. | Top-10 patch | Top-10 robot | Top-10 relevant | Pixel rho | Patch rho |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.004695 | 78.9% | 15.9% | 4.6% | 0.0134 | 0.07172 | 57.7% | 5.2% | 0.605 | 0.836 |
| 1 | 0.001642 | 86.6% | 10.8% | 2.4% | 0.0050 | 0.04062 | 41.6% | 0.0% | 0.641 | 0.740 |
| 2 | 0.004392 | 83.1% | 13.3% | 2.8% | 0.0052 | 0.07705 | 53.3% | 0.0% | 0.486 | 0.719 |
| 3 | 0.005174 | 81.4% | 15.9% | 2.4% | 0.0050 | 0.07847 | 46.0% | 3.0% | 0.395 | 0.655 |
| 4 | 0.006225 | 93.7% | 3.5% | 2.3% | 0.0085 | 0.07659 | 63.3% | 0.0% | 0.604 | 0.800 |
| 5 | 0.003676 | 79.3% | 8.7% | 3.1% | 0.0085 | 0.06479 | 54.6% | 0.0% | 0.549 | 0.794 |
| 6 | 0.008024 | 84.2% | 11.8% | 3.8% | 0.0113 | 0.07332 | 42.0% | 0.8% | 0.577 | 0.794 |
| 7 | 0.005073 | 92.1% | 4.3% | 2.7% | 0.0127 | 0.07983 | 56.8% | 0.0% | 0.569 | 0.895 |
| 8 | 0.002894 | 91.1% | 6.2% | 2.5% | 0.0102 | 0.04851 | 74.2% | 1.6% | 0.501 | 0.791 |
| 9 | 0.003952 | 92.2% | 4.7% | 2.7% | 0.0077 | 0.06170 | 54.5% | 0.0% | 0.557 | 0.771 |

All ten tasks are descriptively robot-energy dominant; zero are
relevant-object dominant. The preregistered task-dependent label requires at
least three tasks on each side and therefore does not fire. This does not
override the failed aggregate robot attribution gate.

## Candidate-pair behavior

| Pair | D_clip mean | Robot energy | Relevant energy | Background energy | Occlusion diff. | Top-10 patch | Top-10 robot | Top-10 relevant | Pixel rho | Patch rho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0-A1 | 0.004153 | 85.5% | 10.2% | 3.0% | 0.0081 | 0.06182 | 54.4% | 1.1% | 0.602 | 0.801 |
| A0-A2 | 0.004049 | 85.4% | 10.0% | 3.2% | 0.0079 | 0.06160 | 54.5% | 1.0% | 0.619 | 0.838 |
| A1-A2 | 0.005522 | 87.9% | 8.4% | 2.6% | 0.0102 | 0.07836 | 54.3% | 1.1% | 0.547 | 0.813 |

The same qualitative pattern appears in all candidate pairs rather than being
an A1-A2 artifact.

## Comparison with the structured-state audit

| Held-out variable vs D_clip | Spearman | 95% CI |
|---|---:|---:|
| EE translation | 0.435 | [0.337,0.525] |
| EE rotation | 0.315 | [0.225,0.405] |
| Object translation max | 0.234 | [0.101,0.362] |
| Relative translation max | 0.441 | [0.342,0.533] |
| Robot pixel mean | 0.530 | [0.458,0.595] |
| Relevant-object pixel mean | 0.382 | [0.265,0.495] |
| Unequal-pixel fraction | 0.636 | [0.573,0.690] |
| Patch mean | 0.826 | [0.790,0.858] |
| Robot patch mean | 0.764 | [0.717,0.810] |
| Relevant-object patch mean | 0.614 | [0.536,0.689] |

Global CLIP separation is more readily predicted by how the rendered image
and patch representations change than by the tested physical state summaries.
This supports spatial compression/mixing as a real issue. It does not prove
that pixel appearance is causally more important than physical state, because
both reflect the same candidate-induced motion and the low-capacity probes
have different feature geometry.

## Task-relevance interpretation

The evidence for task relevance is mixed and does not clear a positive gate:

- relevant-object pixels account for only `9.52%` of raw energy and `1.06%`
  of top-10 dominant patch labels;
- relevant-object patch distance still correlates with `D_clip` at `0.614`,
  showing it is not absent;
- V2 relevant pixels add only `1.43%` after V1 and fail their shuffle gate;
- the preregistered patch-better test requires the relevant-patch Spearman CI
  lower bound to exceed the relevant-pixel CI upper bound by 0.05. The
  measured gap is about `0.0413`, so it fails rather than being rounded up;
- raw robot energy and masked removal indicate a robot-heavy picture, but the
  nested/shuffle robot attribution fails.

Accordingly, the report cannot say that global CLIP primarily represents
task-relevant world change, nor can it mechanically say robot nuisance alone
dominates.

## Why `GLOBAL_CLIP_TARGET_UNSUITABLE_FOR_LOCAL_DYNAMICS` is not assigned

The strong rule required a nuisance mechanism label
(robot/occlusion/background), no object dominance, and either patch-better or
global-compression evidence. Only the final component passes. Robot,
occlusion, and background may be plausible contributors, but none passes its
complete preregistered attribution gate. Declaring the target unsuitable from
energy ratios or OOD masks alone would overstate the evidence.

Conversely, absence of the unsuitable label is **not** a positive validation
of global CLIP. The audit finds no task-relevant alignment label either. The
correct operational state is `BLOCK_VISUAL_TARGET_REDESIGN`: do not preserve
the target by assumption, and do not replace it without a separately
preregistered experiment that establishes natural-image spatial/task
relevance rather than masked-image artifacts.

## Mechanical decision

| Gate | Result |
|---|---|
| V0 full-pixel explanatory value | pass |
| Robot incremental + shuffle | fail |
| Relevant-object incremental + shuffle | fail |
| Visibility incremental + shuffle | fail |
| Patch incremental + shuffle | pass |
| Robot dominance | fail |
| Relevant-object dominance | fail |
| Occlusion driver | fail |
| Background/rendering sensitivity | fail |
| Patch task-relevance advantage | fail |
| Global spatial compression | **pass** |
| Task-dependent target | fail |
| Global CLIP unsuitable strong rule | fail |

Final mechanism verdict: `GLOBAL_CLIP_SPATIAL_COMPRESSION`.

Next-stage authorization: `BLOCK_VISUAL_TARGET_REDESIGN`.

Controller authorization: `false`.

## NaN/Inf and reproducibility audit

All RGB, semantic masks, global and patch features, feature groups,
predictions, bootstrap inputs, and admitted metrics are finite. The constant
baseline has mathematically undefined Pearson/Spearman and records them as
explicit `null`, not as evidence. All 960 held-out identities are task0--9,
init36--39, three pairs, and h1--h8.

| Artifact | SHA-256 |
|---|---|
| Protocol | `af44dc7cd8a405b2b275de3cfeea85b05fa5ef60f953969f1cd886a1daddf984` |
| Corrected mask inventory | `dfc9536005f0ba0e308b65e723e222c33036bb4c80c61c802e9ea4cb52cef1e6` |
| Calibration mask manifest | `361685d0560cf42c1df8851009927a9ee3cc4a54efe2789e8a0862ef7ba501fa` |
| Calibration masks | `230fd5d6e5e00de57216120e400ff7f3e07b5d9f13ddf2d6458b989759f8a528` |
| Calibration freeze | `c16f6ea07083089c88d35782c812746332e85e74895351d772329b13926c8e9e` |
| Calibration weights | `d41a4a95734c495f5b0a83f45c470f251019719f79729781c26553c2e3d0acf6` |
| Calibration features | `7c9d88ce1c2723e6f88eaffcfc19544c69a1cf4fb16c1dc6d035761f6d608e88` |
| Calibration montage | `c92ada49f553f5182cb1adeeaee20e4800247595539581ce19d25df67e3296e9` |
| Evaluation mask manifest | `2ac56f1bd32d2f021d6ab353fac829569eb743211c0ec89e0a2cf35431c950cb` |
| Evaluation masks | `c8346011ddf4a285d755e92687254019f419c00525ec1a9516b9b54ec57f238c` |
| Evaluation features | `3b4c05dd4e6428d835c0f61c1cc86bb79a963a08b259a2cbef223565959dfbdb` |
| Evaluation metrics | `b24f3dec947ce8317a468ef4b54996cd1709ff4feded6c8a2bc019fc570e027c` |
| Compact diagnosis | `1a6a1b7d8c3846c1d9bd29a42c918c711e3b424833165036a88607b2bd389cbe` |

Large masks/features/weights/montage remain ignored and are traced by the
committed manifests/freeze/result. Compact code, configs, JSON, tests, and
this report are tracked.

## Forbidden-activity audit

No new World Model, CLIP, segmentation network, C0 actor, candidate generator,
production scale, reranker, or controller was trained or modified. No C1/C2,
D20-v2, controller success run, seed23/47, reward, done, success, final task
outcome, fake counterfactual target, mid-episode restore, or init40--49 access
occurred. The simulator was rerun only because the frozen RGB/CLIP sidecars
lacked masks; every replayed RGB was independently matched to the frozen
sidecar, and no simulator-derived future mask is proposed as a deployment
input.
