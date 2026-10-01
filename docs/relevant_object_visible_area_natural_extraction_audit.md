# Natural features recover visible-area level but not candidate-sensitive variation

Date: 2026-10-01

Final mechanism labels:

```text
AREA_REGRESSION_ACCURATE_BUT_DECISION_SIGNAL_INSUFFICIENT
NATURAL_SPATIAL_EXTRACTION_BLOCK
```

Next-stage authorization: `BLOCK_SPATIAL_WM_PROTOTYPE`

Selected natural extractor: `none`

Alternative spatial-encoder audit authorization: `false`

Controller authorization: `false`

## Executive answer

Frozen natural CLIP features can recover the absolute relevant-object visible
area substantially better than a calibration-mean constant, but they do not
recover the candidate-sensitive part required by the frozen F1 target design.
The calibration-selected visual-only E0 extractor improves held-out area MAE
by `40.75%` (task-stratified branch-bootstrap 95% CI
`[30.88%,49.27%]`), and the selected visual-plus-language E1 extractor improves
it by `43.57%` (CI `[29.29%,55.81%]`). Their image-level R² values are
`0.665` and `0.665`, and neither suffers aggregate dynamic-range collapse.

That success is mostly scene/task-level area recovery rather than local
A0/A1/A2 counterfactual fidelity. Signed pair-difference Spearman is only
`0.075` for E0 and `0.039` for E1. Direction accuracy after the frozen
near-zero exclusion is `50.49%` (CI `[46.32%,55.24%]`) and `51.14%` (CI
`[48.13%,54.03%]`). Mean three-way candidate rank Spearman is `0.033` and
`0.023`, with both confidence intervals crossing zero.

Replacing oracle F1 with predicted area confirms the break. Adding E0
predicted area to G improves held-out Area RMSE by only `0.35%`, CI
`[-0.92%,1.46%]`; E1 improves it by `0.06%`, CI `[-1.32%,1.16%]`.
Their shuffle intervals cross zero, and Area-residual gains are `-2.11%` and
`+0.24%`, respectively. Neither is positive across enough tasks or all three
candidate pairs. For comparison, frozen oracle F1 improves Area by `59.13%`
and its EE-conditioned residual by `53.81%`.

The formal result is therefore
`AREA_REGRESSION_ACCURATE_BUT_DECISION_SIGNAL_INSUFFICIENT`, not
`NATURAL_VISIBLE_AREA_EXTRACTION_FAILED`: ordinary natural features do predict
the absolute visible-area level. But they do not preserve the variation that
distinguishes the frozen candidates, so `NATURAL_SPATIAL_EXTRACTION_BLOCK` and
`BLOCK_SPATIAL_WM_PROTOTYPE` remain in force. This result does not isolate a
frozen-CLIP information limit and does not authorize another encoder audit.

## Frozen provenance and authorization boundary

The upstream continuous-footprint decomposition is frozen at commit
`863f32450a1a45545ef6fdd283618245f16f0a75` under annotated tag
`continuous-footprint-decomposition-resolved`. Its conclusions remain:

```text
VISIBLE_AREA_DOMINANT_TARGET
MINIMAL_CONTINUOUS_FOOTPRINT_VALID
BINARY_COMPONENT_REDUNDANT_GIVEN_AREA
SOFT_OCCUPANCY_ADDS_VALUE_BEYOND_AREA
OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED
GO_MINIMAL_SPATIAL_EXTRACTION_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

The extraction protocol and first implementation were committed at `003264e`
before calibration or evaluation. A calibration-only preflight exposed a
training-scale defect: a sigmoid initialized near `0.5` had to fit targets
centered near `0.026`, leaving the linear control outside a meaningful range.
Before any init36--39 access, target mean/std normalization, inverse mapping,
and a fixed `[0,1]` clamp were committed at `91ee6e1`. Inputs, candidates,
widths, parameter ceilings, MSE loss, optimizer, epochs, split, selection
order, downstream probes, bootstrap, shuffles, thresholds, labels, and
authorizations did not change. Calibration was rebuilt and frozen at
`d3ac00a`; init36--39 was then evaluated once and the numerical diagnosis was
frozen at `c7f8cb7`.

This round authorized only natural extraction of the already selected 1-D F1
state. It did not authorize target redesign, soft-footprint reconstruction,
another encoder, a World Model, reranker, score, controller, or task-success
experiment.

## Data split and exposure

| Partition | Init states | Branch points | Image states | Pair rows | Use |
|---|---|---:|---:|---:|---|
| Calibration | 31--34 | 40 | 960 | 960 | normalization, OOF form/epoch selection, tie threshold, ridge fit, freeze |
| Frozen development evaluation | 36--39 | 40 | 960 | one-shot metrics, bootstrap, gates, verdict |
| Historical reference | 30/35 | — | — | — | not fitted or re-evaluated |
| Forbidden | 40--49 | 0 | 0 | 0 | never accessed |

This is a frozen development evaluation, not a blind confirmation. The split
and bootstrap unit is the matched branch point within task, retaining its
three candidates and eight horizons. No simulator ran; all natural features,
labels, identities, and downstream targets already existed in frozen
sidecars.

## Exact F1 target and oracle evidence

For each unmodified natural future image, the target is exactly:

```text
F1 = union pixels of runtime parsed_problem["obj_of_interest"] / 16384
```

The two object slots are retained before their mask union. Across this suite,
runtime resolution identifies the black bowl and plate, but names, masks, IDs,
poses, and simulator state never enter an extractor. Segmentation is a frozen
offline label only.

The upstream oracle evidence is strong and remains unchanged:

| Frozen oracle F1 result | Value |
|---|---:|
| Area RMSE improvement over G | 59.13% |
| Area improvement 95% CI | [51.07%,65.09%] |
| Area-residual improvement | 53.81% |
| Shuffle degradation | 214.86% |
| Robustness | 10/10 tasks, 8/8 horizons, 3/3 pairs |

F1 solves only the visible-scale family. It retains `94.40%` of the full F0
Area gain but only `35.63%` of F0's ObjMask gain. It does not recover the
additional shape information in the 49-D soft occupancy field.

## Natural-input contract

All extractor inputs derive from the same upright, unmodified natural RGB:

```text
natural RGB
  -> frozen CLIP ViT-B/32
     V0: full-image global projection [512]
     V1: arithmetic mean of 49 natural patch projections [512]
```

V1 is not re-normalized after averaging. No pooling form was searched on
init36--39. The two conditions are:

| Condition | Input |
|---|---|
| E0 | frozen visual feature only |
| E1 | same visual feature plus the existing frozen 512-D task-language feature |

Language was read from the frozen D13 feature store; it was not re-encoded or
fine-tuned. Masks, depth, state, proprioception, object/EE pose, contact,
future action, candidate identity, and future simulator state are absent.
There was no masked, cropped, object-only, or modified-RGB inference.

## Extractor architectures and calibration

| Head | Architecture | E0 parameters | E1 parameters |
|---|---|---:|---:|
| M1 | Linear(d,1) | 513 | 1,025 |
| M2 | Linear(d,64)-GELU-Linear(64,32)-GELU-Linear(32,1) | 34,945 | 67,713 |

All features use calibration-only mean/std. The raw area target uses the
training-fold mean/std for optimization, is inverse-transformed, and is
clamped to `[0,1]`. AdamW uses learning rate `0.003`, weight decay `1e-4`, MSE,
and at most 240 epochs. Leave-one-init-out calibration selects the best MAE
epoch in each fold; the round-half-up median trains the final head.

Within E0 and E1 separately, the frozen lexicographic rule minimizes OOF area
MAE, then signed pair-difference RMSE, then maximizes calibration downstream
Area gain, then minimizes parameters. It selected:

| Condition | Frozen representative | Epoch | Parameters | Calibration OOF MAE | Pair Δ RMSE |
|---|---|---:|---:|---:|---:|
| E0 | V0 global + M2 | 27 | 34,945 | 0.001660 | 0.000833 |
| E1 | V0 global + M2 | 34 | 67,713 | 0.001490 | 0.000669 |

The calibration-mean baseline is `0.026234`. On evaluation it has MAE
`0.003700` and RMSE `0.004770`. The signed-direction exclusion threshold is
the calibration 10th percentile of strictly positive `|ΔA|`, frozen at
`6.1035e-05`.

## Absolute natural-area recovery

All eight calibration-frozen candidates were evaluated for transparency.
Only the two representatives above enter the authorization gate.

| Frozen candidate | MAE | RMSE | R² | Pearson | Spearman | MAE vs constant | Pair Δ RMSE | Pair Δ ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 V0 M1 | 0.002548 | 0.003207 | 0.544 | 0.759 | 0.683 | 31.13% | 0.001042 | -0.010 |
| **E0 V0 M2 selected** | **0.002192** | **0.002751** | **0.665** | **0.821** | **0.793** | **40.75%** | **0.000793** | **0.075** |
| E0 V1 M1 | 0.002273 | 0.003014 | 0.598 | 0.781 | 0.739 | 38.55% | 0.000946 | 0.049 |
| E0 V1 M2 | 0.002207 | 0.002835 | 0.644 | 0.805 | 0.781 | 40.34% | 0.000653 | 0.122 |
| E1 V0 M1 | 0.002467 | 0.003200 | 0.546 | 0.800 | 0.732 | 33.33% | 0.000893 | 0.113 |
| **E1 V0 M2 selected** | **0.002088** | **0.002748** | **0.665** | **0.825** | **0.761** | **43.57%** | **0.000679** | **0.039** |
| E1 V1 M1 | 0.002463 | 0.003177 | 0.553 | 0.795 | 0.729 | 33.43% | 0.000871 | 0.125 |
| E1 V1 M2 | 0.001995 | 0.002573 | 0.707 | 0.848 | 0.834 | 46.09% | 0.000586 | 0.128 |

The last row is descriptively best on init36--39 but was not selected by the
calibration rule and cannot replace E1 V0 M2 post hoc.

E0 evaluation MAE is `32.02%` above its calibration OOF MAE; E1 is `40.17%`
above. Despite this gap, both clear the frozen absolute-recovery gate by a
wide margin. E1 lowers E0 evaluation MAE by `4.75%` and pair-difference RMSE
by `14.32%`, but those improvements do not restore downstream value.

## Mean, scale, bias, and dynamic range

| Quantity | Real | E0 selected | E1 selected |
|---|---:|---:|---:|
| Mean area | 0.025805 | 0.026137 | 0.026122 |
| Standard deviation | 0.004751 | 0.003567 | 0.004409 |
| Image std ratio | 1.000 | 0.751 | 0.928 |
| Minimum | 0.015564 | 0.019908 | 0.019083 |
| Maximum | 0.039062 | 0.038568 | 0.038652 |
| Predicted-on-real slope | — | 0.617 | 0.766 |
| Intercept | — | 0.01022 | 0.00637 |
| Pair-Δ std ratio | 1.000 | 1.807 | 1.425 |

Both image and pair standard-deviation ratios exceed the frozen `0.25`
collapse threshold, so `AREA_DYNAMIC_RANGE_COLLAPSE` is not assigned. The
positive aggregate range is nevertheless poorly aligned with candidate
direction: excessive pair variance is noise-like rather than useful signal.

## Candidate differential, direction, and ordering

| Metric | E0 selected | E1 selected |
|---|---:|---:|
| Signed ΔA RMSE | 0.000793 | 0.000679 |
| Signed ΔA MAE | 0.000580 | 0.000499 |
| ΔA Pearson | 0.106 | 0.061 |
| ΔA Spearman | 0.075 | 0.039 |
| Direction accuracy | 50.49% | 51.14% |
| Direction 95% CI | [46.32%,55.24%] | [48.13%,54.03%] |
| Direction rows after tie exclusion | 614 | 614 |
| Mean three-way Spearman | 0.033 | 0.023 |
| Spearman 95% CI | [-0.032,0.105] | [-0.038,0.081] |
| Pairwise ordering accuracy | 44.06% | 43.65% |
| Top-area agreement | 34.38% | 33.44% |
| Bottom-area agreement | 35.31% | 31.25% |

The corresponding ordering confidence intervals also remain weak: pairwise
accuracy is `[41.04%,47.40%]` for E0 and `[41.04%,46.25%]` for E1; top/bottom
agreement intervals include the one-in-three chance level. Absolute area
prediction therefore does not imply candidate ranking fidelity.

## Temporal normalization

Real candidate mean separation divided by adjacent-horizon temporal movement
is `0.656`. The predicted ratio is `0.803` for both selected extractors, or
`1.224x` the real ratio. Candidate separation is not numerically zero, so the
failure is not a trivial flat-output artifact. Its direction and ordering are
misaligned with real candidate differences.

## Downstream G versus G plus predicted area

| Condition | Target | G RMSE | G+predArea RMSE | Gain | 95% CI | Shuffle | Shuffle 95% CI |
|---|---|---:|---:|---:|---|---:|---|
| E0 | Area | 0.00026858 | 0.00026765 | +0.35% | [-0.92%,1.46%] | +0.45% | [-0.88%,1.77%] |
| E0 | Area residual | 0.00025697 | 0.00026239 | -2.11% | [-3.67%,-0.18%] | +0.28% | [-0.25%,0.87%] |
| E0 | ObjMask | 0.00060711 | 0.00060678 | +0.06% | [0.01%,0.10%] | +0.02% | [-0.02%,0.06%] |
| E0 | ObjMask residual | 0.00039974 | 0.00041542 | -3.92% | [-5.10%,-2.38%] | +0.11% | [0.04%,0.19%] |
| E1 | Area | 0.00026858 | 0.00026842 | +0.06% | [-1.32%,1.16%] | -0.40% | [-1.98%,1.22%] |
| E1 | Area residual | 0.00025697 | 0.00025635 | +0.24% | [-0.87%,1.17%] | -0.15% | [-1.57%,1.38%] |
| E1 | ObjMask | 0.00060711 | 0.00060889 | -0.29% | [-0.57%,-0.05%] | -0.32% | [-0.70%,-0.04%] |
| E1 | ObjMask residual | 0.00039974 | 0.00040026 | -0.13% | [-0.30%,0.01%] | -0.06% | [-0.28%,0.16%] |

E0 retains only `0.59%` of oracle F1's Area gain and `-3.91%` of its Area
residual gain. E1 retains `0.10%` and `0.45%`. Good image-level MAE is thus
not the same as recovering the frozen oracle target's counterfactual value.

## Task robustness

Entries are G+predArea relative Area-RMSE improvement over G in percent.

| Task | E0 | E1 |
|---:|---:|---:|
| 0 | -5.94 | -16.05 |
| 1 | 2.47 | 2.05 |
| 2 | 1.50 | -5.78 |
| 3 | 0.45 | 0.82 |
| 4 | -2.13 | -2.62 |
| 5 | -3.60 | -3.82 |
| 6 | -1.91 | -0.75 |
| 7 | 1.20 | 1.96 |
| 8 | -3.36 | -1.52 |
| 9 | 2.59 | 3.00 |

E0 is positive on 5/10 tasks and E1 on 4/10, below the required 7/10.
Tasks 0, 4, 5, 6, and 8 expose the instability rather than one isolated task.

## Horizon robustness

| Horizon | E0 | E1 |
|---:|---:|---:|
| h1 | -0.68 | -4.36 |
| h2 | -1.61 | -7.08 |
| h3 | -0.32 | 1.23 |
| h4 | -0.32 | 0.54 |
| h5 | -0.84 | -0.19 |
| h6 | 1.83 | -0.38 |
| h7 | -0.03 | 0.13 |
| h8 | 2.62 | 2.46 |

E0 is positive on 2/8 horizons and E1 on 4/8. E1 clears only the count floor;
it still fails magnitude, CI, shuffle, residual, task, and pair requirements.

## Candidate-pair robustness

| Pair | E0 | E1 |
|---|---:|---:|
| A0-A1 | -1.97 | -2.39 |
| A0-A2 | 0.14 | 0.12 |
| A1-A2 | 1.24 | 0.89 |

Both regress A0-A1, so neither satisfies the all-three-pairs rule. The small
positive A0-A2/A1-A2 values are far below the 10% aggregate requirement.

## Language-conditioning interpretation

Language improves absolute area MAE modestly and brings the predicted
image-level standard deviation closer to the real value. It does not improve
the candidate-sensitive bridge: E1 has lower ΔA Spearman, worse aggregate
Area gain, a negative shuffle point estimate, and still regresses A0-A1.
Therefore the audit assigns neither a language-needed label nor a
visual-only-sufficiency label. The question is not simply whether the relevant
object can be named; the missing property is stable within-task,
within-branch candidate variation.

## Mechanical decision

| Required gate | E0 | E1 |
|---|---|---|
| Absolute MAE improvement >=20%, CI lower >0 | pass | pass |
| Area gain >=10%, CI lower >0 | fail | fail |
| Predicted-area shuffle CI lower >0 | fail | fail |
| Area residual gain >=5%, CI lower >0 | fail | fail |
| >=7 positive tasks | fail | fail |
| >=4 positive horizons | fail | pass count only |
| all 3 pairs positive | fail | fail |
| image and pair std ratios >=0.25 | pass | pass |
| Complete gate | **FAIL** | **FAIL** |

Final mechanism labels:

```text
AREA_REGRESSION_ACCURATE_BUT_DECISION_SIGNAL_INSUFFICIENT
NATURAL_SPATIAL_EXTRACTION_BLOCK
```

Final authorization:

```text
BLOCK_SPATIAL_WM_PROTOTYPE
```

No extractor specification is selected. `FROZEN_CLIP_INFORMATION_LIMIT` is
not claimed: this experiment compares only two fixed summaries and small
heads, and it does not isolate whether candidate information is absent from
the encoder, inaccessible to the heads, or overwhelmed by scene/task-level
variation. Another encoder audit is not authorized here.

## Six required answers

**Q1 — Can ordinary natural RGB stably recover the 1-D continuous visible
area?**

At the absolute image level, yes: both selected conditions improve MAE over
the constant by more than 40% with positive intervals. Under the full
candidate-sensitive meaning required for F1, no.

**Q2 — Is vision alone sufficient to identify the relevant object, or is task
language required?**

Neither conclusion clears the complete gate. Language improves absolute MAE
by `4.75%` relative to selected E0, but it does not restore differential,
shuffle, residual, task, or pair evidence.

**Q3 — Does the predictor recover continuous range rather than collapse to
the mean?**

Yes at the aggregate level. Image std ratios are `0.751/0.928` and pair std
ratios `1.807/1.425`, all above `0.25`. The failure is misalignment, not flat
output.

**Q4 — Does it predict A0/A1/A2 visible-area differences and directions?**

No. ΔA Spearman is `0.075/0.039`, direction accuracy is approximately chance,
and three-way ordering intervals cross zero.

**Q5 — Does predicted natural area preserve the oracle F1 downstream gate?**

No. Area gains are `0.35%/0.06%` rather than at least 10%; residual gains are
`-2.11%/+0.24%` rather than at least 5%; shuffles cross zero, tasks are
insufficient, and A0-A1 regresses.

**Q6 — Is there now a 1-D, natural-RGB-available, task-relevant,
non-EE-only, candidate-sensitive, cross-task-stable state that authorizes a
World Model?**

No. Oracle F1 remains valid, but neither natural proxy transfers its
candidate-sensitive value. `BLOCK_SPATIAL_WM_PROTOTYPE` is the only authorized
decision.

## Candidate headroom remains separate

The frozen A0/A1/A2 best-of-three expert-action MSE oracle gain remains about
`1.648%`. It does not enter any extraction gate. Natural target extraction and
candidate quality/headroom remain distinct questions.

## Artifacts, finite values, and reproducibility

| Artifact | SHA-256 |
|---|---|
| Protocol | `123dd3a9ad7e24dc01cdf886352c1bd8bdd7e26cde9267f9f263994558d59029` |
| Calibration freeze | `6a2d35f9dd087b37002671adf3317399ceb13d92b579ce341cbba27ea8331584` |
| Calibration weights | `65eec73a3a87e8319889c2486cc0f61d71219f4be7aa04cca830bb2e93b0ac69` |
| Calibration derived | `d8311da4d3c8517c2dfdfc00f819d30d7f5007200dc1dd106e57c9129b2b022b` |
| Evaluation derived | `6cd343b36c09751f933206e8d3176026396ac6b154c19c697d343988c1e32ddc` |
| Compact diagnosis | `98c1e65dcaaed3ad8f077a7ab921ba2a491343077d1b04cca4b5b51524ae9946` |
| Ordering bootstrap supplement | `c0072eec18f4ddce358cef360fd7478e1e0f8c54a683a27eb8827cf24ecac2ab` |

All admitted natural features, language features, target labels, parameters,
predictions, residuals, pair differences, bootstrap statistics, and persisted
numeric arrays are finite. Constant three-candidate groups produce
mathematically undefined rank correlation and are assigned zero only inside
the preregistered aggregate ordering statistic; no undefined value is used as
positive evidence.

The focused protocol/result suite verifies frozen hashes, partitions, model
capacity, feature/pair contracts, finite sidecars, exact decision mapping, and
forbidden activity. The complete repository suite passes `168` tests; the
remaining 21 messages are dependency deprecation warnings, not test failures.

## Forbidden-activity audit

No simulator, modified RGB, masked/cropped/object-only inference, soft
footprint extractor, shape reconstruction, SAM, DINO, SigLIP, CNN, new encoder,
or encoder fine-tuning was run. No World Model, GRU, Transformer,
action-conditioned predictor, actor, candidate generator, production scale,
reranker, score, or controller was trained or modified. No C1/C2, D20-v2,
seed23/47, reward, done, success, final task outcome, controller evaluation,
or init40--49 access occurred. Frozen masks were labels only and never
extractor inputs. No upstream artifact was overwritten.
