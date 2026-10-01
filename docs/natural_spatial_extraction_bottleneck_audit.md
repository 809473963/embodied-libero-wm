# Binary localization is recoverable, but it discards the oracle footprint value

Date: 2026-10-01

Final mechanism labels:

```text
SOFT_FOOTPRINT_INFORMATION_REQUIRED
VISIBLE_SCALE_SIGNAL_NEEDED
LOCALIZATION_NOT_SUFFICIENT
NATURAL_SPATIAL_EXTRACTION_BLOCK
OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED
```

Next-stage authorization: `BLOCK_SPATIAL_WM_PROTOTYPE`

Block type: `target simplification failure`

Selected target: `none`

Alternative spatial-encoder audit authorization: `false`

Controller authorization: `false`

## Executive answer

The natural bridge is blocked first by target simplification, not by decoder
capacity. The 7x7 binary footprint does not retain the useful information in
the frozen soft footprint. On held-out init36--39, adding oracle B1 binary
features to global CLIP changes ObjMask and Area RMSE by `-1.04%` and
`+0.39%`, versus `+30.45%` and `+62.64%` for B0 soft occupancy plus continuous
area. B1 retains `-3.42%` and `0.62%` of those gains. Its EE-conditioned
residual improvements are also effectively zero. This assigns
`SOFT_FOOTPRINT_INFORMATION_REQUIRED` and stops the binary-only route before a
natural B1 decoder can be selected.

Visible scale is not optional under the tested representation. B2 keeps the
same binary map but restores exact continuous area; its oracle gains are
`10.92%` for ObjMask and `58.91%` for Area, with positive bootstrap intervals,
strong shuffles, and both EE-conditioned residual gains above 5%. In contrast,
the calibration-frozen three-level area B4 remains near zero. The evidence
therefore assigns `VISIBLE_SCALE_SIGNAL_NEEDED`: the useful scale is not
recovered by occupied-cell count or three coarse area bins.

The capacity ladder does not rescue deployment. C1, C2, and C3 recover binary
location well enough to pass the binary classification thresholds. C2 is the
strongest: IoU `0.716`, balanced accuracy `0.910`, recall `0.871`, and AUROC
`0.976`. Yet its natural B2 downstream gains are only `1.67%` for ObjMask and
`2.60%` for Area, and its area MAE is `78.75%` worse than the calibration
constant. No C0--C3 design reaches the 10% primary or 5% residual bridge gate.
This assigns `LOCALIZATION_NOT_SUFFICIENT`, but it does not establish a frozen
CLIP information limit: the binary teacher itself already failed, so changing
the encoder is not authorized by this experiment.

The final result is `BLOCK_SPATIAL_WM_PROTOTYPE`. No low-dimensional target in
this audit is simultaneously natural-RGB recoverable, object-relevant beyond
EE motion, and stable across tasks, horizons, and candidate pairs.

## Frozen provenance and authorization boundary

The upstream target-design study is frozen at commit
`3792ec5bfc58f540a9b25049f968aa45339cc5b9` under annotated tag
`spatial-target-design-unresolved`. Its conclusions remain:

```text
OBJECT_FOOTPRINT_TARGET_VALID
ORACLE_SPATIAL_TARGET_VALID
NATURAL_SPATIAL_EXTRACTION_BLOCK
OBJECT_AUXILIARY_TARGET_VALID
OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED
SPATIAL_TARGET_DESIGN_UNRESOLVED
BLOCK_SPATIAL_WM_PROTOTYPE
```

The new protocol and implementation were committed at `3acbfdd` before any
new calibration result. After calibration and before init36--39 evaluation, a
decision-mapping omission was corrected at `1f8524f`: the user's explicit
binary-teacher-failure case required `SOFT_FOOTPRINT_INFORMATION_REQUIRED`,
although that label was absent from the shorter allowed-label list. No teacher,
threshold, bin, decoder, loss, training setting, ridge form, bootstrap, shuffle,
numeric gate, partition, or authorization changed. Calibration was rebuilt and
frozen at `d870daf` before evaluation was opened.

The one-shot numerical diagnosis was committed at `13a756b`. A final label-only
amendment added `VISIBLE_SCALE_SIGNAL_NEEDED` and
`LOCALIZATION_NOT_SUFFICIENT` under their already frozen case definitions. The
source diagnosis, sidecar, metrics, intervals, gates, selected target, block
type, and authorization were not changed.

This audit was authorized only to compare compact teachers and frozen-patch
decoders. It did not authorize a World Model, reranker, controller, new
encoder, or success experiment.

## Data split and exposure

| Partition | Init states | Branch points | Pair/horizon rows | Use |
|---|---|---:|---:|---|
| Calibration | 31--34 | 40 | 960 | teacher bins, oracle screen, decoder CV/fit, thresholds, ridge fits |
| Frozen development evaluation | 36--39 | 40 | 960 | one-shot metrics, bootstrap, gates, verdict |
| Historical reference | 30/35 | — | — | not fitted or re-evaluated |
| Forbidden | 40--49 | 0 | 0 | never accessed |

The split and bootstrap unit is the matched branch point within task, retaining
all three candidate pairs and eight horizons. Evaluation is a frozen
development evaluation, not blind confirmation. Existing natural CLIP patch
features, per-object masks, and image-space targets were reused. No simulator
ran.

## B0--B4 exact teacher definitions

Every teacher is a per-image 7x7 state. Pair differences appear only in the
evaluation and probe layers.

| Teacher | Dimension | Exact state | Area/scale field | Calibration role |
|---|---:|---|---|---|
| B0 | 50 | 49 soft relevant-object occupancy fractions | exact continuous visible-area fraction | oracle-soft reference |
| B1 | 49 | cell is 1 iff soft occupancy is greater than zero | none; occupied-cell mean is diagnostic only | binary-only simplification |
| B2 | 50 | same binary map as B1 | exact continuous visible-area fraction | binary plus continuous scale |
| B3 | 49 | 0/0.5/1 ordinal occupancy | none; ordinal-map mean is diagnostic only | single coarse-occupancy form |
| B4 | 50 | same binary map as B1 | 0/0.5/1 coarse visible scale | single coarse-area form |

The B1/B2/B4 binary rule was fixed before calibration as `occupancy > 0`; it
was not tuned. Calibration froze the B3 positive-cell split at
`0.0617283951`. B4 visible-area tertile edges are `0.0235595703` and
`0.0270589193`. No second ordinal or coarse-scale definition was tried.

Pair distance is normalized Hamming distance for B1, normalized ordinal L1 for
B3, and mean absolute difference across map plus scale for B0/B2/B4. The same
six low-capacity pair summaries feed all ridge comparisons: RMSE, MAE, maximum,
top-quartile mean, centroid displacement, and area or footprint-derived scale
difference.

## Oracle sufficiency comparison

All rows below add the teacher summary to the same 10-D global baseline.
Percentages are held-out RMSE improvements; intervals and shuffles use 10,000
task-stratified branch-block replicates.

| Teacher | ObjMask gain | 95% CI | Shuffle | Area gain | 95% CI | Shuffle | Oracle gate |
|---|---:|---|---:|---:|---|---:|---|
| B0 soft+area | 30.45% | [20.25%,40.81%] | 71.34% | 62.64% | [55.95%,67.54%] | 254.83% | pass |
| B1 binary | -1.04% | [-2.76%,1.06%] | -0.36% | 0.39% | [-1.88%,2.20%] | 0.97% | FAIL |
| B2 binary+area | 10.92% | [5.37%,17.26%] | 20.04% | 58.91% | [51.01%,64.85%] | 213.52% | pass |
| B3 ordinal | -0.57% | [-2.38%,1.58%] | -0.54% | 0.17% | [-3.95%,3.28%] | 2.00% | FAIL |
| B4 binary+coarse area | -1.02% | [-2.70%,1.07%] | 0.03% | 0.85% | [-1.15%,2.56%] | 2.59% | FAIL |

B1, B3, and B4 were therefore stopped before natural decoder fitting. B2 was
the only simplified calibration teacher that cleared the preregistered oracle
screen. Its held-out primary and residual gates also pass, but only its Area
gain retains at least 80% of B0.

## B0 to B1 information loss

| Target | B0 gain | B1 gain | B1 retention vs B0 | Soft→binary loss |
|---|---:|---:|---:|---:|
| ObjMask | 30.45% | -1.04% | -3.42% | 103.42% |
| Area | 62.64% | 0.39% | 0.62% | 99.38% |
| ObjMask residual | 18.49% | -0.09% | -0.49% | 100.49% |
| Area residual | 55.66% | 0.38% | 0.68% | 99.32% |

This is Loss A in the requested chain. It is already decisive: binary presence
throws away almost all of the teacher's useful value. Loss B—oracle B1 to a
natural B1 decoder—was not estimated because B1 failed the calibration oracle
screen. Training it after that failure would violate the frozen stop rule.

The evaluated natural B2 models are not substitutes for Loss B because they
also attempt to recover continuous area.

## Visible-area necessity

The occupied-cell count implicit in B1 is insufficient. Its derived scale is
already included in the fixed pair summary, yet B1 remains at zero. B4 shows
that replacing continuous area by three calibration-frozen scale levels also
does not restore value. Only B2, which retains the exact continuous area
scalar, recovers a strong Area signal and enough ObjMask signal to pass.

Therefore `VISIBLE_SCALE_SIGNAL_NEEDED` is assigned. Within the tested forms,
continuous visible scale carries material information that is absent from
binary localization and cannot be reconstructed by a simple count of occupied
cells. This does not claim that a separately learned continuous area head is
the only possible representation; it says the tested binary and coarse forms
are insufficient.

## EE-conditioned object residuals

The EE-only control was fitted on init31--34 and frozen. Held-out increments
below predict the same ObjMask/Area residuals after subtracting that frozen
EE-only expectation.

| Teacher | ObjMask residual | 95% CI | Area residual | 95% CI |
|---|---:|---|---:|---|
| B0 soft+area | 18.49% | [10.95%,25.54%] | 55.66% | [48.53%,60.92%] |
| B1 binary | -0.09% | [-0.28%,0.21%] | 0.38% | [-1.26%,1.73%] |
| B2 binary+area | 9.77% | [3.44%,16.05%] | 53.36% | [45.61%,59.15%] |
| B3 ordinal | 0.59% | [0.12%,1.00%] | 0.22% | [-3.11%,2.82%] |
| B4 binary+coarse area | -0.31% | [-0.87%,0.44%] | 0.77% | [-0.79%,2.16%] |

B0 and B2 retain object information beyond robot EE motion. Binary B1 does
not. `OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED` remains supported at the
oracle level, but no natural simplified target preserves it sufficiently.

## Decoder C0--C3 definitions and calibration

Only frozen natural CLIP ViT-B/32 patch features `[49,512]` enter the decoder.
No mask, object identity, pose, task ID, future simulator state, reward, or
success signal is an input.

| Decoder | Patch path | Parameters | Frozen epochs | Binary threshold |
|---|---|---:|---:|---:|
| C0-linear | Linear(512,1) | 1,026 | 80 | 0.50 |
| C1-width32 | Linear(512,32)-GELU-Linear(32,1) | 32,898 | 80 | 0.60 |
| C2-width128-64 | Linear(512,128)-GELU-Linear(128,64)-GELU-Linear(64,1) | 106,882 | 74 | 0.70 |
| C3-local-context | Linear(512,32), one 3x3 depthwise plus 1x1 pointwise mix, 1x1 output | 18,914 | 80 | 0.70 |

Each B2 decoder also has its preregistered pooled continuous-area head. BCE
uses calibration-derived positive weighting capped at 20; area uses sigmoid
MSE with weight 10. Within each leave-one-init-out calibration fold, the best
validation objective epoch was retained. The final epoch is the round-half-up
median of the four fold epochs. Threshold maximizes calibration OOF IoU, then
balanced accuracy, then favors the lower threshold. No evaluation value changed
an epoch, loss, width, or threshold.

## Natural recovery and calibration/evaluation gap

| Decoder | Cal IoU | Eval IoU | Eval Dice | Eval balanced acc. | Eval recall | Eval precision | Eval AUROC | Cal area MAE | Eval area MAE | Area vs constant | Recovery gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C0 | 0.530 | 0.551 | 0.710 | 0.850 | 0.807 | 0.634 | 0.923 | 0.12520 | 0.12658 | -3321.12% | FAIL |
| C1 | 0.643 | 0.664 | 0.798 | 0.884 | 0.825 | 0.774 | 0.959 | 0.00520 | 0.00438 | -18.44% | FAIL |
| C2 | 0.702 | 0.716 | 0.834 | 0.910 | 0.871 | 0.800 | 0.976 | 0.01057 | 0.00661 | -78.75% | FAIL |
| C3 | 0.689 | 0.651 | 0.789 | 0.877 | 0.811 | 0.768 | 0.962 | 0.02619 | 0.02573 | -595.43% | FAIL |

The held-out constant area MAE is `0.003700`. C1 has the best area MAE but is
still worse than the constant. C2 has the strongest localization. C3 loses
`0.037` IoU from calibration to evaluation, but the experiment is not primarily
blocked by an OOF-to-evaluation collapse: every calibration downstream bridge
gain was already below 1%, and every area recovery gate already failed.
`EXTRACTION_GENERALIZATION_LIMIT` is therefore not assigned.

## Downstream natural object value

Percentages compare global plus the decoded B2 state with global alone.

| Decoder | ObjMask gain | 95% CI | Area gain | 95% CI | ObjMask residual | 95% CI | Area residual | 95% CI |
|---|---:|---|---:|---|---:|---|---:|---|
| C0 | -0.19% | [-1.93%,1.44%] | 4.55% | [1.75%,6.70%] | 0.57% | [-0.60%,1.58%] | 4.35% | [1.72%,6.39%] |
| C1 | -1.72% | [-2.87%,-0.65%] | 0.93% | [-1.25%,3.07%] | -0.37% | [-1.21%,0.30%] | 1.23% | [-0.89%,3.24%] |
| C2 | 1.67% | [-0.36%,3.74%] | 2.60% | [-0.66%,5.46%] | 1.87% | [0.63%,3.10%] | -1.03% | [-2.56%,0.72%] |
| C3 | -2.51% | [-4.43%,-0.05%] | 0.76% | [-0.36%,1.74%] | -4.40% | [-5.65%,-2.75%] | 1.03% | [0.04%,1.90%] |

No natural design reaches 10% on either primary target or 5% on a residual.
C2's primary shuffle degrades ObjMask and Area by `3.28%` and `3.53%`, with
positive intervals, so some candidate-sensitive signal remains. Its magnitude
is still far too small to clear the bridge.

This is `LOCALIZATION_NOT_SUFFICIENT`: C1/C2/C3 correctly classify occupied
cells by the frozen thresholds, but the decoded state does not retain the
continuous object value needed by the downstream targets.

## Candidate separation and temporal normalization

| Representation | Candidate/temporal ratio | ObjMask rho | Area rho | EE rho | ObjMask-residual rho | Area-residual rho |
|---|---:|---:|---:|---:|---:|---:|
| B0 oracle | 0.565 | 0.799 | 0.700 | 0.644 | 0.581 | 0.493 |
| B1 oracle | 0.740 | 0.236 | 0.209 | 0.186 | 0.190 | 0.147 |
| B2 oracle | 0.740 | 0.411 | 0.555 | 0.332 | 0.312 | 0.467 |
| B3 oracle | 0.703 | 0.256 | 0.212 | 0.207 | 0.200 | 0.140 |
| B4 oracle | 0.725 | 0.245 | 0.223 | 0.194 | 0.196 | 0.160 |
| B2 natural C2 | 0.856 | 0.304 | 0.246 | 0.241 | 0.257 | 0.175 |

Every representation has material candidate separation relative to ordinary
adjacent-frame motion. Failure is therefore not a zero-scale artifact. The
problem is alignment and conditional information: B1 separates candidates but
its separation is weakly related to ObjMask/Area, and natural C2 does not
recover B2's oracle alignment.

Candidate means for B2 oracle are stable across the three pairs; the
candidate/temporal test is computed within the same target coordinates and
does not compare absolute distances across representation families.

## Per-horizon primary evidence

Entries are G+state versus G RMSE improvements in percent. C2 is shown as the
strongest localization decoder; it is a diagnostic, not a selected target.

| h | B1 ObjMask | B1 Area | B2 ObjMask | B2 Area | C2 ObjMask | C2 Area |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7 | 6.0 | 28.4 | 65.3 | -4.2 | -3.6 |
| 2 | -2.5 | 2.6 | 18.0 | 52.5 | -2.9 | -11.5 |
| 3 | 3.6 | 1.5 | 8.9 | 57.2 | 1.7 | 1.3 |
| 4 | -0.9 | -0.2 | 14.3 | 59.0 | 1.6 | 1.4 |
| 5 | -5.6 | -2.1 | 15.0 | 64.5 | 1.5 | 4.9 |
| 6 | -1.3 | 0.8 | 15.4 | 62.8 | 1.5 | 2.5 |
| 7 | -0.6 | -1.4 | 3.4 | 54.5 | 3.0 | -0.9 |
| 8 | -0.2 | 2.1 | 5.7 | 53.1 | 3.1 | 12.3 |

| h | B1 Obj residual | B1 Area residual | B2 Obj residual | B2 Area residual | C2 Obj residual | C2 Area residual |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.3 | 3.7 | 8.9 | 42.4 | -5.4 | -14.6 |
| 2 | -0.1 | 1.8 | 6.6 | 37.8 | -1.9 | -10.7 |
| 3 | 0.3 | 1.3 | 0.3 | 52.2 | 1.3 | -4.9 |
| 4 | 0.0 | -0.0 | 16.4 | 58.3 | 2.5 | -0.6 |
| 5 | 0.3 | -1.0 | 19.5 | 59.3 | 2.3 | -0.3 |
| 6 | 0.1 | 0.9 | 15.9 | 56.8 | 2.6 | -0.1 |
| 7 | -0.4 | -0.7 | -0.6 | 48.2 | 1.5 | 0.7 |
| 8 | -0.4 | 1.3 | 7.2 | 48.4 | 3.1 | 2.8 |

B2's Area value is broad across all eight horizons. C2 regresses at h1/h2 and
never approaches the aggregate threshold; capacity does not repair immediate
or later-horizon behavior consistently.

## Per-task natural C2 evidence

| Task | ObjMask | Area | ObjMask residual | Area residual |
|---:|---:|---:|---:|---:|
| 0 | -4.4 | -6.3 | 0.2 | 9.0 |
| 1 | -11.1 | -3.1 | 1.0 | -10.5 |
| 2 | 2.1 | 2.0 | 1.4 | 2.6 |
| 3 | -2.6 | -6.4 | -2.9 | 0.8 |
| 4 | 4.4 | -3.6 | 3.6 | -0.0 |
| 5 | -2.8 | -13.9 | -0.8 | -4.0 |
| 6 | 1.0 | 0.2 | -0.6 | -5.1 |
| 7 | 10.3 | 9.9 | 6.5 | -2.1 |
| 8 | -1.0 | 4.8 | 1.5 | -2.0 |
| 9 | 1.8 | 5.5 | 0.9 | 0.8 |

The previously unstable tasks remain visible: task1 and task5 regress strongly,
and task9 is only modestly positive. C2 is positive on 5/10 tasks for ObjMask
and 5/10 for Area, below the required 7/10. Only task7 approaches a useful
primary gain, so the aggregate is not hiding a broadly stable decoder.

## Candidate-pair robustness

| Pair | B1 ObjMask | B1 Area | B2 ObjMask | B2 Area | C2 ObjMask | C2 Area | C2 Obj residual | C2 Area residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0-A1 | 1.2 | 0.8 | 8.3 | 47.6 | 1.7 | 0.7 | 1.8 | -0.1 |
| A0-A2 | 1.0 | -0.9 | 12.6 | 58.8 | 2.1 | 0.7 | 2.1 | -0.9 |
| A1-A2 | -2.7 | 0.8 | 11.3 | 63.6 | 1.5 | 4.1 | 1.8 | -1.4 |

B1 fails even the all-pairs-positive condition. B2 is strong for Area and
positive for ObjMask, though A0-A1 is below 10%. C2 is positive on all three
raw-target pairs but far below 10%, and Area residual regresses on all three.

## Decoder-capacity diagnosis

The four preregistered cases resolve as follows:

- `DECODER_CAPACITY_LIMIT_SUPPORTED` is not assigned. C2 improves binary IoU,
  but no stronger decoder clears recovery plus downstream value.
- `SOFT_TARGET_OVERPRECISION_SUPPORTED` is not assigned. Binary B1 fails at
  the oracle layer, so coarsening is not a successful replacement.
- `LOCALIZATION_NOT_SUFFICIENT` is assigned. C1/C2/C3 binary localization
  passes, while every downstream bridge fails.
- `FROZEN_CLIP_SPATIAL_INFORMATION_LIMIT` is not assigned. That label requires
  a strong oracle binary target followed by failure of all C0--C3 extractors;
  the prerequisite is false.

The primary block is therefore `target simplification failure`. The continuous
soft/scale content matters, and the tested natural decoders do not recover it.
This result does not authorize an alternative encoder audit because it has not
isolated frozen CLIP as the bottleneck.

## Mechanical decision

| Question | Mechanical result |
|---|---|
| B1 retains at least 80% of B0 and passes oracle gates | FAIL |
| B2 binary+continuous area passes oracle gates | pass |
| B3 ordinal passes oracle gates | FAIL |
| B4 binary+coarse area passes oracle gates | FAIL |
| C0 natural B2 recovery and bridge | FAIL |
| C1 natural B2 recovery and bridge | FAIL |
| C2 natural B2 recovery and bridge | FAIL |
| C3 natural B2 recovery and bridge | FAIL |
| Any one design passes recovery, primary, shuffle, task/horizon/pair, and residual gates | no |

Final mechanism labels:

```text
SOFT_FOOTPRINT_INFORMATION_REQUIRED
VISIBLE_SCALE_SIGNAL_NEEDED
LOCALIZATION_NOT_SUFFICIENT
NATURAL_SPATIAL_EXTRACTION_BLOCK
OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED
```

Final authorization:

```text
BLOCK_SPATIAL_WM_PROTOTYPE
```

No target specification is selected. `GO_ALTERNATIVE_SPATIAL_ENCODER_AUDIT`
is also not authorized by this result.

## Six required answers

**Q1 — Does a 7x7 binary footprint retain the main object information in the
soft footprint?**

No. B1 retains only `-3.42%` of B0's ObjMask gain and `0.62%` of its Area gain;
both primary increments are effectively zero, as are both EE-conditioned
residual increments.

**Q2 — Is continuous soft occupancy necessary, or was the previous target too
precise?**

The tested evidence supports the former. B1 binary and B3 ordinal occupancy
both fail oracle sufficiency, while B0 soft occupancy plus area passes strongly.
`SOFT_TARGET_OVERPRECISION_SUPPORTED` is not assigned.

**Q3 — Must visible area be predicted separately?**

Some continuous visible-scale information is necessary in these forms. B2
passes, B1's occupied-cell-derived scale fails, and B4's three-level scale
fails. This assigns `VISIBLE_SCALE_SIGNAL_NEEDED`, but does not establish that
the exact scalar head is the only possible implementation.

**Q4 — Is natural extraction blocked by decoder capacity or frozen CLIP patch
information?**

Neither is isolated as the primary failure. Capacity improves localization but
not the bridge; frozen-CLIP insufficiency cannot be claimed because the oracle
binary prerequisite fails. The earliest demonstrated break is target
simplification, followed by failure to recover continuous area.

**Q5 — After controlling EE motion, does a natural simplified target retain
independent object information?**

No natural design clears the 5% residual gate. C2 reaches `1.87%` on ObjMask
residual and `-1.03%` on Area residual. The oracle B2 retains strong residual
value, but the natural bridge does not.

**Q6 — Is there now a low-dimensional, natural-RGB-recoverable,
object-relevant, non-robot-only, cross-cut-stable target for a spatial World
Model prototype?**

No. B0/B2 are useful oracle teachers but are not naturally recovered; B1/B3/B4
lose oracle value; all C0--C3 natural B2 bridges fail. The only valid decision
is `BLOCK_SPATIAL_WM_PROTOTYPE`.

## Candidate headroom remains separate

The frozen A0/A1/A2 best-of-three expert-action MSE oracle gain remains about
`1.648%`. It does not enter any extraction gate. This audit diagnoses a
representation/target bridge; it does not show that the current candidate set
has enough downstream control headroom even if that bridge were later solved.

## Artifacts, finite values, and automated verification

| Artifact | SHA-256 |
|---|---|
| Protocol | `09a9af8f6915729a4f0aec36f6e7ddd26c75182d9f1d7dd2e608a5834b99f03c` |
| Calibration freeze | `f440576af2c55582318f7a2d094bbf95f85ab000da7dd721168c0c9a7ce0e198` |
| Calibration weights | `56e2e8a66c1942d7adc7d79fc1c052e31c5c6086c477c2f4e71e8774047063ae` |
| Calibration derived | `5c90f1b05ff2b187fcd8f8524f8cde825e1bf4d8cc61bf9b7d9928a8d10a81c1` |
| Evaluation derived | `6a59755f6f1f87bcbe6b5795058ccf1ebef5ef3d77caf70f9a72899c8f116118` |
| Decision amendment | `6957c45fd36b68f01eea867208da7994286a3ab1ba165347fc6466e11551c657` |
| Source numerical diagnosis | `67c2566f4548a4e259269cf7bc2f4e64523d67cf8b8bf32f824f1a9692dc5d2c` |
| Corrected compact diagnosis | `9b10854d9f980a90076f7f14e2ef07f7df12e045b4a8b66b883dacbcdccbdaff` |

All admitted targets, representations, decoder parameters, predictions,
distances, residuals, bootstrap statistics, and sidecar numeric arrays are
finite. Undefined correlations are explicit `null`; no missing value is
converted into evidence. The ignored sidecars retain all 960 identities,
teacher/natural states, ridge features and predictions, and candidate/temporal
distances required for independent recalculation.

Seven protocol tests and seven result tests independently verify hashes,
partition boundaries, decoder shapes and parameter limits, finite arrays,
recomputed oracle increments, binary/area decisions, binary localization,
corrected label mapping, authorization, and forbidden activity. The complete
repository suite passes `149` tests; the remaining 21 messages are dependency
deprecation warnings, not test failures.

## Forbidden-activity audit

No simulator, modified RGB, masked-image inference, SAM, DINO, SigLIP, new
encoder, CLIP fine-tuning, World Model, GRU, Transformer, action-conditioned
dynamics model, actor, candidate generator, production scale, reranker, score,
or controller was run or modified. No C1/C2, D20-v2, seed23/47, reward, done,
success, final task outcome, init40--49, or controller-success evaluation
occurred. Frozen simulator masks were labels only and never decoder or
deployment inputs. No upstream artifact was overwritten.
