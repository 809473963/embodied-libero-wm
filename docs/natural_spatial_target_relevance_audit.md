# Natural patches carry broader physical signal, but not enough uniquely attributable task relevance to redesign the target

Date: 2026-09-30

Final mechanism verdict:

```text
NATURAL_PATCH_TASK_RELEVANCE_UNRESOLVED
```

Next-stage authorization: `BLOCK_VISUAL_TARGET_REDESIGN`

Controller authorization: `false`

## Executive answer

Natural patch features retain more information about all six tested physical
changes than the global 512-D CLIP distance. Patch-mean Spearman exceeds the
global value by `0.107--0.175`, with positive branch-bootstrap intervals for
EE translation/rotation, object translation/rotation, and EE-object relative
translation/rotation. The strongest low-capacity patch-only probes also beat
the matched global-summary probes on every target.

That descriptive advantage does not clear the preregistered redesign gate.
Adding the frozen 10-D patch summary to the matched 10-D global summary lowers
held-out RMSE by only `4.20--8.77%`, below the frozen `10%` minimum on every
target. Object translation has a positive increment interval (`6.90%`, 95% CI
`[1.42%,10.41%]`) but its full-patch shuffle interval crosses zero. Relative
translation has a large shuffle effect but its increment interval crosses
zero and one candidate pair regresses. Object/relative rotation likewise fail
the magnitude or shuffle condition. No target satisfies the complete
aggregate, shuffle, task, horizon, and pair gate.

The evidence also does not support the simpler claim that global pooling loses
*only* robot appearance. Raw patch advantage is present for object and relative
targets, and patch-only probes improve their R² substantially. But the unique
semantic attribution is weak: shuffling relevant-object or interaction scalar
features never has a positive 95% CI lower bound. The strongest-changing
patches contain about `49.5%` robot and only `3.9%` relevant-object fractional
coverage among the top ten, while the preregistered interaction region is broad.

Therefore the three required answers are:

1. **Q1 — mainly robot/image-motion detail?** Not established. Robot detail is
   prominent, but object and relative-state associations also improve.
2. **Q2 — confirmed object/interaction information lost by global CLIP?** No.
   The evidence is suggestive but fails the held-out increment-plus-shuffle gate.
3. **Q3 — enough evidence to replace global CLIP with a spatial target?** No.
   `BLOCK_VISUAL_TARGET_REDESIGN` remains in force. No World Model or controller
   work is authorized.

## Frozen provenance

The preceding visual-target audit is frozen at commit
`1819a1763c554819ad4f02de70d71f662e3458be` under annotated tags
`visual-target-spatial-compression` and
`global-clip-spatial-compression-confirmed`. Its conclusions remain:

```text
GLOBAL_CLIP_SPATIAL_COMPRESSION
BLOCK_VISUAL_TARGET_REDESIGN
```

The new protocol and implementation were committed as
`916a2b70c539bd6f464733c6a0c70750906ead02` before calibration. Calibration
weights, PCA bases, probe forms, CV-selected alphas, and localization thresholds
were committed at `b9d916eccb901cb3504c88429155db0aba9ece1b` before init36--39
was opened by this analysis.

All inputs are hash-frozen transitive descendants of the C0-v2, C1-v2.1,
D13/D14/D15, WM-v2A/B/B.1/C-EJ, structured-state, and visual-target audits.
No upstream artifact was overwritten.

## Development-data exposure statement

This is a mechanism-development audit, not a blind confirmation.

| Partition | Init states | Branch points | Rows | Use |
|---|---|---:|---:|---|
| Calibration | 31--34 | 40 | 960 | PCA/ridge fit, alpha CV, p75 localization thresholds |
| Frozen development evaluation | 36--39 | 40 | 960 | One-shot metrics, bootstrap, gates, verdict |
| Historical reference only | 30/35 | — | — | Not fitted or re-evaluated |
| Forbidden | 40--49 | 0 | 0 | Never accessed |

Init36--39 had already appeared in earlier mechanism audits and is explicitly
not described as a blind holdout. Every row is one of ten tasks, four init
states per task, three candidate pairs, and h1--h8. The branch point—not an
individual pair/horizon row—is the split and bootstrap unit.

No simulator ran. Existing sidecars already contained natural RGB-derived
global/patch features, fractional semantic masks, and matched future simulator
state. No reward, done, success, task-completion, or controller outcome was
read.

## Natural-image-only representation contract

Every representation comes from the same unmodified upright natural image.
Historical robot-only, object-only, removal, and flat-color masked variants
were not read by this analyzer.

```text
natural RGB [128,128,3]
  -> frozen openai/clip-vit-base-patch32
  -> global visual_projection + L2 normalization       [512]
  -> patch hidden[:,1:] -> post_layernorm
       -> visual_projection + per-token L2 norm       [49,512]
```

For a candidate pair `(i,j)` and horizon `h`:

```text
delta_g       = g_i[h] - g_j[h]
D_global      = 1 - cosine(g_i[h], g_j[h])
delta_P[k]    = P_i[h,k] - P_j[h,k]
D_patch[k]    = 1 - cosine(P_i[h,k], P_j[h,k])
```

The full signed `49x512` patch differential is retained for equal-rank PCA;
it is never globally averaged before that comparison. Patch summaries include
mean, max, top-5 mean, and top-10 mean, while the complete 7x7 distance map is
stored in the ignored derived sidecar. These are feature-difference diagnostics,
not transformer-attention claims.

## Fractional semantic coverage and interaction region

Frozen native MuJoCo geom segmentation supplies per-patch fractions for robot,
relevant object, other object, and background/mixed pixels. Candidate-pair
coverage is the mean of the two natural frames. No hard dominant label is used
for region pooling.

| Patch population | Robot | Relevant object | Other object | Background/mixed |
|---|---:|---:|---:|---:|
| All 49 patches | 16.53% | 2.58% | 2.20% | 78.69% |
| Top-5 changing patches | 52.20% | 5.33% | 1.93% | 40.54% |
| Top-10 changing patches | 49.55% | 3.93% | 1.85% | 44.67% |

The preregistered interaction indicator is active when a patch overlaps both
robot and relevant-object masks, or its center is within 32 native pixels of
both masks; pair coverage is the union of the two candidate indicators. It is
available without drawing a box or altering RGB.

A post-evaluation metadata amendment records that the protocol's prose
incorrectly called 32 native pixels “one mapped patch width.” A 7x7 cell is
about 18.3 native pixels, so the executed radius is about 1.75 cells. The
numeric threshold, arrays, metrics, and verdict were not changed or rerun.
This region is broad: it activates `57.68%` of all patches on average, with a
median 29 of 49 active patches. It covers `86.98%` of top-5 and `79.75%` of
top-10 changing patches. Interaction-specific correlations therefore cannot
be read as tight contact localization.

## External task-relevance targets

Global CLIP distance is not the relevance target. Six matched real simulator
quantities are used as diagnostic oracles:

| Family | Targets | Units/definition |
|---|---|---|
| `T_robot` | EE translation, EE rotation | Train-standardized EE-position norm; rotation geodesic in radians |
| `T_object` | Relevant-object translation, rotation | Max across runtime `obj_of_interest`; world displacement and geodesic |
| `T_relative` | EE-object relative translation, rotation | Max change in object-minus-EE position and EE-relative orientation |
| Secondary | Contact state different | XOR of four real MuJoCo contact states |

Future simulator state is used only as an offline oracle. It is not a proposed
World Model or controller input.

## Fair representation comparisons

The 10-D patch summary is:

```text
[mean, max, top5, top10,
 robot_mean, relevant_mean, interaction_mean,
 centroid_x, centroid_y, spatial_spread]
```

The matched 10-D global summary is:

```text
[global cosine distance, global differential L2,
 first 8 calibration-fitted global-differential PCA scores]
```

`G+P` concatenates them into 20 dimensions for the incremental test. A
separate capacity-matched comparison fits K=16 PCA to signed global and full
flattened patch differentials using calibration-only deterministic randomized
SVD. All target probes are closed-form ridge regressions with intercept;
alpha is chosen by leave-one-init-out CV within init31--34.

The primary target gate was frozen at all of:

- `G+P` relative RMSE improvement at least 10%, with 95% CI lower above zero;
- full patch-group shuffle degradation with 95% CI lower above zero;
- positive improvement on at least 7/10 tasks, 4/8 horizons, and all 3 pairs.

An object or relative target—not a robot target—must pass to authorize
`GO_SPATIAL_TARGET_DESIGN`.

## Raw scalar associations favor patches, but do not establish unique value

`P/S` are Pearson/Spearman. `Delta rho` is patch-mean Spearman minus global
Spearman with a task-stratified branch-bootstrap interval.

| Target | Global P/S | Patch mean P/S | Delta rho | 95% CI |
|---|---:|---:|---:|---:|
| EE translation | 0.347 / 0.435 | 0.578 / 0.610 | +0.175 | [0.129, 0.222] |
| EE rotation | 0.257 / 0.315 | 0.472 / 0.457 | +0.142 | [0.090, 0.192] |
| Object translation | 0.195 / 0.234 | 0.354 / 0.341 | +0.107 | [0.038, 0.176] |
| Object rotation | 0.168 / 0.148 | 0.332 / 0.263 | +0.115 | [0.048, 0.178] |
| Relative translation | 0.355 / 0.441 | 0.596 / 0.614 | +0.174 | [0.126, 0.223] |
| Relative rotation | 0.212 / 0.249 | 0.387 / 0.381 | +0.132 | [0.075, 0.186] |

The complete JSON contains Pearson, Spearman, bootstrap interval, and
patch-minus-global delta for all nine scalar summaries and all horizon/task/pair
cuts. Patch mean has a positive delta-rho interval for all six targets.

Natural region-specific Spearman values show why this is not a simple robot-only
result:

| Target | Robot patch | Relevant patch | Interaction patch | Background/mixed patch |
|---|---:|---:|---:|---:|
| EE translation | 0.597 | 0.411 | 0.631 | 0.621 |
| EE rotation | 0.425 | 0.214 | 0.420 | 0.467 |
| Object translation | 0.366 | 0.410 | 0.433 | 0.374 |
| Object rotation | 0.229 | 0.360 | 0.325 | 0.289 |
| Relative translation | 0.579 | 0.422 | 0.627 | 0.630 |
| Relative rotation | 0.325 | 0.278 | 0.384 | 0.399 |

Relevant-patch distance is stronger than robot-patch distance for both object
targets, but interaction/background summaries are often just as strong. Coarse
patch overlap, common motion intensity, and correlated spatial groups prevent
unique attribution from these raw correlations.

## Equal-capacity probes and the decisive incremental test

| Target | G RMSE / R² | Patch RMSE / R² | G+P RMSE / R² | G+P vs G | 95% CI |
|---|---:|---:|---:|---:|---:|
| EE translation | 0.013491 / 0.119 | 0.013032 / 0.178 | 0.012925 / 0.192 | +4.20% | [-7.84%,13.46%] |
| EE rotation | 0.006362 / 0.062 | 0.005886 / 0.198 | 0.005926 / 0.187 | +6.86% | [4.14%,9.48%] |
| Object translation | 0.001962 / 0.012 | 0.001804 / 0.164 | 0.001827 / 0.143 | +6.90% | [1.42%,10.41%] |
| Object rotation | 0.019701 / -0.048 | 0.018556 / 0.070 | 0.018731 / 0.052 | +4.93% | [-1.00%,9.37%] |
| Relative translation | 0.001732 / 0.103 | 0.001569 / 0.265 | 0.001580 / 0.254 | +8.77% | [-0.38%,16.49%] |
| Relative rotation | 0.018890 / -0.016 | 0.017653 / 0.113 | 0.017883 / 0.089 | +5.33% | [1.77%,8.47%] |

Patch-only summaries improve held-out R² for every target, but the
incremental question is stricter: does patch information add stable value
*after* global features? No target reaches 10%. Object translation and relative
rotation have positive increment intervals but still fail the magnitude gate.

## Patch and semantic shuffle controls

| Target | Full-patch degradation | 95% CI | Full shuffle passes | Relevant-only degradation | Interaction-only degradation |
|---|---:|---:|---|---:|---:|
| EE translation | +27.02% | [13.40%,43.26%] | yes | -0.09% | +0.38% |
| EE rotation | +1.29% | [-2.50%,4.91%] | no | +0.001% | -0.09% |
| Object translation | +5.55% | [-0.24%,12.51%] | no | +1.56% | +3.09% |
| Object rotation | +2.71% | [-1.21%,6.95%] | no | +0.27% | +0.32% |
| Relative translation | +27.21% | [14.81%,41.46%] | yes | +0.07% | +3.82% |
| Relative rotation | +2.18% | [-1.25%,5.65%] | no | +0.08% | +0.09% |

Every relevant-only and interaction-only shuffle interval crosses zero. The
large full-patch shuffle effects for EE and relative translation arise from the
joint spatial summary, not a uniquely attributable relevant-object or
interaction scalar. This is exactly why raw correlation alone is insufficient.

## Equal-rank PCA does not reproduce the hand-summary advantage

Both sides below are K=16 signed-differential PCA followed by the same ridge
family. No semantic masks enter this comparison.

| Target | Global PCA RMSE / Spearman | Patch PCA RMSE / Spearman |
|---|---:|---:|
| EE translation | 0.014407 / -0.068 | 0.014390 / -0.022 |
| EE rotation | 0.006610 / -0.030 | 0.006609 / 0.003 |
| Object translation | 0.001979 / -0.016 | 0.001979 / -0.019 |
| Object rotation | 0.019831 / -0.007 | 0.019836 / -0.038 |
| Relative translation | 0.001843 / -0.063 | 0.001842 / -0.036 |
| Relative rotation | 0.019177 / -0.013 | 0.019178 / -0.048 |

Neither equal-rank signed PCA representation generalizes across the development
split. The positive patch result is therefore carried by invariant distance,
region, and spatial summaries rather than a generic advantage of sixteen
linear patch-difference directions.

## Contact is a secondary, sparse signal

There are 62 contact-different rows among 960. The fixed balanced logistic
probe is secondary and cannot authorize redesign.

| Representation | AUROC | Balanced accuracy |
|---|---:|---:|
| Global summary | 0.515 | 0.520 |
| Patch summary | 0.706 | 0.690 |
| Global + patch | 0.703 | 0.689 |
| Global PCA16 | 0.468 | 0.473 |
| Patch PCA16 | 0.454 | 0.488 |

Patch summaries contain a useful contact-ranking association, but the sample
is sparse and G+P does not exceed patch alone. Per protocol, this remains a
diagnostic rather than a primary target gate.

## Horizon behavior

Table entries are `G+P` relative RMSE improvement over G in percent.

| h | EE trans | EE rot | Obj trans | Obj rot | Rel trans | Rel rot |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.2 | 17.9 | 14.8 | 26.1 | 30.8 | 17.8 |
| 2 | 10.8 | 8.2 | 5.4 | 10.5 | 15.8 | 7.3 |
| 3 | -6.2 | 8.0 | 8.9 | 7.4 | -2.3 | 7.1 |
| 4 | -15.5 | 0.2 | 4.4 | 5.9 | -13.0 | 4.1 |
| 5 | -2.2 | 4.0 | 4.5 | 3.3 | 1.5 | 3.8 |
| 6 | 7.2 | 6.8 | 7.9 | 5.0 | 13.4 | 6.5 |
| 7 | 7.1 | 7.2 | 8.3 | -0.9 | 12.1 | 2.4 |
| 8 | 4.3 | 6.6 | 6.6 | -0.7 | 10.6 | 1.7 |

The largest advantages occur at h1 and sometimes h2. Object/relative gains do
not strengthen monotonically with horizon; long-horizon rotation gains shrink.
This is evidence against interpreting the pooled result as emerging downstream
task-object dynamics.

Patch-mean raw Spearman is higher than global for most object/relative horizon
cuts, but not uniformly: object translation falls at h2 and relative rotation
falls at h2/h4. Full values are stored in the diagnosis.

## Per-task behavior

Again entries are `G+P` versus G RMSE improvement in percent.

| Task | EE trans | EE rot | Obj trans | Obj rot | Rel trans | Rel rot |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -7.0 | -6.7 | 5.2 | -18.2 | -16.4 | -7.3 |
| 1 | -16.8 | -1.8 | 7.9 | 5.1 | -2.7 | 4.7 |
| 2 | 21.2 | 12.0 | 10.4 | 14.6 | 24.5 | 13.2 |
| 3 | 10.2 | 10.4 | 11.1 | -15.0 | 13.1 | 3.0 |
| 4 | -25.7 | 10.1 | -12.4 | 6.8 | 6.0 | 6.1 |
| 5 | 9.5 | 7.4 | 8.4 | -1.4 | 9.3 | 3.7 |
| 6 | 7.3 | 5.8 | 3.2 | -12.6 | 11.2 | 6.0 |
| 7 | 2.8 | 10.1 | 11.8 | 9.9 | 0.6 | 5.8 |
| 8 | 18.0 | 5.8 | 4.6 | 9.1 | 15.3 | 6.4 |
| 9 | -2.7 | 8.8 | 21.0 | 7.8 | 6.2 | 6.9 |

No target passes the full gate. Positive task counts are 6/8/9/6/8/9 for the
six columns, but magnitude, CI, shuffle, and pair conditions still fail.
Task 2 benefits consistently, whereas task 0 regresses on five of six targets;
the aggregate result is not a uniform ten-task mechanism.

For task-level raw relevance, patch mean exceeds global Spearman on object
translation in 9/10 tasks and on relative translation in 9/10. Object and
relative rotation are also usually higher, but task 5 object correlations
remain negative and several differences are small. The full eight-column
global/patch task table is retained in the diagnosis rather than being hidden
behind the aggregate.

## Candidate-pair behavior

| Pair | EE trans | EE rot | Obj trans | Obj rot | Rel trans | Rel rot |
|---|---:|---:|---:|---:|---:|---:|
| A0-A1 | -7.9 | 6.3 | 5.3 | 4.7 | -2.4 | 4.3 |
| A0-A2 | -7.1 | 5.2 | 8.7 | 5.0 | 2.9 | 5.6 |
| A1-A2 | 12.2 | 7.3 | 6.9 | 5.0 | 15.9 | 5.8 |

Only EE rotation, both object targets, and relative rotation are positive on
all three pairs; none reaches the 10% aggregate gate. Translation improvement
is concentrated in the larger A1-A2 contrast, so it cannot claim pair-robust
target relevance.

Raw patch-mean Spearman exceeds global for all object/relative pair cuts—for
example object translation is `0.317/0.350/0.276` for patch versus
`0.209/0.208/0.207` for global—but this does not repair the failed incremental
and shuffle gates.

## Information localization

Rows above calibration-frozen p75 target thresholds do not move the strongest
patches toward relevant objects:

| High-target subset | n | Top-5 robot | Top-5 relevant | Top-5 background | Top-5 interaction |
|---|---:|---:|---:|---:|---:|
| Object translation | 199 | 49.03% | 6.22% | 43.31% | 85.13% |
| Object rotation | 196 | 48.79% | 5.98% | 42.98% | 84.39% |
| Relative translation | 218 | 49.73% | 5.31% | 43.04% | 86.06% |
| Relative rotation | 221 | 48.99% | 5.96% | 43.10% | 86.06% |

These are fractional mask coverages inside the top five changing patches, not
hard labels. High object/relative change raises relevant coverage only modestly
from the all-row `5.33%`. The broad interaction definition includes most top
patches and is not a unique localization result.

## Reproducibility reference to the preceding audit

The previous audit found patch summaries strongly explain global CLIP distance.
This reconstruction is reproduced here without using it as relevance evidence:
patch summaries predict `D_global` with RMSE `0.002012`, R² `0.628`, Pearson
`0.794`, and Spearman `0.857` on init36--39.

The new contribution is the external-target comparison. It shows that natural
patches also track real physical change better in raw association, but it does
not prove that global pooling discarded stable object/interaction information
needed for candidate ranking. Spatial compression is real; task relevance
remains unresolved.

## Mechanical decision

| Target | Increment | Increment CI lower >0 | >=10% | Patch shuffle | Robust tasks/horizons/pairs | Pass |
|---|---:|---|---|---|---|---|
| EE translation | 4.20% | no | no | pass | 6 / 5 / 1 | no |
| EE rotation | 6.86% | yes | no | fail | 8 / 8 / 3 | no |
| Object translation | 6.90% | yes | no | fail | 9 / 8 / 3 | no |
| Object rotation | 4.93% | no | no | fail | 6 / 6 / 3 | no |
| Relative translation | 8.77% | no | no | pass | 8 / 6 / 2 | no |
| Relative rotation | 5.33% | yes | no | fail | 9 / 8 / 3 | no |

No robot, object, or relative family passes. `SPATIAL_REPRESENTATION_ROBOT_SENSITIVE`
is not assigned because even the robot targets fail the complete frozen gate.
`SPATIAL_OBJECT_STATE_SIGNAL_CONFIRMED`,
`SPATIAL_RELATIVE_GEOMETRY_SIGNAL_CONFIRMED`, and
`SPATIAL_INTERACTION_SIGNAL_CONFIRMED` are also not assigned.

The only allowed mechanical label is:

```text
NATURAL_PATCH_TASK_RELEVANCE_UNRESOLVED
```

Authorization remains:

```text
BLOCK_VISUAL_TARGET_REDESIGN
```

This means the next step is not patch-level World Model training. A future
experiment would need a separately preregistered representation comparison
with stronger independent data or a more specific natural object/interaction
target, while preserving capacity and shuffle controls. This report does not
authorize such a design automatically.

## Integrity, NaN/Inf, and artifacts

| Artifact | SHA-256 |
|---|---|
| Protocol | `c5a3d9fc810144271190949d6b2bbb3015f4fc2ac01b1c9c36e1ba4fbd7ea17b` |
| Calibration freeze | `27448a628f41c8a0beed0e9293aaced1767242a911f970c9bb08a5a975cac4e1` |
| Metadata amendment | `8a6878fa46cab1110e1f0bdbd5d4e050ec0aa993f40cfd16a123b70c01d1d6c8` |
| Calibration weights | `b63aafbf11d72d5b626cf1b03875c5569262df08d51281d13cc8793a04c2d5e3` |
| Calibration derived sidecar | `2fd61f169a8645c390c8dc02168a067bcc5fb1bce689caf46edae7103c02c443` |
| Evaluation derived sidecar | `336ccb1507f08645c944af04823df7d907e36f2e9dff527799434b2cd147e85f` |
| Compact diagnosis | `a21e5371927a80b78192afb3c13d6323407907273eee879799d1d0c2d86c4c46` |

The ignored sidecars retain all 960 identities, six external targets, five
representation matrices, all ridge predictions, full 7x7 distance maps,
fractional coverage, interaction indicators, and localization arrays needed
for independent recalculation. All admitted inputs, PCA/ridge/logistic
parameters, predictions, metrics, and bootstrap values are finite. Undefined
correlations are represented as `null`; no missing value is converted into
evidence.

## Forbidden-activity audit

No simulator, masked-image inference, new encoder, World Model training,
patch-level dynamics model, CLIP fine-tuning, actor, candidate generator,
production scale, gripper policy, reranker, or controller was run or modified.
No C1/C2, D20-v2, seed23/47, reward, done, success, final task outcome, or
init40--49 access occurred. Existing configs, reports, results, checkpoints,
sidecars, and frozen verdicts remain unchanged.
