# The frozen world model uses actions, but its local candidate sensitivity is too weak

Date: 2026-09-24

Final verdict: `WM_HAS_ACTION_SIGNAL_BUT_LOW_LOCAL_SENSITIVITY`

Frozen upstream blockers remain: `RERANKER_SCORE_BLOCK`,
`WM_ACTION_SENSITIVITY_BLOCK`.

## Executive summary

The D15 ensemble does not ignore its action input. Replacing the correct
causal chunk with a different whole chunk from the same task raises validation
cosine error from `0.00891906` to `0.01139824` (+27.80%). The correct-action
prediction wins the paired comparison on 78.11% of task/horizon points, with a
task-stratified episode-block 95% interval of [77.04%, 79.13%]. Physical zero
and cross-task chunks give the same qualitative result. All four
preregistered action-signal checks pass.

The signal is sequence-aware. Reversing the action order increases error by
5.04%, while the fixed temporal permutation increases it by 0.93%; both have
strictly positive episode-block intervals. Magnitude changes also produce a
strong monotonic response: prediction separation from 1x tracks
`abs(scale-1)` with Spearman `0.9747`, and the correct 1x chunk has the lowest
error among the five preregistered scales.

The failure is local rather than global. A within-task shuffled chunk moves
the predicted latent by 27.18% of ordinary WM error, but C1-v2.1's production
candidate perturbations move it by only 0.4235% of ordinary error and 0.2392%
of real temporal latent movement. The model distinguishes substantially wrong
or rescaled chunks while barely separating the nearby candidates that C1 asks
it to rank.

This is not an ensemble-cancellation or aggregate-only result. All three
members independently identify correct actions at 77.68--78.77%; all ten
tasks are above 74%; and member action-effect directions are positively
aligned. Real CLIP latent movement is 1.77 times ordinary WM error and h8
median movement is `0.01956`, so the preregistered
`CLIP_STATE_SENSITIVITY_LIMIT` rule does not fire.

The next priority is a future action-aware WM training objective that improves
local action discrimination. This diagnosis does not authorize that training,
candidate changes, reranker tuning, a controller rollout, D20-v2, or C2-v2.

## Freeze, preregistration, and data boundary

C1-v2.1 was frozen before this analysis:

- commit `196e43dbc5bc0989c4a7781fb97d20b626ca8b63`;
- annotated tag `c1-v2-1-discriminability-negative`;
- aggregate result SHA-256
  `6fd0c3488b3e57fa098a8f5b9071e4e99912cc60f8b54774b4b1a857b6014339`;
- report SHA-256
  `6b08ed9524588421622490856b6b7b85e33e652396524ff5cb7fedee8369f210`.

The complete C1-v2.2 protocol, interpretation thresholds, bootstrap design,
condition construction, and implementation were committed as
`b2e0e68469a2c09190599b04aec308b419397b26` before the first real ablation
result was viewed. Protocol SHA-256 is
`b71429a23da51c9b1e1c3864dd76b15ec994cef561f6d8979b56d59da9081b7a`.

The data audit finds exactly 5,869 unique `(task_id, demo_id,
start_timestep)` samples, 50 episodes, all ten tasks, and demo IDs 45--49.
There are no simulator rollouts and no init-state records. The D13 store
SHA-256 remains
`e5599153cf44282242d955be62249bbafc752b782f5ae76a0ec0d58f9146f98e`.
The causal contract remains:

```text
(z[k], proprio[k], language, actions[k+1:k+9]) -> z[k+1:k+9]
```

H is 8. The D15 config and all three member checkpoints are frozen and
hash-checked. The correct-action inference reproduces the frozen D16
per-horizon values within `2e-7` absolute tolerance.

## Exact action ablations

Only raw physical actions change. Every condition follows:

```text
raw physical [8,7]
  -> frozen train-only WM action normalization
  -> frozen D15 members
  -> member predictions [3,8,512]
  -> normalized ensemble mean [8,512]
```

`z0`, proprio, language, target, architecture, checkpoints, and normalization
remain identical for paired conditions.

- **Correct:** original `actions[k+1:k+9]`.
- **Zero:** physical-space zeros, not normalized-space zeros. After frozen
  normalization, physical zero becomes
  `[-0.3622,-0.3876,0.3056,0.1299,0.1553,0.3435,-0.1016]`.
- **Within-task shuffle:** fixed-seed whole-chunk derangement. Every source
  sample differs from its target identity and remains in the same task.
- **Cross-task shuffle:** fixed-seed whole-chunk draw from a different task;
  the original sample's language remains unchanged.
- **Temporal:** reverse `[7,6,5,4,3,2,1,0]`, fixed permutation
  `[2,5,0,7,3,1,6,4]`, and repeat-first.
- **Magnitude:** raw `0x`, `0.25x`, `0.5x`, `1x`, and `1.5x`, without
  clipping, followed by the normal WM interface.

Shuffle always moves a complete eight-step chunk. No timestep-wise mixing is
used.

## Correct actions improve prediction and are identifiable

The primary error is mean cosine distance to the real `z_future`. Delta is
wrong-condition error minus correct error. Positive values favor the correct
action.

| Condition | Error | Delta | Relative delta | Correct-action wins | Episode-block 95% CI |
|---|---:|---:|---:|---:|---:|
| Correct | 0.00891906 | — | — | — | — |
| Physical zero | 0.01123345 | +0.00231439 | +25.95% | 73.59% | [72.09%, 75.03%] |
| Within-task shuffle | 0.01139824 | +0.00247918 | +27.80% | 78.11% | [77.04%, 79.13%] |
| Cross-task shuffle | 0.01138074 | +0.00246168 | +27.60% | 78.85% | [77.78%, 79.87%] |

Delta intervals are strictly positive: `[0.002166,0.002473]` for zero,
`[0.002365,0.002594]` for within-task shuffle, and
`[0.002340,0.002581]` for cross-task shuffle. The bootstrap resamples the five
validation demos with replacement inside every task and preserves all rows in
each sampled episode; 10,000 fixed-seed replicates were used.

| Horizon | Correct | Zero | Within shuffle | Cross shuffle | Delta zero | Delta within | Delta cross | Win zero | Win within | Win cross | Within-win 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h1 | 0.00629586 | 0.00644426 | 0.00656802 | 0.00656701 | 0.00014840 | 0.00027215 | 0.00027115 | 58.60% | 65.02% | 65.82% | [63.69%, 66.36%] |
| h2 | 0.00755894 | 0.00804989 | 0.00827840 | 0.00826792 | 0.00049095 | 0.00071946 | 0.00070898 | 63.25% | 70.93% | 70.63% | [69.60%, 72.28%] |
| h3 | 0.00838467 | 0.00943715 | 0.00970399 | 0.00972248 | 0.00105248 | 0.00131932 | 0.00133781 | 69.14% | 74.82% | 76.32% | [73.35%, 76.27%] |
| h4 | 0.00899007 | 0.01074180 | 0.01100949 | 0.01103339 | 0.00175173 | 0.00201942 | 0.00204332 | 73.10% | 78.34% | 79.03% | [77.00%, 79.65%] |
| h5 | 0.00945218 | 0.01199193 | 0.01219657 | 0.01218286 | 0.00253976 | 0.00274440 | 0.00273068 | 77.36% | 81.41% | 81.29% | [79.97%, 82.84%] |
| h6 | 0.00984455 | 0.01321132 | 0.01336409 | 0.01331477 | 0.00336677 | 0.00351954 | 0.00347022 | 80.37% | 83.06% | 84.02% | [81.95%, 84.19%] |
| h7 | 0.01020906 | 0.01439254 | 0.01446270 | 0.01442712 | 0.00418347 | 0.00425364 | 0.00421806 | 82.64% | 85.13% | 85.89% | [83.94%, 86.21%] |
| h8 | 0.01061716 | 0.01559873 | 0.01560270 | 0.01553040 | 0.00498157 | 0.00498555 | 0.00491324 | 84.27% | 86.13% | 87.80% | [84.84%, 87.30%] |

The action signal strengthens with horizon, as expected when wrong actions
compound. Correct actions outperform wrong actions at every horizon and all
three aggregate identification intervals exclude the 50% random reference.

## Temporal ordering matters

| Temporal condition | Error | Delta | Relative delta | Correct-order wins | 95% win CI |
|---|---:|---:|---:|---:|---:|
| Reverse | 0.00936895 | +0.00044989 | +5.04% | 63.20% | [61.77%, 64.62%] |
| Fixed permutation | 0.00900189 | +0.00008283 | +0.93% | 55.68% | [54.30%, 57.03%] |
| Repeat first | 0.00911572 | +0.00019666 | +2.20% | 49.91% | [48.93%, 50.87%] |

The preregistered temporal check uses the fixed permutation, not whichever
condition looks strongest after analysis. Its delta 95% interval is
`[0.00006394,0.00010221]`, strictly above zero. The per-horizon view explains
the repeat-first accuracy: h1 is exactly tied because the first action is
unchanged, while its error grows increasingly above correct from h2 to h8.

| Horizon | Reverse delta | Permutation delta | Repeat-first delta | Reverse win | Permutation win | Repeat-first win |
|---:|---:|---:|---:|---:|---:|---:|
| h1 | 0.00012340 | 0.00002011 | 0 | 58.78% | 53.64% | 0% |
| h2 | 0.00030861 | 0.00008533 | 0.00000161 | 62.43% | 56.13% | 50.35% |
| h3 | 0.00048720 | 0.00007653 | 0.00001061 | 64.80% | 55.89% | 51.93% |
| h4 | 0.00059832 | 0.00014140 | 0.00004260 | 65.68% | 57.44% | 53.40% |
| h5 | 0.00061949 | 0.00012820 | 0.00011823 | 65.43% | 57.13% | 56.07% |
| h6 | 0.00056270 | 0.00010063 | 0.00025260 | 64.20% | 56.59% | 59.35% |
| h7 | 0.00048121 | 0.00006411 | 0.00044677 | 62.86% | 54.98% | 62.40% |
| h8 | 0.00041819 | 0.00004634 | 0.00070083 | 61.41% | 53.67% | 65.82% |

This rules out the interpretation that the GRU uses only a weak
order-invariant statistic of the entire chunk.

## Action magnitude produces a structured response

Scaling occurs before normalization and is not clipped. At 1.5x, 22.99% of
raw elements have magnitude above 1; this is an offline sensitivity probe, not
an executable-action proposal.

| Raw scale | Prediction error | Predicted displacement from z0 | Separation from 1x | Separation / WM error | Raw range |
|---:|---:|---:|---:|---:|---:|
| 0x | 0.01123345 | 0.00455434 | 0.00238474 | 26.74% | [0,0] |
| 0.25x | 0.01037285 | 0.00505768 | 0.00149488 | 16.76% | [-0.25,0.25] |
| 0.5x | 0.00944342 | 0.00619225 | 0.00054131 | 6.07% | [-0.5,0.5] |
| 1x | 0.00891906 | 0.00818403 | numerical zero | numerical zero | [-1,1] |
| 1.5x | 0.00906901 | 0.00926166 | 0.00012600 | 1.41% | [-1.5,1.5] |

Correct 1x actions give the lowest error. Predicted displacement rises
monotonically with scale (Spearman `1.000`, p=`1.40e-24`). Separation from 1x
rises with absolute scale distance (Spearman `0.9747`, p=`0.00482`), satisfying
the preregistered magnitude-response check. Pairwise scale separation ranges
from `0.0001260` cosine (1x versus 1.5x) to `0.0032395` (0x versus 1.5x).

The asymmetric sensitivity is meaningful: reducing action magnitude below 1x
has a larger latent effect than increasing it to 1.5x. This diagnostic does
not establish that out-of-range 1.5x actions are physically valid.

## Real representation motion is visible; production candidate motion is not

Real latent displacement is `1-cos(z[k],z[k+h])`, computed directly from the
stored validation trajectory without the WM.

| Horizon | Real mean | Median | p75 | p90 | p95 | WM error | Candidate separation | Candidate / WM error | Candidate / real motion |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h1 | 0.00702432 | 0.00581950 | 0.00879514 | 0.01282045 | 0.01620257 | 0.00629586 | 0.00001599 | 0.2540% | 0.2276% |
| h2 | 0.01017509 | 0.00846702 | 0.01285696 | 0.01883307 | 0.02316991 | 0.00755894 | 0.00002445 | 0.3235% | 0.2403% |
| h3 | 0.01295710 | 0.01077676 | 0.01667523 | 0.02405620 | 0.02931496 | 0.00838467 | 0.00003148 | 0.3755% | 0.2430% |
| h4 | 0.01541427 | 0.01303732 | 0.01978004 | 0.02873369 | 0.03490311 | 0.00899007 | 0.00003772 | 0.4196% | 0.2447% |
| h5 | 0.01758326 | 0.01488066 | 0.02247119 | 0.03246148 | 0.03959787 | 0.00945218 | 0.00004282 | 0.4531% | 0.2436% |
| h6 | 0.01946333 | 0.01659626 | 0.02503389 | 0.03563240 | 0.04366117 | 0.00984455 | 0.00004661 | 0.4734% | 0.2395% |
| h7 | 0.02108554 | 0.01809019 | 0.02723151 | 0.03842132 | 0.04603933 | 0.01020906 | 0.00004935 | 0.4834% | 0.2340% |
| h8 | 0.02265955 | 0.01955897 | 0.02924263 | 0.04100457 | 0.04905518 | 0.01061715 | 0.00005378 | 0.5065% | 0.2373% |

Across all horizons, real displacement has mean `0.01579531`, median
`0.01264381`, p75 `0.02078970`, p90 `0.03110445`, and p95 `0.03901073`.
The mean is 1.771 times ordinary WM error. The preregistered representation
limit required both a ratio no greater than 1.25 and h8 median no greater than
0.01; measured values are 1.771 and `0.01956`. Therefore
`CLIP_STATE_SENSITIVITY_LIMIT` is not assigned.

The scale of the local failure is stark:

| Action contrast | Predicted separation / WM error | Predicted separation / real motion |
|---|---:|---:|
| Physical zero vs correct | 26.74% | 15.10% |
| Within-task shuffle vs correct | 27.18% | 15.35% |
| Cross-task shuffle vs correct | 26.65% | 15.05% |
| Reverse vs correct | 4.91% | 2.77% |
| Fixed temporal permutation vs correct | 0.884% | 0.499% |
| Frozen C1-v2.1 production candidates | 0.4235% | 0.2392% |

CLIP captures temporal change at a scale above ordinary WM error. The WM also
reacts to large action interventions. What falls below noise is specifically
the latent difference induced by the current nearby candidate perturbations.

## All three members use actions; averaging only mildly attenuates the effect

| Member seed | Correct error | Within-shuffle error | Delta | Correct-action wins | Action-induced separation |
|---:|---:|---:|---:|---:|---:|
| 11 | 0.00937134 | 0.01213288 | 0.00276154 | 78.77% | 0.00294876 |
| 23 | 0.00947513 | 0.01195647 | 0.00248134 | 77.68% | 0.00266233 |
| 47 | 0.00941024 | 0.01224978 | 0.00283954 | 78.51% | 0.00303855 |

Every member has positive delta at all eight horizons. The ensemble-mean
within-shuffle separation is `0.00242387`; mean individual-member separation
is `0.00288321`, a ratio of 0.841. Averaging attenuates the action effect by
about 16%, but does not erase it.

Member-pair action-effect direction cosine is `0.6689`, `0.6485`, and
`0.6559`; negative-direction fractions are only 0.16%, 0.52%, and 0.35%.
Response directions become more aligned at later horizons. There is no
evidence that one action-aware member is cancelled by two oppositely
responding members.

## The action signal is present across all ten tasks

| Task | n | Correct error | Within shuffle | Delta | Relative delta | Correct-action wins | 95% CI | Real displacement |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 415 | 0.00862349 | 0.01209784 | 0.00347435 | 40.29% | 83.89% | [80.51%,86.96%] | 0.01744844 |
| 1 | 628 | 0.00651472 | 0.00808006 | 0.00156534 | 24.03% | 78.28% | [72.33%,83.66%] | 0.01005330 |
| 2 | 594 | 0.00882847 | 0.01083349 | 0.00200502 | 22.71% | 74.79% | [69.01%,79.34%] | 0.01500397 |
| 3 | 442 | 0.00943375 | 0.01309884 | 0.00366510 | 38.85% | 83.63% | [81.93%,84.94%] | 0.01964903 |
| 4 | 698 | 0.01055961 | 0.01250212 | 0.00194250 | 18.40% | 74.28% | [73.38%,75.50%] | 0.01731996 |
| 5 | 507 | 0.00760456 | 0.00961809 | 0.00201353 | 26.48% | 76.36% | [73.78%,79.38%] | 0.01358102 |
| 6 | 666 | 0.01082024 | 0.01481316 | 0.00399291 | 36.90% | 81.06% | [79.19%,83.07%] | 0.02088407 |
| 7 | 696 | 0.00862525 | 0.01031948 | 0.00169423 | 19.64% | 75.20% | [72.40%,77.94%] | 0.01477467 |
| 8 | 603 | 0.00905333 | 0.01122425 | 0.00217092 | 23.98% | 75.48% | [71.25%,78.56%] | 0.01531674 |
| 9 | 620 | 0.00865712 | 0.01154458 | 0.00288746 | 33.35% | 81.67% | [79.76%,83.79%] | 0.01475481 |

Every task has positive relative delta above 18% and correct-action accuracy
above 74%. The task accuracy range is `0.0960`, below the preregistered 0.10
spread threshold; ten tasks are above 55% and none are at or below 50%.
No task has a non-positive delta. Neither task-dependence rule passes, so
`TASK_DEPENDENT_ACTION_SIGNAL` is not assigned.

## Preregistered interpretation

| Grouped action-signal check | Result | Pass |
|---|---|---|
| Within-shuffle relative error and positive delta CI | +27.80%; CI lower `0.002365` | yes |
| Within-shuffle ranking and CI above random | 78.11%; CI lower 77.04% | yes |
| Fixed-permutation temporal effect | +0.93%; delta CI lower `0.00006394` | yes |
| 0x-vs-1x magnitude separation and monotonicity | 26.74% of WM error; Spearman 0.9747 | yes |

All four grouped checks pass; the rule required three, including one primary
correct-versus-within check. Therefore `WM_ACTION_IGNORING` is rejected.

The frozen production candidate separation/error ratio remains `0.004235`,
below the preregistered `0.01` local-sensitivity threshold. With strong global
action conditioning but weak local candidate response, the only final verdict
is:

`WM_HAS_ACTION_SIGNAL_BUT_LOW_LOCAL_SENSITIVITY`

No representation-limit or task-dependent secondary verdict applies, and the
conditions for `NO_ACTION_CONDITIONING_BLOCK` are not met.

## What should change next

The first priority is **World Model training**, specifically a future
action-aware objective that makes nearby action chunks distinguishable—for
example, local action-contrast or counterfactual discrimination evaluated on
held-out transitions. That work is not implemented here.

- Do not tune the reranker first: it cannot reliably rank trajectories whose
  predicted differences are below ordinary WM error, and C1-v2.1 already
  showed its frozen score has no oracle-ranking signal.
- Do not prioritize a larger-noise candidate generator: the current generator
  already contains modest oracle headroom, while the previous 2x diagnostic
  worsened frozen score ranking and was never authorized for control.
- Do not prioritize state-representation replacement from this evidence:
  real CLIP temporal movement is measurable and exceeds ordinary WM error.
- After a separately authorized action-aware WM experiment passes a local
  sensitivity gate, reranker diagnosis can resume before any controller run.

Expert actions are used only as an imitation-oriented validation target.
Closeness to expert action does not prove that a candidate maximizes closed-loop
task success.

## Optional Jacobian analysis

The preregistered protocol skips the optional Jacobian. A faithful full
`[8,512]` by `[8,7]` Jacobian across representative samples is expensive;
using the gradient of an arbitrary scalar surrogate would not answer the
requested Jacobian-norm question. The finite-difference ablations above
directly measure prediction response at controlled action distances and are
the basis of the verdict. No gradient magnitude is used to design a rule.

## Reproducibility and forbidden-activity audit

Aggregate result:
`results/c1_v2_2/wm_action_conditioning.json`, SHA-256
`1a3e357d7330ef9658a645cbf0b4d3a7f648b5498fd83eb54b608bb10a61528e`.

Ignored sample sidecar:
`artifacts/stage3/c1_v2_2_wm_action_conditioning/samples.npz`, SHA-256
`24f8a8d870237dda04f63065457c92565f9e4b9b0633a8611b6ec23e089f33fd`.
It retains all identities, shuffle source indices, ensemble/member errors,
real and predicted displacements, action-induced separations, scale-pair
separations, and member effect-direction cosines. All numeric arrays are
finite and their shapes are recorded in the aggregate result.

Frozen ensemble member SHA-256 values:

```text
seed 11  12304c9da129f93325201fad84ce38796ca71fcf427821f64644d4203f6b3888
seed 23  cbed3b53a6d770f9995877d6400e188a72b77ef68a3b077159885312df754805
seed 47  4ed392ec7d1204fd3b0e453415608f54f501f9d0aca29615253a32e9462b43a3
```

No World Model was trained or modified. Loss, architecture, candidate
generator, and reranker weights are unchanged. No controller or simulator was
run; init states 0--49 were not read or run; controller success was not used;
D20-v2 and C2-v2 were not started; old C1-v2/C1-v2.1 artifacts were not
overwritten.
