# C1-v2.1 candidate discriminability and reranker diagnosis

Date: 2026-09-24

Final blockers: `RERANKER_SCORE_BLOCK`, `WM_ACTION_SENSITIVITY_BLOCK`

Frozen controller verdict: `BLOCK_C1_V2_RERANKER`

## Executive conclusion

The current N=3 generator does contain useful alternative actions, but the
frozen C1-v2 score cannot identify them and the frozen D15 ensemble responds
too weakly to the production-scale action differences for dependable
model-based ranking.

On all 5,869 legal validation samples from demos 45--49, an alternative is the
expert-action oracle best on 65.87% of samples. Best-of-three reduces raw
action MSE from `0.04198862` to `0.04129654`, a preregistered-gate-passing
1.648% relative gain. The candidate generator therefore does not receive
`CANDIDATE_GENERATOR_BLOCK`.

The frozen score selects the oracle candidate on 32.22% of samples, below both
random top-1 (33.33%) and the trivial majority-oracle-class reference
(34.13%). Default-versus-alternative pairwise accuracy is 49.43%, mean
within-sample Spearman is `-0.0235`, and score-margin versus oracle-improvement
Spearman is `-0.0165`. All four preregistered reranker checks fail.

The actions are not numerically identical: at 1x, mean pairwise normalized
chunk L2 is `1.75294`. Nevertheless, mean pairwise predicted-latent cosine
separation is only `3.7775e-5`, or `0.004235` times ordinary WM validation
cosine error. This is below the preregistered 0.01 ratio. Thus action distance
passes while WM response fails, producing `WM_ACTION_SENSITIVITY_BLOCK`.

No new C1 controller is authorized. D20-v2 and C2-v2 remain blocked; no
controller rollout was run and init states 40--49 were neither read nor run.

## Freeze, preregistration, and scope

The C1-v2 negative result was frozen before this diagnosis:

- commit `bd1dae4698c8659f50830bc92d9c37d552dac109`;
- annotated tag `c1-v2-reranker-negative`;
- frozen verdict `BLOCK_C1_V2_RERANKER`;
- audit SHA-256
  `bca41d9330d57039bbf4e4cff6acd5486bedf25c7256f94456415c1e0aa709e5`;
- gate SHA-256
  `4f56544eaa7c04023342d54c3dd98fd6012522118cf8e6128f9407cb7e2ac2e0`.

Protocol `c1-v2.1-candidate-discriminability-v1` and its implementation were
committed as `4bed772` before any offline result was viewed. The protocol fixes
the production generator, weights, sweep scales, random seed rule, data split,
and numerical decision gates.

The analysis uses exactly 5,869 C0-v2 validation samples: 10 tasks, five demos
per task, demo IDs 45--49. It uses no simulator rollout or init-state record.
The observation store SHA-256 is
`e5599153cf44282242d955be62249bbafc752b782f5ae76a0ec0d58f9146f98e`.
The frozen actor and three frozen ensemble checkpoints are hash-checked before
inference. All candidate construction uses the deployed raw-action,
antithetic-noise, temporal-smoothing, and clipping path.

## Offline candidate oracle

The oracle here is deliberately narrow: it is the candidate with the lowest
raw-action MSE to expert `actions[k+1:k+9]`. It diagnoses whether the generator
contains a useful imitation target; it is not an online oracle available to a
controller.

| Metric | Result |
|---|---:|
| Samples | 5,869 |
| Default raw-action MSE | 0.04198862 |
| Best-of-three raw-action MSE | 0.04129654 |
| Relative oracle gain | 1.648% |
| Default is oracle best | 34.13% |
| Candidate 1 is oracle best | 33.65% |
| Candidate 2 is oracle best | 32.22% |
| Either alternative is oracle best | 65.87% |
| Default normalized-action MSE | 0.35135784 |
| Best-of-three normalized-action MSE | 0.33680890 |

All ten tasks show a positive 1.21--2.37% best-of-three raw-MSE gain. The
alternative oracle-best rate is 57.24--72.80%, so the aggregate result is not
caused by one task.

| Task | n | Default oracle | A1 oracle | A2 oracle | Alt oracle | Oracle gain | Top score=oracle | Pairwise acc. | Within Spearman | Margin Spearman |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 415 | 31.08% | 35.66% | 33.25% | 68.92% | 1.38% | 33.01% | 48.43% | -0.0398 | -0.0028 |
| 1 | 628 | 39.49% | 30.41% | 30.10% | 60.51% | 1.46% | 29.62% | 49.76% | -0.0462 | -0.0182 |
| 2 | 594 | 34.68% | 32.49% | 32.83% | 65.32% | 1.61% | 31.82% | 46.46% | -0.0547 | 0.0619 |
| 3 | 442 | 42.76% | 30.54% | 26.70% | 57.24% | 1.21% | 31.67% | 52.60% | 0.0385 | -0.0739 |
| 4 | 698 | 34.24% | 31.81% | 33.95% | 65.76% | 1.98% | 31.81% | 50.50% | -0.0021 | -0.0310 |
| 5 | 507 | 32.74% | 35.70% | 31.56% | 67.26% | 2.19% | 29.19% | 46.25% | -0.0976 | -0.0244 |
| 6 | 666 | 31.08% | 35.29% | 33.63% | 68.92% | 1.39% | 31.23% | 48.87% | -0.0405 | -0.0248 |
| 7 | 696 | 37.93% | 31.47% | 30.60% | 62.07% | 1.43% | 30.17% | 49.78% | -0.0259 | -0.0423 |
| 8 | 603 | 27.20% | 36.48% | 36.32% | 72.80% | 1.85% | 36.48% | 49.17% | -0.0050 | -0.1157 |
| 9 | 620 | 30.81% | 37.26% | 31.94% | 69.19% | 2.37% | 37.26% | 52.18% | 0.0355 | 0.0905 |

Verdict for the first mechanism: the generator contains measurable useful
alternatives under the preregistered definition. The gain is modest, so this
does not establish that the generator is optimal or sufficient for control.

## Frozen reranker ranking quality

The exact frozen score is:

```text
score = goal_progress
        - 0.25 * uncertainty
        - 0.001 * action_deviation
```

No coefficient was changed. The score selects candidate 0/1/2 on
9.46%/46.31%/44.23% of samples and replaces the default on 90.54%.

| Ranking metric | Result | Preregistered minimum | Pass |
|---|---:|---:|---|
| Top score equals oracle best | 32.22% | 38% | no |
| Random top-1 reference | 33.33% | reference | — |
| Majority-oracle-class reference | 34.13% | reference | — |
| Default/alternative pairwise accuracy | 49.43% | 53% | no |
| Mean within-sample Spearman | -0.0235 | 0.05 | no |
| Mean within-sample Kendall tau | -0.0224 | diagnostic | — |
| Pooled Spearman | 0.00130, p=0.863 | diagnostic | — |
| Pooled Kendall tau | -0.000088, p=0.986 | diagnostic | — |
| Replacement score margin vs oracle improvement Spearman | -0.0165, p=0.230 | 0.05 | no |

The replacement margin is not informative. Across increasing score-margin
quintiles, mean selected-versus-default oracle improvement remains negative
(`-4.83e-4`, `-6.66e-4`, `-5.75e-4`, `-5.68e-4`, `-5.69e-4` raw MSE), while
the fraction where selection is actually better stays near 29.35--33.52%.

Task-level top-score oracle accuracy ranges from 29.19% to 37.26%. No task
reaches the preregistered 38% aggregate threshold, and pairwise accuracy is
near chance on every task. This produces `RERANKER_SCORE_BLOCK`.

## Score-term scale decomposition

The pooled tables below contain all three candidates. “Contribution” is the
term after applying its frozen coefficient.

### Demo validation 45--49

| Component | Coeff. | Mean | Std | p5 | p25 | p50 | p75 | p95 | Contribution mean | Contribution std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| goal progress | 1 | 3.2045e-4 | 8.0717e-3 | -1.3772e-2 | -4.1430e-3 | 5.9822e-4 | 5.0990e-3 | 1.3291e-2 | 3.2045e-4 | 8.0717e-3 |
| uncertainty | -0.25 | 1.4536e-3 | 5.1967e-4 | 7.9618e-4 | 1.0772e-3 | 1.3505e-3 | 1.7300e-3 | 2.4230e-3 | -3.6340e-4 | 1.2992e-4 |
| action deviation | -0.001 | 2.1359e-2 | 1.8291e-2 | 0 | 0 | 2.2783e-2 | 3.3968e-2 | 5.1881e-2 | -2.1359e-5 | 1.8291e-5 |
| total score | 1 | -6.4314e-5 | 8.0640e-3 | -1.4210e-2 | -4.5306e-3 | 2.2460e-4 | 4.7283e-3 | 1.2858e-2 | -6.4314e-5 | 8.0640e-3 |

| Contribution | A0 mean | A1 mean | A2 mean | A1-A0 | A2-A0 | Selected-default on replacements |
|---|---:|---:|---:|---:|---:|---:|
| goal progress | 3.0339e-4 | 3.3617e-4 | 3.2179e-4 | 3.2788e-5 | 1.8399e-5 | 3.5102e-4 |
| uncertainty | -3.6205e-4 | -3.6411e-4 | -3.6404e-4 | -2.0568e-6 | -1.9898e-6 | -1.4147e-6 |
| action deviation | 0 | -3.2040e-5 | -3.2038e-5 | -3.2040e-5 | -3.2038e-5 | -3.1692e-5 |
| total score | -5.8668e-5 | -5.9977e-5 | -7.4297e-5 | -1.3085e-6 | -1.5629e-5 | 3.1791e-4 |

### Controller development 30--39, read-only frozen trace

| Component | Coeff. | Mean | Std | p5 | p25 | p50 | p75 | p95 | Contribution mean | Contribution std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| goal progress | 1 | 8.5997e-4 | 6.5994e-3 | -1.0978e-2 | -2.7228e-3 | 1.4037e-3 | 4.9984e-3 | 1.0637e-2 | 8.5997e-4 | 6.5994e-3 |
| uncertainty | -0.25 | 2.2783e-3 | 1.1430e-3 | 9.5384e-4 | 1.4880e-3 | 1.9874e-3 | 2.7350e-3 | 4.7109e-3 | -5.6959e-4 | 2.8576e-4 |
| action deviation | -0.001 | 2.1611e-2 | 1.8749e-2 | 0 | 0 | 2.2882e-2 | 3.3725e-2 | 5.3296e-2 | -2.1611e-5 | 1.8749e-5 |
| total score | 1 | 2.6877e-4 | 6.5613e-3 | -1.1560e-2 | -3.3207e-3 | 8.3863e-4 | 4.3803e-3 | 9.9601e-3 | 2.6877e-4 | 6.5613e-3 |

| Contribution | A0 mean | A1 mean | A2 mean | A1-A0 | A2-A0 | Selected-default on replacements |
|---|---:|---:|---:|---:|---:|---:|
| goal progress | 8.3924e-4 | 8.7158e-4 | 8.6909e-4 | 3.2348e-5 | 2.9850e-5 | 3.7856e-4 |
| uncertainty | -5.7001e-4 | -5.6912e-4 | -5.6963e-4 | 8.9816e-7 | 3.8552e-7 | 1.5351e-6 |
| action deviation | 0 | -3.2417e-5 | -3.2416e-5 | -3.2417e-5 | -3.2416e-5 | -3.2088e-5 |
| total score | 2.6922e-4 | 2.7005e-4 | 2.6704e-4 | 8.2894e-7 | -2.1805e-6 | 3.4800e-4 |

The historical controller-development replacement gain of about `3.48e-4`
comes almost entirely from selecting positive goal-progress fluctuations
(`+3.7856e-4`). The action-deviation term offsets only `-3.2088e-5`, and the
uncertainty contribution changes by just `+1.5351e-6`. Validation shows the
same mechanism. Therefore `0.001 * action_deviation` is small in ranking
scale, uncertainty usually does not change the ordering, and goal progress
almost alone chooses the candidate. Formula back-check error is zero on
validation and `1.57e-9` on the frozen rollout trace.

## Candidate action separation

At the production 1x perturbation, all-candidate pre-clip element clipping is
0.1082%. The candidates are clearly different in action space:

| Pair | Raw chunk L2 | Normalized chunk L2 | Max absolute difference |
|---|---:|---:|---:|
| A0-A1 | 0.171769 | 1.314761 | 0.073462 |
| A0-A2 | 0.171729 | 1.314728 | 0.073434 |
| A1-A2 | 0.343279 | 2.629329 | 0.146708 |

Mean absolute difference by action dimension `[dx,dy,dz,drx,dry,drz,gripper]`:

```text
A0-A1 = [0.025936, 0.026134, 0.026212, 0.010509, 0.010404, 0.010508, 0]
A0-A2 = [0.025921, 0.026130, 0.026205, 0.010509, 0.010404, 0.010508, 0]
A1-A2 = [0.051857, 0.052264, 0.052416, 0.021019, 0.020807, 0.021016, 0]
```

Mean raw action-vector L2 by horizon:

```text
A0-A1 = [0.068892, 0.053813, 0.054152, 0.054027, 0.053615, 0.053454, 0.053572, 0.069103]
A0-A2 = [0.068841, 0.053764, 0.054150, 0.054039, 0.053580, 0.053439, 0.053579, 0.069129]
A1-A2 = [0.137641, 0.107536, 0.108279, 0.108032, 0.107120, 0.106837, 0.107092, 0.138135]
```

The gripper difference is intentionally zero under the frozen generator.
There is no numerical-identity explanation for the reranker failure.

## Frozen-WM action sensitivity

Each A0/A1/A2 candidate is independently WM-normalized and passed through all
three frozen D15 members. The table reports pairwise separation of ensemble
mean predictions, alongside ordinary D16 ensemble validation error.

| Horizon | Predicted cosine separation | Predicted L2 separation | Ordinary WM cosine error | Separation/error ratio |
|---:|---:|---:|---:|---:|
| h1 | 1.5991e-5 | 0.004895 | 0.006296 | 0.002540 |
| h2 | 2.4451e-5 | 0.006016 | 0.007559 | 0.003235 |
| h3 | 3.1482e-5 | 0.006832 | 0.008385 | 0.003755 |
| h4 | 3.7718e-5 | 0.007471 | 0.008990 | 0.004196 |
| h5 | 4.2824e-5 | 0.007948 | 0.009452 | 0.004531 |
| h6 | 4.6607e-5 | 0.008299 | 0.009845 | 0.004734 |
| h7 | 4.9350e-5 | 0.008556 | 0.010209 | 0.004834 |
| h8 | 5.3780e-5 | 0.008971 | 0.010617 | 0.005065 |
| Mean | 3.7775e-5 | 0.007374 | — | 0.004235 |

Pairwise mean trajectory cosine distances are `1.9740e-5` (A0-A1),
`1.9595e-5` (A0-A2), and `7.3991e-5` (A1-A2). Despite normalized action
separation `1.75294`, predicted separation is only 0.42% of ordinary model
error on average, and no horizon reaches 0.51%. The WM action response is
therefore substantially below its own ordinary prediction-error scale at the
production perturbation, yielding `WM_ACTION_SENSITIVITY_BLOCK`.

## Diagnostic perturbation-scale sweep

The sweep was preregistered and is diagnostic only. It does not select a new
production scale and no controller was run.

| Scale | Pre-clip clipping | Raw action L2 | Norm. action L2 | Oracle gain | Alt oracle | WM cosine separation | Sep./error | Top-score oracle | Pairwise acc. | Within Spearman |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0x | 0 | 0 | 0 | 0 | 0 | numerical zero | numerical zero | degenerate | — | — |
| 0.25x | 0.060% | 0.05730 | 0.43829 | 0.599% | 91.12% | 2.4636e-6 | 0.000276 | 42.73% | 48.12% | -0.0386 |
| 0.5x | 0.074% | 0.11456 | 0.87654 | 1.059% | 82.21% | 9.8716e-6 | 0.001107 | 38.17% | 48.20% | -0.0371 |
| 1x | 0.108% | 0.22893 | 1.75294 | 1.648% | 65.87% | 3.7775e-5 | 0.004235 | 32.22% | 49.43% | -0.0235 |
| 2x | 0.208% | 0.45663 | 3.50493 | 1.981% | 41.40% | 1.3054e-4 | 0.014636 | 27.88% | 52.20% | 0.0156 |

The sweep does not reveal a scale where all three mechanisms are healthy.
Small scales provide little WM separation. At 2x, the sensitivity ratio finally
exceeds 0.01, but the default becomes oracle-best more often, top-score oracle
accuracy falls to 27.88%, and the scale was never authorized for production.
These observations diagnose the present mechanism; they are not a request to
run a 2x controller.

## Interpretation of the old observed-mismatch gate

The C1-v2 audit compared the selected-candidate prediction and the
default-candidate prediction against the same future latent produced by the
selected-policy rollout. The default action was not executed in a matched
counterfactual environment. That comparison is valid as a WM
candidate-discriminability diagnostic, but it cannot establish that the
default candidate would actually have controlled better. It must not, by
itself, drive a controller change.

This offline expert-action oracle answers a different question—whether an
alternative is closer to the expert target—but likewise does not claim that
raw-action MSE is a sufficient online control objective.

## Decision tree and authorization

| Mechanism | Preregistered checks | Result | Verdict |
|---|---|---|---|
| Candidate generator | Oracle gain >=1%; alternative oracle fraction >=10% | 1.648%; 65.87% | pass |
| Reranker score | top-1 >=38%; pairwise >=53%; within Spearman >=0.05; margin Spearman >=0.05 | all four fail | `RERANKER_SCORE_BLOCK` |
| WM action sensitivity | normalized action L2 >=0.25; separation/error >=0.01 | 1.75294 passes; 0.004235 fails | `WM_ACTION_SENSITIVITY_BLOCK` |

Because the candidate set is useful but both recognition mechanisms fail, the
result is not `NO_MECHANISM_BLOCK`. A new C1 controller may be designed only
after separate evidence establishes both meaningful WM action sensitivity and
an oracle-related ranking signal. This round does not change weights, choose a
new scale, train any model, or authorize another controller rollout.

## Reproducibility and forbidden-field audit

Aggregate result:
`results/c1_v2_1/candidate_diagnosis.json`, SHA-256
`6fd0c3488b3e57fa098a8f5b9071e4e99912cc60f8b54774b4b1a857b6014339`.

Ignored sample sidecar:
`artifacts/stage3/c1_v2_1_candidate_diagnosis/samples.npz`, SHA-256
`5870337ad96c58b3cedee4e124cf89018408446ed82fc7a2df52626f482398fd`.
It contains 5,869 task/demo/timestep identities and the candidate errors,
scores, action tensors, and pairwise WM separations needed for independent
recalculation. All stored values are finite; all demo IDs are 45--49 and all
ten tasks are present.

Automated tests revalidate protocol hashes, frozen negative artifacts/tag,
the exact deployed candidate interface, sample identities and array shapes,
sidecar SHA-256, independent oracle gain/top-1 calculations, decision mapping,
finite values, and forbidden activity.

Forbidden-activity audit: score weights were not changed; no production scale
was selected; no controller rollout or 40--49 access occurred; D20-v2 and
C2-v2 were not started; C0-v2 and the world model were not retrained; C1-v2
negative artifacts were not overwritten; controller success was not used for
parameter iteration.
