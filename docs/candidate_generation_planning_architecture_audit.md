# More one-shot candidates add imitation headroom, while iterative search does not justify a model-based planner

Date: 2026-10-02

Final mechanism labels:

```text
CURRENT_N3_CANDIDATE_SET_UNDERPOWERED
MORE_LOCAL_SAMPLES_ADD_HEADROOM
GRIPPER_SEARCH_ADDS_HEADROOM
```

Next-stage planner authorization: `GO_EXPANDED_CANDIDATE_PLANNER_DESIGN`

Spatial World Model authorization: `BLOCK_SPATIAL_WM_PROTOTYPE`

Controller authorization: `false`

## Executive answer

The current three-candidate architecture is too small under the frozen
imitation oracle, but the evidence does not support jumping to a model-based
iterative planner. On the frozen development evaluation, P0 independently
reproduces a best-of-three expert-action MSE gain of `1.683%` (branch-bootstrap
95% CI `[1.565%,1.827%]`), close to the historical `1.648%` measured on a
different split. Keeping the same production perturbation family and increasing
the nested pool raises the gain monotonically to `3.590%`, `4.833%`, `5.894%`,
and `6.819%` at N=8/16/32/64. N=32 and N=64 clear the preregistered 5% gate,
positive-CI condition, and 7/10-task rule. This assigns
`CURRENT_N3_CANDIDATE_SET_UNDERPOWERED` and
`MORE_LOCAL_SAMPLES_ADD_HEADROOM`.

The strongest one-shot diagnostic is gripper-only N=32, with `34.409%` gain,
CI `[33.058%,35.793%]`, positive results on all ten tasks, and `51.940%`
gripper-dimension oracle gain. This assigns `GRIPPER_SEARCH_ADDS_HEADROOM`, but
it is not a production recommendation: the old generator intentionally froze
gripper, the diagnostic spends an entire pool on that dimension, and `5.504%`
of all proposed action components are clipped (`38.53%` of gripper components,
the only perturbed dimension). A future design must treat the binary-like
expert gripper convention and clipping explicitly.

Broader non-gripper search helps, but no tested refinement earns its own
mechanism label. Translation-only N=32 reaches `6.093%`, only `0.198`
percentage points above the matched full non-gripper pool. Rotation-only
reaches only `0.472%` on full-chunk raw MSE. A fixed 2x scale reaches `7.890%`,
whereas the mixed-scale N=32 pool reaches `6.424%`, just `0.529` points above
1x. The broader P5 mixture reaches `7.217%`.

The oracle CEM-like diagnostic rises from `5.820%` after 32 evaluations to
`9.712%` after 128. Its final gain is `1.822` points above the best comparable
non-gripper one-shot result (2x N=32), with paired CI
`[1.521,2.154]` points. That is real incremental value, but it is below the
frozen 5-point requirement for saying iterative search significantly exceeds
one-shot. It also remains below the 10% strong-headroom line. Therefore
`ITERATIVE_ACTION_SEARCH_HAS_ORACLE_VALUE`,
`ONE_SHOT_RERANKING_ARCHITECTURE_LIMIT`, and
`GO_MODEL_BASED_PLANNER_DESIGN` are not assigned.

The mechanically supported next step is only a separately preregistered
expanded-candidate planner design. It may retain C0 as the center/default,
evaluate a larger one-shot pool, and explicitly address gripper proposals. It
does not authorize a World Model, reranker, goal/value function, controller
rollout, or task-success experiment.

## Conceptual separation: three different components

This audit keeps three architectural roles separate:

| Component | Role | Evaluated here |
|---|---|---|
| World Model | Predict consequences of candidate actions | no |
| Planner / proposal search | Construct and search candidate action chunks | yes, through offline oracle headroom only |
| Goal, value, or cost | Judge whether a predicted future is desirable | no; remains unresolved |

The C1-v2.1 system was a local candidate reranker: it generated A0 and one
antithetic pair, then attempted to rank them. A World Model did not itself
search the action space, and this audit does not treat it as a planner.

Expert-action MSE is used only to estimate candidate-search imitation
headroom. It is unavailable at deployment and is not the future planning
objective, task success, reward, or control value.

## Frozen provenance and preregistration

The visible-area extraction block was frozen at commit
`346554a125d83bfbbd7471a8e48385581922c849` and annotated tag
`planning-architecture-reaudit-start`. Its formal state remains:

```text
AREA_REGRESSION_ACCURATE_BUT_DECISION_SIGNAL_INSUFFICIENT
NATURAL_SPATIAL_EXTRACTION_BLOCK
BLOCK_SPATIAL_WM_PROTOTYPE
```

Protocol, candidate families, seed rules, CEM update, metrics, 10,000-draw
bootstrap, thresholds, labels, and decision mapping were committed at
`15194ec` before calibration. Calibration artifacts were frozen at `0d18cd1`
before demos36--39 were opened. The one-shot numerical diagnosis was frozen at
`58fa479` before this report.

The meaningful-headroom rule was frozen as all of:

- relative raw expert-action MSE gain at least 5%;
- task-stratified demo-block bootstrap 95% CI lower bound above zero;
- positive gain on at least 7/10 tasks.

The secondary strong line is 10%. Expanded one-shot authorization additionally
requires at least a 2-percentage-point gain over P0 with positive paired CI.
Model-based authorization requires iteration 4 to exceed the best one-shot by
at least 5 points with positive paired CI. None of these values was changed
after evaluation.

## Data boundary and offline sample construction

| Partition | Demo/init IDs | Tasks | Legal windows | Use |
|---|---|---:|---:|---|
| Calibration | 31--34 | 10 | 4,724 | validity and frozen calibration result |
| Frozen development evaluation | 36--39 | 10 | 4,563 | one-shot metrics, bootstrap, decision |
| Historical reference | 30/35 | 10 | not opened | reference only |
| Forbidden | 40--49 | — | 0 | never accessed |

For every explicitly admitted HDF5 group, legal sample `k` is:

```text
natural vision[k], language[task], proprio[k]
    -> frozen C0-v2
    -> A0 action chunk

expert target = actions[k+1:k+9]
```

The dedicated loader opens only the four admitted demo groups per task. It
does not instantiate the generic C0 training dataset, which would enumerate
forbidden demos40--44. Evaluation is a frozen development evaluation, not a
blind confirmation. No simulator ran.

## P0 exact production reproduction

P0 retains the production architecture exactly:

```text
A0 = clipped frozen C0-v2 action chunk
A1 = clip(A0 + epsilon)
A2 = clip(A0 - epsilon)

epsilon std = [0.05,0.05,0.05,0.02,0.02,0.02,0]
temporal kernel = [0.25,0.5,0.25]
```

| Quantity | Historical demos45--49 | New demos36--39 |
|---|---:|---:|
| A0 raw expert-action MSE | 0.041989 | 0.028485 |
| Best-of-3 raw MSE | 0.041297 | 0.028006 |
| Relative gain | 1.648% | 1.683% |
| Non-default winner fraction | 65.87% | 60.22% |

The `+0.035`-point difference is expected from the different data partition;
the candidate construction itself is unchanged. P0 fails the 5% meaningful
gate even though alternatives sometimes win.

## Fixed candidate architecture families

| Family | Frozen construction | Candidates / oracle evaluations |
|---|---|---:|
| P0 | production antithetic local pair | 3 |
| P1 | nested 1x production-noise pool | 8/16/32/64 |
| P2 | N32 translation, rotation, gripper, translation+rotation, full non-gripper | 32 |
| P3 | N32 at 0.5x/1x/2x and cyclic mixed-scale N32 | 32 |
| P4 | C0-centered oracle CEM, 32 samples/round, elite 4, 1--4 rounds | 32/64/96/128 cumulative |
| P5 | A0 + 8 full@1x + 8 full@2x + 8 translation@2x + 7 rotation@2x | 32 |

Every proposal rule is shared across tasks. For this seven-dimensional action
interface, `translation+rotation` and `full non-gripper` are exact aliases;
the equality is tested rather than presented as two independent findings.

## Aggregate oracle headroom versus candidates and cost

| Family | N / evals | Best MSE | Gain | 95% CI | Positive tasks | Compute vs P0 |
|---|---:|---:|---:|---|---:|---:|
| P0 N3 | 3 | 0.028006 | 1.683% | [1.565%,1.827%] | 10 | 1.00x |
| P1 N8 | 8 | 0.027462 | 3.590% | [3.349%,3.883%] | 10 | 2.67x |
| P1 N16 | 16 | 0.027108 | 4.833% | [4.504%,5.226%] | 10 | 5.33x |
| P1 N32 | 32 | 0.026806 | 5.894% | [5.501%,6.368%] | 10 | 10.67x |
| P1 N64 | 64 | 0.026542 | 6.819% | [6.373%,7.358%] | 10 | 21.33x |
| P2 translation N32 | 32 | 0.026749 | 6.093% | [5.686%,6.580%] | 10 | 10.67x |
| P2 rotation N32 | 32 | 0.028351 | 0.472% | [0.429%,0.518%] | 10 | 10.67x |
| P2 gripper N32 | 32 | 0.018684 | 34.409% | [33.058%,35.793%] | 10 | 10.67x |
| P3 0.5x N32 | 32 | 0.027487 | 3.502% | [3.263%,3.787%] | 10 | 10.67x |
| P3 1x N32 | 32 | 0.026806 | 5.894% | [5.501%,6.368%] | 10 | 10.67x |
| P3 2x N32 | 32 | 0.026237 | 7.890% | [7.350%,8.526%] | 10 | 10.67x |
| P3 mixed N32 | 32 | 0.026655 | 6.424% | [6.001%,6.924%] | 10 | 10.67x |
| P4 iteration 1 | 32 | 0.026827 | 5.820% | [5.436%,6.278%] | 10 | 10.67x |
| P4 iteration 2 | 64 | 0.026164 | 8.147% | [7.608%,8.801%] | 10 | 21.33x |
| P4 iteration 3 | 96 | 0.025883 | 9.136% | [8.529%,9.868%] | 10 | 32.00x |
| P4 iteration 4 | 128 | 0.025718 | 9.712% | [9.066%,10.495%] | 10 | 42.67x |
| P5 mixture N32 | 32 | 0.026429 | 7.217% | [6.738%,7.782%] | 10 | 10.67x |

N=32 is the first P1 pool to clear 5%. The marginal P1 gains are `+1.907`,
`+1.243`, `+1.061`, and `+0.925` percentage points for 3→8, 8→16, 16→32,
and 32→64. Headroom is still increasing, but with diminishing returns and a
21.33x candidate-count cost at N=64.

## Scale, structure, and gripper diagnosis

P3 shows that 2x is better than 1x by `1.996` points, paired CI
`[1.768,2.223]`. This misses the frozen 2-point family-increment threshold by
`0.004` point and is not rounded into a mechanism label. Mixed-scale N32 is
only `0.529` point above 1x, CI `[0.418,0.633]`; it does not establish that a
mixture is more useful than spending all 31 alternatives at the larger scale.

P2 isolates where dimension-specific correction is available:

| Candidate family | Full-chunk gain | Translation-only oracle gain | Rotation-only oracle gain | Gripper-only oracle gain |
|---|---:|---:|---:|---:|
| P0 N3 | 1.683% | 5.738% | 6.251% | 0% |
| P1 N32 | 5.894% | 19.121% | 24.975% | 0% |
| P2 translation | 6.093% | 19.121% | 0% | 0% |
| P2 rotation | 0.472% | 0% | 24.975% | 0% |
| P2 gripper | 34.409% | 0% | 0% | 51.940% |
| P3 2x | 7.890% | 27.326% | 26.735% | 0% |
| P4 iteration 4 | 9.712% | 30.223% | 26.749% | 0% |

Dimension-specific gains use a separately chosen oracle winner for that
dimension and therefore do not add to the full-chunk gain. Translation-only
is only `0.198` point better than full non-gripper N32, far below the 2-point
structured-search rule. Rotation diversity helps rotation error, but rotation
is a small part of unnormalized seven-dimensional MSE; rotation-only proposals
cannot repair translation or gripper error.

The expert gripper channel is exactly `-1/+1`, while frozen C0 is continuous
and at a boundary on `43.75%` of image states. Gripper-only oracle winners
increase that to `77.03%` and halve gripper MSE (`51.940%`). This is strong
evidence that freezing gripper hides imitation headroom. It is not evidence
that Gaussian gripper perturbation is production-ready: its high gripper
clipping rate and binary-like label structure require a dedicated future
proposal contract.

## Candidate validity

All candidates were clipped to `[-1,1]`; all 19 families have zero NaN, Inf,
out-of-range, or otherwise invalid candidates. Selected validity statistics:

| Family | Preclip component rate | Boundary-hit component rate | Invalid |
|---|---:|---:|---:|
| P0 N3 | 0.113% | 6.413% | 0 |
| P1 N64 | 0.169% | 6.422% | 0 |
| P2 rotation N32 | 0% | 6.397% | 0 |
| P2 gripper N32 | 5.504% | 5.846% | 0 |
| P3 2x N32 | 0.329% | 6.584% | 0 |
| P4 iteration 4 | 0.053% | 6.305% | 0 |
| P5 mixture N32 | 0.213% | 6.500% | 0 |

P2 gripper's aggregate clip fraction corresponds to `38.53%` of gripper
components because the other six dimensions are unchanged. Its oracle value
must therefore be carried forward as a diagnostic with a clipping warning.

## Diversity, effective rank, and quality

Pairwise diversity is the exact root-mean-square difference over candidate
pairs and action components. Effective rank is the entropy rank of the global
56-D candidate-deviation covariance.

| Family | Pairwise action RMS | Translation | Rotation | Gripper | Effective rank | Oracle gain |
|---|---:|---:|---:|---:|---:|---:|
| P0 N3 | 0.0329 | 0.0466 | 0.0187 | 0 | 22.82 | 1.683% |
| P1 N64 | 0.0329 | 0.0466 | 0.0187 | 0 | 22.89 | 6.819% |
| P2 translation | 0.0305 | 0.0466 | 0 | 0 | 15.31 | 6.093% |
| P2 rotation | 0.0122 | 0 | 0.0187 | 0 | 15.30 | 0.472% |
| P2 gripper | 0.1199 | 0 | 0 | 0.3173 | 4.92 | 34.409% |
| P3 2x | 0.0655 | 0.0929 | 0.0373 | 0 | 22.92 | 7.890% |
| P3 mixed | 0.0426 | 0.0604 | 0.0242 | 0 | 22.90 | 6.424% |
| P4 iteration 4 | 0.0226 | 0.0324 | 0.0122 | 0 | 18.63 | 9.712% |
| P5 mixture | 0.0498 | 0.0708 | 0.0277 | 0 | 22.64 | 7.217% |

Increasing N at fixed scale barely changes RMS diversity or covariance rank;
it improves coverage of the same distribution. The 2x and P5 pools increase
both diversity and headroom. P4 behaves differently: after elite updates its
cumulative pool has lower aggregate RMS and rank than broad one-shot pools,
yet better non-gripper headroom. Thus candidate count, scale coverage, and
adaptive concentration are distinct.

## Oracle winner displacement and identity

| Family | A0 remains best | Non-default wins | Conditional mean sample gain | Winner RMS from A0 | Trans. RMS | Rot. RMS | Grip. RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 N3 | 39.78% | 60.22% | 6.49% | 0.01794 | 0.02544 | 0.01019 | 0 |
| P1 N8 | 7.93% | 92.07% | 9.75% | 0.02266 | 0.03223 | 0.01260 | 0 |
| P1 N32 | 0.39% | 99.61% | 15.69% | 0.02476 | 0.03552 | 0.01302 | 0 |
| P1 N64 | 0.09% | 99.91% | 18.48% | 0.02548 | 0.03668 | 0.01302 | 0 |
| P2 gripper | 18.83% | 81.17% | 19.07% | 0.06641 | 0 | 0 | 0.17570 |
| P3 2x | 9.99% | 90.01% | 18.31% | 0.04467 | 0.06372 | 0.02443 | 0 |
| P4 iteration 4 | 0% | 100% | 27.91% | 0.02314 | 0.03432 | 0.00848 | 0 |
| P5 mixture | 2.15% | 97.85% | 16.76% | 0.03859 | 0.05684 | 0.01562 | 0 |

The best local action is not usually A0 once enough samples are available.
However, P4 winners remain close to A0 in absolute RMS; the iterative gain
comes from concentrating around a better local mean, not discovering remote
action-space modes.

Mean per-step RMS winner displacement is:

| Family | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.01551 | 0.01222 | 0.01233 | 0.01226 | 0.01218 | 0.01208 | 0.01212 | 0.01549 |
| P1 N64 | 0.02789 | 0.02304 | 0.02284 | 0.02251 | 0.02246 | 0.02259 | 0.02261 | 0.02756 |
| P2 gripper | 0.03150 | 0.02875 | 0.02842 | 0.02767 | 0.03036 | 0.03265 | 0.03577 | 0.04134 |
| P3 2x | 0.04628 | 0.03796 | 0.03804 | 0.03775 | 0.03746 | 0.03752 | 0.03727 | 0.04584 |
| P4 iteration 4 | 0.02783 | 0.02024 | 0.01945 | 0.01911 | 0.01893 | 0.01897 | 0.01962 | 0.02721 |
| P5 | 0.03973 | 0.03261 | 0.03285 | 0.03257 | 0.03217 | 0.03230 | 0.03223 | 0.03942 |

## Gain distribution

Sample-level relative gains are zero whenever A0 wins. Their mean is not the
aggregate ratio-of-means reported above, so the distribution should not be
used to reconstruct the primary metric.

| Family | p25 | Median | p75 | p90 | >1% | >5% | >10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0% | 0.81% | 5.88% | 12.45% | 48.32% | 27.96% | 14.16% |
| P1 N32 | 8.89% | 15.84% | 21.57% | 27.88% | 95.64% | 80.69% | 72.30% |
| P1 N64 | 11.69% | 19.18% | 25.33% | 31.27% | 96.98% | 82.53% | 77.41% |
| P2 rotation | 0.27% | 0.90% | 2.14% | 4.14% | 46.77% | 6.62% | 0.79% |
| P2 gripper | 0.15% | 4.32% | 28.09% | 49.84% | 64.91% | 48.08% | 38.31% |
| P3 2x | 4.24% | 16.40% | 25.47% | 33.13% | 86.13% | 72.47% | 63.27% |
| P4 iteration 4 | 18.76% | 29.49% | 38.30% | 46.85% | 98.60% | 86.04% | 80.67% |
| P5 mixture | 6.55% | 15.89% | 24.16% | 31.74% | 93.27% | 78.46% | 66.93% |

Every family has zero oracle-worse samples by construction and by explicit
implementation assertion.

## Receding-horizon prefix relevance

Each prefix result chooses its own oracle candidate using only that prefix.
It asks whether a perfect short-horizon selector has useful immediate
headroom; it does not claim the H8 winner would realize the H1 number.

| Family | H1 | H2 | H4 | H8 |
|---|---:|---:|---:|---:|
| P0 | 5.27% | 4.11% | 2.86% | 1.68% |
| P1 N32 | 14.76% | 11.73% | 8.61% | 5.89% |
| P1 N64 | 16.66% | 13.25% | 9.77% | 6.82% |
| P2 translation | 14.97% | 11.90% | 8.78% | 6.09% |
| P2 rotation | 1.12% | 0.91% | 0.68% | 0.47% |
| P2 gripper | 51.88% | 45.32% | 38.80% | 34.41% |
| P3 2x | 21.56% | 17.27% | 12.55% | 7.89% |
| P3 mixed | 17.46% | 13.92% | 9.90% | 6.42% |
| P4 iteration 4 | 17.03% | 13.66% | 10.96% | 9.71% |
| P5 | 19.68% | 15.66% | 11.25% | 7.22% |

Headroom is not confined to h7/h8. One-shot 2x and P5 have especially strong
first-action oracle value; P4's advantage grows relative to them as the
evaluated prefix length increases.

## Per-horizon evidence

Entries use a horizon-specific oracle winner and report raw action MSE gain.

| h | P0 | P1 N64 | P2 trans. | P2 rot. | P2 grip. | P3 mixed | P4 iter4 | P5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.27% | 16.66% | 14.97% | 1.12% | 51.88% | 17.46% | 17.03% | 19.68% |
| 2 | 4.22% | 13.03% | 11.70% | 0.92% | 48.14% | 14.47% | 12.52% | 16.28% |
| 3 | 4.21% | 12.81% | 11.49% | 0.92% | 49.18% | 14.12% | 12.13% | 15.94% |
| 4 | 4.03% | 12.42% | 11.10% | 0.92% | 50.13% | 13.67% | 11.69% | 15.36% |
| 5 | 3.71% | 11.84% | 10.62% | 0.89% | 51.64% | 13.10% | 11.22% | 14.68% |
| 6 | 3.70% | 11.66% | 10.50% | 0.89% | 52.90% | 13.07% | 11.07% | 14.63% |
| 7 | 3.87% | 11.82% | 10.54% | 0.90% | 53.53% | 13.09% | 11.25% | 14.64% |
| 8 | 4.44% | 13.82% | 12.38% | 1.02% | 59.53% | 14.49% | 14.19% | 16.24% |

The horizon-specific numbers exceed full-chunk gains because they may select
different candidates at every h. They show broad rather than late-only action
coverage.

## Per-task evidence

| Task | P0 | P1 N64 | P2 trans. | P2 rot. | P2 grip. | P3 mixed | P4 iter4 | P5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.55% | 5.66% | 5.05% | 0.39% | 31.63% | 5.88% | 7.92% | 6.85% |
| 1 | 1.65% | 6.80% | 6.00% | 0.53% | 34.92% | 6.37% | 9.64% | 7.08% |
| 2 | 1.56% | 6.30% | 5.64% | 0.31% | 38.92% | 5.55% | 9.10% | 6.26% |
| 3 | 1.49% | 5.82% | 5.25% | 0.36% | 37.62% | 5.27% | 8.48% | 5.88% |
| 4 | 1.76% | 7.01% | 6.31% | 0.48% | 34.66% | 6.62% | 9.91% | 7.30% |
| 5 | 1.85% | 7.48% | 6.85% | 0.56% | 32.27% | 6.98% | 10.95% | 7.90% |
| 6 | 1.44% | 6.01% | 5.37% | 0.45% | 34.80% | 5.73% | 8.56% | 6.44% |
| 7 | 1.90% | 8.17% | 7.14% | 0.70% | 35.73% | 6.68% | 11.87% | 7.12% |
| 8 | 1.86% | 8.09% | 7.23% | 0.56% | 34.70% | 7.66% | 11.55% | 8.51% |
| 9 | 1.94% | 8.03% | 7.16% | 0.50% | 31.17% | 7.87% | 11.16% | 9.02% |

All shown families are positive on all ten tasks. P1 N64 clears 5% in each
task, not merely the aggregate. Gripper headroom is also broad rather than
coming from one task.

## One-shot versus iterative search

P4 uses the frozen production non-gripper standard deviation, 32 candidates
per round, elite count 4, unweighted elite mean/std updates, and elementwise
std bounds `[0.1x,2x]` of the initial std. It uses expert MSE to update the
distribution and is therefore an oracle diagnostic, not an executable planner.

Iteration 4 is `1.822` points better than the best comparable non-gripper
one-shot family, P3 2x, with paired CI `[1.521,2.154]`. It is `2.496` points
above P5, CI `[2.201,2.826]`. Both increments are statistically stable, but
neither reaches the frozen 5-point model-based threshold. Relative to the
best one-shot diagnostic including gripper, P4 is `24.697` points worse.

The evidence therefore supports a larger one-shot local proposal set, not the
claim that one-shot reranking has hit a structural ceiling. P4's diminishing
iteration increments (`+2.327`, `+0.988`, `+0.577` points after iteration 1)
also argue against authorizing a 42.67x oracle-evaluation architecture from
this experiment.

## Existing real-future evidence is secondary and unchanged

Frozen A0/A1/A2 branch audits already established that production candidates
produce measurable real visual and physical differences: mean global CLIP
distance is `0.004575`, and action rotation is strongly associated with real
EE rotation. They also found that tested structured states did not uniquely
explain visual separation and that natural visible-area extractors failed to
recover candidate-sensitive variation.

Those existing real futures cover P0 only. No P1--P5 candidate was executed in
the simulator, so this report makes no statement about its actual RGB future,
object footprint, task value, reward, or success. The new result changes how
the old tiny-difference problem is prioritized: before replacing the visual
encoder, it is now justified to design a larger one-shot candidate interface
whose action-space headroom is materially greater than P0.

## C0 role, unresolved value, and architectural decision

C0 is not a strong final-action optimum under the imitation oracle: nested
samples around it exceed the 5% gate, 2x local candidates do better still, and
gripper proposals expose a large omitted dimension. At the same time, this
audit does not assign the Case-C label `C0_SHOULD_BE_PROPOSAL_PRIOR`, because
iterative search failed the required advantage. Operationally, the authorized
one-shot design may retain C0 as its default and proposal center without
claiming a model-based planning architecture.

The future goal/value/cost remains completely unresolved. Expert action is an
offline reference, not a deployable score. A later expanded-candidate design
must separately define what predicted future state is represented and how its
desirability is judged before any control experiment can be contemplated.

## Mechanical decision

| Gate | Result |
|---|---|
| P0 meaningful headroom | fail: 1.683% <5% |
| Nested larger-N one-shot | pass: N32 5.894%, N64 6.819%, all 10 tasks |
| N64 improvement over P0 | +5.136 points, CI [4.776,5.557] |
| Structured dimension increment | fail: translation only adds 0.198 point over matched full pool |
| Mixed-scale increment | fail: +0.529 point over 1x |
| Gripper-only diagnostic | pass: 34.409%, but high gripper clipping |
| P5 broader one-shot | pass meaningful gate at 7.217% |
| P4 iteration 4 meaningful gate | pass at 9.712% |
| P4 significant advantage over one-shot | fail: +1.822 points vs comparable P3 2x, required +5 |
| Any non-gripper strong ≥10% result | no |
| Case | B |

Final mechanism labels:

```text
CURRENT_N3_CANDIDATE_SET_UNDERPOWERED
MORE_LOCAL_SAMPLES_ADD_HEADROOM
GRIPPER_SEARCH_ADDS_HEADROOM
```

Final planner authorization:

```text
GO_EXPANDED_CANDIDATE_PLANNER_DESIGN
```

Continuing independent block:

```text
BLOCK_SPATIAL_WM_PROTOTYPE
```

This GO permits only a separately preregistered design of a larger one-shot
candidate interface, including a careful gripper proposal study. It does not
select N=32 versus N=64, freeze a production scale, authorize a learned
proposal policy, train a World Model, construct a reranker/value/cost, or run
a controller.

## Eight required answers

**Q1 — Is P0's roughly 1.65% headroom only a consequence of N=3?**

Partly. The same nested distribution reaches 5.894% at N=32 and 6.819% at
N=64, so N=3 is underpowered. Candidate count is not the whole story: 2x
scale, P5, and especially gripper proposals change the ceiling further.

**Q2 — Does headroom keep growing through N=64 or saturate quickly?**

It grows monotonically, but marginal gains shrink from +1.907 points for 3→8
to +0.925 for 32→64. N32/N64 are meaningful; unlimited N expansion is not
authorized.

**Q3 — Is production scale too small, and is mixed scale better?**

The 2x fixed pool is better than 1x by 1.996 points, suggesting the production
scale is conservative for this oracle. The mixed pool adds only 0.529 point
over 1x and is worse than all-2x, so `MULTISCALE_SEARCH_ADDS_HEADROOM` is not
assigned.

**Q4 — Which action subspace provides real headroom?**

Gripper dominates full raw-MSE headroom at 34.409%. Translation provides a
meaningful 6.093% full-chunk gain. Rotation proposals strongly reduce
rotation-only error but improve full raw MSE by only 0.472%. Gripper remains a
clipping-sensitive diagnostic, not an immediate production change.

**Q5 — How far is the best action from C0?**

P1 N64 winners are locally displaced (`0.0255` action RMS); P3 2x and P5 are
farther (`0.0447` and `0.0386`). Iterative winners return to a concentrated
local displacement (`0.0231`). C0 is a useful center but not a sufficient
three-action final generator.

**Q6 — Does one-shot reranking have a demonstrated structural limit?**

No. P4 reaches 9.712%, but its advantage over the best comparable one-shot is
only 1.822 points, below the frozen 5-point requirement. Iteration helps, but
not enough to assign the iterative or one-shot-limit labels.

**Q7 — Should C0 be the final generator or a proposal prior?**

It should no longer be treated as sufficient with only one antithetic pair.
The authorized interpretation is C0 as the default/center of an expanded
one-shot design. The stronger model-based `C0_SHOULD_BE_PROPOSAL_PRIOR` label
is withheld because Case C failed.

**Q8 — Optimize tiny-candidate World Model sensitivity or move toward broader
planning?**

Move first to a separately frozen expanded one-shot candidate design and
resolve gripper proposals. Do not resume the blocked spatial-WM prototype, and
do not yet move to CEM/MPPI-style model-based planning or controller success.

## Artifacts, finite values, and reproducibility

| Artifact | SHA-256 |
|---|---|
| Protocol | `23aed612f6aba30a6b5472ec5ab59bc10c6d8441a9251589fbf0c7510020a185` |
| Calibration freeze | `f4d2057cb3f6e2b3628b27a3e7bf8aa5f67d1feadc4b8de714dda8db76a145c8` |
| Calibration derived sidecar | `03db7e99d0c85f5cb1b92ac4091a942b201e78c03fd16f075ead2ed7606e97c0` |
| Evaluation diagnosis | `2eac841dc55f32a9e5ba3987cf78daea7cf093720dfb10e033cfef2334eb8eb9` |
| Evaluation derived sidecar | `32c80c1e998b56ecd61b6c0b2cdc0f6670b3f996b305f7edda7d1c51abb90471` |
| Probe implementation | `fb1726598987b26d045f62f5109864117dd5aec03ebdf33cd5c499e7fae181f7` |
| Audit runner | `28f9da1dd8133d6e46521eed8f183a42e0effc6fa6cabe53a519e2454860e47e` |

The ignored sidecars retain all identities, expert chunks, A0 chunks, family
oracle errors, winner IDs/actions, and bootstrap gain arrays; the protocol
freezes the shared bootstrap seed and draw shape. All admitted inputs, candidates, errors, winner
actions, covariance spectra, distributions, bootstrap gains, and reportable
metrics are finite. Every oracle error is no greater than its corresponding A0
error. Exact family aliases and nested monotonicity are independently tested.

Five protocol tests and five result tests independently verify frozen hashes,
partition boundaries, deterministic nested noise, bootstrap structure, sample
identities, exact gain recomputation, oracle non-regression, family aliases,
decision mapping, authorization, finite arrays, and candidate validity. The
complete repository suite passes `178` tests; the remaining 21 messages are
dependency deprecation warnings, not failures.

## Forbidden-activity audit

No simulator, real future, reward, done, success, task outcome, init/demo40--49,
World Model, encoder, spatial extractor, actor, learned proposal policy,
reranker, score, goal, value, cost, controller, C1/C2, D20-v2, or seed23/47
experiment was run or modified. No new candidate was described as having an
actual visual future or task value. Historical demos45--49 metrics were used
only as a committed scalar reference; their numerical arrays were not opened.
No upstream artifact was overwritten.
