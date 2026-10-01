# Expanded candidates have strong oracle headroom, but the exact interface remains unresolved

Date: 2026-10-02

Final mechanism labels:

```text
STRUCTURED_GRIPPER_PROPOSALS_VALID
N64_REQUIRED_FOR_HEADROOM
STRONG_EXPANDED_CANDIDATE_HEADROOM
EXPANDED_CANDIDATE_INTERFACE_UNRESOLVED
DATA_OR_IMPLEMENTATION_BLOCK
```

Next-stage authorization: `BLOCK_CANDIDATE_VALUE_AUDIT`

Spatial World Model authorization: `BLOCK_SPATIAL_WM_PROTOTYPE`

Controller authorization: `false`

Selected production interface: `none`

## Executive answer

The finite one-shot search study finds much more imitation-oracle headroom
than the old three-candidate interface, but it does not freeze a conforming
production candidate contract. The calibration-selected numerical candidate
used 64 total actions: A0, 51 continuous translation/rotation alternatives at
`2.5x` production scale, and 12 structured binary gripper alternatives. On
frozen development demos36--39 it reduces full-H8 expert-action MSE by
`62.835%` (task-stratified demo-block bootstrap 95% CI
`[60.588%,64.745%]`). H1 gain is `80.463%`, all ten tasks are positive, and
the candidate lies on the H1/H8/count Pareto frontier of the executed
families.

Most of that improvement is real gripper-search headroom under the frozen raw
MSE oracle. The matched N64 continuous-only pool gains `9.616%`; the combined
pool adds `53.219` percentage points, paired CI `[50.246,55.761]`. Structured
gripper candidates win `38.72%` of rows, continuous candidates win `58.91%`,
and A0 remains best on `2.37%`. The gripper-only oracle gain is `86.934%`, and
its selected signs agree with the exactly binary expert channel on `99.56%`
of action steps. This assigns `STRUCTURED_GRIPPER_PROPOSALS_VALID` and
`STRONG_EXPANDED_CANDIDATE_HEADROOM` as mechanism findings.

The numerical candidate nevertheless fails the requested exact-interface
contract. An antithetic bank with 51 nonzero continuous alternatives has 25
complete positive/negative pairs plus one unmatched positive direction. It
therefore does not preserve exact sign symmetry. In addition, the
preregistered sign-discretized C0 gripper form was evaluated by itself but
was omitted from every combined G4/G8/G12 allocation. The full gripper
diagnostic later showed that this form wins frequently, so its omission
cannot be dismissed after evaluation and repaired post hoc.

These are design-contract failures, not numerical failures. They were found
after demos36--39 had been opened. The source numerical diagnosis is retained
unchanged, including its provisional positive label, but a committed decision
amendment removes `EXPANDED_CANDIDATE_INTERFACE_VALID`. No new candidate was
constructed or evaluated after seeing the result. The formal outcome is
`EXPANDED_CANDIDATE_INTERFACE_UNRESOLVED` plus
`DATA_OR_IMPLEMENTATION_BLOCK`, and `BLOCK_CANDIDATE_VALUE_AUDIT` remains in
force.

## Scope and architectural boundary

This audit evaluates only proposal generation:

| Component | Role | Evaluated here |
|---|---|---|
| Frozen C0-v2 | Supplies A0 and the local proposal center | yes, unchanged |
| One-shot proposal interface | Produces a finite candidate set | yes |
| World Model | Predicts candidate consequences | no |
| Goal/value/cost/reranker | Chooses a desirable predicted future | no |
| Controller | Executes candidates and measures task success | no |

The primary oracle is raw expert-action MSE. It is unavailable at deployment
and is not reward, task value, or success. No candidate in this audit has a
simulated RGB future, object state, reward, or outcome.

## Frozen provenance and correction boundary

The upstream planning-architecture audit is frozen at commit
`4e43c7add65b13034db4cea1c7715d190e65a426` under annotated tag
`expanded-candidate-interface-start`. Its conclusions remain:

```text
CURRENT_N3_CANDIDATE_SET_UNDERPOWERED
MORE_LOCAL_SAMPLES_ADD_HEADROOM
GRIPPER_SEARCH_ADDS_HEADROOM
GO_EXPANDED_CANDIDATE_PLANNER_DESIGN
BLOCK_SPATIAL_WM_PROTOTYPE
```

The new protocol and first implementation were committed as `1cdedcd` before
calibration. The first calibration invocation produced no artifact: exact
pairwise coverage had been expanded into an unnecessarily large
sample-by-candidate-by-candidate-by-action tensor. The same calculation was
changed to 32-sample batches and committed as `54dd669`; no candidate, metric,
threshold, or decision rule changed.

Calibration was frozen at commit `46be050` before demos36--39 were opened.
The source numerical evaluation was frozen at `bb34786`. Only after that
freeze did the final contract review identify the unmatched antithetic
direction and missing combined G3 form. The decision amendment changes no
candidate, error, bootstrap draw, interval, or sidecar. It prohibits a
post-evaluation repair and maps the result to the formal block.

## Data boundary

| Partition | Demo/init IDs | Legal windows | Use |
|---|---:|---:|---|
| Calibration/design | 31--34 | 4,724 | all scale/style/count/allocation choices |
| Frozen development evaluation | 36--39 | 4,563 | one-shot metrics and gates; not blind confirmation |
| Historical reference | 30/35 | not opened | reference only |
| Forbidden | 40--49 | 0 | never accessed |

For each admitted HDF5 group, natural vision and proprioception at `k` feed
the frozen causal C0-v2 actor, and the expert target is
`actions[k+1:k+9]`. The loader opens only the four explicitly admitted demo
groups per task. No generic dataset enumerated demos40--44.

## Preregistered finite design space

Continuous proposals changed only translation and rotation. Their base
standard deviations remained:

```text
[0.05,0.05,0.05,0.02,0.02,0.02,0]
```

The temporal kernel remained `[0.25,0.5,0.25]`, and every candidate was
clipped to `[-1,1]`. The finite screens were:

- count budgets `16/24/32/48/64`;
- scales `1.0/1.5/2.0/2.5x`;
- one antithetic-versus-independent comparison;
- structured gripper constants, single switches, and sign-discretized C0;
- fixed combined allocations with `0/4/8/12` gripper alternatives;
- no continuous-by-gripper Cartesian product.

The calibration rule selected scale by mean H1/H2 headroom, preferred
antithetic sampling when the two banks differed by less than one point, and
selected N64 only when doubling N32 was materially useful. Combined gripper
allocation required either at least `5` H8 points with a positive paired
interval or at least `10` mean H1/H2 points.

## Scale and sampling-bank results

These are frozen development results; selection occurred on calibration only.

| Scale, N32 | H8 gain | 95% CI | H1 | H2 | Candidate clip | Component clip |
|---|---:|---:|---:|---:|---:|---:|
| 1.0x | 5.894% | [5.501%,6.368%] | 14.764% | 11.729% | 4.616% | 0.167% |
| 1.5x | 7.301% | [6.809%,7.884%] | 18.914% | 15.120% | 6.679% | 0.239% |
| 2.0x | 7.890% | [7.350%,8.526%] | 21.564% | 17.273% | 9.265% | 0.329% |
| 2.5x | 7.845% | [7.265%,8.492%] | 23.085% | 18.470% | 12.230% | 0.434% |

Calibration selected `2.5x` because its mean H1/H2 advantage over `2.0x`
exceeded the frozen one-point tie tolerance. Evaluation H8 is slightly lower
at `2.5x`, but evaluation was not allowed to reselect the scale. Relative to
1x, the selected-scale H8 increment is `1.950` points, CI
`[1.624,2.263]`. It misses the frozen 2-point label threshold, so
`PRODUCTION_SCALE_TOO_CONSERVATIVE` is not assigned.

At the selected scale, antithetic and independent N32 banks are nearly
identical: H8 gains are `7.845%` and `7.814%`; H1 gains are `23.085%` and
`23.198%`. The calibration tie rule therefore retained the deterministic
antithetic bank.

## Candidate count, compute, and diminishing returns

| Nominal N | H8 gain | 95% CI | H1 | Gain/N | Translation oracle | Rotation oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 5.885% | [5.421%,6.398%] | 19.619% | 0.368% | 21.823% | 16.971% |
| 24 | 7.055% | [6.514%,7.660%] | 21.766% | 0.294% | 25.799% | 21.034% |
| 32 | 7.845% | [7.265%,8.492%] | 23.085% | 0.245% | 28.395% | 23.972% |
| 48 | 8.891% | [8.250%,9.610%] | 24.768% | 0.185% | 31.803% | 27.681% |
| 64 | 9.616% | [8.932%,10.376%] | 25.956% | 0.150% | 34.106% | 30.191% |

Marginal H8 gains are `+1.170`, `+0.789`, `+1.046`, and `+0.726` points for
16→24, 24→32, 32→48, and 48→64. Gain per candidate declines monotonically.
Nevertheless, N64 exceeds N32 by `1.772` H8 points, above the frozen `1.5`
point materiality rule, and the maximum translation/rotation oracle increment
is `6.219` points. Calibration showed the same material result. This assigns
`N64_REQUIRED_FOR_HEADROOM`; it does not mean N64 is economical under a future
learned value function.

For historical context, the old production P0 N3 gain was `1.683%`; the old
1x N32/N64 gains were `5.894%/6.819%`. The new continuous result comes from
both more candidates and a broader frozen scale, not from an iterative search.

## Structured gripper family

The expert gripper channel is exactly `-1/+1` on `100%` of admitted steps.
No Gaussian gripper proposal is used in this audit.

| Standalone family | Candidates | H8 gain | 95% CI | Gripper-only gain | Structured winner |
|---|---:|---:|---:|---:|---:|
| G0 keep C0 | 1 | 0% | [0%,0%] | 0% | 0% |
| G1 all-open/all-close | 3 | 30.426% | [26.977%,33.257%] | 45.927% | 76.16% |
| G2 constants + switches | 13 | 57.592% | [54.712%,60.015%] | 86.934% | 85.51% |
| G3 sign-discretized C0 | 2 | 8.316% | [6.987%,9.815%] | 12.553% | 72.15% |
| All admitted forms | 14 | 58.281% | [55.485%,60.647%] | 87.975% | 85.84% |

The switch family adds substantial value beyond constants. In the all-form
diagnostic, however, sign-discretized C0 is the full-MSE winner on `71.55%`
of all rows, while nonconstant switch forms collectively win `9.91%`. This
does not mean G3 alone has the largest aggregate gain: winner frequencies are
conditional on all forms being available, and many rows use G3 for a modest
correction. It does mean omitting G3 from the combined allocation was not a
harmless bookkeeping choice.

## Fixed-budget combined results

All combined rows below use total N64 and the frozen `2.5x` continuous bank.
The gripper candidate modifies only the gripper sequence; there is no
Cartesian product with continuous noise.

| Allocation | Continuous alternatives | Gripper alternatives | H8 gain | H1 | A0 / continuous / gripper winners |
|---|---:|---:|---:|---:|---:|
| G0 | 63 | 0 | 9.616% | 25.956% | 12.43% / 87.57% / 0% |
| G4 | 59 | 4 | 47.072% | 80.929% | 2.54% / 65.18% / 32.28% |
| G8 | 55 | 8 | 56.691% | 80.724% | 2.41% / 61.71% / 35.88% |
| G12 | 51 | 12 | 62.835% | 80.463% | 2.37% / 58.91% / 38.72% |

G12 improves over continuous-only by `53.219` points, CI
`[50.246,55.761]`; H1 and H2 increments are `54.507` and `56.649` points.
It is numerically superior to G4/G8 at the same total count. These results
support structured gripper proposals, but the executed allocation is not the
required final interface because of the contract omissions.

## Horizon and task robustness of the numerical candidate

Each horizon row chooses its own oracle candidate at that step.

| h | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gain | 80.46% | 80.42% | 80.82% | 81.33% | 81.75% | 81.85% | 81.24% | 82.20% |

Prefix gains are `80.463%`, `77.711%`, `68.010%`, and `62.835%` at
H1/H2/H4/H8. Headroom is immediate rather than confined to the tail.

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H8 gain | 63.49% | 64.17% | 68.68% | 67.02% | 66.99% | 56.33% | 66.36% | 55.95% | 58.53% | 55.88% |

All ten tasks are strongly positive. The effect is not carried by one task.

## Validity, coverage, and diversity

The numerical G12 candidate passes every frozen numeric validity limit:

| Check | Result | Limit |
|---|---:|---:|
| Invalid / NaN / Inf / out-of-range | 0 | 0 |
| Preclip component fraction | 0.357% | <=1% |
| Candidate with any clipped component | 10.048% | <=25% |
| Boundary-hit component fraction | 8.144% | <=10% |
| Mean nearest-neighbor action RMS | 0.09348 | descriptive |
| Near-duplicate candidate fraction | 0.418% | descriptive |
| Exact duplicate-pair fraction | 0.0066% | descriptive |

Structured gripper candidates themselves require no clipping; the clip rate
comes from continuous alternatives. The combined pool's pairwise action RMS
is `0.3102`, but its covariance effective rank is only `3.62`, reflecting the
large binary-like gripper variation. Numerical diversity should not be read
as broad semantic planning coverage.

The oracle winner is displaced from A0 by action RMS `0.1418`, decomposed as
`0.0639` translation, `0.0244` rotation, and `0.3560` gripper RMS. Sample-level
gain has median `27.98%`; `90.55%` of rows exceed 5% and `85.54%` exceed 10%.
These sample ratios do not reconstruct the primary ratio-of-mean-MSE metric.

## Pareto analysis and why it cannot authorize the interface

Within the executed finite families, G12 is not dominated by a design with
equal or lower candidate count on both H1 and H8. The continuous count curve
shows diminishing gain per candidate, while G4/G8/G12 produce much larger
H8 gains at fixed N64. Numerically, G12 is therefore on the executed Pareto
frontier.

That frontier is incomplete in exactly the dimensions required for the final
contract:

1. `51` continuous alternatives cannot be partitioned into exact antithetic
   pairs.
2. The combined allocation never admitted `c0_binarized`, even though G3 was
   one of the required gripper forms and the all-form diagnostic shows it is
   often selected.

The missing conforming candidates were not evaluated. A post-hoc candidate
such as an odd realized total or an odd gripper allocation including G3 would
change the design and its metrics. It requires a separately preregistered
restart; this report does not choose between those examples.

## Mechanical decision

| Gate | Result |
|---|---|
| Continuous N32/N64 meaningful headroom | pass |
| N64 material over N32 | pass: +1.772 H8 points |
| Structured gripper increment | pass: +53.219 points, positive CI |
| Numerical G12 H8 >=10% | pass: 62.835% |
| Numerical G12 H1 >=10% | pass: 80.463% |
| >=7 positive tasks | pass: 10/10 |
| Frozen numeric validity | pass |
| Executed-family Pareto frontier | pass |
| Exact positive/negative symmetry | **FAIL** |
| Required gripper-form coverage in combined allocation | **FAIL** |
| Post-evaluation repair permitted | **no** |
| Exact production interface frozen | **no** |

Final mechanism labels:

```text
STRUCTURED_GRIPPER_PROPOSALS_VALID
N64_REQUIRED_FOR_HEADROOM
STRONG_EXPANDED_CANDIDATE_HEADROOM
EXPANDED_CANDIDATE_INTERFACE_UNRESOLVED
DATA_OR_IMPLEMENTATION_BLOCK
```

Final authorization:

```text
BLOCK_CANDIDATE_VALUE_AUDIT
```

`EXPANDED_CANDIDATE_INTERFACE_VALID` is removed by the decision amendment.
There is no selected production interface. The next work, if separately
authorized, is a clean contract-correction restart—not a value model, World
Model, reranker, controller, or success experiment.

## Six required answers

**Q1 — How many candidates should production use?**

Not frozen. The count screen supports the N64 budget tier over N32 because the
H8 increment is `1.772` points and the best translation/rotation increment is
`6.219` points. But the executed N64 composition is structurally invalid, so
this does not authorize an exact total count.

**Q2 — What continuous scale should be used?**

Calibration selected `2.5x` for H1/H2 headroom and it passed validity. On
evaluation its H8 gain is slightly below `2.0x`, and its improvement over 1x
misses the 2-point label threshold. It remains a frozen diagnostic choice,
not a production scale.

**Q3 — Should gripper proposals be included, and in what form?**

Yes at the mechanism level: constants and single switches add very large,
stable oracle headroom without Gaussian gripper noise. The exact finite set is
unresolved because sign-discretized C0 was tested but omitted from the
combined allocation.

**Q4 — Is the combined interface stronger than continuous-only search?**

Numerically yes. G12 adds `53.219` H8 points over matched N64 continuous-only,
with CI `[50.246,55.761]`, and improves all ten tasks. This is imitation-oracle
headroom, not deployed candidate value or success.

**Q5 — Is the result compute-efficient and on a Pareto frontier?**

It is on the frontier of the executed H1/H8/count families, but continuous
gain per candidate falls from `0.368%` at N16 to `0.150%` at N64. Gripper
structure supplies most of the large combined improvement. The incomplete
contract prevents a production compute recommendation.

**Q6 — May the project proceed to a candidate-value audit?**

No. A value audit requires one exact, deterministic, compliant candidate
interface. The current numerical winner violates exact antithetic symmetry
and omits an admitted gripper form. `BLOCK_CANDIDATE_VALUE_AUDIT` is the only
valid authorization.

## Artifacts, finite values, and reproducibility

| Artifact | SHA-256 |
|---|---|
| Protocol | `b537f4abf3ed84d6a7afe9259048b457712a57736eeafdd8ad84eb4f3b546836` |
| Calibration freeze | `49d361c33193517c1d4af1cad65fca7da48f762565ca46afe246ddf818ff80c0` |
| Calibration sidecar | `f858efc4302fce939ca932e820a9c1730a76a092df650f8d5e6e9df6cca48e3f` |
| Source numerical diagnosis | `434d262ca0ea52127b71d31fac579533e646048dac5fe3785e1830a7b8cdc488` |
| Evaluation sidecar | `93f0270c28d93eebb131fda5584eccd5f1c955d3d8d02e351268dc5e5f9ce70f` |
| Decision amendment | `71cf5e2709a614d88b652b6edb792b140e96baf566276328384613fe79568ff8` |
| Corrected compact diagnosis | `91528c70881d228186829e3596d50e99628489949f279c06f1320f6838429659` |
| Probe implementation | `728c28b583f37d18dd466e327fa79d3933d12575013b4674059af084564b0ec1` |
| Audit runner | `0d2b7c79a9c5894f926430a7c5a3de4420c95981ff22964decf75cd84dd22161` |

All admitted inputs, candidates, expert errors, winners, bootstrap arrays,
distances, covariance spectra, and reportable metrics are finite. Every oracle
error is no greater than its corresponding A0 error. The ignored sidecars
retain identities, expert/A0 chunks, per-family errors, winners, winner
categories, and bootstrap gains needed for independent recalculation.

Five protocol tests and six result tests independently verify frozen hashes,
the partition boundary, finite candidate construction, exact gripper forms,
recomputed family gains, oracle non-regression, the strong numerical result,
the unmatched antithetic direction, the missing combined G3 form, and the
corrected authorization mapping. They pass `11/11`. The complete repository
suite passes `190` tests; the remaining 21 messages are dependency
deprecation warnings, not failures.

## Forbidden-activity audit

No simulator, real future, reward, done, success, task outcome, demo/init40--49,
World Model, encoder, spatial extractor, learned proposal policy, reranker,
goal, value, cost, score, controller, C1/C2, D20-v2, or seed23/47 experiment
was run or modified. No candidate was described as having actual visual or
task value. No CEM/MPPI/beam-search refinement was run. The source numerical
artifacts remain unchanged, and no corrected candidate was evaluated after
the contract failure was found.
