# The corrected N64 one-shot interface clears the exact contract gate

Date: 2026-10-02

Final mechanism labels:

```text
EXPANDED_CANDIDATE_INTERFACE_VALID
STRUCTURED_GRIPPER_PROPOSALS_VALID
N64_REQUIRED_FOR_HEADROOM
STRONG_EXPANDED_CANDIDATE_HEADROOM
```

Next-stage authorization: `GO_CANDIDATE_VALUE_AUDIT`

Spatial World Model authorization: `BLOCK_SPATIAL_WM_PROTOTYPE`

Controller authorization: `false`

Selected interface: `Selected Expanded Candidate Interface v1` (`C48_G15`)

## Executive answer

The contract-correction restart freezes one exact, deterministic, finite N64
one-shot candidate interface. It consists of the frozen C0-v2 default, 48
translation/rotation alternatives in 24 complete antithetic pairs, and 15
structured gripper alternatives. The gripper set includes sign-discretized C0,
both constant signs, and twelve calibration-selected single-switch forms. There
is no continuous-by-gripper Cartesian product.

On frozen development demos36--39, the selected interface reduces full-H8 raw
expert-action MSE by `68.096%` (task-stratified demo-block bootstrap 95% CI
`[66.213%,69.821%]`). H1 gain is `80.255%`; all ten tasks are positive. The
result comfortably clears the frozen 10% H8 and H1 gates.

The previous two contract defects are resolved without a post-evaluation
repair. All 48 continuous alternatives form exact construction-level
`+epsilon/-epsilon` pairs; float32 reconstruction error is at most
`5.96e-8`, below the calibration-frozen `2e-7` engineering tolerance, and the
unmatched-direction count is zero. The combined interface includes the
previously omitted `c0_binarized` G3 form. A0 occurs once by contract, all
special forms are unique, and all candidates are finite and within bounds.

The selected pool remains an imitation-oracle diagnostic rather than a
deployable planner. Expert actions are unavailable at deployment, and no
candidate has a simulated future, reward, task value, or success result. The
only new authorization is `GO_CANDIDATE_VALUE_AUDIT`: a separately frozen
offline goal-conditioned progress/value representation study. World Model and
controller work remain blocked.

## Frozen provenance and correction boundary

The blocked interface study is frozen at commit
`241546e9021faac9c1313536aebb3c0962ee32bf` under annotated tag
`expanded-candidate-contract-restart`. Its numerical evidence and formal block
remain historical facts.

The restart protocol and initial implementation were committed at `043a853`
before calibration. The first calibration invocation exposed a validation-only
defect: reconstructing deviations from float32 `A0+epsilon` and
`A0-epsilon` cannot be required to be bitwise negatives after rounding. No
evaluation partition was opened and no candidate, seed, allocation, metric, or
decision rule changed. Commit `a025ecd` froze a `2e-7` absolute reconstruction
tolerance plus maximum-error reporting; the rebuilt calibration observed only
`5.96e-8` error. Calibration and the exact selected interface were frozen at
`face4f4` before demos36--39 were opened. The one-shot numerical evaluation was
then frozen at `d8dbed9`.

## Data boundary

| Partition | Demo IDs | Legal windows | Use |
|---|---:|---:|---|
| Calibration/design | 31--34 | 4,724 | switch subsets, allocation selection, freeze |
| Frozen development evaluation | 36--39 | 4,563 | one-shot metrics and gate; not blind confirmation |
| Historical reference | 30/35 | not opened | scalar/context reference only |
| Forbidden | 40--49 | 0 | never accessed |

For each explicitly admitted HDF5 group, natural vision and proprioception at
`k` feed the frozen causal C0-v2 actor, while the expert diagnostic is
`actions[k+1:k+9]`. The loader never enumerates forbidden demos. No simulator,
reward, done, success, or task outcome was read.

## Exact selected contract

```text
Selected Expanded Candidate Interface v1

H:                              8
N_total:                        64
A0:                             frozen clipped C0-v2 chunk, index 0 exactly once
Continuous alternatives:       48
Continuous antithetic pairs:   24
Continuous scale:              2.5 x frozen production sigma
Continuous dimensions:         translation 0--2, rotation 3--5
Continuous gripper:             unchanged frozen C0
Temporal smoothing:             [0.25, 0.5, 0.25]
Structured gripper alternatives: 15
Required G3:                    c0_binarized; zero tie -> positive
Bounds:                         [-1, 1]
Seed base:                      1901
Candidate order:                A0; +epsilon_i/-epsilon_i pairs; frozen gripper order
Cartesian product:              none
Contract SHA-256:               5cd3311d847c410a75cca6970e7acf4e5ec3af811c739946df7ec18532a9f14a
```

The structured gripper sequence is:

```text
c0_binarized
all_negative
all_positive
plus_to_minus_after_1
minus_to_plus_after_1
plus_to_minus_after_4
minus_to_plus_after_4
plus_to_minus_after_5
minus_to_plus_after_5
plus_to_minus_after_3
minus_to_plus_after_6
plus_to_minus_after_6
minus_to_plus_after_2
minus_to_plus_after_3
plus_to_minus_after_7
```

The repository establishes only negative/positive gripper states, so this
report does not guess open/close semantics.

## Calibration-only allocation selection

All three preregistered arithmetic allocations passed the exact contract and
numeric-validity checks. The frozen lexicographic rule first maximized H8 gain,
then mean H1/H2 gain, then lower candidate clipping, then larger continuous
count.

| Allocation | Composition after A0 | Calibration H8 | H1 | Positive tasks | Candidate clipped | Contract |
|---|---|---:|---:|---:|---:|---|
| C52_G11 | 52 continuous + 11 gripper | 61.928% | 76.506% | 10/10 | 9.570% | pass |
| C50_G13 | 50 continuous + 13 gripper | 64.786% | 76.402% | 10/10 | 9.192% | pass |
| **C48_G15** | **48 continuous + 15 gripper** | **67.420%** | **76.229%** | **10/10** | **8.827%** | **pass/selected** |

No evaluation result selected an allocation or switch form.

## Frozen development results

| Allocation | H8 gain | 95% CI | H1 | Positive tasks | Candidate clipped |
|---|---:|---|---:|---:|---:|
| C52_G11 | 61.278% | [59.003%,63.191%] | 80.513% | 10/10 | 10.251% |
| C50_G13 | 64.901% | [62.832%,66.677%] | 80.398% | 10/10 | 9.850% |
| **C48_G15 frozen** | **68.096%** | **[66.213%,69.821%]** | **80.255%** | **10/10** | **9.462%** |

Evaluation did not reselect the interface. The other two rows are transparent
frozen diagnostics.

## Horizon and task robustness

The selected interface's prefix gains are `80.255%`, `77.515%`, `70.963%`,
and `68.096%` for H1/H2/H4/H8. Horizon-specific oracle gains range from
`80.254%` to `82.025%` across h1--h8.

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H8 gain | 67.44% | 68.94% | 73.92% | 73.72% | 70.71% | 65.49% | 70.86% | 63.16% | 65.18% | 60.40% |

All tasks clear zero by a wide margin. This is broad imitation headroom, not a
claim about task success.

## Candidate composition and gripper diagnosis

| Winner class | Fraction |
|---|---:|
| A0 | 2.37% |
| Continuous | 57.99% |
| Structured gripper | 39.64% |

Within all rows, G3 wins `25.09%`, constants `4.51%`, and single switches
`10.04%`. The gripper-only oracle gain is `95.069%`; the selected gripper sign
agrees with the exactly binary expert channel on `99.81%` of steps. Translation
and rotation dimension-only oracle gains are `31.952%` and `27.845%`.

These diagnostics justify retaining both continuous and structured-gripper
branches. They do not make expert MSE a deployable value function.

## Exactness, validity, and coverage

| Check | Result | Frozen limit |
|---|---:|---:|
| Total candidates | 64 | exactly 64 |
| Continuous pairs | 24 | all complete |
| Unmatched directions | 0 | 0 |
| Max float32 pair error | `5.96e-8` | `2e-7` |
| G3 / both constants present | yes | required |
| Duplicate special-form contract | none | none |
| Invalid / NaN / Inf / out-of-range | 0 | 0 |
| Preclip component fraction | 0.336% | <=1% |
| Candidate-any-clip fraction | 9.462% | <=25% |
| Boundary-hit component fraction | 8.506% | <=10% |
| Mean nearest-neighbor action RMS | 0.09538 | descriptive |
| Near-duplicate candidate fraction | 3.270% | descriptive |
| Exact duplicate-pair fraction | 0.0618% | descriptive |

Numeric duplicates can occur after clipping or when a structured form happens
to match another realized sequence. They do not duplicate the frozen special
form contract or create an unmatched continuous direction.

## Mechanical decision

| Gate | Result |
|---|---|
| Exact deterministic N64 contract | pass |
| A0 exactly once by contract | pass |
| All continuous directions antithetically paired | pass |
| G3 and both constants included | pass |
| Unique special-form contract | pass |
| H8 gain >=10% and CI lower >0 | pass: 68.096%, lower 66.213% |
| H1 gain >=10% | pass: 80.255% |
| At least 7/10 positive tasks | pass: 10/10 |
| Frozen numeric validity | pass |

Final mechanism labels:

```text
EXPANDED_CANDIDATE_INTERFACE_VALID
STRUCTURED_GRIPPER_PROPOSALS_VALID
N64_REQUIRED_FOR_HEADROOM
STRONG_EXPANDED_CANDIDATE_HEADROOM
```

Next-stage authorization:

```text
GO_CANDIDATE_VALUE_AUDIT
```

The authorization permits only the preregistered offline demonstration-based
goal-conditioned progress/value representation audit. It does not authorize
candidate execution, a World Model, a reranker, a controller, or success
evaluation.

## Required answers

**Q1 — Is there now one exact production candidate contract?**

Yes as a frozen interface specification: C48_G15 has exactly 64 ordered
candidates, complete antithetic continuous pairs, all mandatory gripper form
classes, fixed seeds, bounds, and ordering. “Production” here describes the
interface contract, not deployment authorization.

**Q2 — Were the prior unmatched direction and missing G3 repaired cleanly?**

Yes. Both repairs were preregistered before the new calibration/evaluation.
The evaluated contract has zero unmatched directions and includes
`c0_binarized`. No candidate was changed after development evaluation.

**Q3 — Does the corrected interface retain strong headroom?**

Yes under the imitation oracle: H8 gain is `68.096%`, H1 is `80.255%`, the CI
is strongly positive, and all ten tasks improve.

**Q4 — Are continuous and structured-gripper proposals both used?**

Yes. Continuous proposals win `57.99%` of rows and structured gripper proposals
win `39.64%`; G3 alone wins `25.09%`.

**Q5 — Is the interface numerically valid and sufficiently distinct?**

Yes under the frozen validity rules. All arrays are finite and bounded, clip
rates pass, antithetic error is below tolerance, and exact duplicate-pair rate
is only `0.0618%`.

**Q6 — Is this evidence of deployed candidate value or success?**

No. It is expert-action imitation headroom only. The N64 candidates have no
real or predicted future outcomes in this audit.

**Q7 — May the project proceed?**

Only to Stage B, the offline candidate-value representation audit.
`BLOCK_SPATIAL_WM_PROTOTYPE` and controller authorization `false` remain in
force.

## Artifacts and reproducibility

| Artifact | SHA-256 |
|---|---|
| Protocol | `e98fcab27ed22cc20acd3141c6915d7d0a6ec477799e656ff6ec878e5332a542` |
| Calibration freeze | `c65ac858a291fd537effa41d835ca46d3d145abc6e71832090bebc62b7f05072` |
| Calibration sidecar | `d38baaca90fa51c36c32de08fcf7c0317bd6fde9db75d73c7d65cbcdb4e3e2af` |
| Selected interface | `5cd3311d847c410a75cca6970e7acf4e5ec3af811c739946df7ec18532a9f14a` |
| Evaluation diagnosis | `899c9ae6d0ed9c922a87a7aa162834ca967c07ecd3e6978afa6b5f98448c74a4` |
| Evaluation sidecar | `d680d8c2c035816485e07b61cdf5b15277246ef9e473aa10f37ed3fc6f5b9990` |
| Probe implementation | `ad218c6095b76f0d80fde6a10040db2f11604b03526d58d5dfc8d2aa24ca0e10` |
| Audit runner | `84de008061fca102717556345a3a83e300c68db10966a952948077d06ec6b402` |

The ignored sidecars retain sample identities, expert/A0 chunks, family oracle
errors, winners, categories, candidates, and bootstrap gains. All admitted
inputs, candidates, errors, bootstrap arrays, and reported metrics are finite.

Five protocol tests and six result tests independently verify frozen hashes,
partition boundaries, the finite allocation set, G3/constants/switch forms,
exact antithetic construction, sidecar identities, recomputed gains, oracle
non-regression, winner composition, gripper diagnostics, complete gate, and
authorization mapping. They pass `11/11`. The complete repository suite passes
`201` tests; the remaining 21 messages are dependency deprecation warnings,
not failures.

## Forbidden-activity audit

No simulator, real candidate future, reward, done, success, task outcome,
demo/init40--49, World Model, encoder, spatial extractor, learned proposal,
value model, reranker, score, controller, CEM/MPPI/beam search, C1/C2, D20-v2,
or seed23/47 experiment was run or modified. No candidate is described as
having real visual or task value. No upstream artifact was overwritten.
