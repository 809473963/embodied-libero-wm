# C1-v2 rebase and world-model distribution audit

Date: 2026-09-24

Final verdict: `BLOCK_C1_V2_RERANKER`

## Executive summary

C1-v2 preserves aggregate controller-development competence but fails the
preregistered reranker-evidence gate. On init-state indices 30--39, frozen
C0-v2 and C1-v2 both achieve 44/100 successes across 7/10 non-zero tasks.
C1-v2 replaces the actor-default candidate on 2,441/2,728 plans (89.48%), yet
the selected prediction is non-worse than the default prediction against the
subsequently observed latent at only h8. It is slightly worse at h2, h4, and
h6; the gate required at least two non-worse horizons.

The action interface is not the blocker. Every stored plan passes independent
Actor denormalization, physical candidate construction/clipping, WM
normalization, finite-value, and latent-norm back-checks. The WM never reads an
Actor-normalized action directly.

The logging-only distribution audit also gives a useful positive result:
C0-v2 is closer to expert validation than historical C0-v1 at all four audited
horizons for both mismatch and uncertainty. The median excess uncertainty over
expert is reduced by 58.0% on average. It is not eliminated: C0-v2 rollout
uncertainty medians remain 1.39--1.41 times expert, supporting the conclusion
that policy-induced distribution shift is a general residual problem, not only
a causal-label problem.

D20-v2 threshold work and C2-v2 remain blocked. No C1-v2/C2-v2 rollout from
indices 40--49 was viewed or run.

## C0-v2 freeze and independent v2 stack

The frozen causal actor is commit
`20f3166d0a0f4d7ed034425c056f042d85fe077b`, annotated tag
`c0-v2-causal-aligned`. The tag fixes:

- checkpoint `results/c0_v2/seed11/best.pt`, SHA-256
  `2346603c8123e0a1cdee408cb635a2ffd12d6bc4e0cc5d91eacbc1330f5faac0`;
- config SHA-256
  `33bb274fd6413bdd2ccd15158a25272e40f2580c3f33b31feb241f3cced12897`;
- contract `obs[k] -> actions[k+1:k+9]` and its train-only unique-row
  normalization;
- development result SHA-256
  `ca64a2d5169efc4f4db0dccb34c546305a86a922a6648bd4763dd5ecb3ad5a29`;
- confirmatory result SHA-256
  `ed2576d41bb4b02abda6158bea122ec4969354a933c2610ecf1a7cdcea6b494c`.

The new stack is versioned independently as
`C0-v2 -> C1-v2 -> D20-v2 -> C2-v2`. Old C0-v1/C1-v1/C2-v1, D20-v1, and
D21/D22 results remain frozen. This round implements and audits C1-v2 only.

## Online causal mapping

D13 uses the half-open slice:

```text
actions[k+1:k+9] = a[k+1], ..., a[k+8]
```

It contains exactly eight actions. Immediately after online observation `k`,
`candidate[0:8]` naturally supplies those same next eight causal actions. There
is no missing ninth action. The earlier D22.5 wording was corrected without
changing any old rollout artifact.

## C1-v2 definition

C1-v2 uses the frozen C0-v2 actor, frozen D13/D14/D15 three-member ensemble,
H=8, and N=3 candidates. It retains C1-v1's deterministic antithetic smoothed
Gaussian candidate idea and frozen score:

```text
score = goal_progress
        - 0.25 * mean_pairwise_uncertainty
        - 0.001 * normalized_action_deviation
```

Candidate 0 is the clipped actor-default chunk. Candidates 1 and 2 add/subtract
the same temporally smoothed noise; gripper noise is zero. The selected chunk
executes open loop. There is no replanning and no D20-v2 rule.

## Action-space interface audit

The model-facing chain is explicit:

```text
C0-v2 actor-normalized [8,7]
  -> Actor denormalization
  -> raw physical actor chunk
  -> clip actor default to [-1,1]
  -> raw antithetic candidate generation
  -> clip executable candidates to [-1,1]
  -> WM causal-action normalization
  -> frozen D15 ensemble
```

Actor action statistics:

```text
mean = [ 0.15043248,  0.13491899, -0.15583906, -0.00498378,
        -0.01116378, -0.01996221,  0.10107868]
std  = [ 0.41536851,  0.34807537,  0.51000369,  0.03836074,
         0.07189857,  0.05811741,  0.99487844]
count = 55,531
rule = train demos 0-44; each episode actions[1:] row exactly once
```

The WM statistics have the same numeric vectors and count, computed from the
same unique causal-action rows without overlapping-window expansion. Numeric
equality does not collapse the semantic boundary: candidate construction is in
physical action space, and its output is passed through the independent WM
normalization path.

Observed C1-v2 interface ranges and clipping:

| Quantity | Result |
|---|---:|
| Actor-denormalized pre-clip range | [-1.631214, 1.486525] |
| Actor-denormalized element clipping rate | 4.0231% |
| Generated candidate pre-clip range | [-1.104024, 1.097488] |
| All generated-candidate clipping rate | 0.0698% |
| Selected-candidate clipping rate | 0.0825% |
| Executable raw candidate range | [-1.0, 1.0] |
| WM-normalized candidate range | [-3.405956, 3.264605] |

The frozen batched C0-v2 baseline had 45/100 successes, while the new
single-environment logging trace has 44/100. The differences are task 2
(-1), task 5 (-1), and task 6 (+1). The same-observation frozen-plan action
parity test is exact (`max_abs_diff = 0.0`), and every sidecar equation passes;
the one-success difference is therefore reported as an execution/batching
reproduction limitation, not an interface failure. C0-v2 and C1-v2 below use
the same single-environment instrumentation and are directly comparable.

## Preregistered data boundaries and gate

Protocol `c1-v2-rebase-and-wm-distribution-v1` was committed as `de3cb2d`
before any C1-v2 rollout. Its SHA-256 is
`19ce68fabe912d849e65d283e4ce713beda233f4549385c672c8abb655060946`.

| Init states | Role |
|---|---|
| 0--19 | legacy exposed; forbidden for design |
| 20--29 | historical diagnosis/development; C0-v1 WM distribution reference only |
| 30--39 | C0-v2 logging and C1-v2 controller development |
| 40--49 | forbidden for C1/C2 design; future controller evaluation only, not fully blind |

All 200 current episodes use only indices 30--39 with their preregistered
per-task seeds. The gate was frozen before results. Nine of ten checks pass;
only `selected_observed_mismatch_nonworse_horizon_count >= 2` fails (measured
1). This maps mechanically to `BLOCK_C1_V2_RERANKER`.

## Frozen-WM distribution audit

Threshold exceedance uses frozen D20-v1 only as a diagnostic reference:
`tau_mismatch = 0.0290635973`, `tau_uncertainty = 0.00445149094`. No threshold
was changed. `Exceed` is the fraction strictly above the corresponding frozen
threshold.

| Distribution | Metric | h | Median | p75 | p90 | p95 | p99 | Exceed | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Expert validation | mismatch | 2 | 0.006669 | 0.009321 | 0.012557 | 0.014805 | 0.021739 | 0.29% | 5869 |
| Expert validation | uncertainty | 2 | 0.001346 | 0.001707 | 0.002141 | 0.002429 | 0.003147 | 0.00% | 5869 |
| Expert validation | mismatch | 4 | 0.007968 | 0.011039 | 0.014673 | 0.017574 | 0.024294 | 0.34% | 5869 |
| Expert validation | uncertainty | 4 | 0.001302 | 0.001709 | 0.002152 | 0.002485 | 0.003182 | 0.09% | 5869 |
| Expert validation | mismatch | 6 | 0.008733 | 0.011972 | 0.016034 | 0.018956 | 0.026371 | 0.58% | 5869 |
| Expert validation | uncertainty | 6 | 0.001361 | 0.001791 | 0.002293 | 0.002663 | 0.003628 | 0.27% | 5869 |
| Expert validation | mismatch | 8 | 0.009563 | 0.012966 | 0.017170 | 0.020490 | 0.027527 | 0.80% | 5869 |
| Expert validation | uncertainty | 8 | 0.001475 | 0.001941 | 0.002473 | 0.002915 | 0.004127 | 0.65% | 5869 |
| Historical C0-v1 20--29 | mismatch | 2 | 0.007779 | 0.011146 | 0.015280 | 0.018079 | 0.024374 | 0.37% | 3491 |
| Historical C0-v1 20--29 | uncertainty | 2 | 0.002732 | 0.004245 | 0.005901 | 0.007069 | 0.009076 | 22.40% | 3491 |
| Historical C0-v1 20--29 | mismatch | 4 | 0.009986 | 0.014619 | 0.020680 | 0.025123 | 0.033398 | 2.44% | 3487 |
| Historical C0-v1 20--29 | uncertainty | 4 | 0.002631 | 0.004179 | 0.005838 | 0.007007 | 0.008926 | 22.00% | 3487 |
| Historical C0-v1 20--29 | mismatch | 6 | 0.011253 | 0.016650 | 0.023827 | 0.029044 | 0.040947 | 5.00% | 3399 |
| Historical C0-v1 20--29 | uncertainty | 6 | 0.002680 | 0.004146 | 0.005943 | 0.007029 | 0.009129 | 21.48% | 3399 |
| Historical C0-v1 20--29 | mismatch | 8 | 0.012558 | 0.018202 | 0.025858 | 0.031425 | 0.043491 | 7.01% | 3395 |
| Historical C0-v1 20--29 | uncertainty | 8 | 0.002760 | 0.004226 | 0.006022 | 0.007168 | 0.009354 | 22.71% | 3395 |
| C0-v2 30--39 | mismatch | 2 | 0.006993 | 0.009813 | 0.013456 | 0.016081 | 0.022252 | 0.30% | 2710 |
| C0-v2 30--39 | uncertainty | 2 | 0.001900 | 0.002632 | 0.003589 | 0.004183 | 0.005333 | 4.02% | 2710 |
| C0-v2 30--39 | mismatch | 4 | 0.008863 | 0.012696 | 0.016960 | 0.020246 | 0.029191 | 1.04% | 2702 |
| C0-v2 30--39 | uncertainty | 4 | 0.001835 | 0.002576 | 0.003578 | 0.004277 | 0.005943 | 3.85% | 2702 |
| C0-v2 30--39 | mismatch | 6 | 0.010137 | 0.014714 | 0.019692 | 0.023065 | 0.031822 | 1.63% | 2637 |
| C0-v2 30--39 | uncertainty | 6 | 0.001922 | 0.002693 | 0.003762 | 0.004452 | 0.006290 | 5.04% | 2637 |
| C0-v2 30--39 | mismatch | 8 | 0.011273 | 0.015900 | 0.021473 | 0.025333 | 0.035234 | 2.75% | 2622 |
| C0-v2 30--39 | uncertainty | 8 | 0.002057 | 0.002848 | 0.004068 | 0.004872 | 0.006677 | 7.09% | 2622 |
| C1-v2 30--39 | mismatch | 2 | 0.007079 | 0.009959 | 0.013503 | 0.016369 | 0.022838 | 0.33% | 2724 |
| C1-v2 30--39 | uncertainty | 2 | 0.001930 | 0.002674 | 0.003793 | 0.004500 | 0.005844 | 5.21% | 2724 |
| C1-v2 30--39 | mismatch | 4 | 0.009133 | 0.012760 | 0.017511 | 0.021058 | 0.030296 | 1.33% | 2713 |
| C1-v2 30--39 | uncertainty | 4 | 0.001878 | 0.002597 | 0.003842 | 0.004529 | 0.005776 | 5.49% | 2713 |
| C1-v2 30--39 | mismatch | 6 | 0.010290 | 0.014544 | 0.019645 | 0.023581 | 0.034230 | 2.00% | 2644 |
| C1-v2 30--39 | uncertainty | 6 | 0.001958 | 0.002733 | 0.004006 | 0.004816 | 0.006079 | 6.66% | 2644 |
| C1-v2 30--39 | mismatch | 8 | 0.011381 | 0.015782 | 0.020592 | 0.024327 | 0.036710 | 2.55% | 2632 |
| C1-v2 30--39 | uncertainty | 8 | 0.002089 | 0.002877 | 0.004200 | 0.005093 | 0.006536 | 8.62% | 2632 |

Answers to the distribution questions:

1. C0-v2 is closer to expert validation than historical C0-v1 in quantile
   MAE at h2/h4/h6/h8 for both mismatch and uncertainty (8/8 comparisons).
2. The causal Actor reduces the historical median uncertainty excess over
   expert by 60.0%, 59.9%, 57.5%, and 54.7% at h2/h4/h6/h8, respectively;
   mean relief is 58.0%. Frozen-threshold uncertainty exceedance falls from
   21.48--22.71% to 3.85--7.09%.
3. The residual is material: C0-v2 median uncertainty is still 1.39--1.41x
   expert and p95 is 1.67--1.72x expert. Thus causal label alignment explains
   much, but not all, of the old shift. Policy-induced OOD remains a general
   issue.

The C0-v1 and C0-v2 traces use different preregistered partitions (20--29 and
30--39), so these comparisons are descriptive, not paired causal estimates.

## C0-v2 versus C1-v2 development competence

| Metric | C0-v2 | C1-v2 |
|---|---:|---:|
| Successes / episodes | 44/100 | 44/100 |
| Macro success | 44.0% | 44.0% |
| Non-zero tasks | 7/10 | 7/10 |
| Mean steps | 213.62 | 214.47 |
| Plans | 2,713 | 2,728 |
| Actor raw clipping | 3.9854% | 4.0231% |
| Generated-candidate clipping | 0.0000% | 0.0698% |
| Selected-candidate clipping | 0.0000% | 0.0825% |
| Candidate replacements | 0 | 2,441 |
| Replacement rate | 0.0% | 89.48% |
| Mean planning time | 59.50 ms | 65.15 ms |
| p95 planning time | 196.33 ms | 208.40 ms |

Both times include frozen-WM logging/inference; C1 additionally scores three
candidates.

| Task | C0 success | C1 success | Delta | C0 mean steps | C1 mean steps | C0 plans | C1 plans | C1 replace |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9/10 | 9/10 | 0 | 110.6 | 112.3 | 142 | 145 | 92.4% |
| 1 | 7/10 | 8/10 | +1 | 168.2 | 150.3 | 214 | 193 | 88.6% |
| 2 | 6/10 | 4/10 | -2 | 182.8 | 221.8 | 232 | 282 | 94.3% |
| 3 | 8/10 | 9/10 | +1 | 128.5 | 107.8 | 164 | 138 | 91.3% |
| 4 | 0/10 | 0/10 | 0 | 300.0 | 300.0 | 380 | 380 | 91.1% |
| 5 | 4/10 | 5/10 | +1 | 222.3 | 218.5 | 283 | 278 | 86.0% |
| 6 | 6/10 | 4/10 | -2 | 199.0 | 226.7 | 253 | 288 | 93.8% |
| 7 | 0/10 | 0/10 | 0 | 300.0 | 300.0 | 380 | 380 | 95.8% |
| 8 | 4/10 | 5/10 | +1 | 224.8 | 207.3 | 285 | 264 | 83.3% |
| 9 | 0/10 | 0/10 | 0 | 300.0 | 300.0 | 380 | 380 | 80.3% |

C1-v2 preserves aggregate success and task coverage, but its gains on tasks
1/3/5/8 are offset by regressions on tasks 2/6. Tasks 4/7/9 remain at floor.

## Does candidate replacement help?

C1-v2 selected candidates 0/1/2 on 287/1,220/1,221 plans. On the 2,441
replacements:

- selected score is at least the default score in 100% and strictly higher in
  100%; mean gain is `0.0003480`;
- selected goal progress is higher in 99.918% of replacements;
- selected mean uncertainty is lower in only 52.72%, with a very small mean
  change of `-0.00000614`;
- success episodes have an 88.65% replacement rate versus 89.71% for failure
  episodes (difference -1.06 percentage points). This is observational, not a
  randomized treatment effect.

There is no legitimate pre-execution "predicted mismatch" label because no
future observed target latent exists at selection time. The audit does not
fabricate one. Instead it compares selected and actor-default predictions to
the same subsequently observed selected-policy latent:

| Horizon | Checks | Selected mean | Default-prediction mean | Selected - default | Selected lower fraction | Non-worse mean |
|---:|---:|---:|---:|---:|---:|---|
| h2 | 2,438 | 0.00806962 | 0.00806244 | +0.00000717 | 47.05% | no |
| h4 | 2,427 | 0.01035245 | 0.01034130 | +0.00001114 | 48.41% | no |
| h6 | 2,366 | 0.01174424 | 0.01173961 | +0.00000463 | 48.73% | no |
| h8 | 2,355 | 0.01268138 | 0.01269770 | -0.00001632 | 50.06% | yes |

This comparison is diagnostic rather than a counterfactual default-action
rollout. Even under that favorable limitation, only 1/4 horizons is non-worse,
so the preregistered reranker gate correctly blocks progression.

## Verification and artifacts

Per-plan compact JSON records include actor-default score, selected index,
replacement flag, score components, uncertainty, and actual mismatch checks.
Hash-addressed compressed sidecars under ignored
`artifacts/stage3/c1_v2_controller_development/` retain all candidate action
tensors plus every ensemble trajectory `[3,8,512]`. The result audit verifies
all sidecar hashes and equations.

Primary artifacts:

| Artifact | SHA-256 |
|---|---|
| `results/c1_v2/rebase_audit.json` | `bca41d9330d57039bbf4e4cff6acd5486bedf25c7256f94456415c1e0aa709e5` |
| `results/c1_v2/wm_distribution_table.csv` | `d6128e7ae3d5a1ff4691734ec0f96b47dc51090e128686b2efc24de500f7d40f` |
| `configs/c1_v2_gate_decision.json` | `4f56544eaa7c04023342d54c3dd98fd6012522118cf8e6128f9407cb7e2ac2e0` |

The audit additionally revalidates the frozen C0-v2 checkpoint, ensemble
config, all three WM checkpoints, expert-validation samples, D22.5 historical
traces, and D20-v1 threshold result by SHA-256. All values and stored
trajectories are finite.

Tests cover protocol boundaries, exact action-space transformations, collector
semantics, synthetic analysis/gate paths, all 200 final episode identities,
all raw/sidecar hashes, distribution rows, forbidden-activity flags, and the
final report/verdict. Git contains compact JSON/CSV/code/docs only; HDF5,
checkpoints, caches, rollout logs, and trajectory sidecars remain ignored.

## Final gate decision

`BLOCK_C1_V2_RERANKER`

The failure is specific: action-space integration passes and aggregate
competence is retained, but replacement does not provide the preregistered
multi-horizon observed-mismatch evidence. Do not start D20-v2 threshold
selection, C2-v2, old D23, or any 40--49 controller rollout from this state.
