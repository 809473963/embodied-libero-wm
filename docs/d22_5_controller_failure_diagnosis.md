# D22.5 Controller Distribution-Shift Diagnosis

诊断日期：2026-09-23。

## Verdict

**Primary verdict: `BASELINE_COMPETENCE_BLOCK`.** D23 继续保持 BLOCK。冻结的
all-task C0 在完全独立的 controller-validation 上只有 15/100，且 7/10 个
task 为 0/10；它没有为跨任务 C1/C2 结论提供足够的 baseline competence。

**Secondary recommendation: `GO_D20_V2_PROTOCOL_DESIGN_ONLY`.** D20-v1 的
expert-transition calibration 不能迁移到 policy-induced controller distribution。
可以设计 D20-v2，但在 C0-v2 competence 和 controller action causal indexing 问题解决
前，不应再做 fixed final test。D20-v1 threshold 没有修改，本轮也没有选择任何 D20-v2
threshold。

## 1. D22 negative result freeze

- Frozen commit: `3f3633285503fb178b25f260192353e61e2a9d5c`
- Annotated tag: `d22-c2-overreplan-negative`
- C0/C1/C2 task-0 fixed test: `0/20`, `0/20`, `0/20`
- C2: 2,018 / 2,242 plans early replan
- Mean early replans: 100.9 / episode
- Trigger reasons: uncertainty-only 1,859; mismatch-only 37; both 122
- h2 early replans: 1,960 / 2,242 plans
- D20-v1 result SHA-256:
  `3d07d136d9a9ae493ea04ad9bd08ec5f4099a993e301de0906c47996b46c23a1`

D21/D22 negative JSON/JSONL artifacts remain untouched. Their outcomes are retained as
negative evidence but were not inputs to threshold selection or the numerical distribution
analysis below.

## 2. Independent control-validation protocol

The protocol was committed as `713a06ffdb8cf2cf0f6734dd39e4ab931bf31966`
before any rollout was collected. Its SHA-256 is
`4b3c5b914875f4154768d19232d743fdb1bc66ab4a1393447193e1c539782dff`.

Each LIBERO-Spatial task exposes 50 official init states, so a genuinely disjoint split was
possible:

| Partition | Records |
|---|---|
| demo train | demonstrations 0–44 |
| demo validation | demonstrations 45–49 |
| controller validation | init-state indices 20–29; 10 episodes/task; seeds 20000–20909 |
| fixed final test | init-state indices 0–19; immutable seeds 10000–10919 |
| still unused/reserved | init-state indices 30–49 |

Controller-validation and fixed final test have zero seed overlap and zero init-state overlap.
No task was selected from known fixed-test performance. All 10 tasks were collected in the
precommitted order.

Two logging-only distributions were collected:

- **C0 rollout:** execute each frozen C0 H=8 chunk fully; WM only observes.
- **C1-selected rollout:** execute each frozen C1-selected H=8 chunk fully; verification only
  logs and cannot replan.

Both use h2/h4/h6/h8 live upright agentview checks, the frozen D15 ensemble, frozen D20-v1
thresholds, and the same all-task C0 checkpoint. The collector contains no trigger-dependent
control branch. It stores no simulator state, object pose, reward, done, or fixed-test record.

Collection/analysis commits:

- Collector: `9fafe41a169e549fafa095663d03f9c974def33c`
- Unused-camera rendering fix: `c3359694acf8513eba6790cb65437835b965794d`
- Analysis implementation: `d48e83b0a5f72753748e015e0e2f283db269ff99`
- Frozen traces/results: `8a88453d495167ac7864157443e5f1432f374661`

## 3. Distribution shift

The full min/median/p75/p90/p95/p99/max table is in
`results/control_validation/distribution_table.csv`. The key h2 comparison is:

| Distribution | mismatch median | mismatch p95 | mismatch p99 | uncertainty median | uncertainty p95 | uncertainty p99 |
|---|---:|---:|---:|---:|---:|---:|
| expert demo validation | 0.006669 | 0.014805 | 0.021739 | 0.001346 | 0.002429 | 0.003147 |
| C0 controller validation | 0.007779 | 0.018079 | 0.024374 | 0.002732 | 0.007069 | 0.009076 |
| C1-selected controller validation | 0.007929 | 0.017811 | 0.024578 | 0.002700 | 0.006671 | 0.008143 |

At h2:

| Shift metric | C0 vs expert | C1 vs expert |
|---|---:|---:|
| mismatch median ratio | 1.166× | 1.189× |
| mismatch p95 ratio | 1.221× | 1.203× |
| mismatch KS statistic | 0.132 | 0.144 |
| uncertainty median ratio | **2.030×** | **2.006×** |
| uncertainty p95 ratio | **2.910×** | **2.747×** |
| uncertainty KS statistic | **0.604** | **0.611** |

D20-v1's uncertainty threshold is never exceeded at expert-validation h2, but is exceeded by
22.40% of C0 h2 checks and 21.76% of C1 h2 checks. By contrast, the mismatch threshold rate
only moves from 0.290% expert to 0.372% C0 / 0.364% C1. The dominant deployment shift is
therefore ensemble disagreement, not h2 mismatch.

Across later rollout horizons, prediction error grows as expected:

| Distribution | horizon | mismatch median / p95 | uncertainty median / p95 |
|---|---:|---:|---:|
| C0 | h2 | 0.007779 / 0.018079 | 0.002732 / 0.007069 |
| C0 | h4 | 0.009986 / 0.025123 | 0.002631 / 0.007007 |
| C0 | h6 | 0.011253 / 0.029044 | 0.002680 / 0.007029 |
| C0 | h8 | 0.012558 / 0.031425 | 0.002760 / 0.007168 |
| C1 | h2 | 0.007929 / 0.017811 | 0.002700 / 0.006671 |
| C1 | h4 | 0.010250 / 0.024332 | 0.002653 / 0.006527 |
| C1 | h6 | 0.011739 / 0.028615 | 0.002689 / 0.006633 |
| C1 | h8 | 0.012699 / 0.030275 | 0.002774 / 0.006866 |

C0 and C1-selected distributions are similar; the problem is not caused only by C1's small
candidate perturbations. Frozen C0 policy states/actions already lie well outside the expert
uncertainty calibration distribution.

## 4. D20-v1 trigger-source counterfactual

No rule below changed rollout execution. Projected mean plan length uses only plans with all
four h2/h4/h6/h8 observations and stops offline at the first counterfactual trigger.

| Method | Rule | Overall rate | h2 | h4 | h6 | h8 | Projected mean H |
|---|---|---:|---:|---:|---:|---:|---:|
| C0 | mismatch-only | 3.67% | 0.37% | 2.44% | 5.00% | 7.01% | 7.819 |
| C0 | uncertainty-only | 22.15% | 22.40% | 22.00% | 21.48% | 22.71% | 6.557 |
| C0 | OR | **23.91%** | 22.60% | 23.40% | 23.80% | 25.89% | **6.468** |
| C0 | AND | 1.91% | 0.17% | 1.03% | 2.68% | 3.83% | 7.918 |
| C1 | mismatch-only | 3.26% | 0.36% | 2.22% | 4.63% | 5.93% | 7.828 |
| C1 | uncertainty-only | 20.96% | 21.76% | 20.86% | 20.51% | 20.67% | 6.586 |
| C1 | OR | **22.75%** | 21.96% | 22.07% | 22.99% | 24.01% | **6.496** |
| C1 | AND | 1.47% | 0.17% | 1.01% | 2.16% | 2.59% | 7.926 |

On expert validation, OR rates at h2/h4/h6/h8 are only
`0.290% / 0.426% / 0.852% / 1.448%`. The current OR rule is therefore unreasonable for
deployment: its controller-validation rate is dominated by an expert-calibrated uncertainty
tail rather than actual mismatch.

The logging-only open-loop OR rate (about 23%) is lower than D22 C2's 90.01% early-plan rate.
This is not contradictory. Closed-loop C2 repeatedly replans from states reached after only two
actions, creating a feedback distribution that re-enters the high-uncertainty region. The D22
result is an amplification of the same calibration error.

## 5. Is uncertainty useful?

Yes as a moderate ranking/OOD signal; no as a calibrated OR failure trigger.

| Method | current Pearson / Spearman | next-check Pearson / Spearman | current high-error AUROC | next high-error AUROC |
|---|---:|---:|---:|---:|
| C0 | 0.352 / 0.368 | 0.368 / 0.379 | 0.724 | 0.730 |
| C1 | 0.347 / 0.382 | 0.361 / 0.392 | 0.696 | 0.705 |

The label for these ranking metrics is whether current/next mismatch exceeds the frozen D20
mismatch threshold. Positives are sufficiently numerous: 506 current and 493 next for C0; 459
current and 446 next for C1.

For both methods, mean current mismatch and mean next-check mismatch rise strictly across all
ten uncertainty deciles. Top-vs-bottom decile mean-error ratios are 2.09× / 2.13× for C0 and
1.98× / 2.00× for C1. Thus uncertainty is not arbitrary noise. However, high uncertainty says
the predictor is less reliable/OOD; it does not say that replanning with that same predictor is
the correct intervention.

## 6. Confidence-gated mismatch diagnostic

The following `tau_conf` candidates come only from pooled C0+C1 controller-validation
uncertainty. They are an exploratory grid, not selected thresholds:

| Candidate | tau_conf | C0 rate / projected H | C1 rate / projected H |
|---|---:|---:|---:|
| p25 | 0.001931 | 0.18% / 7.985 | 0.21% / 7.988 |
| p50 | 0.002700 | 0.85% / 7.944 | 0.79% / 7.947 |
| p75 | 0.004150 | 1.61% / 7.909 | 1.65% / 7.910 |
| p90 | 0.005783 | 2.44% / 7.867 | 2.49% / 7.862 |

The rule is `mismatch > tau_m AND uncertainty < tau_conf`: only a sufficiently confident model
may use mismatch to request a replan. It is semantically different from D20 AND, which triggers
when both mismatch and uncertainty are high—exactly where the predictor is less trustworthy.

This family deserves a separately precommitted validation experiment, as does mismatch-only.
No row is selected here because tuning and evaluating on the same 100 controller-validation
episodes would overfit. Init-state indices 30–49 remain unused and can support a future,
precommitted calibration/evaluation split.

## 7. C0 competence audit

| Task | C0 | C1-selected |
|---:|---:|---:|
| 0 | 0/10 | 0/10 |
| 1 | 0/10 | 0/10 |
| 2 | 0/10 | 0/10 |
| 3 | 0/10 | 0/10 |
| 4 | 3/10 | 2/10 |
| 5 | 0/10 | 0/10 |
| 6 | 7/10 | 5/10 |
| 7 | 0/10 | 0/10 |
| 8 | 5/10 | 5/10 |
| 9 | 0/10 | 0/10 |
| **Total** | **15/100** | **12/100** |

C0 is not zero everywhere, but success is confined to 3/10 tasks and 7/10 tasks are at floor.
This is insufficient for a broad C1/C2 study: most tasks provide no baseline headroom with
which to separate reranking/replanning effects. `BASELINE_COMPETENCE_BLOCK` therefore applies.
A C0-v2 is required, but this round did not train or alter C0.

## 8. Additional causal-index integration finding

D13's frozen observation-dynamics contract is:

```text
(z[k], p[k], language, actions[k+1:k+9]) -> z[k+1:k+9]
```

The deployed D19/D21 integration instead feeds the immediately executed candidate chunk
`candidate[0:8]` at the current observation. The diagnostic collector deliberately preserved
this behavior so that it measured the actual deployed controller distribution; it did not repair
or reinterpret the trace.

This is an unresolved causal action-index mismatch. D20-v2 must first define how an online C0
chunk maps to the WM's `k+1:k+9` training inputs (including the missing ninth action if needed),
or introduce a versioned WM contract. Otherwise controller mismatch cannot be given a clean
causal interpretation.

## 9. D20-v2 answers

**A. D20-v1's largest problem:** threshold calibration used expert demonstration transitions,
while C2 deployment uses policy-induced states/actions. The effect is especially severe for
ensemble disagreement. The unresolved action-index mapping is a second structural problem.

**B. Move calibration to policy-induced controller-validation?** Yes. D20-v2 should precommit
separate controller calibration and controller evaluation rollouts. It must not tune and report
on this same diagnosis set, and must never use fixed final test outcomes.

**C. What role should uncertainty have?** The evidence rejects `uncertainty OR mismatch` as the
next default. Worth validating are:

1. mismatch-only;
2. uncertainty retained as a reranking score and diagnostic;
3. uncertainty as a confidence gate that vetoes mismatch-based replanning when the predictor is
   not trusted.

D20 AND is not preferred merely because its rate is small: it selects high-mismatch,
high-uncertainty cases. Deleting uncertainty is also unsupported because its ranking signal is
real. No new threshold value is approved by this report.

## 10. Artifact and reproducibility audit

- Machine result: `results/control_validation/d22_5_diagnosis.json`, SHA-256
  `acfbc124f0f5156250d3ffe7cea94c0cba54b8bb8e1a94dd287f9ff287a9f527`
- Distribution table SHA-256:
  `006aef3d6266091b8dfa8b2045d95fe8258c258c98d1118e0be15db9b3242bf3`
- Trigger table SHA-256:
  `8284e7158e92f9df4b06f177bebfa21c9f22ec9da5679ef8cd5fc1a9a966439f`
- Uncertainty-decile table SHA-256:
  `96bb233b687abdbf10bdf55bb6c2ddee2f651f6e52fd0def13e289d2d381b3a8`
- Competence table SHA-256:
  `afa7ad388b3e4ba62ebbea4377699657f156b54eec7520ee8e56ed89d732213e`
- Raw traces: 20 task JSON files, 11,966,120 total bytes; every file hash is embedded in
  the machine result.
- C0: 13,772 checks / 3,493 plans; C1: 14,085 checks / 3,571 plans.
- Strict JSON, protocol identity, seed/init split, finite-value, event-count, source-hash, and
  generated-artifact-hash audits pass.
- Re-running the complete analysis produced byte-identical hashes for all 25 result files.

Commands:

```bash
.venv/bin/python tests/test_control_validation_protocol.py
.venv/bin/python tests/test_control_validation_collector.py
.venv/bin/python scripts/analyze_controller_distribution_shift.py
.venv/bin/python tests/test_controller_distribution_analysis.py
```

Final state: **`BASELINE_COMPETENCE_BLOCK`**, D23 remains blocked, D20-v1 remains frozen, and
`GO_D20_V2_PROTOCOL_DESIGN_ONLY` is the only authorized follow-up from this diagnosis.
