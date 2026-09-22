# D16 Held-out Transition Evaluation Audit

## Verdict

**D16 PASS.** On the frozen D13 validation split, the D15 ensemble predictor
has lower cosine error than copy-current overall, at every horizon, in every
held-out episode, and in every task. The task-stratified episode bootstrap
confidence interval for paired improvement is strictly positive.

This is an offline observation-transition result. It does not claim C1/C2
control success or use simulator evaluation episodes.

## Protocol

- Implementation commit: `66be196392d415850a4caa1e81b267126ba77516`
- Config: `configs/d16_transition_eval.yaml`
- Config SHA-256: `67924e5c1cd5f38a73a10916b84b62c978313f64b1663936610482d812d98b71`
- Predictor: normalized mean of D15 members with seeds `[11,23,47]`
- Baseline: `z0` repeated for all eight horizons
- Split: D13 validation demos 45–49 only
- Samples / episodes / tasks: 5,869 / 50 / 10
- Bootstrap: 10,000 replicates, episode unit, sampled within each task,
  seed 1601
- Forbidden data: no simulator evaluation/test episodes, reward, done,
  success, object pose, simulator state, or future proprio

Episode blocking is required because neighboring WM samples overlap heavily.
The confidence interval therefore resamples the 50 held-out demonstrations,
not the 5,869 correlated windows.

## Primary result

| Metric | Learned ensemble | Copy-current | Copy − learned |
|---|---:|---:|---:|
| Overall cosine error | **0.00891906** | 0.01579532 | **+0.00687626** |
| Relative error reduction | — | — | **43.53%** |

| Horizon | Learned | Copy | Improvement |
|---:|---:|---:|---:|
| h1 | 0.00629586 | 0.00702433 | +0.00072846 |
| h2 | 0.00755894 | 0.01017511 | +0.00261617 |
| h3 | 0.00838467 | 0.01295711 | +0.00457244 |
| h4 | 0.00899007 | 0.01541428 | +0.00642421 |
| h5 | 0.00945218 | 0.01758327 | +0.00813110 |
| h6 | 0.00984455 | 0.01946334 | +0.00961879 |
| h7 | 0.01020906 | 0.02108555 | +0.01087649 |
| h8 | 0.01061715 | 0.02265956 | +0.01204240 |

The gain increases with horizon, which is consistent with a learned action-
conditioned transition model adding more value as copy-current becomes stale.

## Paired robustness checks

- Sample-level improvement fraction: 90.99%
- Episode-level improvement fraction: 50/50 = 100%
- Task-level improvement fraction: 10/10 = 100%
- Episode-block bootstrap mean improvement: `0.00705895`
- 95% bootstrap CI: **`[0.00675743, 0.00735354]`**

Per-task improvement ranges from `+0.00353859` on task 1 to `+0.01021529`
on task 3. No task is carried by a negative task result.

## Frozen gate

| Check | Result |
|---|---|
| overall copy − learned > 0 | PASS |
| episode-bootstrap CI lower bound > 0 | PASS |
| every h1–h8 improvement > 0 | PASS |
| h8 improvement > h1 improvement | PASS |

Final gate: **PASS**.

## Reproducibility and integrity

- Result: `results/wm/transition_eval.json`
- Result SHA-256: `4e03b91c16d2458d489fc4924b01a4113595016e45709c961153e6e4429d8544`
- Per-sample record: `results/wm/transition_eval_samples.jsonl`
- Per-sample SHA-256: `d0f71bf9ade25116dd465bbecb09214b9b8544e2f853802ffa7116430da4ee3a`
- Unique `(task_id,demo_id,start_timestep)` rows: 5,869
- Demo IDs: exactly 45–49; task IDs: exactly 0–9
- Every learned/copy/improvement/uncertainty vector: length 8 and finite
- Aggregate values independently recomputed from JSONL match the result JSON
  within `1e-12`
- Repeating the full command produced byte-identical result and sample hashes
- NaN/Inf audit: pass

The JSON result is versioned. The larger per-sample JSONL remains ignored but
is hash-addressed by the result and retained locally for D17 sample selection.

## Decision

The learned predictor has a clear, paired, statistically stable advantage over
copy-current on held-out transitions. D16 therefore permits D17 qualitative
sample review. This result alone does not authorize C2 or establish that latent
error corresponds to online control failure.
