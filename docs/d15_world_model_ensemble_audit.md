# D15 World Model Ensemble Audit

## Verdict

**D15 PASS.** A three-member latent dynamics ensemble is trained, independently
reloadable, finite, and evaluated on all 5,869 held-out D13 validation samples.
This day defines epistemic disagreement and records resources; it does not make
the formal D16 copy-current gate decision.

## Frozen inputs and implementation

- D13 contract: `d13-v1`
- D13 store SHA-256: `e5599153cf44282242d955be62249bbafc752b782f5ae76a0ec0d58f9146f98e`
- D13 contract SHA-256: `e988b7e1078b3c89614bc83225ba677045cfda3b854098a3e1180b85cca32181`
- D14 implementation commit: `c276bbcced898b2df7b41cf95e964390dd01fb81`
- D15 implementation commit: `b6cbe73109f4cec30367441ec93b6074cbfa337f`
- Members: seeds `[11, 23, 47]`
- Per-member parameters: 792,000; total parameters: 2,376,000
- Each member uses the frozen D14 architecture, `H=8`, identical train/validation
  episode split, and identical train-only normalization.

The ensemble prediction is the L2-normalized arithmetic mean of the member
latents. D15 uncertainty is defined as mean pairwise cosine distance:

```text
u[b,h] = 2 / (M(M-1)) * sum_{i<j}(1 - cosine(pred_i[b,h], pred_j[b,h]))
```

It has shape `[B,8]`. It is an epistemic member-disagreement score, not a
prediction error, probability, or calibrated failure score.

## Member checkpoints and training resources

| Seed | Best epoch | Validation error | Train seconds | Peak allocated / reserved | Checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---|
| 11 | 13 | 0.00937134 | 275.22 | 75,717,120 / 123,731,968 | `12304c9da129f93325201fad84ce38796ca71fcf427821f64644d4203f6b3888` |
| 23 | 12 | 0.00947513 | 260.43 | 75,717,120 / 123,731,968 | `cbed3b53a6d770f9995877d6400e188a72b77ef68a3b077159885312df754805` |
| 47 | 14 | 0.00941024 | 268.91 | 75,717,120 / 123,731,968 | `4ed392ec7d1204fd3b0e453415608f54f501f9d0aca29615253a32e9462b43a3` |

Total member-training wall time was 804.56 seconds. The seed-11 member is the
already frozen D14 predictor; seeds 23 and 47 were trained for D15 with the
same 15-epoch/AdamW/AMP/gradient-clipping setup.

All checkpoint-required fields are present. All model and optimizer tensors in
all three best checkpoints are finite. Their model schema, data contract,
store hash, contract hash, and full normalization dictionaries match exactly.

## Shape and behavior tests

Synthetic test:

- member predictions: `[3,2,8,512]`
- mean prediction: `[2,8,512]`, finite and L2-normalized
- uncertainty: `[2,8]`, finite and nonnegative
- independently initialized members produce nonzero disagreement
- three identical prediction tensors produce uncertainty approximately zero

Real-checkpoint test:

- loaded seeds are exactly `[11,23,47]`
- all members reference the expected D13 store
- member predictions, ensemble prediction, and uncertainty are finite
- D13 dataset and D14 model tests continue to pass

The first full evaluator run exposed a float32/float64 mismatch in the
`torch.quantile` probability tensor. The probability tensor is now explicitly
float64, and the full evaluation plus regressions pass.

## Held-out ensemble validation

| Horizon | Ensemble cosine error | Mean uncertainty |
|---:|---:|---:|
| h1 | 0.00629586 | 0.00168742 |
| h2 | 0.00755894 | 0.00145110 |
| h3 | 0.00838467 | 0.00141173 |
| h4 | 0.00899007 | 0.00142430 |
| h5 | 0.00945218 | 0.00145625 |
| h6 | 0.00984455 | 0.00149920 |
| h7 | 0.01020906 | 0.00155342 |
| h8 | 0.01061715 | 0.00162266 |
| overall | **0.00891906** | **0.00151326** |

The best individual member is seed 11 at `0.00937135` when recomputed by the
ensemble evaluator. The ensemble mean lowers that held-out error by
`0.00045228`, or **4.83%** relative. This is a D15 aggregation diagnostic, not
the D16 copy-current gate.

Uncertainty distribution over 46,952 validation horizon points:

| Statistic | Value |
|---|---:|
| standard deviation | 0.00060019 |
| minimum | 0.00043350 |
| p50 | 0.00138547 |
| p90 | 0.00228827 |
| p95 | 0.00263665 |
| p99 | 0.00354049 |
| maximum | 0.00630381 |

The diagnostic Pearson correlation between disagreement and ensemble
prediction error is `0.25012`. This confirms the score is nonconstant and has
some positive association with prediction error; it does not establish a
deployment threshold or failure detector.

## Ensemble inference resources

- Total GPU inference time across validation batches: 0.37565 seconds
- Model inference time per validation sample: 0.06401 ms
- Peak GPU allocated / reserved: 76,961,792 / 98,566,144 bytes
- NaN/Inf audit: pass

The timing excludes DataLoader time and includes CUDA synchronization around
the three-model forward call.

## Artifacts and version control

- Ensemble config: `configs/world_model_ensemble.yaml`
- Aggregation implementation: `src/world_models/ensemble.py`
- Evaluator: `scripts/evaluate_world_model_ensemble.py`
- Tests: `tests/test_world_model_ensemble.py`
- Checkpoint index: `results/wm/d15_ensemble/ensemble_manifest.json`
- Validation/resource record: `results/wm/d15_ensemble/ensemble_validation.json`
- Manifest SHA-256: `1bf33109378c00565bc3beb1391c017cf5bcc57172d6c89ce891280525e39e41`
- Validation SHA-256: `46f20b84d78ff4b319083901b09bc50822507dd39848fccee3abdc122c01234d`

The three `best.pt`/`last.pt` files and training JSONL logs remain ignored.
Lightweight configs, run metadata, best-validation records, checkpoint hashes,
and ensemble results are versioned. After the D15 audit commit, tracked and
untracked status is expected to be clean, with only the established data,
checkpoint, JSONL, environment, and cache paths ignored.

## Decision

D15 supplies three valid independent predictors and a precisely defined,
measurable uncertainty signal. It therefore satisfies the prerequisites for
D16 held-out comparison. No D16 gate is claimed in this report.
