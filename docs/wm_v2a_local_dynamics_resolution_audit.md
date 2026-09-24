# WM-v2A local dynamics resolution experiment

Date: 2026-09-24

Final verdict: `DELTA_OBJECTIVE_INSUFFICIENT`

## Executive conclusion

WM-v2A preserves ordinary prediction quality and the previously established
global action signal, and its imitation-oriented local direction checks pass.
It does **not** resolve production-scale local action sensitivity. The frozen
C1 candidate separation/error ratio rises from `0.00423536` to `0.00530693`
(1.253x), but remains below both preregistered requirements: ratio `>=0.01`
and improvement `>=2.0x`. Seed23/47 are not authorized, no WM-v2A ensemble is
built, and WM-v2B is not implemented.

The negative result has a direct mathematical explanation. Because the same
`z0` is subtracted from prediction and target,

```text
pred_delta - target_delta
= (pred_z - z0) - (z_future - z0)
= pred_z - z_future.
```

With L2-normalized `pred_z` and `z_future`, and residuals in SmoothL1's
quadratic region, the specified mean-over-512 delta loss is approximately the
cosine loss divided by 512. At the selected checkpoint, the measured ratio is
`L_future / L_delta = 511.997`; it is 511.996--511.998 at every horizon.
Thus `lambda_delta=1` changes the effective gradient scale by only about
0.195%, rather than adding an independent local-dynamics constraint. The
experiment cleanly shows that this objective alone is insufficient; it does
not justify changing the controller, candidate generator, or score.

No simulator or controller rollout was run. Only D13 train demos 0--44 and
validation demos 45--49 were used. Init-state outcomes, controller success,
D20-v2, C2-v2, and the 40--49 partition were not accessed.

## Frozen provenance and preregistration

C1-v2.2 was frozen before this experiment at commit
`bd749c79f665c6a0eb3f5d5ebee82c3d64b03a98` with annotated tag
`wm-v1-low-local-action-sensitivity`. The tag resolves to that commit and
freezes the verdict `WM_HAS_ACTION_SIGNAL_BUT_LOW_LOCAL_SENSITIVITY` plus the
upstream blockers `RERANKER_SCORE_BLOCK`, `WM_ACTION_SENSITIVITY_BLOCK`, and
`BLOCK_C1_V2_RERANKER`.

The WM-v2A protocol, implementation, config, and tests were committed before
training as `d2767d8488cb71c6050f3626ea2c527729fc0a0f`. The checkpoint records
that exact source commit.

| Frozen input | SHA-256 |
|---|---|
| D13 observation store | `e5599153cf44282242d955be62249bbafc752b782f5ae76a0ec0d58f9146f98e` |
| D14 seed11 checkpoint | `12304c9da129f93325201fad84ce38796ca71fcf427821f64644d4203f6b3888` |
| D15 transition result | `4e03b91c16d2458d489fc4924b01a4113595016e45709c961153e6e4429d8544` |
| C1-v2.1 production-candidate sidecar | `5870337ad96c58b3cedee4e124cf89018408446ed82fc7a2df52626f482398fd` |
| C1-v2.2 action-conditioning sidecar | `24f8a8d870237dda04f63065457c92565f9e4b9b0633a8611b6ec23e089f33fd` |
| WM-v2A config | `0403f3267527ac624ba79d926383fc763b0a8189ef04f6a741c80283b65ca855` |
| WM-v2A protocol | `856d76349d775ccdddc11832835fe5a02e38530d1ef569025f19b9482d50ec59` |

All D14/D15 checkpoints and all C1-v2/v2.1/v2.2 artifacts remain unchanged.

## Unchanged contract and model

The causal contract remains:

```text
(z[k], proprio[k], language, actions[k+1:k+9])
    -> z[k+1:k+9]
```

H is 8. The split is 52,381 training samples from demos 0--44 and 5,869
validation samples from demos 45--49. Frozen upright agentview CLIP ViT-B/32
512-D image latents, 512-D language, 12-D proprio, 7-D raw actions, train-only
normalization, camera, and action indexing are unchanged. No simulator state,
object pose, reward, done, or success field is used.

The model is the exact D14 `LatentDynamicsModel`: the same projections, GRU,
normalized residual latent output, and **792,000 parameters**. Optimization is
also unchanged: AdamW, learning rate `3e-4`, weight decay `1e-4`, batch 256,
AMP fp16, gradient clipping at 10, seed 11, and 15 epochs. The run used an
NVIDIA GeForce RTX 4070 Laptop GPU; peak allocated/reserved memory was
79,919,616 / 121,634,816 bytes.

## Exact objective and selection rule

The only changed factor is the added loss term:

```text
target_delta = z_future - z0[:,None,:]
pred_delta   = pred_z    - z0[:,None,:]

L_future = mean(1 - cosine(pred_z, z_future))
L_delta  = SmoothL1(pred_delta, target_delta, beta=1, reduction=mean)
L_total  = L_future + 1.0 * L_delta
```

All horizons are equally weighted. `lambda_delta=1.0` was the single value
committed before training; there was no sweep or post-result selection. The
best checkpoint minimizes validation total loss, with lower validation future
cosine error as the frozen tie-breaker. Epoch 13 / step 2665 is selected.

The cancellation identity above is important: the added target is real and
causal, but this particular loss does not retain `z0` after subtraction.
Numerically, `L_delta=1.83049e-5` while `L_future=0.00937206`, almost exactly a
factor of 512 apart. Therefore the total loss is effectively a tiny rescaling
of the original cosine objective.

## Training and validation curve

| Epoch | Train future | Train delta | Train total | Validation future | Validation delta | Validation total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01259233 | 2.45944945e-5 | 0.01261693 | 0.01148009 | 2.24221580e-5 | 0.01150251 |
| 2 | 0.01075782 | 2.10114640e-5 | 0.01077883 | 0.01064190 | 2.07850668e-5 | 0.01066268 |
| 3 | 0.01003516 | 1.96000169e-5 | 0.01005476 | 0.01025601 | 2.00313679e-5 | 0.01027604 |
| 4 | 0.00963032 | 1.88093101e-5 | 0.00964913 | 0.00997461 | 1.94817665e-5 | 0.00999409 |
| 5 | 0.00931461 | 1.81926870e-5 | 0.00933280 | 0.00979390 | 1.91288123e-5 | 0.00981303 |
| 6 | 0.00909471 | 1.77632036e-5 | 0.00911247 | 0.00966946 | 1.88857772e-5 | 0.00968835 |
| 7 | 0.00887718 | 1.73383347e-5 | 0.00889452 | 0.00954937 | 1.86512153e-5 | 0.00956802 |
| 8 | 0.00871340 | 1.70184559e-5 | 0.00873042 | 0.00970141 | 1.89481818e-5 | 0.00972036 |
| 9 | 0.00857159 | 1.67414824e-5 | 0.00858833 | 0.00949846 | 1.85517917e-5 | 0.00951702 |
| 10 | 0.00846597 | 1.65351862e-5 | 0.00848250 | 0.00947146 | 1.84990604e-5 | 0.00948996 |
| 11 | 0.00833197 | 1.62734662e-5 | 0.00834824 | 0.00948125 | 1.85181746e-5 | 0.00949977 |
| 12 | 0.00820872 | 1.60327498e-5 | 0.00822475 | 0.00941581 | 1.83903681e-5 | 0.00943420 |
| **13** | **0.00813207** | **1.58830465e-5** | **0.00814795** | **0.00937206** | **1.83048976e-5** | **0.00939036** |
| 14 | 0.00802623 | 1.56763357e-5 | 0.00804191 | 0.00939033 | 1.83405990e-5 | 0.00940867 |
| 15 | 0.00792915 | 1.54867098e-5 | 0.00794463 | 0.00945773 | 1.84722407e-5 | 0.00947621 |

Historical-best protection retained epoch 13 when epochs 14--15 were worse.
Training and validation losses, gradients, parameters, predictions, and all
diagnostic arrays pass the NaN/Inf audit.

## Ordinary prediction quality

| Metric | Result | Reference / gate | Pass |
|---|---:|---:|---|
| WM-v2A correct-action cosine error | 0.00937206 | — | — |
| D14 same-seed error | 0.00937134 | relative degradation <=5% | yes: +0.0076% |
| D15 three-member ensemble error | 0.00891906 | report-only | +5.079% |
| Copy-current error | 0.01579531 | WM-v2A must be lower | yes |
| Latent-delta SmoothL1 | 1.83049e-5 | diagnostic | finite |

The D15 comparison is intentionally not a gate because it compares a single
WM-v2A member against a three-member ensemble.

| Horizon | WM-v2A error | D15 ensemble | Copy current | Delta SmoothL1 |
|---:|---:|---:|---:|---:|
| h1 | 0.00686935 | 0.00629586 | 0.00702430 | 1.34168e-5 |
| h2 | 0.00801587 | 0.00755894 | 0.01017508 | 1.56561e-5 |
| h3 | 0.00880892 | 0.00838467 | 0.01295711 | 1.72050e-5 |
| h4 | 0.00940663 | 0.00899007 | 0.01541429 | 1.83724e-5 |
| h5 | 0.00986658 | 0.00945218 | 0.01758325 | 1.92707e-5 |
| h6 | 0.01027020 | 0.00984455 | 0.01946334 | 2.00591e-5 |
| h7 | 0.01065441 | 0.01020906 | 0.02108558 | 2.08096e-5 |
| h8 | 0.01108450 | 0.01061715 | 0.02265953 | 2.16495e-5 |

## Frozen global action-conditioning diagnostics

All actions pass through the unchanged raw-physical-action to frozen WM
normalization interface. Whole-chunk shuffles, temporal conditions, and scale
conditions exactly reuse the C1-v2.2 construction and seeds.

| Wrong-action condition | Error | Relative increase | Correct wins | Delta 95% CI lower | Win 95% CI lower |
|---|---:|---:|---:|---:|---:|
| Physical zero | 0.01190049 | 26.98% | 76.84% | 0.002377 | 75.24% |
| Within-task shuffle | 0.01213448 | 29.48% | 78.75% | 0.002636 | 77.71% |
| Cross-task shuffle | 0.01214821 | 29.62% | 80.01% | 0.002645 | 79.06% |
| Reverse | 0.00984796 | 5.08% | 63.32% | 0.000420 | 61.88% |
| Fixed permutation | 0.00944668 | 0.80% | 54.57% | 0.0000555 | 53.15% |
| Repeat first | 0.00960034 | 2.44% | 50.42% | 0.000195 | 49.24% |

The preregistered global gate uses within-task shuffle. Its minimums are 0.5%
relative increase, positive delta interval, 52.5% correct-action ranking, and
ranking interval above 50%; all pass. Magnitude response also remains
structured: errors at scales 0/0.25/0.5/1/1.5 are
`0.0119005/0.0109924/0.0100185/0.0093721/0.0095213`, and separation from 1x
tracks `abs(scale-1)` with Spearman `0.9747` (p=`0.00482`).

## Production-candidate local sensitivity

The candidate generator and 1x production scale are frozen. The same A0/A1/A2
actions used by C1-v2.1 have mean pairwise raw chunk L2 `0.22893` and
normalized chunk L2 `1.75294`; action separation itself is not the failure.

| Horizon | Predicted cosine separation | Separation / WM error | Separation / real motion |
|---:|---:|---:|---:|
| h1 | 2.78206e-5 | 0.4050% | 0.3961% |
| h2 | 3.58700e-5 | 0.4475% | 0.3525% |
| h3 | 4.25083e-5 | 0.4826% | 0.3281% |
| h4 | 4.88315e-5 | 0.5191% | 0.3168% |
| h5 | 5.41664e-5 | 0.5490% | 0.3081% |
| h6 | 5.84969e-5 | 0.5696% | 0.3005% |
| h7 | 6.19357e-5 | 0.5813% | 0.2937% |
| h8 | 6.82651e-5 | 0.6159% | 0.3013% |
| **Mean** | **4.97368e-5** | **0.5307%** | **0.3149%** |

WM-v1's corresponding separation was `3.7775e-5` and its ratio was
`0.00423536`. WM-v2A increases absolute separation by about 31.7% and the
error-normalized ratio by 25.3%, but the new ratio `0.00530693` fails the
preregistered `>=0.01` threshold and is only 1.253x the reference, failing the
`>=2x` threshold. No horizon reaches 0.01.

## Local directional validity

For each pair among A0/A1/A2, lower expert-action MSE is expected to coincide
with lower predicted-future cosine error. This is an imitation-oriented sanity
check, not a claim of counterfactual or closed-loop optimality.

| Metric | Result | Gate | Pass |
|---|---:|---:|---|
| Pairwise ranking accuracy | 55.52% | >=53% | yes |
| Episode-block 95% CI | [54.56%, 56.47%] | lower >50% | yes |
| Mean within-sample Spearman | 0.1115 | >=0.05 | yes |
| Spearman 95% CI | [0.0918, 0.1313] | lower >0 | yes |
| Mean within-sample Kendall tau | 0.1104 | diagnostic | positive |

Pairwise ranking rises from 51.26% at h1 to a peak of 54.96% at h6 and is
54.17% at h8. The aggregate direction gate passes even though the absolute
prediction differences remain below the local-sensitivity floor. This rules
out `LOCAL_SENSITIVITY_ARTIFACT` as the primary verdict: the extra separation
is directionally useful, just too small.

## Per-task breakdown

| Task | n | Correct error | Delta error | Local separation | Sep./error | Pairwise rank | Spearman |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 415 | 0.009096 | 1.7767e-5 | 5.6492e-5 | 0.6210% | 52.77% | 0.0578 |
| 1 | 628 | 0.006765 | 1.3214e-5 | 4.5003e-5 | 0.6652% | 51.22% | 0.0255 |
| 2 | 594 | 0.009302 | 1.8169e-5 | 6.7135e-5 | 0.7217% | 56.85% | 0.1473 |
| 3 | 442 | 0.010002 | 1.9535e-5 | 5.4948e-5 | 0.5494% | 55.96% | 0.1324 |
| 4 | 698 | 0.011245 | 2.1962e-5 | 4.2263e-5 | 0.3758% | 53.77% | 0.0759 |
| 5 | 507 | 0.008102 | 1.5824e-5 | 4.9486e-5 | 0.6108% | 56.87% | 0.1321 |
| 6 | 666 | 0.011316 | 2.2101e-5 | 4.6559e-5 | 0.4115% | 57.71% | 0.1592 |
| 7 | 696 | 0.009071 | 1.7717e-5 | 5.6030e-5 | 0.6177% | 54.17% | 0.0776 |
| 8 | 603 | 0.009363 | 1.8288e-5 | 4.4432e-5 | 0.4745% | 57.66% | 0.1509 |
| 9 | 620 | 0.009003 | 1.7585e-5 | 3.9754e-5 | 0.4415% | 58.06% | 0.1573 |

Every task remains below the 1% local sensitivity threshold. Directional
quality is heterogeneous—task 1 remains near chance—but the aggregate
episode-block intervals meet the frozen gate.

## Gate decision

| Gate group | Result | Pass |
|---|---|---|
| Prediction quality | +0.0076% vs same-seed D14; beats copy-current | yes |
| Global action conditioning | within-shuffle +29.48%, 78.75% correct wins | yes |
| Local sensitivity | ratio 0.005307; 1.253x WM-v1 | **no** |
| Local directional validity | 55.52% ranking; Spearman 0.1115; positive intervals | yes |

The frozen decision order maps this pattern to:

`DELTA_OBJECTIVE_INSUFFICIENT`

This is not `PREDICTION_QUALITY_REGRESSION`: ordinary same-seed prediction is
essentially unchanged. It is not `LOCAL_SENSITIVITY_ARTIFACT`: directional
checks pass. It is not `WM_V2A_LOCAL_SENSITIVITY_PASS`: both local magnitude
checks fail.

Accordingly, seed23 and seed47 remain untrained and no WM-v2A ensemble is
authorized. A separately preregistered WM-v2B experiment may later test
future-proprio auxiliary supervision, but this report does not implement or
authorize it. Controller/reranker work remains blocked by the frozen upstream
verdicts.

## Artifacts, integrity, and forbidden-activity audit

| Artifact | SHA-256 |
|---|---|
| Best checkpoint, epoch 13 | `1e3741c1d5307014b070c6256af0753c8cb86f4c591b78c738324161223e1fe5` |
| Last checkpoint, epoch 15 | `a83eede02360294d7f6cc051ff7ddb8b6958565ee46640262c28db187ad9190a` |
| Training metrics JSONL | `300758e256263852558271a3b77cc6df2aecedbfed6ebed9fb208caf1c5b34d3` |
| Run metadata | `4488d8418475f9f2579e12136f3193ef55718f243d83976946ebe2e3c6f510bb` |
| Best-validation record | `5fc112f65cd8c291952769c6fd82046162033c4468251db371c13c26e09aafeb` |
| Aggregate diagnosis | `68f5d7b8fb01440918cc2dc7b8d4557eaf752362c8610085e5a6aede2373889c` |
| Ignored sample sidecar | `7f3e82eded3f31c3a8e5ed4ff6d14e74a7cae584afabedc2623aa554b79c6331` |

The compact diagnosis and metadata are tracked; checkpoints, JSONL, and the
3.5 MB sample sidecar remain ignored. The sidecar contains all 5,869 unique
validation identities, global-condition errors/separations, production-
candidate errors/separations, delta errors, and directional bootstrap inputs
for independent recalculation.

Forbidden-activity audit: C0-v2, candidate generation, production noise
scale, reranker score, CLIP, camera, architecture, D13 pairing, and D14/D15
checkpoints were not modified. Seed23/47 were not trained. No controller,
simulator, success-rate selection, init-state access, D20-v2, C2-v2, or
WM-v2B action occurred.
