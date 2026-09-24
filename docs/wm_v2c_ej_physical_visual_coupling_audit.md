# Explicit EE+joint coupling physical to visual transfer still insufficient

Date: 2026-09-25

Final verdict: `PHYSICAL_TO_VISUAL_TRANSFER_STILL_INSUFFICIENT`

Authorization: `BLOCK_RERANKER_REDIAGNOSIS`

Legacy 12-D mediator: `BLOCK_WM_V2C_PHYSICAL_MEDIATOR`

Legacy reranker: `RERANKER_SCORE_BLOCK`

## Executive summary

WM-v2C-EJ initializes every existing module from frozen WM-v2B and adds only
a 75,072-parameter predicted-EE/joint-to-visual residual adapter. Gripper is
excluded by fixed indices and future physical state remains target-only.

The selected seed-11 checkpoint has ordinary validation cosine error
`0.00968781` (+3.38%
versus D14) and matched counterfactual transfer ratio
`0.019030`, or
`1.32x` frozen WM-v2B. Mean matched candidate
error changes by `+1.43%`.

The mechanical verdict is `PHYSICAL_TO_VISUAL_TRANSFER_STILL_INSUFFICIENT`. Even a PASS authorizes only
offline reranker rediagnosis; no controller, D20-v2, C2-v2, seed23/47, or
init-state 40--49 work occurs here.

## Frozen provenance and experiment boundary

- Protocol commit: `1fbf8f2c8acd52afa0300b014ce9bac4aa9ed048`; protocol SHA-256 `cff07db51aa682aa6cd8d0823af9f898c23fc5e5ff1836aa64f6e1cc71933636`.
- WM-v2B initialization checkpoint SHA-256: `eb13775493a13d845e93db195c25d513b2fa8f679ac8578456fb8c64b2573848`.
- D13 store SHA-256: `e5599153cf44282242d955be62249bbafc752b782f5ae76a0ec0d58f9146f98e`.
- Matched branch sidecar SHA-256: `6c81a692bac418a9defe3a8abc0086408b909ce1c99b9241c49361a7c298e12d`.
- Train demos 0--44; validation demos 45--49; H=8; 52,381/5,869 samples.
- Matched init 30/35 is final mechanism audit only. Init 40--49 was not accessed.

The prior `PARTIAL_PHYSICAL_MEDIATOR_PASS` and complete 12-D
`BLOCK_WM_V2C_PHYSICAL_MEDIATOR` are unchanged.

The first train-only calibration invocation stopped before writing a result:
two gradient norms had been requested from one released autograd graph. The
implementation was changed to use two independent forward graphs, matching
the frozen WM-v2B calibration procedure, covered by a regression test, and
committed as `48d2fef9f79dc0055f86330bd1110f16a76c329a`. No validation or
matched-branch result had been read. The successful calibration and its fixed
sample identities were then frozen before training.

## Architecture and parameter difference

```text
shared GRU h_t
  +--> frozen-form base visual delta head
  +--> frozen-form 12-D physical head
          +--> select ee_pos[0:3] + joints[5:12]
                  +--> Linear(10,64)-GELU-Linear(64,128)-GELU-Linear(128,512)
```

`pred_z = normalize(z0 + base_visual_delta + physical_visual_residual)`.
WM-v2B has 826,444 parameters; the adapter adds 75,072; total is 901,516.
The final adapter layer starts at exact zero, so initialization preserves the
WM-v2B visual prediction. No warm-start stage, new GRU, Transformer, camera,
or encoder is used.

## EE/joint target and explicit gripper exclusion

The target is standardized future-minus-current proprio for dimensions
`[0,1,2,5,6,7,8,9,10,11]`: 3-D EE position plus seven joints. Gripper
dimensions `[3,4]` remain in the legacy physical head only and have no edge
into the adapter. Future proprio is training target only.

## Gradient calibration and fixed losses

Train-only shared-GRU gradient norms are `g_z=0.0060594918`
and `g_EJ=0.0040077765`. The frozen coefficient
is `lambda_EJ=1.5119335449`; `lambda_res=0.001`
was fixed before calibration. The objective is `L_z + lambda_EJ*L_EJ +
lambda_res*L_res` with no fake counterfactual target.

## Train and validation curves

| epoch | train z | train EJ | train res | val z | val EJ | val res | val total |
|---|---|---|---|---|---|---|---|
| 1 | 0.008434 | 0.000771 | 3.270e-07 | 0.009826 | 0.001258 | 3.123e-07 | 0.011728 |
| 2 | 0.008278 | 0.000655 | 3.344e-07 | 0.009740 | 0.001259 | 3.835e-07 | 0.011643 |
| 3 | 0.008228 | 0.000603 | 4.847e-07 | 0.009710 | 0.001231 | 6.102e-07 | 0.011571 |
| 4 | 0.008178 | 0.000553 | 7.785e-07 | 0.009717 | 0.001202 | 9.324e-07 | 0.011534 |
| 5 | 0.008136 | 0.000517 | 1.151e-06 | 0.009757 | 0.001190 | 1.345e-06 | 0.011557 |
| 6 | 0.008087 | 0.000490 | 1.591e-06 | 0.009736 | 0.001157 | 1.819e-06 | 0.011485 |
| 7 | 0.008036 | 0.000464 | 2.049e-06 | 0.009698 | 0.001141 | 2.314e-06 | 0.011423 |
| 8 | 0.007999 | 0.000438 | 2.572e-06 | 0.009693 | 0.001147 | 2.842e-06 | 0.011427 |
| 9 | 0.007945 | 0.000421 | 3.145e-06 | 0.009771 | 0.001109 | 3.414e-06 | 0.011447 |
| 10 | 0.007898 | 0.000398 | 3.696e-06 | 0.009735 | 0.001138 | 4.090e-06 | 0.011456 |
| 11 | 0.007872 | 0.000395 | 4.318e-06 | 0.009882 | 0.001118 | 4.661e-06 | 0.011572 |
| 12 | 0.007823 | 0.000376 | 4.958e-06 | 0.009784 | 0.001074 | 5.298e-06 | 0.011409 |
| 13 | 0.007782 | 0.000363 | 5.634e-06 | 0.009780 | 0.001071 | 6.032e-06 | 0.011399 |
| 14 | 0.007747 | 0.000356 | 6.319e-06 | 0.009773 | 0.001101 | 6.723e-06 | 0.011438 |
| 15 | 0.007705 | 0.000342 | 7.037e-06 | 0.009688 | 0.001124 | 7.323e-06 | 0.011388 |

Best checkpoint: epoch 15, step 3075,
SHA-256 `dae59cf73c099a39faf49c86c4e7bdcf94630193222c79378cbfd6dbb2491302`. GPU peak allocated memory:
98592256 bytes.

## Ordinary visual prediction

Overall correct-action cosine error is `0.00968781` versus D14
`0.00937134`, WM-v2B `0.00962736`,
and copy-current `0.01579531`.

## Global action conditioning

Within-task shuffled actions change error by
`34.29%` with paired
accuracy `80.11%`.
Fixed temporal permutation delta CI lower is
`0.00008713`;
magnitude-distance Spearman is `0.9747`.

| Condition | Visual cosine error |
|---|---:|
| correct | 0.00968781 |
| physical zero | 0.01291997 |
| within-task shuffle | 0.01300940 |
| cross-task shuffle | 0.01308584 |
| reverse | 0.01039599 |
| fixed temporal permutation | 0.00979565 |
| repeat first | 0.00991325 |
| scale 0.25x | 0.01181148 |
| scale 0.5x | 0.01051522 |
| scale 1.5x | 0.01003806 |

Correct actions beat physical-zero, within-task, and cross-task alternatives by
33.36%, 34.29%, and 35.08% relative error, respectively. The production-scale
model therefore retains the preregistered global action signal; the failed
gate below is specifically local matched-counterfactual transfer.

## Frozen production candidates on demo validation

Mean candidate separation/error ratio is
`0.007359`. References are
WM-v1 `0.004235`, WM-v2A `0.005307`, and WM-v2B `0.005157`.

## Matched real counterfactual visual fidelity

| mode | pred sep | transfer | vs v2B | candidate error | error delta | direction |
|---|---|---|---|---|---|---|
| full_ej | 7.923953e-05 | 0.0190 | 1.32x | 0.016080 | 1.43% | 0.0203 |
| ee_only | 8.077350e-05 | 0.0194 | 1.35x | 0.016103 | 1.58% | 0.0208 |
| joints_only | 8.004214e-05 | 0.0192 | 1.34x | 0.016114 | 1.65% | 0.0208 |

The final gate uses only the preregistered `full_ej` row. EE-only and
joints-only are inference ablations and do not redefine the architecture.

## Candidate-specific matched future errors

Full-EJ A0/A1/A2 errors are
`0.016012` /
`0.015858` /
`0.016369`.
Frozen WM-v2B aggregate reference is
`0.015853`.

Candidate-specific errors by horizon:

| h | A0 | A1 | A2 |
|---:|---:|---:|---:|
| 1 | 0.010628 | 0.010905 | 0.010848 |
| 2 | 0.014032 | 0.012830 | 0.014844 |
| 3 | 0.016368 | 0.016662 | 0.016246 |
| 4 | 0.017092 | 0.017369 | 0.017696 |
| 5 | 0.018162 | 0.017805 | 0.018379 |
| 6 | 0.018218 | 0.017743 | 0.018224 |
| 7 | 0.017200 | 0.016722 | 0.018263 |
| 8 | 0.016398 | 0.016832 | 0.016455 |

Candidate-specific errors by task:

| Task | A0 | A1 | A2 |
|---:|---:|---:|---:|
| 0 | 0.017044 | 0.017225 | 0.016195 |
| 1 | 0.010982 | 0.010223 | 0.010777 |
| 2 | 0.015242 | 0.017208 | 0.016237 |
| 3 | 0.019197 | 0.018137 | 0.019783 |
| 4 | 0.020141 | 0.020499 | 0.020621 |
| 5 | 0.014552 | 0.014648 | 0.013145 |
| 6 | 0.015971 | 0.015380 | 0.016830 |
| 7 | 0.016094 | 0.013559 | 0.017804 |
| 8 | 0.015534 | 0.016729 | 0.015824 |
| 9 | 0.015366 | 0.014975 | 0.016478 |

## Visual-delta direction fidelity

Aggregate A0-relative direction is
`0.0203` with branch-bootstrap
95% CI `[0.0025,
0.0377]`.

A0-A1 direction is `0.0093`, with 95% CI `[-0.0233, 0.0404]`; A0-A2 is
`0.0313`, with CI `[0.0062, 0.0559]`. The aggregate direction is positive, but
the preregistered requirement that both A0-relative pairs have positive lower
bounds fails for A0-A1. This is the second failed major gate, independent of
the separation-ratio failure.

## Base/residual contribution and adapter action sensitivity

Validation residual/base L2 mean is `0.0377`
and p95 is `0.0605`. Mean adapter candidate
pair L2 is `0.003664` versus base
`0.034862`, ratio
`0.1051`. Base/residual direction
cosine is `0.7372`.

## Per-horizon matched analysis

| h | val error | pred sep | real sep | transfer | matched error | direction |
|---|---|---|---|---|---|---|
| 1 | 0.007327 | 6.816797e-05 | 2.417768e-03 | 0.0282 | 0.010793 | 0.0238 |
| 2 | 0.008377 | 7.208337e-05 | 3.545901e-03 | 0.0203 | 0.013902 | 0.0037 |
| 3 | 0.009134 | 7.259430e-05 | 4.276665e-03 | 0.0170 | 0.016425 | -0.0177 |
| 4 | 0.009696 | 7.562943e-05 | 4.593353e-03 | 0.0165 | 0.017385 | 0.0673 |
| 5 | 0.010140 | 7.786602e-05 | 4.930369e-03 | 0.0158 | 0.018115 | 0.0509 |
| 6 | 0.010537 | 8.184185e-05 | 4.261248e-03 | 0.0192 | 0.018062 | -0.0052 |
| 7 | 0.010928 | 8.816172e-05 | 4.502458e-03 | 0.0196 | 0.017395 | 0.0186 |
| 8 | 0.011362 | 9.757157e-05 | 4.784324e-03 | 0.0204 | 0.016562 | 0.0213 |

## Per-task matched analysis

| task | val error | pred sep | real sep | transfer | matched error | direction |
|---|---|---|---|---|---|---|
| 0 | 0.009111 | 6.321555e-05 | 4.128163e-03 | 0.0153 | 0.016821 | 0.0420 |
| 1 | 0.007257 | 6.404711e-05 | 3.452377e-03 | 0.0186 | 0.010660 | 0.0421 |
| 2 | 0.009420 | 1.056642e-04 | 4.569171e-03 | 0.0231 | 0.016229 | 0.0265 |
| 3 | 0.009677 | 7.254087e-05 | 4.251629e-03 | 0.0171 | 0.019039 | -0.0356 |
| 4 | 0.011782 | 6.583711e-05 | 4.394850e-03 | 0.0150 | 0.020420 | 0.0234 |
| 5 | 0.008569 | 7.295726e-05 | 3.824139e-03 | 0.0191 | 0.014115 | 0.0326 |
| 6 | 0.011455 | 8.383065e-05 | 4.367510e-03 | 0.0192 | 0.016061 | 0.0125 |
| 7 | 0.009391 | 9.337474e-05 | 5.946933e-03 | 0.0157 | 0.015819 | 0.0127 |
| 8 | 0.010010 | 6.960477e-05 | 3.019443e-03 | 0.0231 | 0.016029 | -0.0038 |
| 9 | 0.009479 | 1.013231e-04 | 3.685891e-03 | 0.0275 | 0.015606 | 0.0510 |

## Mechanical gate decision

| gate | result |
|---|---|
| A_ordinary_visual | pass |
| B_global_action | pass |
| C_local_predicted_real_separation | FAIL |
| D_candidate_specific_prediction | pass |
| E_visual_delta_direction | FAIL |
| F_residual_non_degeneracy | pass |

Gate C is the primary failure: `0.019030` is only `1.323x` WM-v2B, below both
the absolute `0.043143` floor and the `3x` improvement rule. Gate E also fails
because the A0-A1 direction interval crosses zero. Gates A/B/D/F pass: ordinary
prediction and global action conditioning remain valid, matched candidate
error changes only +1.43%, and the adapter is finite, candidate-sensitive,
and non-dominant. Thus this is neither arbitrary latent inflation nor a total
absence of local signal; the explicit path simply transfers too little of the
real candidate-induced visual difference.

The EE-only and joints-only transfer ratios (`0.019398` and `0.019222`) are
nearly the same as full EJ (`0.019030`). These inference ablations provide no
evidence that the joint full-EJ path creates the required additional coupling;
they do not authorize an architecture change or post-hoc subspace selection.

Final verdict: `PHYSICAL_TO_VISUAL_TRANSFER_STILL_INSUFFICIENT`. Authorization:
`BLOCK_RERANKER_REDIAGNOSIS`. Controller authorization is always false.

## Integrity and forbidden-activity audit

- Config SHA-256: `aae1cb7f822470be16ead5c82f41f24c211b2fe89f99c2d37b5f7599e72cb728`.
- Result: `results/world_model_v2c_ej/seed11/diagnosis.json`.
- Result SHA-256: `39bd7716b8e939da10b4ae953c9496833c9c216d5bae764ee80062e4d2f454dc`.
- Diagnostic sidecar SHA-256: `133557d9cf9559115bb11b0eb1786e9eaf439d8e769b28b796675a4a2d051865`.
- NaN/Inf audit: `pass`.
- Gripper adapter input: `false`.
- No simulator rerun, training on matched branches, controller success, C1/C2,
  D20-v2, reward/done/success, init 40--49, seed23/47, ensemble, or upstream
  overwrite occurred.

After the immutable diagnosis was written, the report presentation was
completed with A0/A1/A2-by-horizon and by-task tables derived directly from
the already hashed sidecar. The diagnosis JSON, sidecar, metrics, gates, and
verdict were not changed; automated tests independently recompute these added
table entries from the sidecar.
