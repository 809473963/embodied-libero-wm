# Matched simulator branches reveal the local candidate dynamics mechanism

Date: 2026-09-24

Final verdict: `VISUAL_TRANSFER_FAILURE_CONFIRMED`

## Executive summary

This audit uses 40 preregistered branch points across all
ten LIBERO Spatial tasks. Each A0/A1/A2 future comes from a fresh environment
that starts at the same official init state and replays the same stored raw
prefix actions. No mid-episode simulator-state restore, reward, success, or
task-completion signal is used.

Real candidate futures have mean visual cosine separation
`0.0041640106` and standardized-proprio RMSE separation
`0.015533`. Relative to the frozen
D15 ordinary validation error, real visual separation is
`46.687%`; relative to the
frozen WM-v2B proprio prediction RMSE, real physical separation is
`33.25%`.

The preregistered rule assigns `VISUAL_TRANSFER_FAILURE_CONFIRMED`. This is a
mechanism conclusion only. It does not rank candidates by task success. It
clears the mechanism-evidence prerequisite for a separately preregistered
WM-v2C physical-to-visual coupling design, but this round does not implement
or train WM-v2C and does not authorize controller, threshold, candidate-scale,
or reranker changes.

## Frozen provenance and data boundary

- WM-v2B freeze commit: `5393c5c95dd7c65347bc11f572f4168681121ed6`.
- Annotated tag: `wm-v2b-proprio-transfer-insufficient`.
- Counterfactual protocol commit: `8fe800d7343f4969756dba60be4182ad98c60b5e`.
- Protocol SHA-256: `b1c8c622e23c8c6879f6209ba93bf1be76b370909c1d1285c5b2d4406760159c`.
- Frozen C0-v2, C1-v2 generator, D15 ensemble, WM-v2A, and WM-v2B hashes all passed before execution.
- Only controller-development init states 30 and 35 were used; 0--29 and 40--49 were not run.
- No reward, done, success, or final task completion value is present in the result or sidecar.

## Preflight correction history

Two replay-identity implementation checks stopped before any mechanism metric
was admitted. The original v1 preflight encoded identical RGB in different
CLIP batch shapes, exposing a numerical batch-shape artifact rather than a
simulator mismatch. V1.1 then incorrectly included a separate prefix-source
renderer in the three-branch identity gate; the three actual branches were
already RGB-, proprio-, observable-, and simulator-state-identical. Both
blocked results and reports remain frozen at commits `2702525` and `9804a2e`.

The final v1.2 amendment, committed as
`8fe800d7343f4969756dba60be4182ad98c60b5e` before any real-future metric
was viewed, restricts the gate to the three actual fresh branches and encodes
their identical RGB in one shared CLIP batch. It changes no task, init state,
seed, planning index, candidate, model, metric, denominator rule, threshold,
or verdict mapping.

## Preregistered branch identities and candidate hashes

| Task | Init | Seed | Plan | Prefix steps | Candidate SHA-256 |
|---|---|---|---|---|---|
| 0 | 30 | 30000 | 0 | 0 | 6fbad2bfbc0e1c55… |
| 0 | 30 | 30000 | 8 | 64 | edd9b95752d88ef2… |
| 0 | 35 | 30005 | 0 | 0 | 138ef38b54eeb10f… |
| 0 | 35 | 30005 | 8 | 64 | d39ebe045f3d35a4… |
| 1 | 30 | 30100 | 0 | 0 | 154a5b52f83fe08f… |
| 1 | 30 | 30100 | 8 | 64 | 0032639ff4fa1417… |
| 1 | 35 | 30105 | 0 | 0 | cb0506eff4f9d5c9… |
| 1 | 35 | 30105 | 8 | 64 | c04827bf5bb1b95a… |
| 2 | 30 | 30200 | 0 | 0 | 1e9a227fd20e9818… |
| 2 | 30 | 30200 | 8 | 64 | 65028f9f962a8df3… |
| 2 | 35 | 30205 | 0 | 0 | 8a6c8e536d47222a… |
| 2 | 35 | 30205 | 8 | 64 | 0ad130ec669c1aca… |
| 3 | 30 | 30300 | 0 | 0 | 2abc3d6ef21c5613… |
| 3 | 30 | 30300 | 8 | 64 | e0f4599f124a6527… |
| 3 | 35 | 30305 | 0 | 0 | a198d0713d1430d1… |
| 3 | 35 | 30305 | 8 | 64 | c43c97ee3d0f6319… |
| 4 | 30 | 30400 | 0 | 0 | 7bda192d9e0455a7… |
| 4 | 30 | 30400 | 8 | 64 | 11cc436076ed59a2… |
| 4 | 35 | 30405 | 0 | 0 | 0a9f0e128c5e2688… |
| 4 | 35 | 30405 | 8 | 64 | 02a98a9333acd1c5… |
| 5 | 30 | 30500 | 0 | 0 | ff8326ccea06df89… |
| 5 | 30 | 30500 | 8 | 64 | e981f9b2dad735dc… |
| 5 | 35 | 30505 | 0 | 0 | 3dff30ada26bbf07… |
| 5 | 35 | 30505 | 8 | 64 | 27670f61fe43b57a… |
| 6 | 30 | 30600 | 0 | 0 | 2bbe0d08ef393491… |
| 6 | 30 | 30600 | 8 | 64 | d96247a75a41d92a… |
| 6 | 35 | 30605 | 0 | 0 | 81fd84a084c6eefb… |
| 6 | 35 | 30605 | 8 | 64 | 6a1ff7f4b100baf2… |
| 7 | 30 | 30700 | 0 | 0 | 84f7799654ec81e0… |
| 7 | 30 | 30700 | 8 | 64 | 389cc0f6a5688755… |
| 7 | 35 | 30705 | 0 | 0 | cfd25c87b9a6c3f5… |
| 7 | 35 | 30705 | 8 | 64 | fee6a21811b79c9d… |
| 8 | 30 | 30800 | 0 | 0 | 7a86b7f5c80ba9d3… |
| 8 | 30 | 30800 | 8 | 64 | 3be9af2d08653205… |
| 8 | 35 | 30805 | 0 | 0 | 6d491e64846db302… |
| 8 | 35 | 30805 | 8 | 64 | d0b253c1c3abf641… |
| 9 | 30 | 30900 | 0 | 0 | 2f818633d7ecca21… |
| 9 | 30 | 30900 | 8 | 64 | 8c4e04f865de3bfc… |
| 9 | 35 | 30905 | 0 | 0 | 304913e4cef31976… |
| 9 | 35 | 30905 | 8 | 64 | 09ef5da19f53b377… |

The ordered 40-candidate hash manifest is
`601c02fe0e3cc63a095549bfa4973d2d28036e1ad65a791d741474d64e3b67da`. A candidate tensor
was generated once from the three branches' verified common planning
observation and copied unchanged to all three independent branches.

## Prefix determinism passed before every branch

All 40 branch points pass. RGB and every planning
observation array are bitwise identical. Maximum proprio difference is
`0`, maximum flattened simulator-state
difference is `0`, and minimum
same-batch CLIP cosine to branch 0 is
`0.999999821`. Flattened state is used only for this
identity audit and never for restore.

## Real and predicted local visual separation

| Model | Predicted cosine | Real cosine | Pred/real | A0 error | A1 error | A2 error |
|---|---|---|---|---|---|---|
| wm_v1_d15_ensemble | 4.1923034e-05 | 0.0041640106 | 0.0101 | 0.015290 | 0.015135 | 0.015612 |
| wm_v2a_seed11 | 6.2641879e-05 | 0.0041640106 | 0.0150 | 0.015789 | 0.015729 | 0.016060 |
| wm_v2b_seed11 | 5.9883112e-05 | 0.0041640106 | 0.0144 | 0.015794 | 0.015688 | 0.016077 |

Overall real-future distribution summaries:

| Metric | n | Mean | Median | p75 | p90 | p95 |
|---|---|---|---|---|---|---|
| visual cosine | 960 | 0.0041640106 | 0.0030939214 | 0.005114207 | 0.0082215151 | 0.010668313 |
| visual L2 | 960 | 0.085400963 | 0.07866284 | 0.10113562 | 0.12823031 | 0.14607052 |
| standardized proprio RMSE | 960 | 0.015532955 | 0.01249912 | 0.020039899 | 0.029132601 | 0.038593024 |
| raw proprio L2 | 960 | 0.012640877 | 0.010507112 | 0.016335148 | 0.024693675 | 0.030827475 |
| pixel MAE | 960 | 0.73315455 | 0.57696533 | 0.95384216 | 1.3597555 | 1.7087809 |
| pixel RMSE | 960 | 4.5050348 | 4.140415 | 5.5416467 | 7.0610836 | 8.2220065 |
| unequal pixel fraction | 960 | 0.073082203 | 0.051289876 | 0.11090597 | 0.1335022 | 0.14594014 |

Ratios use ratio-of-means and are withheld when the matching real group mean
is below the preregistered `1e-5` denominator floor. Absolute predicted, real,
and signed differences remain in the JSON for every horizon, task, and pair.

## Per-horizon dynamics

| h | Visual mean | Median | p75 | p90 | p95 | Proprio mean | Pixel MAE | WM-v1 ratio | WM-v2A ratio | WM-v2B ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0024177676 | 0.0018747807 | 0.0030798726 | 0.0045307468 | 0.0055564533 | 0.003885 | 0.3001 | 0.009 | 0.018 | 0.020 |
| 2 | 0.0035459011 | 0.0025482399 | 0.0049112573 | 0.0080033358 | 0.0095560854 | 0.008638 | 0.5152 | 0.008 | 0.014 | 0.015 |
| 3 | 0.0042766652 | 0.0030970155 | 0.0055804288 | 0.0079198211 | 0.0089794254 | 0.011773 | 0.6532 | 0.008 | 0.013 | 0.013 |
| 4 | 0.0045933529 | 0.0035639874 | 0.0057077883 | 0.0087565355 | 0.011956427 | 0.014137 | 0.7421 | 0.009 | 0.013 | 0.012 |
| 5 | 0.0049303686 | 0.0035338541 | 0.005870705 | 0.0099506929 | 0.012075322 | 0.016903 | 0.8138 | 0.009 | 0.013 | 0.012 |
| 6 | 0.0042612476 | 0.0034041674 | 0.0050186793 | 0.0074732995 | 0.0099029601 | 0.020051 | 0.8542 | 0.012 | 0.017 | 0.015 |
| 7 | 0.0045024577 | 0.0032861357 | 0.0051016246 | 0.0078345446 | 0.010966595 | 0.023009 | 0.9347 | 0.012 | 0.017 | 0.015 |
| 8 | 0.0047843238 | 0.0034812856 | 0.0059643273 | 0.0093166784 | 0.010499591 | 0.025868 | 1.0521 | 0.012 | 0.017 | 0.016 |

## Per-task dynamics

| Task | Visual mean | Median | p75 | p90 | p95 | Proprio mean | WM-v1 ratio | WM-v2A ratio | WM-v2B ratio |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0041281632 | 0.0033790361 | 0.0052199695 | 0.0072006065 | 0.0085699895 | 0.013808 | 0.008 | 0.010 | 0.012 |
| 1 | 0.0034523772 | 0.0021954353 | 0.0032579761 | 0.0059282551 | 0.012325243 | 0.019518 | 0.009 | 0.015 | 0.014 |
| 2 | 0.0045691711 | 0.0032942357 | 0.0055888634 | 0.0080425161 | 0.011592865 | 0.015997 | 0.011 | 0.015 | 0.014 |
| 3 | 0.0042516287 | 0.0035023129 | 0.0052585307 | 0.0079662052 | 0.010627631 | 0.012736 | 0.009 | 0.013 | 0.013 |
| 4 | 0.0043948503 | 0.0035946075 | 0.0056131515 | 0.008784583 | 0.010282801 | 0.014250 | 0.008 | 0.013 | 0.011 |
| 5 | 0.0038241393 | 0.0030101709 | 0.004520512 | 0.0071596138 | 0.0089183542 | 0.013188 | 0.009 | 0.013 | 0.016 |
| 6 | 0.0043675096 | 0.0034078927 | 0.0053995128 | 0.0079830777 | 0.01089663 | 0.018871 | 0.011 | 0.015 | 0.016 |
| 7 | 0.0059469327 | 0.0042495605 | 0.0071798882 | 0.011987218 | 0.017156964 | 0.017953 | 0.015 | 0.020 | 0.013 |
| 8 | 0.0030194428 | 0.0026859138 | 0.0035564773 | 0.0048390661 | 0.0067037054 | 0.014241 | 0.011 | 0.018 | 0.015 |
| 9 | 0.0036858907 | 0.002545851 | 0.0053921383 | 0.0081770718 | 0.0093719584 | 0.014768 | 0.009 | 0.019 | 0.022 |

Each task contributes four branch points: init states 30/35 crossed with
planning indices 0/8. Per-pair mean/median/p75/p90/p95 and model ratios are in
the table below.

## Candidate-pair dynamics and transfer

| Pair | Visual mean | Median | p75 | p90 | p95 | Proprio mean | WM-v1 ratio | WM-v2A ratio | WM-v2B ratio |
|---|---|---|---|---|---|---|---|---|---|
| A0-A1 | 0.0036064795 | 0.0027219578 | 0.0045316341 | 0.006218143 | 0.008659744 | 0.011272 | 0.0059 | 0.0088 | 0.0085 |
| A0-A2 | 0.0038552304 | 0.002903665 | 0.0045027529 | 0.0073517083 | 0.0088502041 | 0.012284 | 0.0062 | 0.0091 | 0.0083 |
| A1-A2 | 0.0050303218 | 0.0037675127 | 0.0065656438 | 0.010251211 | 0.012355791 | 0.023043 | 0.0160 | 0.0241 | 0.0232 |

## Candidate-specific future prediction fidelity

| Model | A1/A2 vs validation | A1/A2 vs matched A0 | Alt worse fraction | Fail |
|---|---|---|---|---|
| wm_v1_d15_ensemble | 1.724x | 1.005x | 51.88% | no |
| wm_v2a_seed11 | 1.696x | 1.007x | 52.19% | no |
| wm_v2b_seed11 | 1.650x | 1.006x | 51.09% | no |

The prediction-failure gate requires all three preregistered conditions. It
therefore distinguishes general policy-rollout error from an additional error
specific to nearby A1/A2 counterfactual actions.

WM-v2B standardized-proprio RMSE is
`0.166381` for A0,
`0.166532` for A1,
and `0.166234` for A2.

Candidate-specific mean visual cosine errors and WM-v2B standardized-proprio
RMSE by horizon:

| h | v1 A0 | v1 A1 | v1 A2 | v2A A0 | v2A A1 | v2A A2 | v2B A0 | v2B A1 | v2B A2 | p A0 | p A1 | p A2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.009470 | 0.009812 | 0.009474 | 0.010110 | 0.010483 | 0.010125 | 0.010160 | 0.010526 | 0.010397 | 0.039923 | 0.040269 | 0.039973 |
| 2 | 0.012731 | 0.011393 | 0.013226 | 0.013228 | 0.012039 | 0.013913 | 0.013215 | 0.012172 | 0.013741 | 0.066940 | 0.067439 | 0.067018 |
| 3 | 0.015253 | 0.015791 | 0.015399 | 0.015754 | 0.016435 | 0.015983 | 0.016058 | 0.016423 | 0.015994 | 0.108802 | 0.109034 | 0.109331 |
| 4 | 0.016544 | 0.016682 | 0.016955 | 0.016936 | 0.017210 | 0.017310 | 0.017019 | 0.017191 | 0.017305 | 0.161476 | 0.161553 | 0.161845 |
| 5 | 0.017519 | 0.017242 | 0.018136 | 0.018138 | 0.017626 | 0.018360 | 0.018136 | 0.017661 | 0.018395 | 0.208359 | 0.208412 | 0.208627 |
| 6 | 0.017774 | 0.017369 | 0.017990 | 0.018135 | 0.017722 | 0.018202 | 0.018047 | 0.017699 | 0.018296 | 0.236451 | 0.236582 | 0.235962 |
| 7 | 0.017007 | 0.016633 | 0.017697 | 0.017468 | 0.017334 | 0.018098 | 0.017342 | 0.017111 | 0.017958 | 0.251099 | 0.251105 | 0.250027 |
| 8 | 0.016025 | 0.016157 | 0.016016 | 0.016539 | 0.016985 | 0.016489 | 0.016372 | 0.016720 | 0.016531 | 0.257999 | 0.257859 | 0.257090 |

The same candidate-specific errors by task:

| Task | v1 A0 | v1 A1 | v1 A2 | v2A A0 | v2A A1 | v2A A2 | v2B A0 | v2B A1 | v2B A2 | p A0 | p A1 | p A2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.016694 | 0.016734 | 0.015595 | 0.018345 | 0.018385 | 0.017244 | 0.017021 | 0.016982 | 0.016022 | 0.242233 | 0.243720 | 0.240215 |
| 1 | 0.009816 | 0.009076 | 0.009722 | 0.009841 | 0.009240 | 0.009696 | 0.010583 | 0.009947 | 0.010403 | 0.134492 | 0.137373 | 0.129401 |
| 2 | 0.013873 | 0.015720 | 0.015379 | 0.015302 | 0.017056 | 0.016340 | 0.014928 | 0.016782 | 0.015962 | 0.164238 | 0.162190 | 0.166992 |
| 3 | 0.019214 | 0.017829 | 0.019392 | 0.019507 | 0.018179 | 0.019749 | 0.019028 | 0.017923 | 0.019314 | 0.211350 | 0.211444 | 0.210343 |
| 4 | 0.019003 | 0.019413 | 0.019255 | 0.018929 | 0.019666 | 0.019407 | 0.018843 | 0.019275 | 0.019403 | 0.143531 | 0.139365 | 0.146038 |
| 5 | 0.013120 | 0.013369 | 0.011576 | 0.013666 | 0.013928 | 0.012076 | 0.013913 | 0.014099 | 0.012370 | 0.121331 | 0.117656 | 0.127540 |
| 6 | 0.014929 | 0.014280 | 0.015947 | 0.015126 | 0.014703 | 0.016173 | 0.015640 | 0.015364 | 0.016504 | 0.132820 | 0.138188 | 0.131014 |
| 7 | 0.016068 | 0.014159 | 0.017322 | 0.015860 | 0.014134 | 0.017085 | 0.015250 | 0.013058 | 0.016646 | 0.178136 | 0.175105 | 0.180168 |
| 8 | 0.014456 | 0.015672 | 0.014688 | 0.014531 | 0.015785 | 0.014700 | 0.015961 | 0.017201 | 0.016253 | 0.153088 | 0.153852 | 0.152411 |
| 9 | 0.015733 | 0.015098 | 0.017240 | 0.016780 | 0.016217 | 0.018132 | 0.016769 | 0.016249 | 0.017892 | 0.182589 | 0.186424 | 0.178220 |

## Does local physical motion map into CLIP motion?

Overall Pearson correlation is
`0.1900135151673456` and Spearman
correlation is `0.3108560773177922`.

| h | Pearson | Spearman | n |
|---|---|---|---|
| 1 | 0.3474 | 0.4038 | 120 |
| 2 | 0.3066 | 0.3604 | 120 |
| 3 | 0.0070 | 0.0765 | 120 |
| 4 | 0.2359 | 0.2117 | 120 |
| 5 | 0.1414 | 0.1652 | 120 |
| 6 | 0.0758 | 0.0664 | 120 |
| 7 | 0.0795 | 0.1145 | 120 |
| 8 | 0.1372 | 0.1770 | 120 |

This correlation is descriptive across candidate pairs and branch points; it
does not make future proprio a task-success objective.

## Mechanical decision

- Real visual meaningful: `True`.
- Real physical meaningful: `True`.
- WM-v2B predicted separation is below real separation on `8/8` horizons.
- `VISUAL_TRANSFER_FAILURE_CONFIRMED`: `True`.
- `LOCAL_CLIP_COUNTERFACTUAL_LIMIT`: `False`.
- Any model meets `WM_LOCAL_COUNTERFACTUAL_PREDICTION_FAIL`: `False`.

Final verdict: `VISUAL_TRANSFER_FAILURE_CONFIRMED`.

This verdict permits the next experiment to *design* a separately
preregistered WM-v2C physical-to-visual coupling. It does not itself authorize
training that model, changing the controller/reranker, D20-v2, C2-v2, or any
40--49 rollout.

## Integrity and forbidden-activity audit

- Compact result: `results/local_candidate_counterfactual/diagnosis.json`, SHA-256 `e3fede26f63f480df815c9a6bc78a658cfd15f7439f135194318e1002ef5f517`.
- Ignored sidecar: `artifacts/stage3/local_candidate_counterfactual/branches.npz`, SHA-256 `6c81a692bac418a9defe3a8abc0086408b909ce1c99b9241c49361a7c298e12d`.
- Sidecar arrays: all 35 named arrays are finite where numeric; RGB is uint8.
- NaN/Inf audit: `pass`.
- Simulator/controller success comparison: not performed.
- Model training, candidate modification, reranker modification, D20-v2, C2-v2, and init-state 40--49 access: not performed.
