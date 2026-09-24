# Experiment log

## 2026-09-20 — Stage 0 environment preparation

- Confirmed WSL2 GPU access: RTX 4070 Laptop GPU (8,188 MiB), driver 595.79; PyTorch CUDA probe passed.
- Created Python 3.10.12 environment at `.venv` and installed LIBERO from upstream commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Used official PyTorch CUDA 12.4 wheels for the pinned `torch` / `torchvision` / `torchaudio` trio; used official PyPI for LIBERO dependencies and official Hugging Face dataset hosting.
- Pinned MuJoCo to 2.3.7 because robosuite 1.4 fails with the newer MuJoCo 3.x joint API. Restored LIBERO's required NumPy 1.22.4 after that install.
- Downloaded only the 10-task official `libero_spatial` demonstrations. The source revision, file sizes, and SHA-256 values are in `artifacts/stage0/data_manifest.json`.
- Ran `scripts/run_smoke.sh`: a Spatial task reset, returned RGB, stepped with a 7-D zero action, queried task success, and saved a screenshot plus JSON evidence.
- Added `scripts/run_teleop.sh`, which uses WSLg + MuJoCo GLFW rendering and robosuite's official keyboard input mapping for manual, non-benchmark interaction.
- Verified the teleoperation launcher can construct and render the task window. The test was stopped by a timed `SIGINT`; its log is in `artifacts/stage0/teleop_gui_probe.log`.
- Added `scripts/run_policy_viewer.sh`: a live policy adapter executes RGB + language + robot-proprioception to 7-D action in the GUI loop. A zero-action fixture validates wiring only; no learned checkpoint exists yet.

## Compatibility notes

- LIBERO 0.1.0's editable package discovery is incomplete with current setuptools. The environment adds the fixed upstream repository root through a project-local `.pth` file; no upstream source was modified.
- PyTorch 2.6 defaults `torch.load(..., weights_only=True)`. The smoke script scopes `weights_only=False` only to LIBERO's Git-pinned task-init files, which are trusted local benchmark assets.
- robosuite warns that an optional private macro file is absent; this does not affect headless benchmark rendering.

## 2026-09-20 — Stage 1, D7 data contract

- Froze the C0 data boundary in `docs/data_contract.md`: external `agentview` RGB, task language, and declared robot proprioception are the only policy inputs. Full MuJoCo state, object poses, reward/success data, future frames/actions, and the wrist camera are explicitly prohibited.
- Froze a whole-demonstration split per task: demonstrations 0–44 train and 45–49 validation. This prevents transition-level leakage into model selection.
- Added the immutable 20-episode-per-task evaluation seed record at `configs/eval_seeds_spatial.json`; all C0/C1/C2 evaluations must share it.
- Added `scripts/audit_spatial_dataset.py` to validate the 10 official HDF5 files, their declared observation/action schema, finite actions, and C0 action-chunk sample counts without materializing RGB into a duplicate dataset.

## 2026-09-20 — Stage 1, D8 frozen feature extraction (implementation)

- Added `scripts/extract_features.py`. It pins official OpenAI CLIP ViT-B/32 revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, freezes both encoders, vertically aligns offline RGB with the online MuJoCo convention, and writes only policy-visible features/action labels plus provenance metadata.
- The script creates separate train/validation feature files atomically, derives action normalization statistics from training demonstrations only, and writes SHA-256 values into a feature manifest. It preserves a `.part` file on an interrupted run instead of silently treating it as a usable feature set.
- Added `scripts/verify_feature_store.py` for an independent pre-training check of feature SHA-256 values, tensor shapes, finite values, CLIP feature norms, policy-field allowlist, and whole-demonstration split disjointness.
- Generated and verified the contract-v1 12-D feature store: 52,831 training and 5,919 validation action-chunk samples. Its feature manifest records the train SHA-256 `7cbd8c5c10a04d64b07a3f39b2eff5384a4097198f80f9af93fe6fcad7203487` and validation SHA-256 `cfe1eb6b78508d4299d01f64a65bf761286a4ca667993694ba80f9cc4a4ec38d`.

## 2026-09-20 — Stage 1, D9 Actor implementation

- Added the compact `src/c0/C0Actor`, which consumes only frozen 512-D image/text features and normalized declared proprioception, then predicts the common 8×7 action block. No world model is present in C0.
- Added an HDF5-safe lazy dataset, train-only proprio/action normalization, an AMP-capable training launcher with config/feature-manifest hashes, RNG state, `last.pt`/`best.pt`, metrics JSONL, and hash-checked checkpoint resume.
- Added `tests/test_c0_actor.py` for one actual feature-store batch through forward, loss, backward, and optimizer step.
- Offline `ee_ori` and online `robot0_eef_quat` were explicitly compared on reconstructed simulator states. Their 3-D/4-D conventions were not directly interchangeable, so C0 was tightened to the reproducible 12-D `ee_pos + gripper_states + joint_states` interface and its feature store was regenerated. This avoids silently deploying a representation that training did not use.
- Ran the 12-D GPU AMP/backward/checkpoint probe and verified a checkpoint resume. The probe is strictly wiring evidence, not a trained policy result. Added `src/policies/c0_actor_policy.py` and `tests/test_c0_policy_adapter.py` so a genuinely trained C0 checkpoint can execute its buffered 8-step action blocks in the existing GUI.
- Added task-filtered in-memory C0 training and `scripts/evaluate_c0.py`. The evaluator applies the immutable 20 seeds, pairs them with the first 20 fixed official task initial states, exposes only the C0 observation contract to the policy, and records every episode's official `check_success()` result.

## 2026-09-20 — Stage 1, D10 task-0 C0 result

- Trained C0 on task 0 only, training seed 11, for 50 epochs / 3,400 optimizer steps using 4,298 training action chunks. The held-out task-0 validation set has 420 chunks; the best raw action MSE was `0.04644001616912634`.
- Evaluated `results/c0/task0_seed11/best.pt` on the immutable 20 task-0 episodes (seeds 10000–10019; fixed initial states 0–19). Official `check_success()` produced 12 successes / 20 = `0.60`; every episode, including all eight failures at the 300-step limit, is in `results/c0/task0_seed11/eval_task0_seed11.json`.
- Checkpoint SHA-256: `4f2c78847c149806f9c24e2a3ab6489ec30a7c49694ae36cb358de4a09856287`. This is a single-task, single-training-seed D10 result only; it is not the required all-task C0 result and is not used to claim a C1/C2 improvement.

## 2026-09-20 — Stage 1, D10 upright-RGB correction

- Image alignment audit established that official `agentview_rgb` and live `agentview_image` match directly (MAE about 2.4) and do not need a vertical flip (flipped MAE about 55.4). Version 1 had applied an identical but semantically harmful flip to both paths, presenting upside-down images to frozen CLIP.
- Raised the data contract to version 2 and introduced `scripts/audit_image_alignment.py`. The corrected features and task-0 result are stored under distinct `*_upright` / `task0_seed11_upright` paths; the earlier result remains untouched as a preprocessing ablation.
- The corrected task-0 / seed-11 run completed 50 epochs / 3,400 steps and achieved 11/20 = `0.55` on the exact same 20 official episodes, compared with 12/20 = `0.60` for the flipped-input run. The direct-RGB fix is kept because it is semantically and geometrically correct, but it did not improve this single-task C0 result. The main remaining limitation is therefore open-loop behavior cloning, not image orientation.

## 2026-09-20 — Stage 1, D11 all-task C0 launcher

- Added an all-task evaluator that batches independent environment observations only for frozen-model inference. Each MuJoCo environment remains separate, uses its own fixed seed/init state, receives its own action chunk, and calls its own official `check_success()`; batching cannot exchange episode state or oracle data.

## 2026-09-22 — Stage 2, D13 observation-aligned world-model data

- Froze `d13-v1`: `(z[k], proprio[k], language, actions[k+1:k+9]) -> z[k+1:k+9]`, while leaving the C0 imitation-learning convention unchanged.
- Created and audited an episode-grouped feature store with 62,250 unique observations, 52,381 train samples, and 5,869 validation samples. The store SHA-256 is `e5599153cf44282242d955be62249bbafc752b782f5ae76a0ec0d58f9146f98e`.
- Verified multiple source indices, final legal starts, episode boundaries, split disjointness, and the oracle-field denylist. D13 commit `e37073e588505b4461d11dc81b174032ff297f2d` is tagged `d13-wm-data-v1`.

## 2026-09-22 — Stage 2, D14 single latent predictor

- Implemented the 792,000-parameter GRU latent-delta predictor and train-only unique-row normalization. Proprio count is 55,981; causal action count is 55,531.
- Trained seed 11 for 15 epochs / 3,075 steps. The epoch-13 best checkpoint has validation cosine error `0.0093713445`, versus copy-current diagnostic `0.0157953181`.
- Verified forward/backward, L2 output norms, finite gradients and checkpoint tensors, safe resume, historical-best protection, and independent checkpoint reload. Full evidence is in `docs/d14_world_model_single_predictor_audit.md`.

## 2026-09-22 — Stage 2, D15 three-member ensemble

- Trained independent seed-23 and seed-47 members and combined them with the frozen seed-11 D14 member. All three share the exact D13 store, split, architecture, and normalization contract.
- Defined uncertainty as the mean pairwise cosine distance between three member latent predictions. On 5,869 validation samples its mean is `0.0015132597`, p95 is `0.0026366502`, and its diagnostic Pearson correlation with prediction error is `0.2501194`.
- Ensemble mean prediction obtains validation cosine error `0.0089190605`, 4.83% below the best individual member. Training/member resources and checkpoint hashes are recorded in `docs/d15_world_model_ensemble_audit.md`; no D16 gate or C1/C2 result is claimed.

## 2026-09-22 — Stage 2, D16 held-out transition gate

- Compared the frozen D15 ensemble against copy-current on all 5,869 D13 validation windows. Learned/copy cosine errors are `0.0089190605` / `0.0157953181`, a 43.53% relative reduction.
- All eight horizons, all 50 held-out episodes, and all 10 tasks improve. A 10,000-replicate task-stratified episode bootstrap gives paired-improvement 95% CI `[0.00675743, 0.00735354]`.
- The formal four-check gate passes. The full result is `results/wm/transition_eval.json`; the 5,869-row sample record is locally retained and hash-addressed for D17. A repeated run produced byte-identical hashes.

## 2026-09-22 — Stage 2, D17 qualitative sample review

- Selected two lowest-error and two highest-error validation samples per task from distinct demos, yielding 20 low-error and 20 high-error cases with complete 10-task coverage.
- Generated four RGB contact sheets showing `k`, `k+1`, `k+4`, and `k+8`, then manually reviewed every sample. High errors often coincide with larger motion/contact-phase appearance changes, but static high-error and dynamic low-error counterexamples are present.
- Concluded that mismatch reflects transition difficulty but is not a control-failure label. Exact per-sample metrics, panel hashes, and all 40 manual notes are in `docs/wm_sample_review.md`.

## 2026-09-22 — Stage 2, D18 offline foundation gate

- Re-audited D13 data/split/index fields, unique-row normalization, all three finite ensemble checkpoints, D16 held-out results, D17 sample coverage, and C0 immutability. The full test suite passes.
- The offline world-model gate passes and authorizes D19. Because the stage-level G2 definition also requests a C1 output but the day schedule implements C1 on D19, final project G2 remains `PENDING_C1_OUTPUT_D19`; C2 remains unauthorized.

## 2026-09-22 — Stage 3, D19 C1 pre-execution reranking

- Implemented deterministic N=3 candidate generation around the unchanged C0 chunk and D15-ensemble scoring by language progress, disagreement penalty, and a small base-deviation penalty. C1 executes the selected H=8 chunk fully and has no early replan path.
- Completed all 20 fixed task-0 episodes: 760 planning events, candidate counts `[20,366,374]`, no invalid values/events, and 0/20 success versus C0's same-protocol 0/20. This is functional C1 evidence, not an improvement claim.
- The C1 output completes the stage-level G2 evidence. Online selected-candidate uncertainty is above the held-out D15 distribution and is carried forward as an explicit D20/D22 risk; no test-driven tuning was performed.

## 2026-09-22 — Stage 3, D20 validation-only threshold selection

- Evaluated 36 mismatch/uncertainty quantile pairs on 23,476 h2/h4/h6/h8 points from validation demos 45–49. No fixed simulator episode or success result entered selection.
- Froze mismatch threshold `0.0290635973` (p99.5) and uncertainty threshold `0.0044514909` (p99.75), giving 0.75396% nominal OR-trigger rate against a ≤1% target.
- A post-selection excluded diagnostic shows 59.47% of D19 plans have mean selected-candidate uncertainty above the threshold, warning of possible C2 over-replanning. Thresholds remain frozen; D24 must audit actual loop behavior.

## 2026-09-22 — Stage 3, D21 C2 prediction verification

- Implemented C2 verification after every 2 actions at h2/h4/h6/h8. It compares the selected D15 ensemble prediction against a newly encoded live RGB observation and applies the immutable D20 mismatch-or-uncertainty rule.
- Early triggers discard only a non-empty chunk suffix and replan on the next control step. The summary retains every plan/check and the ignored `replan_events.jsonl` is count- and SHA-addressed; synthetic, real-D13, resume, boundary, and finite-value checks pass.
- A one-episode task-0 production smoke executed all 300 environment actions with no zero-action loop, but 90 of 103 plans ended early (84 at h2), including 86 uncertainty-only triggers. D21 is functionally valid but exposes the predeclared over-replanning risk. Only the narrow D22 task-0 comparison is authorized before deciding whether broader evaluation must stop.

## 2026-09-22 — Stage 3, D22 task-0 ablation

- Protocol-checked the all-task seed-11 C0 actor, C1, and C2 on the same task-0 episode indices, seeds 10000–10019, fixed initial states, 300-step limit, checkpoint hash, action clip, and official success API.
- C0/C1/C2 all obtain 0/20, so no success-rate benefit or harm is identifiable. C2 performs 2,242 plans and 3,000 checks; 2,018 plans end early, averaging 100.9 replans per episode, with 1,859 uncertainty-only triggers.
- All 6,000 actions are accounted for and no plan executes zero actions, ruling out an implementation-level infinite loop. The 90.01% early-plan rate is nevertheless an operational high-frequency replanning loop with no observed benefit. The frozen D20 thresholds were not changed, D22 status is `BLOCK_D23_HIGH_FREQUENCY_REPLANNING`, and the ten-task D23 expansion is not authorized.

## 2026-09-23 — Stage 3, D22.5 controller distribution-shift diagnosis

- Tagged the preserved D22 negative result as `d22-c2-overreplan-negative`, then precommitted an isolated controller-validation split: init-state indices 20–29 and new seeds 20000–20909, with zero overlap against fixed-test indices 0–19/seeds 10000–10919. Indices 30–49 remain unused.
- Collected logging-only C0 and C1-selected rollouts on all 10 tasks without any replan or trigger-dependent control. C0/C1 produced 13,772/14,085 check events. At h2, policy uncertainty median is about 2× expert validation and p95 is 2.75–2.91×, whereas mismatch shifts only about 1.17–1.19× at the median.
- Frozen D20 OR would trigger on 23.91% of C0 and 22.75% of C1 checks, dominated by uncertainty-only (22.15% / 20.96%); mismatch-only rates are 3.67% / 3.26%. Uncertainty still has moderate useful ranking signal: current/next high-error AUROC is about 0.70–0.73 and mismatch rises monotonically across uncertainty deciles. It should not be treated as a calibrated OR replan probability.
- Independent C0 competence is 15/100 with per-task successes `[0,0,0,0,3,0,7,0,5,0]`; 7/10 tasks remain at floor. Final verdict is `BASELINE_COMPETENCE_BLOCK`, with secondary `GO_D20_V2_PROTOCOL_DESIGN_ONLY`. D23 stays blocked, D20-v1 is unchanged, and no C0-v2 was trained.
- The original D22.5 write-up described D13 `actions[k+1:k+9]` versus online `candidate[0:8]` as an unresolved mapping. This was later corrected: both contain the same eight next actions, `a[k+1]...a[k+8]`; there is no missing ninth action. The measured policy-induced WM distribution shift remains valid independently of that interpretation.

## 2026-09-23 — C0-v2 causal-alignment preregistration

- D22.5 remains frozen with `BASELINE_COMPETENCE_BLOCK`; D23 and D20-v2 threshold work remain blocked. New experimental status is `GO_C0_V2_CAUSAL_ALIGNMENT`.
- Corrected the interpretation of Python's half-open slice: `actions[k+1:k+9]` contains exactly eight actions, `a[k+1]...a[k+8]`, so an online eight-action candidate does not lack a ninth action. The isolated question is whether changing C0 supervision from `actions[k:k+8]` to `actions[k+1:k+9]` improves baseline competence.
- Before any rollout from init states 30–49, preregistered 30–39 as controller development and locked 40–49 as a one-shot confirmatory holdout. The numerical GO criteria are frozen in `configs/c0_v2_causal_alignment_protocol.json`.

## 2026-09-23 — C0-v2 development gate

- Trained the single-factor causal-label actor with the same C0 architecture and budget. The data layer contains exactly 52,381 train and 5,869 validation samples; best demo-validation raw action MSE is 0.042468849 at epoch 3.
- On preregistered init-state indices 30–39, frozen C0-v1 obtains 11/100 over 3/10 non-zero tasks; C0-v2 obtains 45/100 over 7/10 non-zero tasks. Mean steps change from 281.88 to 210.58 and C0-v2 clipping is 4.00%.
- All six preregistered gate checks pass. Before observing any init state in 40–49, froze checkpoint SHA-256 `2346603c8123e0a1cdee408cb635a2ffd12d6bc4e0cc5d91eacbc1330f5faac0` and recorded `GO_CONFIRMATORY_HOLDOUT`.

## 2026-09-23 — C0-v2 one-shot confirmatory holdout

- Ran the already-frozen C0-v1 and C0-v2 checkpoints once on preregistered init-state indices 40–49, 10 episodes for every LIBERO-Spatial task. The analyzer independently verified all 200 seed/index identities, checkpoint/protocol hashes, completion markers, and finite values.
- C0-v1 obtains 10/100 over 3/10 non-zero tasks; C0-v2 obtains 38/100 over 7/10 non-zero tasks, a +28-success and +4-task improvement. Mean steps fall from 284.09 to 225.62; C0-v2 planned-action clipping is 4.119%, below the frozen 10% ceiling.
- Paired outcomes are 5 both-success, 5 C0-v1-only, 33 C0-v2-only, and 57 neither; exact two-sided McNemar `p=4.25596e-6`. All six precommitted acceptance checks pass, so the final verdict is `CAUSAL_ALIGNMENT_IMPROVES_BASELINE`.
- The effect is broad but not uniform: C0-v2 remains 0/10 on tasks 7 and 9 and trails C0-v1 on tasks 4 and 6 in this holdout. The result supports causal label alignment for this fixed C0 setup, not a claim that all tasks or the WM controller are solved. D23 and D20-v2/C2-v2 remain blocked/not run.

## 2026-09-24 — C1-v2 rebase and WM distribution audit

- Froze C0-v2 at annotated tag `c0-v2-causal-aligned` and established the independent `C0-v2 -> C1-v2 -> D20-v2 -> C2-v2` stack. Corrected the online mapping: eight-element `candidate[0:8]` maps to D13's eight causal actions `a[k+1]...a[k+8]`; there is no missing ninth action.
- Preregistered C1-v2 and its gate before rollout, then used only controller-development init-state indices 30–39. No C1-v2/C2-v2 result from 40–49 was viewed; D20-v1 thresholds remained diagnostic-only and unchanged.
- Independently verified the complete Actor-normalized -> Actor-denormalized -> raw candidate -> WM-normalized -> ensemble path for every stored plan. The sidecar audit proves that the WM never receives Actor-normalized actions directly; the final selected-candidate clipping rate is 0.0825%.
- C0-v2 and C1-v2 both obtain 44/100 successes over 7/10 non-zero tasks. C1-v2 replaces the Actor default on 2,441/2,728 plans (89.48%), but gains on tasks 1/3/5/8 are offset by regressions on tasks 2/6.
- C0-v2 rollout mismatch and uncertainty are closer to expert validation than historical C0-v1 at all h2/h4/h6/h8 quantile comparisons. Median excess uncertainty is reduced by 58.0% on average, but C0-v2 medians remain 1.39–1.41× expert, so residual policy-induced OOD remains.
- Replacement predictions are non-worse than the Actor-default prediction against the subsequent observed latent only at h8, not h2/h4/h6. The preregistered requirement was at least two horizons; final verdict is `BLOCK_C1_V2_RERANKER`. D20-v2, C2-v2, old D23, and 40–49 controller evaluation were not started.

## 2026-09-24 — C1-v2.1 candidate discriminability diagnosis

- Froze the C1-v2 negative result at annotated tag `c1-v2-reranker-negative`, then committed the complete offline diagnosis protocol and implementation before viewing results. The analysis uses all 5,869 demo-validation samples from demos 45–49 and performs no simulator rollout or init-state access.
- The production N=3 generator contains useful alternatives under the preregistered expert-action oracle: an alternative is best on 65.87% of samples and best-of-three raw-action MSE improves 1.648% over the default. `CANDIDATE_GENERATOR_BLOCK` does not apply.
- The frozen score cannot identify those alternatives: top-score oracle accuracy is 32.22%, below the 33.33% random reference; pairwise accuracy is 49.43%, mean within-sample Spearman is -0.0235, and score-margin/improvement Spearman is -0.0165. The old ~3.48e-4 selected-score gain is almost entirely goal-progress selection, while uncertainty and action-deviation contributions barely affect ordering.
- Production candidates have mean normalized action-chunk separation 1.75294, but their predicted latent cosine separation is only 0.004235 times ordinary WM validation error. The diagnostic scale sweep finds no preregistered scale where generator, score, and WM sensitivity all work; it does not select a new scale.
- Final blockers are `RERANKER_SCORE_BLOCK` and `WM_ACTION_SENSITIVITY_BLOCK`. The frozen C1-v2 verdict remains `BLOCK_C1_V2_RERANKER`; no new C1 controller, D20-v2, C2-v2, or 40–49 evaluation is authorized.

## 2026-09-24 — C1-v2.2 frozen-WM action-conditioning diagnosis

- Froze C1-v2.1 at annotated tag `c1-v2-1-discriminability-negative`, then committed the complete action-ablation protocol, bootstrap, interpretation thresholds, implementation, and synthetic tests before viewing real results. The diagnosis uses only all 5,869 D13 validation samples from demos 45–49 and performs no simulator/init-state access.
- The D15 ensemble clearly uses actions: within-task whole-chunk shuffle raises cosine error from 0.00891906 to 0.01139824 (+27.80%), and correct-action paired ranking accuracy is 78.11% with task-stratified episode-block 95% CI [77.04%,79.13%]. Physical zero and cross-task shuffle give similar results.
- Temporal ordering and magnitude matter. Reverse/fixed-permutation raise error 5.04%/0.93%, with positive bootstrap intervals. Correct 1x actions have the lowest error across 0x/0.25x/0.5x/1x/1.5x, and prediction separation from 1x tracks absolute scale distance with Spearman 0.9747.
- All three members and all ten tasks show the same positive action signal; ensemble averaging attenuates but does not cancel it. Real CLIP temporal displacement is 1.771× ordinary WM error, so neither `TASK_DEPENDENT_ACTION_SIGNAL` nor `CLIP_STATE_SENSITIVITY_LIMIT` applies.
- The remaining failure is local: within-task shuffle changes predictions by 27.18% of ordinary WM error, while frozen production candidate changes are only 0.4235%. Final verdict is `WM_HAS_ACTION_SIGNAL_BUT_LOW_LOCAL_SENSITIVITY`; future priority is a locally action-sensitive WM training objective, not immediate reranker tuning. No training or controller work is authorized by this result.

## 2026-09-24 — WM-v2A local dynamics resolution experiment

- Froze WM-v1/C1-v2.2 at annotated tag `wm-v1-low-local-action-sensitivity`, then committed the complete WM-v2A objective, gate, diagnostics, and tests before training. The only changed factor is `SmoothL1(pred_z-z0, z_future-z0)` with preregistered `lambda_delta=1`; data, split, normalization, 792,000-parameter architecture, optimizer, and seed-11 budget remain D14-equivalent.
- The epoch-13 checkpoint preserves ordinary prediction: validation cosine error is `0.00937206`, only +0.0076% versus D14 seed11 and below copy-current `0.01579531`. Within-task shuffled actions raise error by 29.48%, so the global action-conditioning gate remains healthy.
- Production-candidate cosine separation rises from `3.7775e-5` to `4.9737e-5`, but separation/error is only `0.0053069` (1.253x the frozen `0.0042354` reference), below both local gates (`>=0.01`, `>=2x`). Directional sanity checks pass at 55.52% pairwise ranking and mean Spearman 0.1115, so the improvement is directionally valid but too small.
- The reason is structural: the shared `z0` cancels from the delta residual, and for normalized outputs in SmoothL1's quadratic region the new mean-over-512 term is approximately cosine loss divided by 512 (`L_future/L_delta=511.997`). It therefore adds almost no independent supervision. Final verdict is `DELTA_OBJECTIVE_INSUFFICIENT`; seed23/47, an ensemble, WM-v2B, controllers, D20-v2, and C2-v2 are not authorized or run.

## 2026-09-24 — WM-v2B future-proprio auxiliary dynamics experiment

- Froze WM-v2A at annotated tag `wm-v2a-delta-objective-insufficient`, then preregistered a target-only future-proprio head and all offline gates. The D13 causal contract, split, H=8, frozen CLIP representation, D14 visual backbone, optimizer, candidate generator, and controller remain unchanged.
- Calibrated the auxiliary coefficient exactly once on 1,024 fixed train-only samples using separate shared-GRU gradient norms: `g_z=0.01429453`, `g_p=0.03348240`, yielding unclipped/final `lambda_p=0.42692670`. No validation metric entered calibration.
- The epoch-14 seed11 checkpoint preserves ordinary visual quality (`0.00962736`, +2.73% versus D14) and learns future proprio strongly (`0.00187477` versus zero-delta `0.03250166`). Correct actions beat within-task shuffled actions for proprio by 2,832.5%, with 98.85% paired accuracy.
- Frozen production candidates become clearly separable in predicted proprio: standardized trajectory separation/error is `0.4550`, and proprio oracle-ranking accuracy/Spearman are 62.12%/0.2531. Visual separation/error remains only `0.005157`, below `0.01`, only 1.218x WM-v1, and slightly below WM-v2A.
- Final verdict is `PHYSICAL_SIGNAL_LEARNED_VISUAL_TRANSFER_INSUFFICIENT`: shared dynamics learned nearby physical action differences but the visual head did not express them above its prediction-noise scale. Seed23/47, an ensemble, controller/reranker changes, D20-v2, and C2-v2 are not authorized or run.
