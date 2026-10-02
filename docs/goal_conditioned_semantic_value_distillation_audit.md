# Frozen natural features do not yet support a valid semantic value student

Date: 2026-10-03

Final mechanism labels:

```text
SEMANTIC_VALUE_DISTILLATION_BLOCKED
VALUE_STUDENT_LANGUAGE_INSENSITIVE
VALUE_STUDENT_VISUAL_INFORMATION_INSUFFICIENT
FROZEN_VISUAL_VALUE_INFORMATION_INSUFFICIENT
VALUE_STUDENT_SCENE_SHORTCUT
```

Next-stage authorization: `BLOCK_REAL_CANDIDATE_VALUE_BRIDGE`

Continuing authorizations:

```text
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
```

Selected semantic value student: `none`

Frozen calibration representative: `S2`

Controller authorization: `false`

## Executive answer

The frozen T1-M2 teacher does not distill into a student that clears the full
deployment gate under the tested frozen natural representation. Calibration
mechanically selected the direct antisymmetric S2 comparator. On held-out
demo36--39, S2 reaches `69.81%` ordinary local non-tie accuracy
(task-stratified demo-block 95% CI `[68.14%,71.46%]`) and exceeds `60%` on
9/10 tasks. The point estimate misses the frozen `70%` requirement by `0.19`
percentage point, although its CI lower bound clears the separate `65%` gate.

The more important language bridge also fails. Across `74,645` same-pair
applications for which frozen T1 changes under a real alternate instruction,
S2 changes probability in the teacher's ordinal direction in `63.99%`
(CI `[62.59%,65.35%]`), below the required `70%`. All ten tasks are above
chance, and the paired advantage over image-only is `63.99` points with a
wholly positive interval, so the result is not language blindness in the
literal zero-effect sense. It is nevertheless insufficient transfer under the
frozen gate. On the rarer `3,258` true better/worse reversals, both predictions
are correct only `18.57%` of the time (CI `[14.20%,23.58%]`).

Tie behavior is acceptable but does not rescue the model. On `15,835` held-out
local T1 ties, mean `|P-0.5|` is `0.0923` (CI `[0.0885,0.0961]`), below the
`0.15` limit. When language changes a non-tie into a tie, distance from 0.5
falls by `0.0478` on average and improves in `63.06%` of cases. Same-referent
language swaps have mean probability change `0.0955`, narrowly passing the
`0.10` invariance gate, although their p90 change is `0.2062` and sign flips
remain common.

The cross-demo control exposes an additional shortcut. On 94 same-task,
different-demo pairs whose T1 Boolean matches and M2 differs by at most the
frozen epsilon, S2's mean `|P-0.5|` is `0.1681`, above the `0.15` limit; p90 is
`0.3424`. On 1,280 clear semantic-difference cross-demo pairs it reaches only
`55.31%` accuracy. S2's direct-pair cycle rate is low (`3.75%`), but the model
is not sufficiently semantic, language-sensitive, or scene-invariant.

The scalar alternatives do not provide a valid fallback. S1.5 reaches
`65.44%` ordinary accuracy and `60.74%` counterfactual agreement; S1 reaches
`61.15%` and `55.72%`. Both are materially below S2, so the frozen 2-point
non-inferiority rule cannot replace the O(N²) comparator with an O(N) scalar.
No candidate is selected, no positive distillation label is assigned, and no
real candidate future, N64 tournament, reranker, World Model, or controller is
run.

## 1. Frozen T1 provenance

The privileged teacher is unchanged from commit
`1655b5866c9f33f70c3f86efb9828b0e85ce38f1` under annotated tag
`t1-semantic-teacher-full-evaluation-valid`. The audit verified the frozen G2
resolver, persistent-referent rule, official `On` source, M2 equation,
`epsilon_goal=0.004065508647683736`, T1 lexicographic comparison, and both T1
sidecars.

The new study began at annotated tag `semantic-value-distillation-start`.
Protocol and implementation were committed at `416f629` before student
training. A calibration preflight revealed only that the optional `>16`
ordinary evaluation bin is absent from the frozen T1 sidecar. Commit `4b19a04`
changed empty-bin serialization from NaN to `null`; no sample, model, loss,
temperature, threshold, selection rule, bootstrap, or authorization changed.
Calibration was rebuilt and frozen at `faeb366` before demo36--39 was loaded.
The one-shot numerical evaluation was frozen at `571d60f`.

## 2. Data boundary

| Partition | Demo IDs | Frames | T1 pair rows | Use |
|---|---|---:|---:|---|
| Calibration | 31--34 | 5,044 | 75,264 | fit, demo34 selection, temperature, final refit |
| Frozen development evaluation | 36--39 | 4,883 | 72,688 | one-shot metrics and decision |
| Historical only | 30/35 | 0 | 0 | not opened |
| Forbidden | 40--49 | 0 | 0 | no numeric or metadata access |

The evaluation is frozen development evaluation, not blind confirmation.
Exact-path guards opened 40 admitted `vision` datasets per partition plus one
global language matrix. They did not enumerate groups or read attributes from
any forbidden demo.

## 3. Frozen natural input representation

Every deployable student input is:

```text
upright, unmodified natural RGB
  -> frozen CLIP ViT-B/32 global image projection [512]

task instruction
  -> frozen CLIP text projection [512]
```

The inherited D13 store hash is
`e5599153cf44282242d955be62249bbafc752b782f5ae76a0ec0d58f9146f98e`.
The exact admitted-array digests are
`af6c327df0f8ee8b24d1e7b3428e84fd4420f5405e332786a9414d2be1bac13b`
for calibration and
`1045ea73fe7cd2633d15ba9842b4c23526dd98ba69055a67fa2b3e0804996eb3`
for evaluation.

No student sees task ID, demo ID, time, normalized progress, proprioception,
action, simulator state, object identity, G2, `On`, M2, or T1 coordinates.
Privileged arrays are used only to construct labels and frozen strata.

## 4. Student families

| Family | Form | Parameters | Deployment scoring |
|---|---|---:|---|
| S0 | language-selected terminal visual prototype cosine | 0 | O(N) |
| S1 | visual+language, Boolean head plus M2 head with exact predicted Boolean priority | 65,730 | O(N) |
| S1.5 | visual+language scalar, Bradley--Terry pair loss | 65,665 | O(N) |
| S2 | antisymmetric `(za,zb,zb-za,l)` MLP | 131,201 | O(N²) |
| Image-only | antisymmetric visual pair MLP | 98,433 | diagnostic |
| Language-only | one linear language prior | 513 | diagnostic |

All trainable families remain below 200,000 parameters. No Transformer,
encoder update, or architecture search was admitted.

## 5. Pair construction

The frozen sampler stratifies ordinary pairs by task, demo, gap, and T1 label.
Counterfactuals are separately stratified by same/different referent and the
full original-to-alternate label transition. Every sampled non-self pair has
an exact reversed row with negated label. The final calibration refit uses
83,474 rows: 37,188 ordinary/reverse rows, 40,028 counterfactual/anchor/reverse
rows, 3,778 robot-control rows, and 2,480 same-state ties. S1 additionally uses
38,809 state-language targets.

The primary evaluation uses every frozen ordinary T1 pair with gap at most 8,
not the balanced training subsample.

## 6. Semantic tie construction

T1 ties are retained with target `P(B>A)=0.5`; they are never converted to a
random better/worse label. The sampler includes ordinary ties, exact reverse
ties, same-state self-pairs under every uniquely grounded instruction, and
semantic-tie/robot-different controls.

## 7. Language-counterfactual construction

For each local pair, the correct instruction and every other real instruction
that G2 grounds uniquely in the same initial physical scene are evaluated with
their own persistent referent. The image pair is identical; only frozen text
features and T1 labels change. Evaluation contains 252,300 applications, of
which 74,645 change T1 preference.

Teacher-change agreement is deliberately ordinal: it requires the student's
probability change to have positive product with the T1 label change. A tiny
change in the wrong direction fails. True reversals and non-tie-to-tie changes
are reported separately.

## 8. Same-referent control

There are 117,880 alternate-language applications that retain the same
persistent bowl. For S2, mean/median/p90 absolute probability changes are
`0.0955 / 0.0746 / 0.2062`; the mean passes the frozen `0.10` gate. The false
preference-flip rate is `26.47%`, so the point gate should not be read as strong
invariance.

## 9. Different-referent control

There are 134,420 different-referent applications; all 74,645 teacher-changing
rows occur in this stratum. S2 follows the teacher's ordinal change in `63.99%`.
This is materially above an image-only model, but below the `70%` requirement.

## 10. Image-only baseline

Image-only reaches `68.46%` ordinary local accuracy, only `1.35` points below
S2. It cannot change predictions when language changes, so its teacher-change
agreement is exactly zero by construction. Its ordinary strength demonstrates
that substantial within-demo phase information is present without goal
language.

## 11. Language-only baseline

Language-only reaches `43.80%` ordinary local accuracy and `51.04%`
teacher-change agreement. Cross-demo semantic-difference accuracy is exactly
`50%`. Its strong task-to-task variation does not translate into a useful
pairwise value; the balanced reverses prevent a task prior from solving the
audit.

## 12. S1 absolute semantic potential

S1 predicts an official-goal class and standardized M2 coordinate. Evaluation
uses an exact two-unit Boolean class offset plus a bounded sigmoid margin, so a
predicted goal-positive state always outranks a predicted goal-negative state;
there is no arbitrary Boolean/M2 weighted tradeoff.

S1 reaches `61.15%` ordinary accuracy, `55.72%` language-change agreement,
tie deviation `0.00738`, and cross-demo matched deviation `0.0629`. Its strong
neutral calibration comes at inadequate discrimination.

## 13. S1.5 scalar pairwise potential

S1.5 uses the same natural inputs but learns a scalar only through
Bradley--Terry differences. It reaches `65.44%` ordinary accuracy, `60.74%`
language-change agreement, tie deviation `0.0915`, and matched cross-demo
deviation `0.1539`. It is transitive and O(N), but loses `4.37` ordinary points
and `3.26` language points to S2, exceeding both 2-point tolerances.

## 14. S2 direct pairwise model

S2 applies one shared raw MLP to `(za,zb,zb-za,l)` and its swapped ordering,
then subtracts the two logits. Reverse probabilities are complementary by
construction. It is the calibration-frozen representative, not a selected
production student.

## 15. Calibration

Models were fit on demo31--33 and compared on demo34. S2 led the three student
families on the first two lexicographic criteria:

| Family | Language-change agreement | Ordinary accuracy | Tie mean | Tasks >60% |
|---|---:|---:|---:|---:|
| S1 | 57.99% | 62.56% | 0.0076 | 8 |
| S1.5 | 60.24% | 65.20% | 0.1025 | 8 |
| S2 | 61.92% | 68.05% | 0.1049 | 8 |

Calibration selected temperature `0.25` for S1.5, S2, and both learned
ablations; S1 uses `0.5` and S0 uses `4.0`. The fixed models were then refit on
all demo31--34 with no re-selection.

## 16. Model selection

S1.5 is within two points of S2 on language change (`1.68` points behind) but
not ordinary accuracy (`2.85` points behind) on demo34. S1 is not within the
tolerance on either. The frozen rule therefore falls through to the
lexicographic winner S2. Evaluation never revisits this choice.

## 17. Frozen development evaluation

The frozen weights, normalization, language mapping, temperatures, selection,
tie target, gates, bootstrap seed, and 10,000-draw shape were applied once to
demo36--39. All persisted probability arrays are finite and in `[0,1]`.

## 18. Ordinary local semantic results

| Family | Accuracy | AUROC | Log loss | Brier | 95% CI for accuracy |
|---|---:|---:|---:|---:|---|
| S0 | 54.06% | 0.524 | 0.691 | 0.249 | [52.68%,55.45%] |
| S1 | 61.15% | 0.590 | 0.686 | 0.246 | [59.90%,62.48%] |
| S1.5 | 65.44% | 0.656 | 0.611 | 0.212 | [64.13%,66.80%] |
| **S2** | **69.81%** | **0.706** | **0.566** | **0.193** | **[68.14%,71.46%]** |
| Image-only | 68.46% | 0.686 | 0.601 | 0.206 | [66.89%,70.10%] |
| Language-only | 43.80% | 0.456 | 0.694 | 0.250 | [42.06%,45.52%] |

S2 has the strongest tested natural representation, but `69.81%` is below the
preregistered `70%` point gate.

## 19. Per-gap results

S2 non-tie accuracy increases with temporal separation:

| Gap | Rows | Accuracy |
|---|---:|---:|
| 1--2 | 5,157 | 62.58% |
| 3--4 | 5,410 | 68.34% |
| 5--8 | 11,222 | 73.85% |
| 9--16 | 23,011 | 79.20% |

The frozen ordinary sidecar has no `>16` rows; that optional cell is explicit
`null`, not evidence.

## 20. Per-task results

| Task | S2 local non-tie accuracy |
|---:|---:|
| 0 | 54.58% |
| 1 | 63.87% |
| 2 | 74.44% |
| 3 | 72.91% |
| 4 | 73.02% |
| 5 | 69.20% |
| 6 | 72.97% |
| 7 | 71.38% |
| 8 | 62.30% |
| 9 | 78.32% |

Nine tasks exceed `60%`; task0 is the sole failure. This passes the 8/10 task
count but does not override the aggregate point gate.

## 21. Language-counterfactual transfer

| Family | Changed rows | Agreement | 95% CI | Positive tasks |
|---|---:|---:|---|---:|
| S0 | 74,645 | 51.20% | [50.21%,52.17%] | 7 |
| S1 | 74,645 | 55.72% | [54.76%,56.71%] | 10 |
| S1.5 | 74,645 | 60.74% | [59.50%,62.00%] | 10 |
| **S2** | **74,645** | **63.99%** | **[62.59%,65.35%]** | **10** |
| Image-only | 74,645 | 0% | [0%,0%] | 0 |
| Language-only | 74,645 | 51.04% | [49.75%,52.34%] | 4 |

S2 contains real language-conditioned signal, but it does not reach 70%.

## 22. True reversal

The `3,258` true better-to-worse or worse-to-better rows are strong secondary
evidence. S2 gets both sides correct in `18.57%`, CI `[14.20%,23.58%]`.
S1/S1.5 reach `21.42%/16.11%`. No reversal threshold was used as a sole hard
gate, as preregistered.

## 23. Non-tie-to-tie recovery

There are 71,083 non-tie-to-tie swaps. For S2, distance to 0.5 falls by
`0.0478` on average and `0.0379` at the median; `63.06%` improve. The effect is
directionally useful but far from universal.

## 24. Tie calibration

| Family | Local tie rows | Mean `|P-0.5|` | Median | p90 |
|---|---:|---:|---:|---:|
| S0 | 15,835 | 0.0095 | 0.0065 | 0.0214 |
| S1 | 15,835 | 0.0074 | 0.0035 | 0.0105 |
| S1.5 | 15,835 | 0.0915 | 0.0756 | 0.1932 |
| **S2** | **15,835** | **0.0923** | **0.0733** | **0.2007** |
| Image-only | 15,835 | 0.0610 | 0.0481 | 0.1304 |
| Language-only | 15,835 | 0.0023 | 0.0030 | 0.0042 |

S2's mean CI is `[0.0885,0.0961]`; it passes the `0.15` gate.

## 25. Cross-demo semantic-match control

Strict semantic matching yields 94 pairs. S2 mean/median/p90 `|P-0.5|` is
`0.1681 / 0.1381 / 0.3424`. The mean exceeds the frozen `0.15` line, assigning
`VALUE_STUDENT_SCENE_SHORTCUT`. S1 is neutral (`0.0629`) but too weak on the
primary tasks; S1.5 is also just above the line (`0.1539`).

## 26. Cross-demo semantic-difference control

On 1,280 direction-balanced, clear T1-difference pairs, S2 reaches `55.31%`
accuracy, AUROC `0.592`, log loss `0.723`, and Brier `0.259`. This prevents a
false interpretation that neutralizing every cross-demo pair would be useful:
the model is only weakly able to order genuine cross-demo semantic differences.

## 27. Image-only shortcut

S2 exceeds image-only ordinary accuracy by only `1.35` points. On the decisive
teacher-changing counterfactuals, however, image-only cannot move at all, and
S2's paired advantage is `63.99` points, CI `[62.59,65.35]`. The critical
language-evidence gate therefore passes even though absolute language transfer
does not.

## 28. Language-only shortcut

Language-only is weak overall and exactly chance on cross-demo semantic
differences. Its `51.04%` counterfactual direction score shows that instruction
priors alone do not explain S2's `63.99%`; neither do they provide a valid
semantic value.

## 29. Transitivity and cycles

S2 has `9/240` cyclic triplets, a `3.75%` cycle rate below the frozen `10%`
limit. S1 and S1.5 are structurally scalar and have zero cycles. The future S2
aggregation rule was preregistered as Borda/mean pairwise probability, but it
was not applied to real candidates because the value-student gate failed.

## 30. Compute

Measured in one CPU process on frozen held-out embeddings:

| Family | Parameters | FP32 parameter bytes | N64 work | Measured latency |
|---|---:|---:|---:|---:|
| S1 | 65,730 | 262,920 | 64 state scores | 0.124 ms |
| S1.5 | 65,665 | 262,660 | 64 state scores | 0.159 ms |
| S2 | 131,201 | 524,804 | 2,016 pair scores | 9.22 ms |

Single-state latency is `0.0767 ms` for S1 and `0.0380 ms` for S1.5. These are
diagnostic CPU timings, not production throughput guarantees.

## 31. Gates

| Required gate | Frozen result |
|---|---|
| T1 hash unchanged | pass |
| RGB+language deployable inputs only | pass |
| Local non-tie accuracy >=70% | **FAIL: 69.81%** |
| Local CI lower >=65% | pass: 68.14% |
| >=8 tasks above 60% | pass: 9/10 |
| Language-change agreement >=70% | **FAIL: 63.99%** |
| >=8 positive language tasks | pass: 10/10 |
| Significantly beats image-only on changed CFs | pass |
| Same-referent mean change <=0.10 | pass: 0.0955 |
| Tie mean deviation <=0.15 | pass: 0.0923 |
| Cross-demo matched mean <=0.15 | **FAIL: 0.1681** |
| S2 cycle rate <=0.10 | pass: 0.0375 |
| All reportable arrays finite | pass |
| Forbidden access zero | pass |

Three independent gates fail, so the complete gate is false.

## 32. Final labels

```text
SEMANTIC_VALUE_DISTILLATION_BLOCKED
VALUE_STUDENT_LANGUAGE_INSENSITIVE
VALUE_STUDENT_VISUAL_INFORMATION_INSUFFICIENT
FROZEN_VISUAL_VALUE_INFORMATION_INSUFFICIENT
VALUE_STUDENT_SCENE_SHORTCUT
```

`VALUE_STUDENT_LANGUAGE_INSENSITIVE` means insufficient label-changing
transfer under the 70% gate, not zero statistical language response.
`FROZEN_VISUAL_VALUE_INFORMATION_INSUFFICIENT` is limited to this frozen global
CLIP representation and the preregistered small heads; it does not authorize a
new encoder in this task.

## 33. Selected student

```text
Selected semantic value student: none
Frozen calibration representative: S2
Input: frozen natural global CLIP [512] + frozen task-language CLIP [512]
Architecture: antisymmetric Linear(2048,64)-GELU-Linear(64,1)
Trainable parameters: 131,201
Score semantics: calibrated probability that state B is T1-better than A
Pairwise semantics: raw(a,b,l)-raw(b,a,l)
```

S2 is retained only as the evaluated diagnostic representative. It is not a
validated candidate-value model.

## 34. Next authorization

```text
BLOCK_REAL_CANDIDATE_VALUE_BRIDGE
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE
controller authorization = false
```

No C48_G15/N64 future may be collected or ranked from this result. No encoder
replacement is authorized automatically.

## 35. Hashes and automated verification

| Artifact | SHA-256 |
|---|---|
| Protocol | `78324f900bf9911e9d691bdabe082a25d6d8f26babc5f5a2f942bb5a092b30fd` |
| Architectures | `dd11f6526964692ab762e2dac5b7a5785afb59bd0a32a79f9991aabac9711990` |
| Pair sampler | `aa88e8c2f657eaee79611bd22e77ee56f1a50b5ec7392a1bb5b8a998210db595` |
| Student selection | `dd77c76a37130cd05c64dd480e68459b8ca09e2035a4be050d6916e4f9242fe7` |
| Calibration | `527c8e2639104240a7a7af19c5689fd9306e9b9e40ab855d8f46b9b20a1028fa` |
| Calibration freeze | `b8b1dc819c536f2d8ed58e19d58f270c25e0cdf54a4eef1d796fbbc6e117ce11` |
| Evaluation | `3eb4796d15892cfd77a941aa8a2e92dcfa2ac9752c454fd8d6940344aed100f8` |
| Language evaluation | `20fa4b9710530a060fcb78e185f6f3416a82cdc79d873e1d827bc44d90677e59` |
| Tie evaluation | `f43274bbde43ed8e4d20f654c5965320248599a1bfb95f950f92f17eb89ac006` |
| Cross-demo controls | `57fa0e9956b2c918b7f1d98d1669876719f572787df3a3593c092ae23accc8c8` |
| Weights | `cfe76ba01d71ae46ed2ca25eeeba392f92f914145c2f39ad4d1d71dd937cec13` |
| Calibration sidecar | `40d1f23179d723e541e5e8c5c2a7ee996734193bdb15a1eeaf23670a724f5bbe` |
| Evaluation sidecar | `a7ad95443dac1209c33ab3abf465c43162030930e826370e7718af02aa88e7d1` |
| Final decision | `93aa58007843c3aa3ceb6a6f1f72e5773280704d0eb8c8ac77fb5b1075a5417d` |

Nineteen focused protocol/calibration/evaluation/final-decision tests independently verify
hashes, capacity, antisymmetry, exact row identities, selection logic,
recomputed accuracy and bootstrap intervals, counterfactual agreement, tie
calibration, decision mapping, and access boundaries. The complete repository
suite passes `297` tests; the remaining 21 messages are dependency deprecation
warnings, not failures.

## 36. Forbidden-activity audit

No demo30/35 or demo40--49 numeric array or metadata was opened. No D13 action,
proprioception, reward, done, success, object state, or task outcome entered a
student. No simulator, candidate execution, N64 future, candidate reranking,
World Model, learned proposal, new encoder, encoder fine-tuning, controller,
CEM, MPPI, beam search, C1/C2, D20-v2, seed23/47 control experiment, or task
success evaluation was run. The privileged T1-M2 teacher and G2 resolver were
not modified.

## Required answers

**Q1 — Was frozen T1-M2 completely unchanged?**

Yes. All frozen source and artifact hashes match; G2, persistent referents,
official `On`, M2, epsilon, T1 comparison, and counterfactual eligibility were
unchanged.

**Q2 — Which deployable inputs did the student use?**

Only the frozen 512-D global CLIP projection of unmodified natural RGB and the
frozen 512-D task-language CLIP projection.

**Q3 — How did S0, S1, S1.5, and S2 perform?**

Their ordinary accuracy / language-change agreement was respectively
`54.06/51.20%`, `61.15/55.72%`, `65.44/60.74%`, and `69.81/63.99%`. S2 was
best but failed the final gates.

**Q4 — What is ordinary held-out local semantic non-tie accuracy?**

For frozen S2, `69.81%`, CI `[68.14%,71.46%]`.

**Q5 — What are the 1--2, 3--4, and 5--8 gap results?**

`62.58%`, `68.34%`, and `73.85%` accuracy.

**Q6 — How do the ten tasks perform?**

Tasks0--9 reach `54.58, 63.87, 74.44, 72.91, 73.02, 69.20, 72.97, 71.38,
62.30, 78.32%`; nine exceed 60%.

**Q7 — Is the student close to 0.5 on T1 ties?**

Yes by the mean gate: mean/median/p90 deviation is
`0.0923/0.0733/0.2007` on 15,835 ties.

**Q8 — When language changes the teacher preference, how often does the student follow?**

`63.99%`, CI `[62.59%,65.35%]`, across 74,645 rows—below the required 70%.

**Q9 — How does it perform on true better/worse reversals?**

`18.57%`, CI `[14.20%,23.58%]`, on 3,258 reversals.

**Q10 — Does it move toward 0.5 on teacher non-tie-to-tie changes?**

Partly. Mean/median distance reduction is `0.0478/0.0379`, and `63.06%` of
rows improve.

**Q11 — Is it approximately invariant on same-referent language swaps?**

It narrowly passes the mean rule at `0.0955`, but p90 is `0.2062` and the
false sign-flip rate is `26.47%`; the control is not uniformly strong.

**Q12 — Does it respond correctly on different-referent swaps?**

Only `63.99%` of the teacher-changing rows, which is positive but insufficient.

**Q13 — How strong is image-only?**

`68.46%` ordinary local accuracy, only 1.35 points below S2, but zero ability
to respond to a language swap on an identical visual pair.

**Q14 — Does primary RGB+language significantly beat image-only on true teacher-changing counterfactuals?**

Yes. The paired agreement advantage is `63.99` points, CI
`[62.59,65.35]`.

**Q15 — How strong is language-only?**

`43.80%` ordinary accuracy and `51.04%` counterfactual directional agreement;
it is not a usable shortcut.

**Q16 — Is the student still overconfident on semantic-match cross-demo states?**

Yes. Mean `|P-0.5|` is `0.1681`, above the `0.15` gate, with p90 `0.3424`.

**Q17 — Can it order semantic-difference cross-demo states?**

Only weakly: `55.31%` accuracy and AUROC `0.592`.

**Q18 — Is pairwise still better than absolute after semantic supervision?**

Yes in this audit. S2 exceeds S1 by `8.66` ordinary points and `8.27` language
points, but S2 itself remains invalid.

**Q19 — Can S1.5 replace the O(N²) direct evaluator without material loss?**

No. It trails S2 by `4.37` ordinary points and `3.26` language points, beyond
the frozen 2-point tolerances.

**Q20 — What is S2's cycle rate?**

`3.75%` over 240 preregistered triplets, below the 10% limit.

**Q21 — What is the final selected student?**

`none`. S2 remains only the frozen calibration representative.

**Q22 — May the project assign valid-student labels and `GO_REAL_CANDIDATE_VALUE_BRIDGE`?**

No. The only valid decision is:

```text
SEMANTIC_VALUE_DISTILLATION_BLOCKED
BLOCK_REAL_CANDIDATE_VALUE_BRIDGE
```
