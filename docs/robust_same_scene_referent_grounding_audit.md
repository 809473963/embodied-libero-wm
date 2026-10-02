# Normalized source geometry robustly grounds same-scene referents

Date: 2026-10-02

Final mechanism label: `SAME_SCENE_LANGUAGE_GROUNDING_VALID`

Next-stage authorization: `GO_T1_SEMANTIC_TEACHER_FULL_EVALUATION`

Value distillation authorization: `false`

## Executive answer

The independently preregistered Stage G audit resolves the brittle exact-predicate
failure. Calibration selected G2, a lower-is-better normalized semantic geometry
score that uses no task ID, bowl suffix, or known construction identity. With its
frozen zero gap threshold, G2 is uniquely correct in all 40 calibration episodes
and all 40 frozen development episodes. Precision and coverage are both `100%`,
all ten tasks have `100%` coverage, and agreement on the subset accepted by the
installed exact predicate is `100%`.

The held-out mean/median best-versus-second score gaps are `17.124/10.446`.
Shifting each source anchor by ±1, ±2, and ±5 mm on every axis changes none of
the 720 held-out correct-instruction decisions. Alternate real instructions
produce 271 eligible held-out cases across 10/10 source tasks; `53.51%` select
the other bowl. Alternate-instruction identity is never scored against a hidden
label.

The exact access guard opened only demo31--34 `states` during calibration and
demo36--39 `states` during evaluation. Demo40--49 have zero numeric and zero
metadata access. Stage G therefore authorizes only the separately frozen T1
semantic-teacher evaluation. No value model, World Model, reranker, candidate
future, or controller was run.

## Geometry contract

For site/region relations, bowl centers are expressed in the source site's
rotated local frame. `On` uses horizontal region half-sizes plus a vertical
support band; `In` uses the region volume. For object supports, G2 combines
horizontal center distance normalized by summed object radii with the absolute
orientation-aware bowl-bottom/support-top gap normalized by object height.

The score is explicit and deterministic. The resolver sorts the two bowl scores,
requires the best score to be finite and within the frozen family maximum, and
requires the second-minus-first confidence gap to be strictly above `0.0`.
Equality is therefore ambiguous; no tie is resolved by object name.

## Frozen selection and results

| Partition | Episodes | Precision | Coverage | Tasks >=75% | Exact-subset agreement | Mean gap | Median gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibration demo31--34 | 40 | 100% | 100% | 10/10 | 100% | 17.055 | 11.159 |
| Frozen development demo36--39 | 40 | 100% | 100% | 10/10 | 100% | 17.124 | 10.446 |

Calibration selected G2 lexicographically by unique precision, coverage, task
coverage, exact-subset agreement, confidence, and family simplicity. T1 signal
or any temporal-progress metric was not available to this choice.

| Diagnostic | Calibration | Evaluation |
|---|---:|---:|
| Wrong unique decisions | 0 | 0 |
| Ambiguous / not found | 0 / 0 | 0 / 0 |
| Eligible alternate cases | 270 | 271 |
| Source tasks with an eligible alternate | 10/10 | 10/10 |
| Alternate referent-switch rate | 52.59% | 53.51% |
| ±1/2/5 mm decision stability | 100% | 100% |

The switch rate is descriptive rather than an accuracy score: counterfactual
instructions intentionally have no construction-identity ground truth.

## Access and mechanical decision

Both ledgers contain only the exact allowlisted `states` datasets. Calibration
opened demos31--34; evaluation opened demos36--39. Neither ledger records an
existence probe, group enumeration, other metadata operation, forbidden attempt,
or demo40--49 identifier.

| Gate | Calibration | Evaluation |
|---|---|---|
| Unique precision >=95% | pass | pass |
| Coverage >=90% | pass | pass |
| >=9 tasks with coverage >=75% | pass | pass |
| Exact-success agreement >=95% | pass | pass |
| Eligible counterfactuals in >=7 tasks | pass: 10 | pass: 10 |
| Finite and forbidden access zero | pass | pass |

Final result: `SAME_SCENE_LANGUAGE_GROUNDING_VALID`.

Authorization: `GO_T1_SEMANTIC_TEACHER_FULL_EVALUATION` only.

## Required answers

**GQ1 — What failed in exact grounding?** Strict sites and contact predicates
reject millimeter-scale restored-state offsets; they are brittle Boolean
initialization checks rather than robust referent rankings.

**GQ2 — What resolver was selected?** G2 normalized semantic compatibility.

**GQ3 — Does it use object identity?** No. Bowl identity is used only after
prediction to score the correct real instruction.

**GQ4 — How are region relations handled?** In the rotated source-site frame,
using explicit region half-sizes or containment volume.

**GQ5 — How are support relations handled?** With horizontal support geometry
and an orientation-aware surface-height gap; simple anchor proximity is not used.

**GQ6 — What are the held-out precision and coverage?** Both are `100%` over
40 episodes and ten tasks.

**GQ7 — Is the result stable to small geometry shifts?** Yes. Every frozen
correct-instruction decision survives all ±1/2/5 mm source-anchor shifts.

**GQ8 — Do real alternate instructions remain usable?** Yes. Every source task
has eligible held-out alternates; 271 cases are admitted.

**GQ9 — Was the forbidden partition untouched?** Yes, including metadata.

**GQ10 — What is authorized?** Only the preregistered full T1 semantic-teacher
evaluation. Value distillation and all downstream modeling remain blocked.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| Protocol | `cf5a07f23401640515e0d0226d416b5efedc91bc2c66fbddbbf3442919ea80c6` |
| Geometry contract | `982daa94bb27cae1749a2196d127e7b0d437f719676695cbbf65aad1a9cbc8c8` |
| Frozen resolver | `337ebe063792402369e779bcec4800cb0ce542ea10fd8992eb8f66796dd56881` |
| Calibration result | `59f1476fb8f2590c2be9299c6e9a7e8baf9d64aa444a93b9ab54285044322d98` |
| Evaluation result | `0738f6756b24af87b1b6c4a51fdbb1a2768ca0820c5419d6d55f27a10c3b7c63` |
| Calibration sidecar | `19252fdd297fa1d791f3a60117a629650a413a39a76fb52c073d79121ebf8fe4` |
| Evaluation sidecar | `dbc64847ebcc42c862a682d26a6d29a764f276dcf1c9da6e3e7eb27596e42ce7` |

## Forbidden-activity audit

No reward, done, success, task outcome, image, action, value distillation,
temporal V2 repair, candidate future, World Model, learned proposal, reranker,
score, controller, CEM, MPPI, or beam search was accessed or run.
