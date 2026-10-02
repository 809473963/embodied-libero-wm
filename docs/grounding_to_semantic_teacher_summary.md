# Grounding-to-semantic-teacher decision summary

Date: 2026-10-02

The two-stage clean audit passes end to end.

Stage G replaces brittle exact source-predicate satisfaction with an explicit,
identity-free G2 semantic geometry ranking. It achieves `100%` precision and
coverage on both calibration and frozen development data, agrees with every
exact-success case, repairs tasks1/5/8, remains stable under ±1/2/5 mm source
perturbations, and yields counterfactual instructions in all ten tasks.

Stage T freezes M2 from official-goal separation rather than temporal behavior.
The selected T1-M2 teacher has held-out local non-tie density `57.91%`, including
`12.97%` regressions, and is usable on 10/10 tasks. Same-scene language changes
semantic state in `53.29%` and pair preference in `29.59%`. Both robot/semantic
controls pass on all tasks, while proprio-only reaches `80.01%`, below the 90%
severe-shortcut gate.

Final state:

```text
ROBUST_SAME_SCENE_GROUNDING_VALID
ROBUST_REFERENT_MARGIN_VALID
COUNTERFACTUAL_GROUNDING_SET_VALID
PERSISTENT_REFERENT_CONTRACT_VALID
OFFICIAL_GOAL_MARGIN_VALID
SEMANTIC_LOCAL_PROGRESS_SIGNAL_VALID
SEMANTIC_LANGUAGE_COUNTERFACTUAL_VALID
SEMANTIC_ROBOT_PHASE_INVARIANCE_SUPPORTED
SEMANTIC_GOAL_PROGRESS_TEACHER_VALID

GO_VALUE_DISTILLATION
BLOCK_WORLD_MODEL_INTERFACE_AUDIT
BLOCK_SPATIAL_WM_PROTOTYPE

Controller authorization: false
```

The GO is an authorization only. No value distillation, candidate bridge,
World Model, reranker, or controller was run.

Primary reports:

- `docs/robust_same_scene_referent_grounding_audit.md`
- `docs/t1_semantic_goal_progress_teacher_full_evaluation.md`

The prior exact-grounding blocked audit and clean-restart result remain frozen
and were not overwritten.
