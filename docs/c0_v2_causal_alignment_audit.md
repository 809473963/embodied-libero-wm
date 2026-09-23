# C0-v2 causal-alignment audit

Date: 2026-09-23

Experiment: `c0-v2-causal-alignment`

Final verdict: `CAUSAL_ALIGNMENT_IMPROVES_BASELINE`

## Executive summary

This is a single-factor experiment. C0-v1 supervises
`obs[k] -> actions[k:k+8]`; C0-v2 changes only the label offset to
`obs[k] -> actions[k+1:k+9]`. Python's half-open interval `[k+1:k+9]`
contains exactly eight actions, `a[k+1] ... a[k+8]`; neither the offline
world-model contract nor the online eight-action candidate is missing a ninth
action.

On the preregistered controller-development partition (init-state indices
30--39), C0-v1 achieved 11/100 successes over 3/10 non-zero tasks and C0-v2
achieved 45/100 over 7/10 non-zero tasks. The +34-success, +4-task result passed
all six frozen GO criteria. The C0-v2 checkpoint and interpretation criteria
were then frozen before using init-state indices 40--49. On the one-shot
confirmatory holdout, C0-v1 achieved 10/100 and C0-v2 achieved 38/100, again
covering 3/10 versus 7/10 non-zero tasks. All six precommitted acceptance checks
pass. The paired outcomes are 5 both-success, 5 C0-v1-only, 33 C0-v2-only, and
57 neither; the exact two-sided McNemar p-value is `4.25596e-6`.

D22.5 remains `BASELINE_COMPETENCE_BLOCK`, D23 remains blocked, and this work
does not modify D20-v1 or authorize D20-v2/C2-v2.

## Frozen experiment chronology

| Event | Commit | Evidence |
|---|---|---|
| D22.5 diagnosis frozen | `0d5654b` | D22 negative tag `d22-c2-overreplan-negative` retained |
| 30--39 / 40--49 protocol preregistered | `cf359fd` | `configs/c0_v2_causal_alignment_protocol.json` |
| guarded rollout evaluator | `62f4747` | refuses holdout without a committed GO decision |
| causal dataset/trainer added | `a3f7ae5` | only action-label start offset changes |
| training safety fixes | `99b530e`, `65fb087` | historical best and recoverable AMP overflow handling |
| completed checkpoint audited | `be0bfcb` | checkpoint/data/config provenance verified |
| development decision frozen | `d1cadf7` | `GO_CONFIRMATORY_HOLDOUT`; holdout not observed |
| confirmatory interpretation preregistered | `ff718ea` | acceptance mapping frozen before any complete holdout task |
| confirmatory analyzer | `8827539`, `b04bf66` | exact identity and paired-outcome checks |

The immutable rollout protocol SHA-256 is
`24ebc5f9d870a21293bf93c5129ce33376a908f9ea1f3726de0f6cc37e5c9a0c`.
Legacy indices 0--19 and D22.5 indices 20--29 are exposed historical data and
did not participate in this experiment's model or gate selection. Indices
30--39 are development data. Indices 40--49 are the one-shot confirmatory
holdout.

## Controlled change and data contract

The following remain identical to C0-v1: upright `agentview`, frozen
`openai/clip-vit-base-patch32` at revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, 512-D image embedding, 512-D
language embedding, 12-D `ee_pos[3] + gripper[2] + joint[7]` proprioception,
the 739,512-parameter MLP, horizon 8, AdamW, batch size 64, learning rate
`3e-4`, weight decay `1e-4`, 50 epochs, FP16, gradient clipping at 10, seed 11,
and demo splits 0--44 train / 45--49 validation.

The sole experimental intervention is the causal label contract:

```text
C0-v1: obs[k] -> actions[k:k+8]
C0-v2: obs[k] -> actions[k+1:k+9]
```

As required by that contract, the target-action normalization is recomputed
from the corresponding unique causal rows (`actions[1:]`) rather than reusing
C0-v1's overlapping-chunk statistics. This is a label-consistency consequence,
not an additional architecture or controller change; the causal attribution in
this report therefore means the label contract together with its required
train-only normalization.

For an episode with `T` observations, C0-v2 uses legal starts
`0 <= k <= T-9` and contributes `T-8` samples. The audited totals are exactly
52,381 train and 5,869 validation samples. Back-checks at train
`(task,demo,k) = (0,0,0), (4,17,5), (8,44,last)` and validation `(6,47,5)`
verify that the input is `obs[k]`, target row 0 is source `action[k+1]`, target
row 7 is source `action[k+8]`, the final target is in range, and no sample
crosses an episode. Train and validation episode IDs are disjoint.

The observation-aligned source store SHA-256 is
`e5599153cf44282242d955be62249bbafc752b782f5ae76a0ec0d58f9146f98e`;
its manifest SHA-256 is
`7c374d187e06c6ef99851c159134e722fa96acca9d69461bd1541e1a2895b0dc`.

## Train-only normalization

Statistics are computed once from unique source rows rather than expanding
overlapping chunks. Proprioception uses every unique observation row from the
450 train episodes (count 55,981). Actions use each episode's `actions[1:]`
exactly once, excluding `action[0]` (count 55,531). Population mean/std and a
minimum std of `1e-6` are used; validation and deployment reuse the train-only
statistics.

```text
proprio mean = [-0.02578376, 0.10606899, 1.05645103, 0.01967004,
                -0.01986377, 0.11607296, 0.63062408, 0.05883583,
                -1.65001935, -0.15051613, 2.30217306, 1.11536489]
proprio std  = [0.11198754, 0.13838893, 0.10302571, 0.01709845,
                0.01692213, 0.13876829, 0.41810828, 0.12199119,
                0.57098416, 0.20987855, 0.32549205, 0.47887201]

action mean  = [0.15043248, 0.13491899, -0.15583906, -0.00498378,
                -0.01116378, -0.01996221, 0.10107868]
action std   = [0.41536851, 0.34807537, 0.51000369, 0.03836074,
                0.07189857, 0.05811741, 0.99487844]
```

The statistics, counts, computation rules, source-store hash, and manifest
hash are embedded in both run metadata and checkpoints.

## Training audit

Training completed 50 validation epochs and 40,934 valid optimizer steps.
Sixteen recoverable AMP overflow steps were skipped; every saved metric and
checkpoint tensor is finite. Historical-best protection selected epoch 3 by
demo-validation raw action MSE, independently of any controller rollout.

| Epoch | Validation raw action MSE | Historical best |
|---:|---:|---:|
| 0 | 0.0464149874 | 0.0464149874 |
| 1 | 0.0436488320 | 0.0436488320 |
| 2 | 0.0431686086 | 0.0431686086 |
| 3 | **0.0424688490** | **0.0424688490** |
| 49 | 0.0504258709 | 0.0424688490 |

Best-epoch validation MSE by task:

| Task | MSE | Task | MSE |
|---:|---:|---:|---:|
| 0 | 0.0559907450 | 5 | 0.0307622883 |
| 1 | 0.0379334345 | 6 | 0.0594496230 |
| 2 | 0.0434417541 | 7 | 0.0456537539 |
| 3 | 0.0352387842 | 8 | 0.0521090532 |
| 4 | 0.0338614527 | 9 | 0.0303054523 |

Best checkpoint: `results/c0_v2/seed11/best.pt`

SHA-256: `2346603c8123e0a1cdee408cb635a2ffd12d6bc4e0cc5d91eacbc1330f5faac0`

Training source commit: `99b530ea232a7ed94ddc0259bda2aa03c61597a3`

Config SHA-256: `33bb274fd6413bdd2ccd15158a25272e40f2580c3f33b31feb241f3cced12897`

## Controller-development result: indices 30--39

| Task | C0-v1 | C0-v2 | Delta |
|---:|---:|---:|---:|
| 0 | 0/10 | 9/10 | +9 |
| 1 | 0/10 | 7/10 | +7 |
| 2 | 0/10 | 7/10 | +7 |
| 3 | 0/10 | 8/10 | +8 |
| 4 | 3/10 | 0/10 | -3 |
| 5 | 0/10 | 5/10 | +5 |
| 6 | 4/10 | 5/10 | +1 |
| 7 | 0/10 | 0/10 | 0 |
| 8 | 4/10 | 4/10 | 0 |
| 9 | 0/10 | 0/10 | 0 |
| **All** | **11/100** | **45/100** | **+34** |

| Aggregate | C0-v1 | C0-v2 | Delta |
|---|---:|---:|---:|
| Macro success | 11% | 45% | +34 pp |
| Non-zero tasks | 3/10 | 7/10 | +4 tasks |
| Mean steps | 281.88 | 210.58 | -71.30 |
| Planned-action clipping | 2.488% | 4.002% | +1.514 pp |

All preregistered development checks passed: C0-v2 successes at least 20,
non-zero tasks at least 5, at least +10 successes and +2 non-zero tasks over
C0-v1, finite actions/metrics, and clipping no greater than 10%. The GO decision
and C0-v2 checkpoint were committed before holdout use. Development already
shows that improvement is not uniform: task 4 regresses, while tasks 7 and 9
remain at floor.

## One-shot confirmatory holdout: indices 40--49

Both arms completed exactly 100 episodes. The independent analyzer verified the
partition, all task/episode identities, seeds, init-state indices, protocol
hash, checkpoint hashes, finite values, and completion markers before applying
the precommitted acceptance criteria.

| Task | C0-v1 | C0-v2 | Delta |
|---:|---:|---:|---:|
| 0 | 0/10 | 8/10 | +8 |
| 1 | 0/10 | 6/10 | +6 |
| 2 | 0/10 | 3/10 | +3 |
| 3 | 0/10 | 5/10 | +5 |
| 4 | 2/10 | 0/10 | -2 |
| 5 | 0/10 | 5/10 | +5 |
| 6 | 7/10 | 5/10 | -2 |
| 7 | 0/10 | 0/10 | 0 |
| 8 | 1/10 | 6/10 | +5 |
| 9 | 0/10 | 0/10 | 0 |
| **All** | **10/100** | **38/100** | **+28** |

| Aggregate | C0-v1 | C0-v2 | Delta |
|---|---:|---:|---:|
| Macro success | 10% | 38% | +28 pp |
| Non-zero tasks | 3/10 | 7/10 | +4 tasks |
| Mean steps | 284.09 | 225.62 | -58.47 |
| Planned-action clipping | 2.713% | 4.119% | +1.406 pp |

All acceptance checks pass: C0-v2 has at least 20 successes (38), at least 5
non-zero tasks (7), improves by at least 10 successes (+28) and 2 non-zero
tasks (+4), contains only finite actions/metrics, and keeps clipping below 10%
(4.119%). At the paired episode level, the counts are:

```text
both success = 5
C0-v1 only   = 5
C0-v2 only   = 33
neither      = 57
exact two-sided McNemar p = 4.255962267052382e-06
```

Result SHA-256 values:

- C0-v1 raw holdout: `a9086a524cc0eda1663e38e0045ef82f38f219928d7b245f77230d92a3ebdfef`
- C0-v2 raw holdout: `6bcf619618dcd8327a35f8a2e480206f9fb7cfc932653ffd3ae9646bdb245652`
- confirmatory comparison: `ed2576d41bb4b02abda6158bea122ec4969354a933c2610ecf1a7cdcea6b494c`

The confirmatory result replicates the development direction, breadth, and
remaining weaknesses. It is not uniform: C0-v2 still fails tasks 7 and 9,
regresses on tasks 4 and 6 on this 10-episode partition, and has a higher but
well-within-gate clipping ratio.

## Answers to the experimental questions

1. **Does causal label alignment improve baseline competence?** Yes within this
   controlled C0 setup: +34/100 on development and +28/100 on the untouched
   confirmatory holdout, with 33 C0-v2-only versus 5 C0-v1-only paired wins.
2. **Is success still limited to a few tasks?** It is broader (7 rather than 3
   non-zero tasks on both partitions), but not uniform: tasks 7 and 9 remain
   zero; confirmatory tasks 4 and 6 regress relative to C0-v1.
3. **How many tasks have non-zero success?** Development: C0-v1 3/10, C0-v2
   7/10. Confirmatory: C0-v1 3/10, C0-v2 7/10.
4. **Was confirmatory holdout warranted?** Yes. All frozen development GO checks
   passed before indices 40--49 were viewed. The holdout is not available for
   further tuning.

## Scope and invariants

- C0-v1 checkpoints/results, D13/D14/D15 world-model checkpoints, D20-v1, and
  D21/D22 negative artifacts remain immutable.
- No fixed legacy 0--19 or D22.5 20--29 outcome selected this model or its gate.
- No world-model causal contract, C1/C2 controller, threshold, architecture,
  camera, or horizon was changed.
- D20-v2 threshold design, C2-v2, and D23 were not run.
- This controlled experiment supports an attribution to label alignment within
  this exact C0 setup; it does not prove that every task or controller stack is
  solved.

## Verification and repository status

The following eight executable checks pass on 2026-09-23:

- `tests/test_c0_v2_protocol.py`
- `tests/test_c0_v2_confirmatory_protocol.py`
- `tests/test_c0_v2_dataset.py`
- `tests/test_c0_v2_training.py`
- `tests/test_c0_v2_checkpoint.py`
- `tests/test_c0_causal_alignment_evaluator.py`
- `tests/test_c0_v2_development_gate.py`
- `tests/test_c0_v2_confirmatory_analysis.py`

They cover protocol disjointness, exact sample counts and indices, unique-row
normalization, real-batch forward/backward/optimizer behavior, finite checkpoint
history and provenance, holdout locking, development boundaries, and paired
confirmatory accounting. `git diff --check` also passes. Large HDF5 stores,
checkpoints, caches, and rollout logs remain ignored; only source, protocols,
compact JSON evidence, and documentation are committed.
