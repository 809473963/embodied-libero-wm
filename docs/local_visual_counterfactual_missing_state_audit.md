# The tested structured state variables track visual branching but do not explain it out of sample

Date: 2026-09-27

Final mechanism labels:

```text
ROBOT_KINEMATICS_ONLY_INSUFFICIENT
NO_TESTED_STATE_EXPLAINS_VISUAL_COUNTERFACTUAL
```

Next-stage authorization: `BLOCK_STRUCTURED_PHYSICAL_WM_ROUTE`

Controller authorization: `false`

## Executive answer

The real production-scale A0/A1/A2 visual counterfactual differences are not
explained sufficiently by EE position and joints, but this audit also cannot
attribute the missing part to EE orientation, object motion, contact
branching, or EE-object relative geometry under the preregistered held-out
gates.

Several variables are genuinely associated with visual separation. On the
held-out init36--39 branches, Spearman correlation with real CLIP separation
is `0.315` for EE rotation, `0.234` for maximum object translation, and
`0.441` for maximum relative translation. Candidate rotational-action
difference also strongly predicts real EE rotation (Spearman `0.776`). These
associations are not enough to identify a missing mechanism: the corresponding
nested ridge improvements are only `2.50%`, `0.52%`, `-0.08%`, and `1.41%`,
all below the frozen 5% threshold, and none of the four held-out group
shuffles has a positive 95% CI lower bound.

The strongest full S4 probe reaches RMSE `0.003172` and R² `0.0758`, versus
`0.003311` for the calibration-mean constant. Its apparent gain is spread
across correlated state variables and does not survive the required
incremental/shuffle attribution. Contact branching is particularly weak:
contact-different rows have mean visual separation `0.004769` versus
`0.004561` for contact-same rows, but the branch-bootstrap mean-difference CI
is `[-0.000528, 0.000953]`.

The scientifically supported conclusion is therefore narrower than “these
states do not matter.” They covary with local visual change, especially
relative translation and rotation, but this experiment finds no stable,
non-redundant held-out evidence that any tested group explains the residual
visual counterfactual. The structured-physical route remains blocked; the
next diagnosis should reconsider the CLIP latent or prediction target rather
than adding another physical adapter from these results.

## Frozen provenance and corrections

The upstream physical/visual sufficiency failure was frozen at commit
`c561e86198f70804d08e5734bbe713a893fa4335` under annotated tags
`counterfactual-physical-visual-probe-negative` and
`physical-visual-sufficiency-negative`. The new protocol was preregistered at
`ffd9627f11058e2bac86e2dc67d93d06a7a664e5` before state inventory or branch
replay.

The protocol transitively hash-checks C0-v2, the frozen C1-v2.1 candidate
generator and production scale, D13/D14/D15, WM-v2A, WM-v2B, WM-v2B.1,
WM-v2C-EJ, the matched branch data, the prior sufficiency probe, and their
reports/results/checkpoints. No upstream artifact was overwritten.

Two implementation-only stops are retained:

- The first calibration command stopped inside policy construction before an
  environment was created because it referenced the C1-v2 protocol JSON
  instead of the frozen deployment YAML. Protocol v1.1 corrected only that
  path and was committed as `f4efb7304f4b593384bb740881d2a0b53909d6e8`.
- The first evaluation-analysis command reached the mandatory finite-value
  guard before writing a diagnosis because SciPy represented a constant-input
  correlation as NaN. Amendment v1.2, committed as
  `fba942339b0161047f5b357f707c0f26e03bc4f8`, serializes undefined point
  correlations as `null` and constant bootstrap correlations as zero. It did
  not change features, data, fits, weights, thresholds, bootstrap draws,
  gates, or decision mapping; evaluation state was not replayed or used for
  refitting.

## Split, branch identity, and one-shot boundary

| Partition | Init states | Tasks | Plan | Prefix | Branch points | Pair/horizon rows | Use |
|---|---|---:|---:|---:|---:|---:|---|
| Calibration | 31--34 | 10 | 8 | 64 | 40 | 960 | CV, fit, standardization, object bins |
| Held-out evaluation | 36--39 | 10 | 8 | 64 | 40 | 960 | One-shot metrics only |
| Historical | 30/35 | 10 | 0/8 | mixed | 40 | — | Reference only; not fitted |
| Forbidden | 40--49 | — | — | — | 0 | 0 | Never accessed |

Every candidate branch used a fresh environment, the official init state,
deterministic replay of the same 64 raw prefix actions, and one copied frozen
candidate. Mid-episode simulator-state restore was not used. All 80 new
branch points passed:

- bitwise-equal prefix actions, planning RGB, and observable arrays across
  A0/A1/A2 branches;
- zero maximum proprio difference and simulator-state difference within the
  frozen `1e-10` tolerance;
- common planning CLIP cosine at least `0.999999`;
- exact regenerated candidate tensors;
- bitwise equality of all 3×8 future RGB frames and zero 12-D proprio
  difference against the already frozen branch sidecars.

Candidate identities and hashes are retained in the two compact manifests.
The ignored state sidecars contain no reward, done, success, or task outcome.

## Runtime state inventory

The inventory was performed first on official init31 for all ten tasks and is
stored in `results/missing_state_audit/state_inventory.json`. All tasks expose
one identical 31-key observation schema. Robot keys are:

```text
robot0_eef_pos, robot0_eef_quat,
robot0_gripper_qpos, robot0_gripper_qvel,
robot0_joint_pos, robot0_joint_pos_cos, robot0_joint_pos_sin,
robot0_joint_vel, robot0_proprio-state
```

Object observations expose position, quaternion, object-to-EEF position, and
object-to-EEF quaternion for every scene object plus `object-state`. The
analysis does not select objects from names or language. For every task,
`parsed_problem["obj_of_interest"]` resolves at runtime to the two relevant
objects `akita_black_bowl_1` and `plate_1`; those exact models, root bodies,
body IDs, and contact geom sets are recorded in the inventory.

Verified simulator fields are:

| Runtime field | Shape | dtype | Meaning | Observation-accessible | Simulator-only | Future use is oracle |
|---|---:|---|---|---|---|---|
| `sim.data.body_xpos` | [38,3] | float64 | Body world positions | no | yes | yes |
| `sim.data.body_xquat` | [38,4] | float64 | Body world quaternion, wxyz | no | yes | yes |
| `sim.data.site_xpos` | [31,3] | float64 | Site world positions | no | yes | yes |
| `sim.data.site_xmat` | [31,9] | float64 | Site world rotation matrices | no | yes | yes |
| `sim.data.qpos` | [48] | float64 | Generalized positions | no | yes | yes |
| `sim.data.qvel` | [43] | float64 | Generalized velocities | no | yes | yes |
| `sim.data.contact[:ncon]` | variable | MuJoCo contact | Active geom contacts | no | yes | yes |

The EEF is verified as site `gripper0_grip_site` (site ID 13); both the
observation quaternion and simulator rotation matrix exist. Robot-model and
gripper contact geom sets exist, relevant object contact geoms exist, and
`_check_grasp` is callable for every task. Contact is derived only from actual
MuJoCo geom pairs and `_check_grasp`; no distance threshold is used as a
contact proxy.

All four new slots are therefore available. Future EE/object/contact/relative
values remain diagnostic oracles even when a corresponding quantity is also
observable at the current time.

## Targets and feature hierarchy

For each branch point, candidate pair, and horizon:

```text
D_visual = 1 - cosine(z_real_i[h], z_real_j[h])
delta_z  = z_real_i[h] - z_real_j[h]
```

The held-out `D_visual` distribution has mean `0.004575`, median `0.003702`,
p75 `0.006012`, p90 `0.009045`, p95 `0.011218`, and maximum `0.021249`.

| Group | Dimensions | Exact construction |
|---|---:|---|
| F_EJ | 10 | Signed train-standardized pair difference: EE position 3 + joints 7; gripper excluded |
| F_ROT | 4 | Relative rotvec 3 + rotation geodesic angle; rotation matrices remove q/-q ambiguity |
| F_OBJECT | 10 | Mean object translation/rotvec plus max/mean translation and rotation magnitudes; object-wise poses retained |
| F_CONTACT | 13 | Four actual contact-state splits, any split, four onset differences, four duration differences |
| F_REL | 10 | Object-minus-EE relative position/orientation differentials and max/mean magnitudes |

The fixed nesting is S0=EJ, S1=+ROT, S2=+OBJECT, S3=+CONTACT, and S4=+REL,
with dimensions 10/14/24/37/47. Quaternion geodesic uses
`2*acos(abs(dot(q_i,q_j)))`; Euler subtraction is never used.

## Calibration was frozen before held-out state replay

All continuous columns use calibration-only mean/std. Ridge alpha was chosen
by leave-one-init-out CV within 31--34 from
`[1e-4,1e-3,1e-2,1e-1,1,10]`; exact ties favor larger alpha.

| Set | Dim. | Scalar α | Best scalar CV RMSE | 512-D α |
|---|---:|---:|---:|---:|
| S0 | 10 | 10 | 0.003071 | 10 |
| S1 | 14 | 1 | 0.002965 | 10 |
| S2 | 24 | 1 | 0.002931 | 10 |
| S3 | 37 | 1 | 0.002968 | 10 |
| S4 | 47 | 0.1 | 0.002859 | 10 |

The object-motion max-translation bin edges were frozen from calibration at
`0.0005253 m` and `0.0019967 m`. The calibration weights SHA-256 is
`3a0f0820e52c0e81b45857fdc764236ac18668d547cf06d384444494b9f6a2cd`.

## Raw state associations are real but not sufficient attribution

| State magnitude ↔ D_visual | Pearson | Spearman | Branch-bootstrap Spearman 95% CI |
|---|---:|---:|---:|
| EE translation | 0.347 | 0.435 | [0.337, 0.525] |
| EE rotation | 0.257 | 0.315 | [0.225, 0.405] |
| Candidate action rotation | 0.316 | 0.359 | [0.277, 0.445] |
| Object translation max | 0.195 | 0.234 | [0.101, 0.362] |
| Object rotation max | 0.168 | 0.150 | [0.005, 0.301] |
| Contact different | 0.015 | 0.029 | [-0.025, 0.078] |
| Relative translation max | 0.354 | 0.441 | [0.342, 0.533] |
| Relative rotation max | 0.212 | 0.249 | [0.142, 0.357] |

Candidate rotational-action difference strongly maps to real EE orientation
difference (Pearson `0.803`, Spearman `0.776`, bootstrap Spearman CI
`[0.718,0.831]`). Thus the first two links of the requested rotation chain
hold: action rotation changes real EE orientation, and real EE rotation is
associated with visual separation. The third link fails because S1 improves
S0 by only 2.50%, below the frozen 5% gate, and rotation shuffling does not
worsen held-out prediction.

The complete JSON includes Pearson/Spearman cuts for h1--h8, task0--9, and all
three candidate pairs for all twelve recorded scalar summaries. Selected
Spearman cuts are shown below.

| h | EE trans. | EE rot. | Object trans. | Object rot. | Contact split | Relative trans. | Relative rot. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.558 | 0.295 | 0.289 | 0.063 | -0.145 | 0.550 | 0.198 |
| 2 | 0.525 | 0.359 | 0.244 | 0.208 | 0.015 | 0.521 | 0.267 |
| 3 | 0.236 | 0.311 | 0.042 | -0.065 | 0.060 | 0.234 | 0.088 |
| 4 | 0.383 | 0.145 | 0.252 | 0.067 | 0.086 | 0.406 | 0.075 |
| 5 | 0.394 | 0.188 | 0.108 | -0.058 | -0.018 | 0.348 | 0.049 |
| 6 | 0.385 | 0.132 | 0.043 | -0.090 | 0.024 | 0.374 | 0.031 |
| 7 | 0.210 | 0.109 | 0.039 | 0.057 | 0.092 | 0.253 | 0.068 |
| 8 | 0.119 | -0.001 | 0.109 | 0.100 | 0.190 | 0.167 | 0.043 |

| Task | EE trans. | EE rot. | Object trans. | Object rot. | Contact split | Relative trans. | Relative rot. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.463 | 0.141 | 0.434 | 0.143 | undefined | 0.443 | 0.084 |
| 1 | 0.620 | 0.416 | 0.558 | 0.428 | 0.207 | 0.640 | 0.583 |
| 2 | 0.512 | 0.424 | 0.520 | 0.383 | 0.082 | 0.517 | 0.381 |
| 3 | 0.410 | 0.272 | 0.408 | 0.273 | undefined | 0.414 | 0.320 |
| 4 | 0.506 | 0.538 | 0.309 | 0.188 | 0.230 | 0.516 | 0.543 |
| 5 | 0.414 | 0.284 | -0.296 | -0.305 | 0.087 | 0.416 | 0.109 |
| 6 | 0.448 | 0.170 | 0.364 | 0.171 | -0.242 | 0.406 | 0.134 |
| 7 | 0.537 | 0.691 | 0.542 | 0.506 | 0.100 | 0.574 | 0.651 |
| 8 | 0.571 | 0.243 | 0.527 | 0.527 | undefined | 0.586 | 0.425 |
| 9 | 0.593 | 0.594 | 0.515 | 0.501 | 0.078 | 0.601 | 0.626 |

| Pair | EE trans. | EE rot. | Object trans. | Object rot. | Contact split | Relative trans. | Relative rot. |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0-A1 | 0.392 | 0.271 | 0.209 | 0.138 | -0.012 | 0.391 | 0.223 |
| A0-A2 | 0.411 | 0.257 | 0.208 | 0.079 | 0.014 | 0.416 | 0.166 |
| A1-A2 | 0.386 | 0.261 | 0.206 | 0.146 | 0.072 | 0.412 | 0.223 |

## Contact and object-motion conditioning

Actual contact-state differences occur in 62/960 pair/horizon rows. The
remaining 898 rows have the same aggregate contact state. Their visual
distributions are:

| Contact state | n | Mean | Median | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| Same | 898 | 0.004561 | 0.003656 | 0.006004 | 0.009109 |
| Different | 62 | 0.004769 | 0.004340 | 0.006060 | 0.008468 |

The mean difference is `+0.000208`, but its 95% CI crosses zero
(`[-0.000528,0.000953]`). Contact-different samples therefore do not show the
required stable elevation.

The calibration-frozen object-motion bins show a monotonic descriptive trend:

| Max object translation | n | Mean D_visual | Median | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| Low | 351 | 0.003829 | 0.002991 | 0.005381 | 0.007463 |
| Medium | 325 | 0.004524 | 0.003537 | 0.006017 | 0.009078 |
| High | 284 | 0.005554 | 0.004444 | 0.006884 | 0.010512 |

This association is not enough for the object-driver label because S2 adds
only 0.52% over S1 and the object shuffle CI crosses zero.

## Held-out scalar sufficiency and nested increments

| Probe | Dim. | RMSE | MAE | R² | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Calibration-mean constant | 1 | 0.003311 | 0.002438 | -0.0072 | undefined | undefined |
| S0 EE+joints | 10 | 0.003314 | 0.002441 | -0.0093 | -0.036 | -0.014 |
| S1 + EE orientation | 14 | 0.003231 | 0.002368 | 0.0406 | 0.213 | 0.290 |
| S2 + object motion | 24 | 0.003215 | 0.002339 | 0.0506 | 0.243 | 0.349 |
| S3 + contact | 37 | 0.003217 | 0.002337 | 0.0492 | 0.241 | 0.335 |
| S4 + relative geometry | 47 | 0.003172 | 0.002259 | 0.0758 | 0.319 | 0.431 |

| Increment | Relative RMSE improvement | Branch-bootstrap 95% CI | Frozen 5% + positive-CI gate |
|---|---:|---:|---|
| S0 vs constant | -0.10% | [-0.34%, 0.13%] | fail |
| S1 vs S0 | +2.50% | [0.66%, 4.41%] | fail: below 5% |
| S2 vs S1 | +0.52% | [-0.90%, 1.89%] | fail |
| S3 vs S2 | -0.08% | [-1.24%, 1.44%] | fail |
| S4 vs S3 | +1.41% | [-1.43%, 4.15%] | fail |

S0 does not outperform the calibration-mean constant, which mechanically
assigns `ROBOT_KINEMATICS_ONLY_INSUFFICIENT`.

## Group shuffle controls reject unique attribution

Each new group was deranged only across branch points with the same task,
horizon, and candidate pair. Parent features and targets stayed fixed.

| Group shuffled in its child set | Original RMSE | Shuffled RMSE | Relative degradation | 95% CI | Pass |
|---|---:|---:|---:|---:|---|
| EE orientation in S1 | 0.003231 | 0.003226 | -0.17% | [-2.83%, 2.53%] | no |
| Object motion in S2 | 0.003215 | 0.003238 | +0.72% | [-1.27%, 2.67%] | no |
| Contact in S3 | 0.003217 | 0.003187 | -0.92% | [-2.74%, 1.66%] | no |
| Relative geometry in S4 | 0.003172 | 0.003228 | +1.76% | [-1.77%, 5.01%] | no |

The higher-dimensional sets therefore cannot claim success merely from more
columns. Every candidate mechanism fails either the 5% nested-increment gate,
the group-shuffle gate, or both.

## Per-horizon, task, and pair scalar evidence

Held-out RMSE for the same calibration-frozen probes:

| h | S0 | S1 | S2 | S3 | S4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.002975 | 0.002704 | 0.002600 | 0.002667 | 0.002437 |
| 2 | 0.002547 | 0.002439 | 0.002387 | 0.002429 | 0.002392 |
| 3 | 0.003489 | 0.003406 | 0.003421 | 0.003440 | 0.003436 |
| 4 | 0.002941 | 0.002910 | 0.002854 | 0.002837 | 0.002811 |
| 5 | 0.003951 | 0.003820 | 0.003802 | 0.003806 | 0.003655 |
| 6 | 0.002856 | 0.002877 | 0.002908 | 0.002946 | 0.002856 |
| 7 | 0.003791 | 0.003760 | 0.003787 | 0.003709 | 0.003706 |
| 8 | 0.003686 | 0.003634 | 0.003620 | 0.003603 | 0.003722 |

| Task | S0 | S1 | S2 | S3 | S4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.003091 | 0.003161 | 0.003093 | 0.003104 | 0.003073 |
| 1 | 0.002825 | 0.002894 | 0.002761 | 0.002748 | 0.002698 |
| 2 | 0.002010 | 0.001798 | 0.001777 | 0.001788 | 0.002046 |
| 3 | 0.002929 | 0.002869 | 0.002805 | 0.002822 | 0.002812 |
| 4 | 0.004067 | 0.003719 | 0.003897 | 0.003846 | 0.003650 |
| 5 | 0.002410 | 0.002410 | 0.002632 | 0.002681 | 0.002547 |
| 6 | 0.005844 | 0.005874 | 0.005831 | 0.005831 | 0.005878 |
| 7 | 0.003325 | 0.003009 | 0.002942 | 0.002942 | 0.002828 |
| 8 | 0.002399 | 0.002448 | 0.002336 | 0.002245 | 0.002198 |
| 9 | 0.002525 | 0.002325 | 0.002212 | 0.002338 | 0.002140 |

| Pair | S0 | S1 | S2 | S3 | S4 |
|---|---:|---:|---:|---:|---:|
| A0-A1 | 0.002764 | 0.002728 | 0.002732 | 0.002763 | 0.002762 |
| A0-A2 | 0.003013 | 0.002962 | 0.002937 | 0.002964 | 0.002893 |
| A1-A2 | 0.004029 | 0.003887 | 0.003861 | 0.003824 | 0.003766 |

Performance varies by task and horizon rather than showing one stable group
increment. In particular, S4 helps some tasks but regresses task 2 and does
not improve h8, consistent with the aggregate shuffle uncertainty.

## Secondary 512-D direction probes were correctly withheld

The protocol permits a multi-output ridge from a new feature set to
`delta_z_real` only after that extension clears its scalar held-out gate. None
does, so `secondary_direction_probes` is empty. Calibration-frozen secondary
weights exist only to avoid post-evaluation hyperparameter fitting; they were
not used to claim a visual-direction mechanism.

## Mechanical decision

| Candidate mechanism | Required evidence | Result | Label |
|---|---|---|---|
| EE orientation | action→rotation association, rotation→visual association, S1 ≥5% over S0 | first two pass; S1 gain 2.50%; rotation shuffle fails | not assigned |
| Object dynamics | S2 ≥5%, significant object shuffle, positive association across tasks | association passes; gain 0.52%; shuffle fails | not assigned |
| Contact branching | contact-different visual elevation, S3 ≥5%, contact shuffle | all three fail | not assigned |
| Relative geometry | S4 ≥5%, significant relative shuffle, positive association | association passes; gain 1.41%; shuffle fails | not assigned |
| Robot kinematics alone | S0 ≥5% over constant with positive CI | -0.10%; CI crosses zero | `ROBOT_KINEMATICS_ONLY_INSUFFICIENT` |

Because no added group passes its complete preregistered gate, the second
label is `NO_TESTED_STATE_EXPLAINS_VISUAL_COUNTERFACTUAL`; this is not a
`MULTI_FACTOR_VISUAL_COUNTERFACTUAL` result. The next-stage authorization is
`BLOCK_STRUCTURED_PHYSICAL_WM_ROUTE`.

This answers the central question: production-scale real visual branching is
not explained by robot EE/joint motion alone, but the held-out evidence also
does not isolate omitted EE orientation, object motion, contact branching, or
relative geometry as a sufficient non-redundant cause. Their raw associations
likely reflect shared motion/interaction intensity, while the residual may
depend on visual representation details, untested scene variables, nonlinear
interactions beyond this deliberately low-capacity probe, or the prediction
target itself. Those are hypotheses, not results from this audit.

## Mechanism variables are not deployment inputs

CLIP `z0` can already contain information about objects in the current image.
This audit asks a different question: after the same current state, which
actual future state variables diverge when candidates differ, and whether
that divergence tracks the real visual future. Simulator future object pose,
contact, and relative geometry are used only to identify such mechanisms.
They are unavailable at decision time and cannot be passed directly to a
deployable World Model or controller. In short, a mechanism variable is not a deployable World Model or controller input. A positive future experiment would
first need to test whether a state can be predicted from current observation
plus actions; this audit authorizes no such design because no group passed.

## Integrity, artifacts, and forbidden activity

| Artifact | SHA-256 |
|---|---|
| Protocol | `d515f4071d2c729621c43a471f33408ab9d3f881d91a9306135384e1e5db6708` |
| State inventory | `13a3f7635dbd45bb1987e045028b27d1dbb6f2a495f0f9364479c86030dbfc3c` |
| Calibration state sidecar | `b1f86ea8f3819b31d54058d8912160fa17507bc0ed3454cc8d89ad33e4f64ca8` |
| Evaluation state sidecar | `0fe5fe70cd1463886c2cbc75f2ba8181922103fe939b110d5a97249b31f345af` |
| Calibration weights | `3a0f0820e52c0e81b45857fdc764236ac18668d547cf06d384444494b9f6a2cd` |
| Evaluation derived metrics | `6367c0a748d1175497ee50ef8639588b050debc2f7d0b7d04ae62e238fa826ed` |
| Compact diagnosis | `382309ac8c00e617a158281e455ba6a0b8b545ed4edfd59a8bd4ea20859bbbe9` |

All collected numeric arrays, predictions, residuals, correlations admitted
to the result, and bootstrap statistics are finite. Undefined constant-input
correlations are explicit `null`, never fabricated as evidence.

No World Model, CLIP encoder, C0 actor, candidate generator, production noise,
reranker, controller, or seed23/47 model was trained or modified. No C1/C2,
D20-v2, controller success, reward, done, task outcome, fake counterfactual
target, mid-episode restore, or init40--49 access occurred. Existing reports,
results, checkpoints, and sidecars remain unchanged.
