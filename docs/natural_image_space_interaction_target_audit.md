# Natural image-space interaction target audit stops at the frozen camera gate

Date: 2026-09-30

Final mechanism verdict: `DATA_OR_IMPLEMENTATION_BLOCK`

Blocking condition: `CAMERA_PROJECTION_BLOCK`

Next-stage authorization: `BLOCK_VISUAL_TARGET_REDESIGN`

Controller authorization: `false`

## Executive answer

This audit stopped before constructing any image-space target. The
calibration-only camera inventory selected the only geometrically plausible
projection convention, but the selected convention did not clear every
preregistered authorization check. Its projected EE point is close to the
natural robot mask (median `1.0948 px`, p95 `1.4111 px`), while the alternative
has median `3.7477 px` and p95 `12.3070 px`. The selected convention therefore
passes both absolute-distance limits. Its median advantage over the alternative
is only `2.6529 px`, however, below the frozen `4 px` margin requirement.

The simulator's visible `gripper0_grip_site` segmentation provides a strong
secondary sanity check: selected-convention site distances are
`0.085--0.626 px`, versus `59.40--68.23 px` for the alternative. That evidence
cannot retroactively replace or lower the committed gate after its result has
been viewed. The mechanical result is therefore `CAMERA_PROJECTION_BLOCK`,
mapped to the allowed final verdict `DATA_OR_IMPLEMENTATION_BLOCK`.

No per-object mask replay, 2D target construction, ridge fitting, bootstrap,
shuffle analysis, or init36--39 evaluation was performed. The experiment does
not answer whether patches uniquely preserve natural object or interaction
geometry. The upstream conclusions remain frozen:

```text
GLOBAL_CLIP_SPATIAL_COMPRESSION
NATURAL_PATCH_TASK_RELEVANCE_UNRESOLVED
BLOCK_VISUAL_TARGET_REDESIGN
```

## Frozen provenance

The upstream natural-spatial audit was frozen at commit
`9d4ac3cf89ebff35a161ea8a337fc569c89affd2` under annotated tag
`natural-patch-relevance-unresolved`. Protocol, target definitions, camera
gate, split, bootstrap, ridge forms, shuffle rules, decision mapping, and
implementation were committed as
`ba419ca096a6c8794095ee5fb00bfe23b6458dba` before camera inventory.

The protocol transitively hash-checks C0-v2, the frozen C1-v2.1 candidate
generator and production scale, D13/D14/D15, WM-v2A/B/B.1/C-EJ, the
structured-state and visual-target audits, all frozen natural RGB futures,
global/patch features, semantic masks, state sidecars, and upstream
reports/results/checkpoints. No upstream artifact was overwritten.

## Development-data exposure and stopping boundary

| Partition | Intended role | Actual access |
|---|---|---|
| init31 | camera inventory only | ten tasks, planning observation only |
| init31--34 | calibration/fitting | target construction and fitting not run |
| init36--39 | frozen development evaluation, not blind | not opened |
| init30/35 | historical reference | not opened |
| init40--49 | forbidden | not accessed |

The inventory used one official init31 state per task. It did not execute an
A0/A1/A2 candidate. No development-evaluation feature, state, or target was
read after the gate failed.

## Natural-image-only contract

The planned primary representation comparison remains exactly the frozen
natural-image contract:

```text
natural upright RGB
  -> frozen CLIP ViT-B/32 global projection + L2 normalization [512]
  -> frozen patch tokens, post_layernorm, visual_projection,
     per-token L2 normalization [49,512]
```

No robot-removed, object-removed, robot-only, object-only, or flat-color image
was read or generated. The inventory used native segmentation only to validate
camera geometry; it performed no CLIP inference.

## Camera inventory

All ten tasks expose the same `agentview` camera:

| Field | Value |
|---|---|
| Image | upright RGB, `128x128`, top-left origin |
| Camera ID | 2 |
| Vertical field of view | 45 degrees |
| Position in MuJoCo world | `[0.658613, 0, 1.610350]` |
| Projection input | `gripper0_grip_site` world xyz |
| Projected coordinate order | `(u,v)`, u rightward, v downward |
| Intrinsic focal lengths | `fx=fy=154.509668`, principal point `(64,64)` |
| Transform source | robosuite intrinsic and corrected camera extrinsic |

The common camera rotation matrix is:

```text
[[-0.000000, -0.628266,  0.777998],
 [ 1.000000,  0.000000,  0.000000],
 [-0.000000,  0.777998,  0.628266]]
```

Direct simulator renders were bitwise equal to the upright observation RGB on
all ten inventory tasks. Depth was positive and both candidate projections
fell inside the image on every task.

## Camera projection validation

The protocol compared the robosuite projection as returned with the same
projection vertically converted to the direct upright render. Selection was
calibration-only and fixed as the lower median distance to the natural robot
mask.

| Convention | Median robot distance | p95 | Maximum |
|---|---:|---:|---:|
| `vertical_flip_to_direct_render` | 1.0948 px | 1.4111 px | 1.4536 px |
| `direct_render` | 3.7477 px | 12.3070 px | 14.1469 px |

| Frozen check | Threshold | Result | Pass |
|---|---:|---:|---|
| Selected median distance | <=8 px | 1.0948 px | yes |
| Selected p95 distance | <=18 px | 1.4111 px | yes |
| Median advantage over alternative | >=4 px | 2.6529 px | **no** |

The third row blocks the audit. The narrower gripper-site segmentation was
recorded before the gate decision and overwhelmingly favors the selected
convention, but the protocol authorized target construction from the
three-part robot-mask gate, not from a post-hoc site-based substitute.

## Planned EE and relevant-object extraction

The blocked implementation had fixed the following definitions before the
inventory:

- Project real `gripper0_grip_site` xyz to normalized `(u/128,v/128)` without
  clipping.
- Select relevant objects only from runtime
  `parsed_problem["obj_of_interest"]`; union all geoms per object.
- Record visible area, centroid, and bounding box per natural frame. If an
  object is invisible, record visibility zero and leave its centroid missing;
  never insert a fake coordinate.
- Retain both object-wise values and use the calibration-frozen maximum across
  valid objects for each scalar target.

These definitions were not executed on A0/A1/A2 futures because of the camera
block.

## Planned object and interaction targets

The primary families were preregistered as:

| Family | Frozen targets |
|---|---|
| Robot | projected EE image displacement |
| Object | centroid displacement, per-object mask symmetric difference, visible-area difference |
| Relative | EE-object relative-vector displacement, EE-object centroid-distance change |
| Interaction | EE-to-object-mask proximity change, object-boundary/robot-mask adjacency change |

Per-object `1-IoU`, bounding boxes, relative visible-area change, and
near-contact image status were diagnostic fields. The strict proximity target
used an 8-pixel near-contact threshold; adjacency used a 4-pixel dilation. A
new secondary strict interaction patch scalar used relevant-object fractional
coverage weighted by a 16-pixel Gaussian around projected EE. The frozen 10-D
patch summary itself was not changed.

## Target redundancy matrix

Not computed. It required valid projected EE and per-object A0/A1/A2 masks.
The protocol required this matrix on calibration data before interpreting
multiple target passes as independent mechanisms; the camera gate prevented
that stage.

## Global and patch feature definitions

The intended fair comparison was unchanged from the upstream audit:

```text
G   = frozen 10-D global summary
P   = frozen 10-D patch summary
G+P = frozen 20-D concatenation
```

`G` consists of global cosine distance, global differential L2, and eight
calibration-fitted global PCA scores. `P` consists of mean, max, top-5,
top-10, robot/relevant/broad-interaction regional means, and spatial
centroid/spread. Independent ridge probes and leave-one-init-out alpha
selection were restricted to init31--34. Frozen global/patch PCA16 remained
secondary. None of these probes was fitted in this blocked run.

## Raw associations, probes, and attribution controls

The following requested analyses were correctly withheld:

- global/patch raw Pearson and Spearman associations and bootstrapped
  `delta_rho`;
- G versus P versus G+P RMSE/MAE/R²/correlation;
- G+P incremental gain under the unchanged 10% gate;
- full-patch, relevant-patch, and strict-interaction-patch shuffles;
- h1--h8 and h1--2/h3--5/h6--8 cuts;
- task0--9 and A0-A1/A0-A2/A1-A2 robustness;
- comparison with frozen 3D object/relative targets.

Running any of them would require proceeding past a failed prerequisite. No
image-space mechanism label and no target-mismatch label is assigned.

## Four required answers

**Q1 — Is the stronger patch signal mainly immediate 2D robot motion?**

Unresolved. Projected EE targets were not admitted after the camera gate.

**Q2 — Does it stably encode relevant-object natural image-space change?**

Unresolved. Per-object natural masks and held-out object targets were not
constructed.

**Q3 — Does it stably encode EE-object 2D relative or interaction geometry?**

Unresolved. These targets depend on the blocked projection contract.

**Q4 — Did a 2D-representation/3D-target mismatch partly explain the prior
weak relevance result?**

Not tested. `IMAGE_SPACE_TARGET_MISMATCH_SUPPORTED` is not assigned.

## Candidate headroom constraint

The frozen C1-v2.1 offline oracle found only about `1.648%` best-of-three
expert-action MSE improvement over the actor default. Even if a future audit
resolves the representation question, the current candidate set offers modest
imitation-oriented headroom. This is not a controller-success or
counterfactual-task-outcome claim, and no success metric was read here.

## Mechanical decision

| Gate | Result |
|---|---|
| Camera absolute alignment | pass |
| Projection-convention margin | **FAIL** |
| Per-object target construction | not authorized |
| Calibration probes | not run |
| Frozen development evaluation | not opened |
| Object/relative/interaction family pass | not established |

Final mechanism verdict: `DATA_OR_IMPLEMENTATION_BLOCK`.

Blocking condition: `CAMERA_PROJECTION_BLOCK`.

Next-stage authorization: `BLOCK_VISUAL_TARGET_REDESIGN`.

Controller authorization: `false`.

The result does not authorize a threshold amendment, another representation
probe, a spatial target design, a World Model, a reranker, or a controller.

## Integrity, artifacts, and forbidden activity

| Artifact | SHA-256 |
|---|---|
| Protocol | `cbe7448503ffbbcf69d5135dc1454492a5ccf97f132c4e8a99c152ff7eaedc47` |
| Camera inventory | `b927dd4020123ffd546181d868108ad68f32b944f9bbc5e959e8209acb0b7e20` |
| Compact blocked diagnosis | `93cc90746f79c2dee8a340440bb01fac34e6b859ad982a610f0e24b401ba4141` |

All camera matrices, points, depths, and distance statistics are finite. No
per-object sidecar, calibration weights, calibration freeze, or evaluation
sidecar was produced. No modified RGB, masked CLIP inference, reward, done,
success, final task outcome, controller result, init40--49 access, C1/C2,
D20-v2, seed23/47, new encoder, patch World Model, candidate/reranker change,
or upstream overwrite occurred.
