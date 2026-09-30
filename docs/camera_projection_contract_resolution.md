# Tight landmarks uniquely resolve the upright agentview projection contract

Date: 2026-09-30

Final verdict: `CAMERA_PROJECTION_CONTRACT_VALIDATED`

Authorization: `GO_IMAGE_SPACE_TARGET_AUDIT_RESTART`

Spatial-target design authorization: `false`

World-Model/reranker/controller authorization: `false`

## Executive summary

The unique validated contract for the installed LIBERO/robosuite/MuJoCo
pipeline leaves the direct `agentview` RGB array unchanged and converts only
the projected vertical coordinate:

```text
u_upright = u_raw
v_upright = H - 1 - v_raw
```

Across ten tasks and four calibration init states, the converted projection of
`gripper0_grip_site` lies within its exact native type-6 site segmentation with
pooled median error `0.393 px`, p95 `0.611 px`, and maximum `0.629 px`. It
passes the preregistered rasterization-aware limits of `1.0/1.5/2.0 px`.
Using raw `v` directly gives median `64.011 px`, p95 `69.660 px`, and maximum
`71.500 px`, and therefore clearly fails the same absolute gate.

The result is independently overdetermined. All 40 direct simulator renders
are bitwise equal to `observation["agentview_image"]`; the image is never
flipped. An explicit projection derived from `cam_xpos`, `cam_xmat`, and fovy
matches the robosuite matrix calculation within `6.04e-14 px`, with depth
difference at most `2.22e-16`. Every point is finite, has positive depth, and
is inside the frame.

This is a new validation design using a tight rendered landmark. It does not
change the old robot-mask gate or its permanent negative result:

```text
DATA_OR_IMPLEMENTATION_BLOCK
CAMERA_PROJECTION_BLOCK
```

The only new authorization is `GO_IMAGE_SPACE_TARGET_AUDIT_RESTART`. It does
not authorize target redesign, representation conclusions, model work,
reranking, or control.

## Frozen upstream provenance

The old block is frozen at commit
`d4dd36113d4381b821238dcd29101546f64860f1` under annotated tag
`camera-projection-gate-blocked`. Its `4 px` robot-mask convention-margin
threshold and measured `2.6529 px` result remain unchanged.

The independent contract-resolution protocol, source trace requirements,
landmark selection, two convention candidates, rasterization thresholds, and
decision tree were committed as
`f1c362e7e2505da6f55a716b94fab4225cfcede9` before landmark inventory or new
quantitative comparison. Source/camera/landmark inventory was then frozen at
`249646b987992019e547036edc82adcecf0d1584` before init31--34 projection
evaluation.

## Why the old robot-mask margin is not reused

The old validator measured a point's distance to a large robot region. Both a
correct and an incorrect point can lie on or near some robot pixel, so the
difference between their errors is not a clean point-correspondence test. Its
frozen `>=4 px` margin nevertheless remains the correct mechanical rule for
that old experiment and continues to fail.

The new experiment changes the validator, not the old threshold: a known 3D
site is compared with pixels carrying that exact MuJoCo site ID. The new gate
asks whether one convention passes absolute rasterization error and the other
fails the same absolute gate. No old artifact or verdict was recomputed.

## Calibration-only boundary

| Scope | Use | Accessed |
|---|---|---|
| init31, tasks0--9 | source/camera/landmark inventory | yes |
| init31--34, tasks0--9 | frozen projection evaluation | yes |
| init30/35 | forbidden historical states | no |
| init36--39 | reserved for restarted image-space audit | no |
| init40--49 | forbidden | no |
| init0--30 | outside this calibration-only experiment | no |

Only the current planning observation was rendered. No action or A0/A1/A2
candidate was executed. Reward, done, success, and task outcome were not read.

## Source-level camera inventory

All ten tasks expose the same runtime camera:

| Field | Value |
|---|---|
| Camera | `agentview`, ID 2 |
| Image | `uint8 [128,128,3]` |
| Vertical fovy | 45 degrees |
| Position | `[0.658613175, 0, 1.610350024]` |
| Intrinsic | `fx=fy=154.509667992`, `cx=cy=64` |
| MuJoCo rotation source | `sim.data.cam_xmat[2]` |
| Position source | `sim.data.cam_xpos[2]` |
| Axis correction | `diag(1,-1,-1,1)` |
| Robosuite image convention | `opengl`, mapped to slice `+1` |

The camera rotation from local coordinates to world is:

```text
[[-0.000000073, -0.628266450,  0.777998244],
 [ 1.000000000,  0.000000073,  0.000000152],
 [-0.000000152,  0.777998244,  0.628266450]]
```

The fovy-derived intrinsic is:

```text
[[154.509667992,   0,            64],
 [  0,           154.509667992,  64],
 [  0,             0,             1]]
```

The frozen contract contains the complete corrected pose and 4x4
world-to-raw-pixel matrix.

## Coordinate Convention Trace

The installed source path resolves the apparent “RGB does not flip, but v
does” tension as follows:

1. `sim.data.cam_xpos` and `cam_xmat` give camera center `C` and local-to-world
   rotation `R_wc` in the MuJoCo convention.
2. `get_camera_extrinsic_matrix` forms the camera pose and postmultiplies it by
   `diag(1,-1,-1,1)`, producing a +z-forward pinhole frame.
3. `get_camera_intrinsic_matrix` computes
   `f=0.5*H/tan(fovy*pi/360)` and principal point `(W/2,H/2)`.
4. `get_camera_transform_matrix` returns `K4 @ inverse(corrected_pose)`.
5. The raw utility projection divides its first two homogeneous coordinates by
   positive depth. Its raw vertical coordinate corresponds to the vertically
   reversed view used by robosuite's standalone camera-segmentation helper.
6. The observation camera sensor uses the configured `IMAGE_CONVENTION` slice.
   Here `opengl -> +1`, so the direct `sim.render` array is returned unchanged.
7. Empirically, that direct array is already identical to the upright LIBERO
   observation. Therefore the image is not transformed. Only the projected
   coordinate is converted to the direct array's row convention.

Thus “vertical flip” names a coordinate-system conversion, not an RGB
operation:

```text
u_final = u_raw
v_final = 127 - v_raw
```

for `H=128`.

## RGB orientation verification

All 40 `sim.render(camera_name="agentview", 128x128)` arrays are bitwise equal
to their corresponding `observation["agentview_image"]`:

| Check | Result |
|---|---:|
| Shapes | all `[128,128,3]` |
| Dtype | all `uint8` |
| Bitwise equality | 40/40 |
| Maximum absolute pixel difference | 0 |

The direct render is the final upright image. Applying `[::-1]` to RGB would be
wrong for this pipeline.

## Landmark inventory and frozen selection

The inventory enumerated every runtime site with name, ID, finite
`site_xpos`, site type/group/size, and native type-6 segmentation visibility.
The complete list is in `landmark_inventory.json`.

| Landmark | ID | Visible tasks | Pixel range | Primary | Reason |
|---|---:|---:|---:|---|---|
| `gripper0_grip_site` | 13 | 10/10 | 4--7 | yes | exact semantic point, compact, finite, visible everywhere |
| `gripper0_ft_frame` | 12 | 10/10 | 1--35 | no | not in the frozen exact-point allowlist; marker extent varies |
| `gripper0_grip_site_cylinder` | 17 | 10/10 | 6--43 | no | extended cylinder rather than tight point reference |
| `gripper0_ee_x/y/z` | 14--16 | 0/10 | 0 | no | invisible |
| Object/region/table sites | various | mostly 0/10 | 0--4 | no | absent, task-dependent, or not a tight universal reference |

Only `gripper0_grip_site` satisfies all primary requirements. The protocol
explicitly allowed one site type when repeated across multiple tasks and
calibration states; it supplies 40 independent planning-state checks.

No pixel was manually clicked or selected after viewing a projection.

## Exact projection formula

Let `X` be a world point, `C` the camera world position, and `R_wc` the MuJoCo
camera local-to-world rotation. First compute:

```text
X_mj = R_wc^T (X - C)
X_cv = diag(1,-1,-1) X_mj = [x_c, y_c, z_c]
```

The forward depth is `z_c` and must be positive. With:

```text
fx = fy = 0.5 H / tan(fovy*pi/360)
cx = W/2
cy = H/2
```

the raw utility coordinates are:

```text
u_raw = fx*x_c/z_c + cx
v_raw = fy*y_c/z_c + cy
```

The unique upright-array coordinates are:

```text
u_upright = u_raw
v_upright = H - 1 - v_raw
```

Coordinates remain floating point and unclipped in the frozen contract.
Rounding/clipping is inappropriate for later continuous target construction;
the official integer helper is retained only as a source diagnostic.

## Independent implementation comparison

The independent implementation uses only `cam_xpos`, `cam_xmat`, fovy, image
shape, and the explicit equations above. It does not multiply by the
robosuite-provided 4x4 projection matrix. Across all 40 primary samples:

| Comparison | Maximum difference | Frozen limit | Pass |
|---|---:|---:|---|
| Raw `(u,v)` vs robosuite matrix | `6.04e-14 px` | `1e-9 px` | yes |
| Positive depth | `2.22e-16` | `1e-12` | yes |
| Camera projection matrix drift | `0` | `1e-12` | yes |

This agreement rules out a matrix-assembly or transpose error independently of
the site segmentation test.

## Candidate conventions and primary errors

Only two source-plausible candidates were admitted:

- `direct_render`: use raw `(u_raw,v_raw)` on the direct image.
- `vertical_flip_to_direct_render`: use `(u_raw,127-v_raw)` while leaving RGB
  unchanged.

| Convention | n | Mean | Median | p95 | Maximum | Absolute gate |
|---|---:|---:|---:|---:|---:|---|
| `vertical_flip_to_direct_render` | 40 | 0.374 | 0.393 | 0.611 | 0.629 | pass |
| `direct_render` | 40 | 64.618 | 64.011 | 69.660 | 71.500 | fail |

The selected contract passes by a wide absolute margin, and the alternative
fails the same thresholds. No selected-versus-alternative error-difference
threshold is used.

## Rasterization-aware tolerance rationale

The gate was frozen before init32--34 comparison. A floating center compared
with integer raster indices incurs up to `sqrt(0.5²+0.5²)=0.707 px` under
pixel-center quantization. Integer-index versus center semantics and native
site-boundary discretization can add about one pixel. The conservative limits
were therefore fixed as:

```text
median <= 1.0 px
p95   <= 1.5 px
max   <= 2.0 px
```

They are engineering bounds derived from the 128x128 raster interface, not a
fit to the old observed `0.085--0.626 px` range.

## Per-task consistency

| Task | Selected median | Selected p95 | Selected max | Alternative median |
|---:|---:|---:|---:|---:|
| 0 | 0.434 | 0.480 | 0.486 | 64.011 |
| 1 | 0.357 | 0.456 | 0.473 | 63.094 |
| 2 | 0.473 | 0.611 | 0.626 | 63.510 |
| 3 | 0.526 | 0.602 | 0.610 | 64.512 |
| 4 | 0.325 | 0.497 | 0.528 | 66.003 |
| 5 | 0.399 | 0.486 | 0.493 | 66.758 |
| 6 | 0.262 | 0.454 | 0.472 | 64.847 |
| 7 | 0.429 | 0.608 | 0.629 | 65.805 |
| 8 | 0.275 | 0.451 | 0.479 | 64.033 |
| 9 | 0.316 | 0.497 | 0.511 | 62.712 |

Every task independently passes all selected-convention limits.

## Per-state and per-landmark consistency

| Init | Selected median | Selected p95 | Selected max | Alternative median |
|---:|---:|---:|---:|---:|
| 31 | 0.403 | 0.574 | 0.626 | 63.768 |
| 32 | 0.445 | 0.620 | 0.629 | 64.653 |
| 33 | 0.308 | 0.543 | 0.555 | 64.094 |
| 34 | 0.352 | 0.514 | 0.528 | 64.195 |

The single primary landmark's 40-sample row equals the pooled result. Site
visibility is 4--7 pixels in every sample. Camera-space depth ranges from
`0.9266` to `0.9823`, always positive.

## Historical robot-mask reproduction

The old broad-mask statistic was recomputed only as a secondary reproducibility
check:

| Convention | Median | p95 | Maximum |
|---|---:|---:|---:|
| Selected converted coordinate | 1.159 | 1.429 | 1.476 |
| Raw coordinate | 3.748 | 10.077 | 14.147 |

The init31 median values reproduce the old `~1.09` versus `~3.75 px` pattern;
the 40-state selected median changes slightly because init32--34 are now also
included. These values do not enter the new decision.

## Relevant-object and geom secondary checks

Relevant-object root-body centers are not strict pixel ground truth, but they
provide an independent secondary sanity check. Across 80 visible object/state
rows, the selected projected center lies inside its own visible bounding box
in `100%`; distance to the exact object mask has median `0.373 px`, p95
`0.609 px`, and maximum `0.684 px`. Exact inside-mask rate is zero because the
floating projected center is near, rather than exactly equal to, an integer
mask coordinate. This result does not enter the primary gate.

`GEOM_FOOTPRINT_CHECK_NOT_IMPLEMENTED`: a new mesh/primitive footprint
rasterizer would add unnecessary implementation risk. Native exact site-ID
segmentation already provides the tight primary reference.

## NaN, depth, and in-frame audit

All 40 primary world points, matrices, raw/final pixels, errors, and depths are
finite. All depths are positive. Every selected projection is within
`0<=u<128`, `0<=v<128` while the site segmentation is visible. No camera
matrix drift was observed.

## Four required answers

**Q1 — What is the orientation relationship between observation RGB and the
direct simulator render?**

They are bitwise identical in all 40 checks. The direct RGB is already the
upright observation and must not be flipped.

**Q2 — How does the robosuite/MuJoCo projected v coordinate map to upright
RGB?**

After the robosuite camera-axis correction and perspective divide, convert
only the coordinate: `v_upright=H-1-v_raw`; keep `u` unchanged. This is not an
image transformation.

**Q3 — Which convention is independently supported by tight landmarks?**

`vertical_flip_to_direct_render`, more precisely defined as “unchanged direct
RGB plus vertical coordinate conversion,” is uniquely supported. It has
maximum site error `0.629 px`; the raw convention has median error `64.011 px`.

**Q4 — Is the evidence strong enough to freeze one contract for later 2D
targets?**

Yes. Source tracing, independent formula agreement, exact site segmentation,
ten tasks, four states, positive depth, and in-frame checks all agree. The
contract is frozen at `artifacts/camera_projection_contract/frozen_contract.json`.

## Mechanical decision and authorization

| Gate | Result |
|---|---|
| Source-level coordinate trace complete | pass |
| Direct RGB orientation verified | pass |
| Independent projection agrees with robosuite | pass |
| Selected convention clears absolute site gate | pass |
| Alternative fails the same absolute gate | pass |
| Multi-task/state/landmark consistency | pass |
| Finite, positive-depth, in-frame validity | pass |

Final verdict: `CAMERA_PROJECTION_CONTRACT_VALIDATED`.

Authorization: `GO_IMAGE_SPACE_TARGET_AUDIT_RESTART`.

This permits only a separately frozen restart of the previously blocked
Natural Image-Space Interaction Target Audit using the saved contract. It does
not authorize `GO_SPATIAL_TARGET_DESIGN`, a World Model, a reranker, or a
controller.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| Protocol | `ed5ba8561f504762e140694e7d02e9dccd49961b8147259d62f8f04106b25907` |
| Camera inventory | `10838796b774dbab59c8a2be2e22c367e360fd10350035ba689b453a28535653` |
| Landmark inventory | `c0f2f7df0ae0ed4774826a335d7d8e505b168d7fc950e4c824957088514efca8` |
| Compact diagnosis | `b66536051ee2f71e0fd28ec06bb33e4c3d4851194d7bc8956a82ded8b7470008` |
| Frozen contract | `5eb187c5b3b7eab7a8f2503f9ee8e9dc4bb12e2e65cfb5d017b1913c2d614e0e` |

Installed source hashes are retained in both inventories. The primary files
are `robosuite/utils/camera_utils.py`, `robot_env.py`, `mjcf_utils.py`, and
`macros.py`; their hashes make the contract sensitive to a future dependency
change.

## Automated verification

Twelve focused tests independently cover the protocol boundary and result:
projection-formula reconstruction for all 40 primary rows, exact RGB
orientation, finite world points, positive depth, in-frame coordinates,
selected/alternative absolute gates, source-versus-independent agreement,
camera stability, frozen upstream hashes, and the exact verdict/authorization
mapping. The complete repository suite passes `105` tests. The remaining 21
messages are dependency deprecation warnings, not test failures.

## Forbidden-activity audit

No candidate action, reward, done, success, task outcome, init0--30,
init35--49, image-space target, global/patch feature analysis, ridge, PCA,
shuffle, target redesign, encoder, World Model, actor, candidate generator,
reranker, C1/C2, D20-v2, seed23/47, or controller was run or modified. The old
camera block and every upstream artifact remain unchanged.
