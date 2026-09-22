# D17 World Model Sample Review

## Verdict

**D17 PASS as a qualitative review; latent mismatch is related to transition
difficulty but is not equivalent to control deviation or failure.**

The review contains exactly 20 relatively low-error and 20 relatively
high-error held-out samples, two from distinct validation demos for every one
of the ten tasks. No sample comes from simulator evaluation episodes.

## Selection and evidence

- Implementation commit: `3cea629225ad3dad6e5edae368a5a0f1e8354b70`
- Source: D16's 5,869 validation sample records
- “Correct”: two lowest mean learned-error samples per task from distinct demos
- “Incorrect”: two highest mean learned-error samples per task from distinct demos
- Displayed frames: `obs[k]`, `obs[k+1]`, `obs[k+4]`, `obs[k+8]`
- Every panel also records mean prediction error, copy error, ensemble
  uncertainty, h8 pixel MAE, h8 end-effector displacement, and gripper changes.

The words “correct” and “incorrect” denote relative error tails. They are not
semantic ground-truth labels and do not imply task success/failure.

## Quantitative contrast

| Selected-tail statistic | Low-error 20 | High-error 20 |
|---|---:|---:|
| learned cosine error | 0.003583 | 0.019269 |
| copy-current error | 0.005233 | 0.026748 |
| ensemble uncertainty | 0.001220 | 0.001856 |
| pixel MAE, k to k+8 | 2.898 | 8.382 |
| end-effector displacement, k to k+8 | 0.0358 | 0.0563 |
| mean translation-action L2 | 0.5029 | 0.7474 |
| mean rotation-action L2 | 0.0607 | 0.1229 |
| gripper sign changes per sample | 0.30 | 0.40 |

Across these deliberately selected tails, learned error has correlation 0.793
with h8 pixel MAE, 0.541 with ensemble uncertainty, 0.493 with rotation-action
magnitude, 0.427 with translation-action magnitude, and 0.375 with h8
end-effector displacement. These are descriptive tail correlations and are
selection-biased; they are not population calibration metrics. Action
saturation is constant in the selected set, so its correlation is explicitly
undefined rather than recorded as NaN.

## Manual review: low-error samples

| ID | task/demo/k | Manual observation |
|---:|---|---|
| 00 | 0/45/37 | Scene and arm change only slightly; a straightforward slow approach. |
| 01 | 0/46/0 | End effector translates visibly but background/object layout remains stable. |
| 02 | 1/46/52 | Almost static scene; learned and copy errors are both tiny, with copy marginally lower. |
| 03 | 1/48/60 | Small arm motion without an abrupt object-state change. |
| 04 | 2/49/13 | Visually stationary over all four frames; near-zero error is expected. |
| 05 | 2/45/57 | Small, smooth approach motion; no visible grasp/release event. |
| 06 | 3/47/79 | Visible interaction-phase motion and one gripper change, yet prediction remains accurate. |
| 07 | 3/48/80 | Arm is close to the plate/object and gripper changes; this is a useful low-error dynamic counterexample. |
| 08 | 4/46/2 | Robot moves from an early pose, but pixel change is small and smooth. |
| 09 | 4/48/1 | Similar early smooth motion; stable object arrangement. |
| 10 | 5/49/40 | Minor wrist/arm displacement with no abrupt scene change. |
| 11 | 5/47/41 | Moderate approach motion; object layout remains visually consistent. |
| 12 | 6/48/2 | Relatively large end-effector displacement but only moderate pixel change; still low error. |
| 13 | 6/47/107 | Late interaction pose with a gripper transition but little visible scene displacement. |
| 14 | 7/49/44 | Nearly stationary approach/hold phase. |
| 15 | 7/46/85 | Small motion near the interaction area, no abrupt visual event. |
| 16 | 8/48/1 | Early smooth motion and stable scene. |
| 17 | 8/49/41 | Small approach motion toward the plate/object. |
| 18 | 9/46/2 | End effector moves substantially while scene appearance changes little. |
| 19 | 9/49/41 | Larger arm displacement and visible foreground change, yet the model remains in the low-error tail. |

The low-error set therefore is not merely “no action”: IDs 06, 07, 12, 18,
and 19 show meaningful motion or gripper change.

## Manual review: high-error samples

| ID | task/demo/k | Manual observation |
|---:|---|---|
| 00 | 0/47/75 | Late manipulation with large foreground-arm change and a gripper transition. |
| 01 | 0/45/75 | Large arm/wrist displacement near the target objects and a gripper transition. |
| 02 | 1/49/114 | Robot occupies much of the foreground; moderate visible motion despite no gripper change. |
| 03 | 1/46/77 | Large end-effector displacement during approach/manipulation. |
| 04 | 2/49/109 | Strong robot/object-region appearance change; one of the largest pixel changes. |
| 05 | 2/45/38 | Moderate motion; learned error exceeds copy error, so this is an important predictor counterexample. |
| 06 | 3/48/35 | Active interaction, large arm motion, and a gripper transition. |
| 07 | 3/47/30 | Large arm motion near the plate/object and a gripper transition. |
| 08 | 4/48/51 | Small measured end-effector displacement but sizable foreground visual change near contact. |
| 09 | 4/45/56 | Large wrist/arm appearance change near the interaction region. |
| 10 | 5/49/90 | Late active manipulation with large visual displacement and high uncertainty. |
| 11 | 5/45/0 | Scene is nearly static and early in the episode; a clear high-error counterexample without obvious control deviation. |
| 12 | 6/46/60 | Very large robot/scene change; learned is much better than copy but absolute error remains high. |
| 13 | 6/49/64 | Moderate interaction motion and high uncertainty; learned and copy errors are nearly equal. |
| 14 | 7/49/80 | Only modest pixel change; high latent error is not visually explained by a dramatic event. |
| 15 | 7/45/120 | End-effector displacement is tiny, with moderate foreground change; another control-deviation counterexample. |
| 16 | 8/49/92 | Large foreground arm/wrist movement near the objects. |
| 17 | 8/46/43 | Moderate-to-large approach motion; no discrete gripper event. |
| 18 | 9/47/105 | Large active manipulation change; learned and copy errors are nearly tied. |
| 19 | 9/46/78 | Large end-effector and foreground displacement in an active phase. |

## Interpretation

The dominant pattern is sensible: many high-error cases occur during fast
visual change, close interaction, grasp/release, or large foreground robot
motion. This supports using mismatch as a signal of difficult dynamics.

It is not a control-failure label:

- These are successful demonstration trajectories, not failed online control
  episodes with deviation annotations.
- High-error IDs 11, 14, and 15 do not show a dramatic visible deviation.
- Low-error IDs 06, 07, 12, 18, and 19 contain meaningful motion.
- Some high-error samples (05, 13, 18) give little or no advantage over copy.
- A frozen CLIP latent can react to semantic/appearance changes that are not
  themselves harmful to control.

Therefore D17 establishes qualitative relevance but not failure-detection
validity or a replan threshold. Later threshold selection must use validation
episodes, report false triggers, and preserve these counterexamples.

## Artifacts and integrity

| Artifact | SHA-256 |
|---|---|
| `correct_1.png` | `139976ae47fb422bd24ad5dcd6d0356e9b612d2c67340fa59df63d81bf076548` |
| `correct_2.png` | `eb7b5abf9da522e3c297958cd63d8c9c3ed6bf3f4cb3e939ef85062ebc5ae44d` |
| `incorrect_1.png` | `202928892fc1f28a353591856bc0a8754be0c83266245beb36460d603fad831e` |
| `incorrect_2.png` | `8de4f38e20cf512d08e0a29a13268275b8f6d2169f3070c8c2957db90c2189ac` |
| `review_samples.json` | `73faa1975d5409b4025fa38fe45e8143c532b08e308a6222810a60b24b2841c6` |

All 40 records have finite metrics, task coverage 0–9, demo IDs 45–49, and
distinct demos within each task/category. The source D16 sample hash is checked
before generation. RGB/proprio are used only for this audit and remain outside
the learned predictor's deployment inputs.
