#!/usr/bin/env python3
"""D21 pure threshold checks and one real plan/verification call."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.control import verification_decision
from src.policies.c2_verify_replan_policy import C2VerifyReplanPolicy


def synthetic_test() -> dict[str, object]:
    predicted = F.normalize(torch.randn(1, 512), dim=-1)
    same = predicted.clone()
    no_trigger = verification_decision(predicted, same, 0.001, 0.02, 0.004, 6)
    assert not no_trigger["threshold_exceeded"] and not no_trigger["early_replan"]
    uncertainty_trigger = verification_decision(predicted, same, 0.005, 0.02, 0.004, 6)
    assert uncertainty_trigger["uncertainty_exceeded"] and uncertainty_trigger["early_replan"]
    opposite = -predicted
    mismatch_trigger = verification_decision(predicted, opposite, 0.001, 0.02, 0.004, 0)
    assert mismatch_trigger["mismatch_exceeded"] and mismatch_trigger["threshold_exceeded"]
    assert not mismatch_trigger["early_replan"]  # h8 has no remaining action.
    return {
        "no_trigger": no_trigger,
        "uncertainty_trigger": uncertainty_trigger,
        "h8_trigger_without_early_replan": mismatch_trigger,
    }


def real_test() -> dict[str, object]:
    source_path = PROJECT_ROOT / "data/LIBERO/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    language = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    with h5py.File(source_path, "r") as source:
        demo = source["data/demo_45"]
        def observation(index: int) -> dict[str, np.ndarray]:
            return {
                "agentview_image": demo["obs/agentview_rgb"][index],
                "robot0_eef_pos": demo["obs/ee_pos"][index],
                "robot0_gripper_qpos": demo["obs/gripper_states"][index],
                "robot0_joint_pos": demo["obs/joint_states"][index],
            }
        first = observation(0)
        future = observation(2)
    policy = C2VerifyReplanPolicy(PROJECT_ROOT / "results/c0/seed11/best.pt")
    policy.reset(seed=10000)
    chunk, plan = policy.plan_chunk(first, language)
    verification = policy.verify(future, horizon=2, remaining_actions=6)
    assert chunk.shape == (8, 7) and np.isfinite(chunk).all()
    assert plan["selected_candidate"] in {0, 1, 2}
    assert np.isfinite(verification["mismatch"])
    assert verification["horizon"] == 2 and verification["remaining_actions"] == 6
    return {"selected_candidate": plan["selected_candidate"], "verification": verification}


def main() -> int:
    print({"synthetic": synthetic_test(), "real": real_test()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
