#!/usr/bin/env python3
"""One real observation through both logging-only controller observers."""

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

from scripts.collect_control_validation import (
    encode_image,
    plan_c0_observed,
    plan_c1_observed,
)
from src.policies.c1_rerank_policy import C1RerankPolicy


def main() -> int:
    source = PROJECT_ROOT / (
        "data/LIBERO/datasets/libero_spatial/"
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    )
    language = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    with h5py.File(source, "r") as handle:
        demo = handle["data/demo_45"]

        def observation(index: int) -> dict[str, np.ndarray]:
            return {
                "agentview_image": demo["obs/agentview_rgb"][index],
                "robot0_eef_pos": demo["obs/ee_pos"][index],
                "robot0_gripper_qpos": demo["obs/gripper_states"][index],
                "robot0_joint_pos": demo["obs/joint_states"][index],
            }

        current = observation(0)
        future = observation(2)
    policy = C1RerankPolicy(PROJECT_ROOT / "results/c0/seed11/best.pt")
    policy.reset(seed=20000)
    c0_chunk, c0_prediction, c0_uncertainty, c0_diagnostics = plan_c0_observed(
        policy, current, language
    )
    direct_c0 = np.clip(policy._predict_raw_batch([current], [language])[0], -1.0, 1.0)
    assert np.allclose(c0_chunk, direct_c0, rtol=0.0, atol=1.0e-6)
    assert c0_prediction.shape == (8, 512) and c0_uncertainty.shape == (8,)
    assert c0_diagnostics["selected_candidate"] == 0
    actual = encode_image(policy, future)
    mismatch = 1.0 - F.cosine_similarity(c0_prediction[1:2], actual, dim=-1)
    assert torch.isfinite(mismatch).all()

    policy.reset(seed=20000)
    c1_chunk, c1_prediction, c1_uncertainty, c1_diagnostics = plan_c1_observed(
        policy, current, language
    )
    assert c1_chunk.shape == (8, 7) and np.isfinite(c1_chunk).all()
    assert c1_prediction.shape == (8, 512) and c1_uncertainty.shape == (8,)
    assert c1_diagnostics["selected_candidate"] in {0, 1, 2}
    assert torch.isfinite(c1_prediction).all() and torch.isfinite(c1_uncertainty).all()
    print(
        {
            "c0_chunk_matches_frozen_actor": True,
            "c0_h2_mismatch": float(mismatch.item()),
            "c1_selected_candidate": c1_diagnostics["selected_candidate"],
            "c0_uncertainty_h2": float(c0_uncertainty[1]),
            "c1_uncertainty_h2": float(c1_uncertainty[1]),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
