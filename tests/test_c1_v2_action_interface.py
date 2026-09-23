#!/usr/bin/env python3
"""Prove the C0-v2 raw-action to WM-normalized-action interface."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
sys.path.insert(0, str(ROOT))

from src.control import prepare_candidate_action_interface
from src.policies.c1_v2_rerank_policy import C1V2RerankPolicy


def synthetic_check() -> None:
    actor_normalized = torch.linspace(-2.0, 2.0, 56).reshape(8, 7)
    actor_mean = torch.tensor([0.2, 0.1, -0.2, 0.0, 0.01, -0.01, 0.1])
    actor_std = torch.tensor([0.4, 0.3, 0.5, 0.04, 0.07, 0.06, 0.9])
    wm_mean = torch.tensor([-0.1, 0.3, 0.2, 0.02, -0.03, 0.04, -0.2])
    wm_std = torch.tensor([0.2, 0.6, 0.4, 0.08, 0.09, 0.1, 0.7])
    noise_std = torch.tensor([0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.0])
    interface = prepare_candidate_action_interface(
        actor_normalized,
        actor_mean,
        actor_std,
        wm_mean,
        wm_std,
        noise_std,
        torch.Generator().manual_seed(1901),
    )
    expected_raw = actor_normalized * actor_std + actor_mean
    assert torch.allclose(interface["actor_denormalized_preclip"], expected_raw)
    assert torch.allclose(interface["candidates"][0], expected_raw.clamp(-1.0, 1.0))
    expected_wm = (interface["candidates"] - wm_mean.reshape(1, 1, 7)) / wm_std.reshape(
        1, 1, 7
    )
    assert torch.allclose(interface["wm_normalized_candidates"], expected_wm)
    assert not torch.allclose(
        interface["actor_normalized"], interface["wm_normalized_candidates"][0]
    )
    assert interface["candidates"].shape == (3, 8, 7)
    assert torch.isfinite(interface["wm_normalized_candidates"]).all()


def real_chain_check() -> dict[str, object]:
    data_path = ROOT / (
        "data/LIBERO/datasets/libero_spatial/"
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    )
    with h5py.File(data_path, "r") as source:
        observation = {
            "agentview_image": source["data/demo_45/obs/agentview_rgb"][0],
            "robot0_eef_pos": source["data/demo_45/obs/ee_pos"][0],
            "robot0_gripper_qpos": source["data/demo_45/obs/gripper_states"][0],
            "robot0_joint_pos": source["data/demo_45/obs/joint_states"][0],
        }
    language = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = C1V2RerankPolicy(ROOT / "results/c0_v2/seed11/best.pt", device=device)
    policy.reset(seed=30000)
    selected, diagnostics = policy.plan_chunk(observation, language)
    trace = policy._last_plan_tensors
    assert trace is not None
    assert selected.shape == (8, 7) and np.isfinite(selected).all()
    assert trace["actor_normalized"].shape == (8, 7)
    assert trace["actor_denormalized_preclip"].shape == (8, 7)
    assert trace["candidates"].shape == (3, 8, 7)
    assert trace["wm_normalized_candidates"].shape == (3, 8, 7)
    assert trace["ensemble_prediction"].shape == (3, 8, 512)
    assert trace["ensemble_uncertainty"].shape == (3, 8)
    actor_mean = policy.action_mean.detach().cpu().numpy().reshape(7)
    actor_std = policy.action_std.detach().cpu().numpy().reshape(7)
    wm_mean = policy.wm_action_mean.detach().cpu().numpy().reshape(7)
    wm_std = policy.wm_action_std.detach().cpu().numpy().reshape(7)
    assert np.allclose(
        trace["actor_denormalized_preclip"],
        trace["actor_normalized"] * actor_std + actor_mean,
    )
    assert np.allclose(
        trace["wm_normalized_candidates"],
        (trace["candidates"] - wm_mean.reshape(1, 1, 7)) / wm_std.reshape(1, 1, 7),
    )
    assert not np.allclose(
        trace["actor_normalized"], trace["wm_normalized_candidates"][0]
    )
    norms = np.linalg.norm(trace["ensemble_prediction"].astype(np.float32), axis=-1)
    assert np.allclose(norms, 1.0, atol=1.0e-4)
    assert diagnostics["selected_candidate"] in {0, 1, 2}
    assert diagnostics["replaced_actor_default"] == (
        diagnostics["selected_candidate"] != 0
    )
    return {
        "selected_candidate": diagnostics["selected_candidate"],
        "actor_action_mean": actor_mean.tolist(),
        "actor_action_std": actor_std.tolist(),
        "wm_action_mean": wm_mean.tolist(),
        "wm_action_std": wm_std.tolist(),
        "raw_candidate_range": diagnostics["interface"]["raw_candidate_range"],
        "candidate_clipping_ratio": diagnostics["interface"]["candidate_preclip"][
            "clipped_element_ratio"
        ],
        "wm_normalized_candidate_range": diagnostics["interface"][
            "wm_normalized_candidate_range"
        ],
    }


def main() -> None:
    synthetic_check()
    print({"status": "PASS", "real_chain": real_chain_check()})


if __name__ == "__main__":
    main()
