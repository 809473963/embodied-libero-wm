#!/usr/bin/env python3
"""D19 pure candidate tests plus one real C1 planning call."""

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

from src.control import generate_antithetic_candidates, score_candidates
from src.policies.c1_rerank_policy import C1RerankPolicy


def synthetic_test() -> dict[str, object]:
    base = torch.zeros(8, 7)
    std = torch.tensor([0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.0])
    first_generator = torch.Generator().manual_seed(1901)
    second_generator = torch.Generator().manual_seed(1901)
    candidates = generate_antithetic_candidates(base, std, first_generator)
    repeated = generate_antithetic_candidates(base, std, second_generator)
    assert candidates.shape == (3, 8, 7)
    assert torch.equal(candidates, repeated)
    assert torch.equal(candidates[0], base)
    assert torch.allclose(candidates[1, :, :6], -candidates[2, :, :6])
    assert torch.equal(candidates[:, :, 6], torch.zeros(3, 8))
    z0 = F.normalize(torch.randn(1, 512), dim=-1)
    language = F.normalize(torch.randn(1, 512), dim=-1)
    predictions = F.normalize(torch.randn(3, 8, 512), dim=-1)
    uncertainty = torch.rand(3, 8) * 0.01
    scores = score_candidates(
        z0, language, candidates, predictions, uncertainty, torch.ones(1, 1, 7)
    )
    assert all(value.shape == (3,) for value in scores.values())
    assert all(torch.isfinite(value).all() for value in scores.values())
    return {"candidate_shape": list(candidates.shape), "scores": scores["score"].tolist()}


def real_plan_test() -> dict[str, object]:
    source_path = PROJECT_ROOT / "data/LIBERO/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    with h5py.File(source_path, "r") as source:
        observation = {
            "agentview_image": source["data/demo_45/obs/agentview_rgb"][0],
            "robot0_eef_pos": source["data/demo_45/obs/ee_pos"][0],
            "robot0_gripper_qpos": source["data/demo_45/obs/gripper_states"][0],
            "robot0_joint_pos": source["data/demo_45/obs/joint_states"][0],
        }
    language = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    policy = C1RerankPolicy(PROJECT_ROOT / "results/c0/seed11/best.pt")
    policy.reset(seed=10000)
    base = policy.plan_batch([observation], [language])[0]
    policy.reset(seed=10000)
    selected, diagnostics = policy.plan_chunk(observation, language)
    policy.reset(seed=10000)
    repeated_selected, repeated_diagnostics = policy.plan_chunk(observation, language)
    assert selected.shape == (8, 7) and np.isfinite(selected).all()
    assert np.array_equal(selected, repeated_selected)
    assert diagnostics["selected_candidate"] == repeated_diagnostics["selected_candidate"]
    assert diagnostics["scores"] == repeated_diagnostics["scores"]
    assert diagnostics["selected_candidate"] in {0, 1, 2}
    assert len(diagnostics["scores"]) == 3
    assert diagnostics["early_replan"] is False
    # Candidate zero is bitwise the frozen C0 clipped action chunk. If it is
    # selected, the executed chunk must therefore equal the C0 baseline.
    if diagnostics["selected_candidate"] == 0:
        assert np.array_equal(selected, base)
    return {
        "selected_candidate": diagnostics["selected_candidate"],
        "scores": diagnostics["scores"],
        "planning_ms": diagnostics["planning_ms"],
    }


def main() -> int:
    print({"synthetic": synthetic_test(), "real_plan": real_plan_test()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
