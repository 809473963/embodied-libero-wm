"""Protocol-boundary tests for the offline planning architecture audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.probes.planning_architecture import (
    antithetic_noise,
    block_bootstrap_draws,
    generate_unit_noise,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/candidate_generation_planning_architecture_protocol.json"


def load() -> dict:
    return json.loads(PROTOCOL.read_text())


def test_partitions_and_forbidden_boundary_are_exact() -> None:
    protocol = load()
    assert protocol["partitions"] == {
        "calibration_demo_ids": [31, 32, 33, 34],
        "evaluation_demo_ids": [36, 37, 38, 39],
        "historical_reference_demo_ids": [30, 35],
        "forbidden_demo_ids": list(range(40, 50)),
    }
    assert protocol["decision_rules"]["spatial_world_model_authorization_always"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert protocol["decision_rules"]["controller_authorization_always"] is False


def test_all_frozen_input_and_implementation_hashes_match() -> None:
    protocol = load()
    for record in protocol["frozen_inputs"] + protocol["implementation"]:
        assert sha256(ROOT / record["path"]) == record["sha256"]
    assert protocol["frozen_upstream"]["commit"] == "346554a125d83bfbbd7471a8e48385581922c849"


def test_candidate_family_and_gate_contract_is_frozen() -> None:
    protocol = load()
    family = protocol["candidate_families"]
    assert family["production_noise_std"] == [0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.0]
    assert family["P4"]["candidates_per_iteration"] == 32
    assert family["P4"]["elite_count"] == 4
    assert protocol["decision_rules"]["meaningful_relative_gain"] == 0.05
    assert protocol["decision_rules"]["strong_relative_gain"] == 0.10
    assert protocol["decision_rules"]["iterative_vs_one_shot_min"] == 0.05


def test_noise_is_deterministic_nested_antithetic_and_smoothed() -> None:
    first = generate_unit_noise(3, 4, 1901)
    second = generate_unit_noise(3, 4, 1901)
    assert np.array_equal(first, second)
    alternatives = antithetic_noise(first, 7)
    assert alternatives.shape == (3, 7, 8, 7)
    assert np.array_equal(alternatives[:, 0], -alternatives[:, 1])
    assert np.array_equal(alternatives[:, 2], -alternatives[:, 3])


def test_bootstrap_draws_preserve_task_and_four_block_shape() -> None:
    draws = block_bootstrap_draws(10_000, 20261002)
    assert draws.shape == (10_000, 10, 4)
    assert draws.min() == 0 and draws.max() == 3

