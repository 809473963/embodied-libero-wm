"""Protocol and generator-contract checks for the expanded interface audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.probes.expanded_candidate_interface import (
    ALLOCATION_ORDER,
    combined_candidates,
    gripper_form_bank,
)
from src.probes.planning_architecture import generate_unit_noise, sha256


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/expanded_candidate_interface_design_protocol.json"


def load() -> dict:
    return json.loads(PROTOCOL.read_text())


def test_partition_and_authorization_boundary() -> None:
    protocol = load()
    assert protocol["partitions"] == {
        "calibration_demo_ids": [31, 32, 33, 34],
        "evaluation_demo_ids": [36, 37, 38, 39],
        "historical_reference_demo_ids": [30, 35],
        "forbidden_demo_ids": list(range(40, 50)),
    }
    assert protocol["authorizations"]["spatial_world_model_always"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert protocol["authorizations"]["controller_always"] is False


def test_frozen_hashes_match() -> None:
    protocol = load()
    for record in protocol["frozen_inputs"] + protocol["implementation"]:
        assert sha256(ROOT / record["path"]) == record["sha256"]
    assert protocol["frozen_upstream"]["commit"] == "4e43c7add65b13034db4cea1c7715d190e65a426"


def test_finite_design_and_gates_are_exact() -> None:
    protocol = load()
    design = protocol["candidate_design"]
    assert design["budgets"] == [16, 24, 32, 48, 64]
    assert design["scales"] == [1.0, 1.5, 2.0, 2.5]
    assert design["fixed_budget_allocations"] == [0, 4, 8, 12]
    assert tuple(design["gripper_contract"]["allocation_order"]) == ALLOCATION_ORDER
    assert protocol["selection_rules"]["meaningful_h8_gain"] == 0.05
    assert protocol["selection_rules"]["strong_h8_gain"] == 0.10
    assert protocol["selection_rules"]["doubling_h8_material_pp"] == 0.015


def test_gripper_forms_are_binary_and_leave_continuous_actions_unchanged() -> None:
    default = np.linspace(-0.5, 0.5, 2 * 8 * 7, dtype=np.float32).reshape(2, 8, 7)
    bank = gripper_form_bank(default)
    assert set(np.unique(np.concatenate(list(bank.values())))) == {-1.0, 1.0}
    assert np.all(bank["minus_to_plus_after_4"][:, :4] == -1.0)
    assert np.all(bank["minus_to_plus_after_4"][:, 4:] == 1.0)


def test_selected_n64_g12_shape_exposes_unmatched_antithetic_direction() -> None:
    default = np.zeros((2, 8, 7), dtype=np.float32)
    unit = generate_unit_noise(2, 32, 1901)
    pre, candidates, category, forms = combined_candidates(
        default, unit, 64, 12,
        np.asarray([0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.0], dtype=np.float32),
        2.5, "antithetic",
    )
    assert pre.shape == candidates.shape == (2, 64, 8, 7)
    assert np.sum(category == 1) == 51
    assert np.sum(category == 2) == 12
    for index in range(1, 51, 2):
        assert np.allclose(pre[:, index] - default, -(pre[:, index + 1] - default))
    unmatched = pre[:, 51] - default
    assert not any(np.allclose(unmatched, -(pre[:, index] - default)) for index in range(1, 51))
    assert "c0_binarized" not in forms

