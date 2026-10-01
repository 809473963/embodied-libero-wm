"""Protocol and construction checks for the exact N64 contract restart."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.probes.expanded_candidate_contract_restart import (
    build_contract_candidates,
    gripper_universe,
    validate_exact_contract,
)
from src.probes.planning_architecture import generate_unit_noise, sha256


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/expanded_candidate_contract_restart_protocol.json"


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
    assert protocol["decision_rules"]["spatial_world_model_always"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert protocol["decision_rules"]["controller_always"] is False


def test_frozen_input_and_implementation_hashes_match() -> None:
    protocol = load()
    for record in protocol["frozen_inputs"] + protocol["implementation"]:
        assert sha256(ROOT / record["path"]) == record["sha256"]
    assert protocol["frozen_upstream"]["commit"] == "241546e9021faac9c1313536aebb3c0962ee32bf"


def test_design_space_has_only_three_arithmetic_contracts() -> None:
    design = load()["candidate_design"]
    assert design["N_total"] == 64
    assert design["scale"] == 2.5
    assert design["allocations"] == [
        {"name": "C52_G11", "continuous": 52, "gripper": 11},
        {"name": "C50_G13", "continuous": 50, "gripper": 13},
        {"name": "C48_G15", "continuous": 48, "gripper": 15},
    ]
    assert all(item["continuous"] % 2 == 0 for item in design["allocations"])
    assert all(1 + item["continuous"] + item["gripper"] == 64 for item in design["allocations"])
    assert design["no_cartesian_product"] is True


def test_gripper_bank_contains_g3_constants_and_both_switch_directions() -> None:
    default = np.linspace(-0.5, 0.5, 2 * 8 * 7, dtype=np.float32).reshape(2, 8, 7)
    bank = gripper_universe(default)
    assert {"c0_binarized", "all_negative", "all_positive"} <= set(bank)
    assert np.array_equal(
        bank["c0_binarized"],
        np.where(default[:, :, 6] >= 0, 1.0, -1.0).astype(np.float32),
    )
    assert np.all(bank["minus_to_plus_after_4"][:, :4] == -1.0)
    assert np.all(bank["minus_to_plus_after_4"][:, 4:] == 1.0)
    assert np.all(bank["plus_to_minus_after_4"][:, :4] == 1.0)
    assert np.all(bank["plus_to_minus_after_4"][:, 4:] == -1.0)


def test_selected_contract_is_exact_and_antithetic() -> None:
    contract = json.loads((ROOT / "results/expanded_candidate_contract_restart/selected_interface.json").read_text())
    default = np.linspace(-0.8, 0.8, 3 * 8 * 7, dtype=np.float32).reshape(3, 8, 7)
    unit = generate_unit_noise(3, 24, contract["seed_base"])
    std = np.asarray([0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.0], dtype=np.float32)
    preclip, candidates, category, forms = build_contract_candidates(
        default,
        unit,
        contract["continuous_alternatives"],
        tuple(contract["structured_gripper_forms"]),
        std,
        contract["continuous_scale"],
    )
    exact = validate_exact_contract(preclip, candidates, category, forms, default)
    assert candidates.shape == (3, 64, 8, 7)
    assert np.sum(category == 1) == 48
    assert np.sum(category == 2) == 15
    assert exact["continuous_pair_count"] == 24
    assert exact["unmatched_continuous_direction_count"] == 0
    assert exact["max_float32_pair_antisymmetry_error"] <= 2e-7
    assert exact["g3_included"] is True
    assert exact["pass"] is True
