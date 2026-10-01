"""Independent checks of the numerical result and formal decision amendment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.probes.planning_architecture import sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/expanded_candidate_interface_design/diagnosis.json"
CORRECTED = ROOT / "results/expanded_candidate_interface_design/corrected_diagnosis.json"
AMENDMENT = ROOT / "configs/expanded_candidate_interface_decision_amendment.json"


def load_source() -> tuple[dict, np.lib.npyio.NpzFile]:
    result = json.loads(SOURCE.read_text())
    sidecar = ROOT / result["sidecar"]["path"]
    assert sha256(sidecar) == result["sidecar"]["sha256"]
    return result, np.load(sidecar)


def test_scope_and_finite_identity_boundary() -> None:
    result, arrays = load_source()
    assert result["scope"]["demo_ids"] == [36, 37, 38, 39]
    assert result["scope"]["sample_count"] == 4563
    assert set(arrays["demo_id"].tolist()) == {36, 37, 38, 39}
    assert set(arrays["task_id"].tolist()) == set(range(10))
    assert result["scope"]["simulator_runs"] == 0
    assert result["scope"]["reward_done_success_reads"] == 0
    assert result["scope"]["forbidden_demo_access"] is False
    assert all(np.isfinite(value).all() for value in arrays.values())


def test_all_family_gains_recompute_and_oracle_never_regresses() -> None:
    result, arrays = load_source()
    for name, family in result["families"].items():
        default = arrays[f"{name}__default_error"]
        oracle = arrays[f"{name}__oracle_error"]
        measured = (default.mean() - oracle.mean()) / default.mean()
        assert np.isclose(measured, family["relative_mse_gain"], atol=1e-12)
        assert np.all(oracle <= default + 1e-15)


def test_numerical_candidate_is_strong_but_contract_is_not_final() -> None:
    result, _ = load_source()
    family = result["families"]["ALLOC_G12"]
    assert np.isclose(family["relative_mse_gain"], 0.6283492932364835)
    assert family["relative_mse_gain_ci95"][0] > 0.60
    assert family["positive_task_count"] == 10
    assert family["prefix_oracle_relative_gain"]["1"] > 0.80
    assert family["continuous_candidate_count"] == 51
    assert family["structured_gripper_candidate_count"] == 12


def test_count_and_structured_gripper_findings() -> None:
    result, _ = load_source()
    family = result["families"]
    assert family["COUNT_N64"]["relative_mse_gain"] - family["COUNT_N32"]["relative_mse_gain"] > 0.015
    assert family["GRIP_G2_switches"]["relative_mse_gain"] > 0.57
    assert family["GRIP_Gall"]["relative_mse_gain"] > family["GRIP_G2_switches"]["relative_mse_gain"]
    assert family["ALLOC_G12"]["relative_mse_gain"] > family["ALLOC_G8"]["relative_mse_gain"] > family["ALLOC_G4"]["relative_mse_gain"]


def test_decision_amendment_hash_and_formal_authorization() -> None:
    corrected = json.loads(CORRECTED.read_text())
    amendment = json.loads(AMENDMENT.read_text())
    assert sha256(SOURCE) == amendment["source_numerical_diagnosis"]["sha256"]
    assert sha256(AMENDMENT) == corrected["decision_amendment"]["sha256"]
    assert corrected["contract_gate"]["pass"] is False
    assert corrected["selected_interface"] is None
    assert corrected["final_mechanism_labels"] == [
        "STRUCTURED_GRIPPER_PROPOSALS_VALID",
        "N64_REQUIRED_FOR_HEADROOM",
        "STRONG_EXPANDED_CANDIDATE_HEADROOM",
        "EXPANDED_CANDIDATE_INTERFACE_UNRESOLVED",
        "DATA_OR_IMPLEMENTATION_BLOCK",
    ]
    assert corrected["candidate_value_authorization"] == "BLOCK_CANDIDATE_VALUE_AUDIT"
    assert corrected["spatial_world_model_authorization"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert corrected["controller_authorization"] is False


def test_source_positive_mapping_is_explicitly_removed_not_rewritten() -> None:
    source = json.loads(SOURCE.read_text())
    amendment = json.loads(AMENDMENT.read_text())
    assert "EXPANDED_CANDIDATE_INTERFACE_VALID" in source["decision"]["final_mechanism_labels"]
    assert amendment["removed_label"] == "EXPANDED_CANDIDATE_INTERFACE_VALID"
    assert amendment["contract_failures"]["post_evaluation_repair_allowed"] is False

