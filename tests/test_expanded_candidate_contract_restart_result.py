"""Independent result checks for the corrected N64 candidate contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.probes.planning_architecture import sha256


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/expanded_candidate_contract_restart/diagnosis.json"
CALIBRATION = ROOT / "results/expanded_candidate_contract_restart/calibration_freeze.json"
CONTRACT = ROOT / "results/expanded_candidate_contract_restart/selected_interface.json"


def load_result() -> tuple[dict, np.lib.npyio.NpzFile]:
    result = json.loads(RESULT.read_text())
    sidecar = ROOT / result["sidecar"]["path"]
    assert sha256(sidecar) == result["sidecar"]["sha256"]
    return result, np.load(sidecar)


def test_artifact_hash_chain_and_partition_boundary() -> None:
    result, arrays = load_result()
    assert result["calibration_freeze"]["sha256"] == sha256(CALIBRATION)
    calibration = json.loads(CALIBRATION.read_text())
    assert calibration["selected_interface"]["sha256"] == sha256(CONTRACT)
    assert result["scope"]["demo_ids"] == [36, 37, 38, 39]
    assert result["scope"]["sample_count"] == 4563
    assert set(arrays["demo_id"].tolist()) == {36, 37, 38, 39}
    assert set(arrays["task_id"].tolist()) == set(range(10))
    assert result["scope"]["simulator_runs"] == 0
    assert result["scope"]["reward_done_success_reads"] == 0
    assert result["scope"]["forbidden_demo_access"] is False


def test_all_arrays_are_finite_and_oracle_errors_never_regress() -> None:
    result, arrays = load_result()
    assert all(np.isfinite(value).all() for value in arrays.values())
    for name, family in result["families"].items():
        default = arrays[f"{name}__default_error"]
        oracle = arrays[f"{name}__oracle_error"]
        assert np.all(oracle <= default + 1e-15)
        gain = (default.mean() - oracle.mean()) / default.mean()
        assert np.isclose(gain, family["relative_mse_gain"], atol=1e-12)


def test_calibration_selection_is_frozen_before_evaluation() -> None:
    calibration = json.loads(CALIBRATION.read_text())
    contract = json.loads(CONTRACT.read_text())
    assert calibration["selected_family"] == "C48_G15"
    assert contract["selected_family"] == "C48_G15"
    assert contract["N_total"] == 64
    assert contract["continuous_antithetic_pairs"] == 24
    assert contract["structured_gripper_alternatives"] == 15
    assert "c0_binarized" in contract["structured_gripper_forms"]


def test_selected_result_clears_every_stage_a_gate() -> None:
    result, _ = load_result()
    selected = result["families"]["C48_G15"]
    assert selected["relative_mse_gain"] >= 0.10
    assert selected["relative_mse_gain_ci95"][0] > 0
    assert selected["prefix_oracle_relative_gain"]["1"] >= 0.10
    assert selected["positive_task_count"] == 10
    assert selected["invalid_candidate_count"] == 0
    assert selected["clip_fraction"] <= 0.01
    assert selected["candidate_clip_fraction"] <= 0.25
    assert selected["boundary_hit_fraction"] <= 0.10
    assert selected["exact_contract"]["pass"] is True


def test_winner_composition_and_gripper_diagnostics() -> None:
    result, _ = load_result()
    selected = result["families"]["C48_G15"]
    winner = selected["winner_category_fraction"]
    assert np.isclose(sum(winner.values()), 1.0)
    assert winner["continuous"] > 0.50
    assert winner["structured_gripper"] > 0.35
    subtype = selected["structured_subtype_winner_fraction"]
    assert subtype["g3"] > 0.20
    assert subtype["single_switch"] > 0.05
    assert selected["gripper_diagnostics"]["binary_expert_fraction"] == 1.0
    assert selected["gripper_diagnostics"]["oracle_sign_accuracy"] > 0.99


def test_final_decision_and_authorization_are_exact() -> None:
    result, _ = load_result()
    decision = result["decision"]
    assert decision["pass"] is True
    assert decision["final_mechanism_labels"] == [
        "EXPANDED_CANDIDATE_INTERFACE_VALID",
        "STRUCTURED_GRIPPER_PROPOSALS_VALID",
        "N64_REQUIRED_FOR_HEADROOM",
        "STRONG_EXPANDED_CANDIDATE_HEADROOM",
    ]
    assert decision["candidate_value_authorization"] == "GO_CANDIDATE_VALUE_AUDIT"
    assert decision["spatial_world_model_authorization"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert decision["controller_authorization"] is False
