"""Independent result checks for the Stage-B progress audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.probes.planning_architecture import sha256


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/goal_conditioned_pairwise_progress/diagnosis.json"
CALIBRATION = ROOT / "results/goal_conditioned_pairwise_progress/calibration_freeze.json"


def load_result() -> tuple[dict, np.lib.npyio.NpzFile]:
    result = json.loads(RESULT.read_text())
    sidecar = ROOT / result["sidecar"]["path"]
    assert sha256(sidecar) == result["sidecar"]["sha256"]
    return result, np.load(sidecar)


def test_hash_chain_scope_and_finite_arrays() -> None:
    result, arrays = load_result()
    assert result["calibration_freeze"]["sha256"] == sha256(CALIBRATION)
    calibration = json.loads(CALIBRATION.read_text())
    assert sha256(ROOT / calibration["weights"]["path"]) == calibration["weights"]["sha256"]
    assert sha256(ROOT / calibration["sidecar"]["path"]) == calibration["sidecar"]["sha256"]
    assert result["scope"]["demo_ids"] == [36, 37, 38, 39]
    assert result["scope"]["state_count"] == 4883
    assert result["scope"]["trajectory_count"] == 40
    assert result["scope"]["expert_action_inputs"] == 0
    assert result["scope"]["forbidden_demo_access"] is False
    assert all(np.isfinite(value).all() for value in arrays.values())


def test_pair_identity_and_local_accuracy_recompute() -> None:
    result, arrays = load_result()
    assert result["pair_count"] == 15360
    assert set(result["pair_counts_by_gap"].values()) == {2560}
    local = np.isin(arrays["gap_bin"], [0, 1, 2])
    labels = arrays["label"]
    for name in ("V0", "V1", "V2_visual_language", "V2_proprio_language", "V2_visual_proprio_language"):
        probability = arrays[f"{name}__probability"]
        accuracy = np.mean((probability[local] >= 0.5) == labels[local])
        assert np.isclose(accuracy, result["metrics"][name]["local_le8"]["accuracy"], atol=1e-12)


def test_visual_pairwise_signal_and_absolute_comparison() -> None:
    result, _ = load_result()
    primary = result["metrics"]["V2_visual_language"]
    v1 = result["metrics"]["V1"]
    assert primary["local_le8"]["accuracy"] > 0.74
    assert primary["local_accuracy_ci95"][0] > 0.73
    assert primary["positive_task_count"] == 10
    assert v1["local_le8"]["accuracy"] >= 0.65
    difference = primary["paired_local_differences"]["V1"]
    assert difference["accuracy_difference"] > 0.09
    assert difference["ci95"][0] > 0.07
    assert primary["symmetry"]["max_abs_complement_error"] < 1e-12


def test_language_gate_fails_materiality_despite_positive_drop() -> None:
    result, arrays = load_result()
    primary = result["metrics"]["V2_visual_language"]
    drop = primary["wrong_language_local_accuracy_drop"]
    assert 0 < primary["wrong_language_drop_ci95"][0]
    assert drop < 0.05
    local = np.isin(arrays["gap_bin"], [0, 1, 2])
    label = arrays["label"]
    correct = np.mean((arrays["V2_visual_language__probability"][local] >= 0.5) == label[local])
    wrong = np.mean((arrays["V2_visual_language__wrong_language_probability"][local] >= 0.5) == label[local])
    assert np.isclose(correct - wrong, drop, atol=1e-12)


def test_proprio_shortcut_is_decisive() -> None:
    result, _ = load_result()
    primary = result["metrics"]["V2_visual_language"]
    proprio = result["metrics"]["V2_proprio_language"]
    assert proprio["local_le8"]["accuracy"] > 0.96
    difference = primary["paired_local_differences"]["V2_proprio_language"]
    assert difference["accuracy_difference"] < -0.21
    assert difference["ci95"][1] < -0.20
    assert primary["cross_demo"]["matched_progress_mean_abs_probability_from_half"] > 0.45


def test_final_decision_blocks_stage_c() -> None:
    result, _ = load_result()
    decision = result["decision"]
    assert decision["pass"] is False
    assert decision["gates"] == {
        "local_pair": True,
        "language": False,
        "severe_robot_shortcut": True,
        "pairwise_noninferior_to_absolute": True,
        "absolute_baseline_valid": True,
        "pairwise_outperforms_absolute": True,
        "terminal_only": False,
    }
    assert decision["final_mechanism_labels"] == [
        "PAIRWISE_PROGRESS_SIGNAL_VALID",
        "ABSOLUTE_PROGRESS_BASELINE_VALID",
        "PAIRWISE_PROGRESS_OUTPERFORMS_ABSOLUTE",
        "PROGRESS_MODEL_LANGUAGE_INSENSITIVE",
        "PROGRESS_MODEL_ROBOT_PHASE_DOMINATED",
        "CANDIDATE_VALUE_AUDIT_BLOCKED",
    ]
    assert decision["world_model_interface_authorization"] == "BLOCK_WORLD_MODEL_INTERFACE_AUDIT"
    assert decision["spatial_world_model_authorization"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert decision["selected_value_model"] is None
    assert decision["controller_authorization"] is False
