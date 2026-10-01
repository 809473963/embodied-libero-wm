import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/continuous_footprint_decomposition/diagnosis.json"
DERIVED = ROOT / "artifacts/stage3/continuous_footprint_decomposition/evaluation_derived.npz"
FREEZE = ROOT / "configs/continuous_footprint_information_decomposition_calibration_freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_improvement(parent: np.ndarray, child: np.ndarray, target: np.ndarray) -> float:
    parent_rmse = float(np.sqrt(np.mean(np.square(parent - target))))
    child_rmse = float(np.sqrt(np.mean(np.square(child - target))))
    return (parent_rmse - child_rmse) / parent_rmse


def test_artifact_identity_and_one_shot_boundary():
    result = json.loads(RESULT.read_text())
    freeze = json.loads(FREEZE.read_text())
    assert result["artifacts"]["evaluation_derived"]["sha256"] == sha256(DERIVED)
    assert result["calibration_freeze_sha256"] == sha256(FREEZE)
    assert freeze["partition"]["evaluation_numeric_arrays_read"] is False
    assert result["data_boundary"]["development_evaluation_init_states"] == [36, 37, 38, 39]
    assert result["data_boundary"]["init40_49_accessed"] is False


def test_every_derived_numeric_array_is_finite():
    with np.load(DERIVED, allow_pickle=False) as data:
        assert data.files
        for name in data.files:
            assert np.isfinite(data[name]).all(), name


def test_all_primary_and_residual_increments_recompute():
    result = json.loads(RESULT.read_text())
    targets = ("object_mask_difference", "visible_area_difference", "object_mask_residual", "visible_area_residual")
    with np.load(DERIVED, allow_pickle=False) as data:
        for representation in ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "B2"):
            for target_name in targets:
                target = data[f"target__{target_name}"].astype(np.float64)
                parent = data[f"prediction__G__{target_name}"].astype(np.float64)
                child = data[f"prediction__{representation}__{target_name}"].astype(np.float64)
                actual = relative_improvement(parent, child, target)
                expected = result["incremental"][representation][target_name]["overall"]
                assert abs(actual - expected) < 1.0e-7


def test_f0_reproduces_upstream_and_f1_is_unique_minimum():
    result = json.loads(RESULT.read_text())
    assert abs(result["incremental"]["F0"]["object_mask_difference"]["overall"] - 0.30445) < 1.0e-4
    assert abs(result["incremental"]["F0"]["visible_area_difference"]["overall"] - 0.62641) < 1.0e-4
    f1 = result["gates"]["F1"]["targets"]["visible_area_difference"]
    assert f1["complete"] is True
    assert f1["positive_tasks"] == 10
    assert f1["positive_horizons"] == 8
    assert f1["positive_pairs"] == 3
    assert result["retention_vs_F0"]["F1"]["visible_area_difference"] >= 0.80
    assert result["decision"]["selected_minimal_oracle_target"] == "F1"
    assert result["decision"]["selected_dimension"] == 1


def test_area_does_not_replace_full_objmask_information():
    result = json.loads(RESULT.read_text())
    assert result["retention_vs_F0"]["F1"]["object_mask_difference"] < 0.80
    assert result["retention_vs_F0"]["F2"]["object_mask_difference"] >= 0.99
    binary = result["nested_analysis"]["area_to_binary_plus_area"]["object_mask_difference"]
    assert binary["overall"] < 0.05 or binary["ci95"][0] <= 0.0
    soft = result["nested_analysis"]["area_to_soft_map_plus_area"]["object_mask_difference"]
    assert soft["overall"] >= 0.05 and soft["ci95"][0] > 0.0


def test_decision_mapping_and_permanent_prototype_block():
    result = json.loads(RESULT.read_text())
    assert result["decision"]["mechanism_labels"] == [
        "VISIBLE_AREA_DOMINANT_TARGET",
        "MINIMAL_CONTINUOUS_FOOTPRINT_VALID",
        "BINARY_COMPONENT_REDUNDANT_GIVEN_AREA",
        "SOFT_OCCUPANCY_ADDS_VALUE_BEYOND_AREA",
        "OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED",
    ]
    assert result["decision"]["next_stage_authorization"] == "GO_MINIMAL_SPATIAL_EXTRACTION_AUDIT"
    assert result["decision"]["spatial_world_model_prototype_authorization"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert result["decision"]["alternative_spatial_encoder_audit_authorized"] is False
    assert result["decision"]["controller_authorization"] is False


def test_source_diagnosis_is_hash_frozen_in_git():
    result = json.loads(RESULT.read_text())
    source = subprocess.check_output(["git", "show", f"565a2c6df920b246a324c9f093a8ad3e51f59735:{RESULT.relative_to(ROOT)}"], cwd=ROOT)
    assert hashlib.sha256(source).hexdigest() == sha256(RESULT)
    assert result["nan_inf_audit"] == "pass"
    assert not any(result["forbidden_activity_audit"].values())
