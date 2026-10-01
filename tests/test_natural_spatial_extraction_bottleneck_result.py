import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/natural_spatial_extraction/diagnosis.json"
DERIVED = ROOT / "artifacts/stage3/natural_spatial_extraction/evaluation_derived.npz"
FREEZE = ROOT / "configs/natural_spatial_extraction_calibration_freeze.json"
AMENDMENT = ROOT / "configs/natural_spatial_extraction_decision_amendment.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def relative_improvement(parent: np.ndarray, child: np.ndarray, target: np.ndarray) -> float:
    parent_rmse = float(np.sqrt(np.mean(np.square(parent - target))))
    child_rmse = float(np.sqrt(np.mean(np.square(child - target))))
    return (parent_rmse - child_rmse) / parent_rmse


def test_artifact_identity_and_partition_boundary():
    result = json.loads(RESULT.read_text())
    freeze = json.loads(FREEZE.read_text())
    assert result["artifacts"]["evaluation_derived"]["sha256"] == sha256(DERIVED)
    assert result["calibration_freeze_sha256"] == sha256(FREEZE)
    assert result["decision_amendment"]["sha256"] == sha256(AMENDMENT)
    assert freeze["partition"]["evaluation_numeric_arrays_read"] is False
    assert result["data_boundary"]["calibration_init_states"] == [31, 32, 33, 34]
    assert result["data_boundary"]["development_evaluation_init_states"] == [36, 37, 38, 39]
    assert result["data_boundary"]["init40_49_accessed"] is False


def test_every_derived_numeric_array_is_finite():
    with np.load(DERIVED, allow_pickle=False) as data:
        assert data.files
        for name in data.files:
            assert np.isfinite(data[name]).all(), name


def test_oracle_increments_recompute_from_sidecar():
    result = json.loads(RESULT.read_text())
    with np.load(DERIVED, allow_pickle=False) as data:
        for representation in ("oracle__B0_soft_area", "oracle__B1_binary", "oracle__B2_binary_area"):
            for target_name in ("object_mask_difference", "visible_area_difference", "object_mask_residual", "visible_area_residual"):
                target = data[f"target__{target_name}"].astype(np.float64)
                parent = data[f"prediction__global__{target_name}"].astype(np.float64)
                child = data[f"prediction__{representation}__{target_name}"].astype(np.float64)
                actual = relative_improvement(parent, child, target)
                expected = result["incremental"][representation][target_name]["overall"]
                assert abs(actual - expected) < 1.0e-7


def test_binary_simplification_and_continuous_area_decision():
    result = json.loads(RESULT.read_text())
    b1 = result["gates"]["oracle__B1_binary"]
    b2 = result["gates"]["oracle__B2_binary_area"]
    assert b1["passes"] is False
    assert b1["primary_passing_targets"] == []
    assert b2["passes"] is True
    assert "visible_area_difference" in b2["oracle_retention_passing_targets"]
    assert result["incremental"]["oracle__B2_binary_area"]["visible_area_difference"]["overall"] >= 0.10


def test_binary_localization_passes_but_every_natural_bridge_fails():
    result = json.loads(RESULT.read_text())
    for capacity in ("C1_width32", "C2_width128_64", "C3_local_context"):
        name = f"natural__B2_binary_area__{capacity}"
        occupancy = result["natural_recovery"][name]["occupancy"]
        assert occupancy["iou"] >= 0.60
        assert occupancy["balanced_accuracy"] >= 0.80
        assert occupancy["positive_recall"] >= 0.70
        assert result["gates"][name]["passes"] is False
    assert not any(record["passes"] for name, record in result["gates"].items() if name.startswith("natural__"))


def test_corrected_decision_mapping_and_source_preservation():
    result = json.loads(RESULT.read_text())
    amendment = json.loads(AMENDMENT.read_text())
    expected = [
        "SOFT_FOOTPRINT_INFORMATION_REQUIRED",
        "VISIBLE_SCALE_SIGNAL_NEEDED",
        "LOCALIZATION_NOT_SUFFICIENT",
        "NATURAL_SPATIAL_EXTRACTION_BLOCK",
        "OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED",
    ]
    assert result["decision"]["mechanism_labels"] == expected
    assert amendment["corrected_mechanism_labels"] == expected
    assert result["decision"]["selected_target"] is None
    assert result["decision"]["next_stage_authorization"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert result["decision"]["block_type"] == "target simplification failure"
    assert result["decision"]["alternative_spatial_encoder_audit_authorized"] is False
    source = subprocess.check_output(
        ["git", "show", f"{amendment['source_numerical_diagnosis']['commit']}:{amendment['source_numerical_diagnosis']['path']}"],
        cwd=ROOT,
    )
    assert hashlib.sha256(source).hexdigest() == amendment["source_numerical_diagnosis"]["sha256"]


def test_forbidden_activity_mapping():
    result = json.loads(RESULT.read_text())
    assert result["nan_inf_audit"] == "pass"
    assert not any(result["forbidden_activity_audit"].values())
    assert result["decision"]["controller_authorization"] is False
