import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/spatial_visual_target_design/diagnosis.json"
DERIVED = ROOT / "artifacts/stage3/spatial_visual_target_design/evaluation_derived.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result() -> dict:
    return json.loads(RESULT.read_text())


def test_result_and_sidecar_hashes_are_frozen() -> None:
    result = _result()
    assert _sha256(RESULT) == "9d788cc7a6d64a29deede070b8b756e48543a6c89b685d25f3151a049cf72e97"
    assert _sha256(DERIVED) == "7cab9829b8c5b25bb9116acc4216382206d346b7caf7755e207edb3e14670754"
    assert result["artifacts"]["evaluation_derived"]["sha256"] == _sha256(DERIVED)


def test_partition_identity_and_all_numeric_arrays_are_finite() -> None:
    boundary = _result()["data_boundary"]
    assert boundary["calibration_init_states"] == [31, 32, 33, 34]
    assert boundary["development_evaluation_init_states"] == [36, 37, 38, 39]
    assert boundary["init40_49_accessed"] is False
    with np.load(DERIVED, allow_pickle=False) as arrays:
        for name in arrays.files:
            value = arrays[name]
            if np.issubdtype(value.dtype, np.number):
                assert np.isfinite(value).all(), name
        assert set(np.unique(arrays["metadata__task"]).tolist()) == set(range(10))
        assert set(np.unique(arrays["metadata__init_state"]).tolist()) == {36, 37, 38, 39}
        assert set(np.unique(arrays["metadata__horizon"]).tolist()) == set(range(1, 9))
        assert set(np.unique(arrays["metadata__pair"]).tolist()) == {0, 1, 2}


def test_reported_oracle_increment_recomputes_from_sidecar() -> None:
    result = _result()
    with np.load(DERIVED, allow_pickle=False) as arrays:
        for target in ("object_mask_difference", "visible_area_difference", "object_mask_residual", "visible_area_residual"):
            truth = arrays[f"target__{target}"].astype(np.float64)
            parent = arrays[f"prediction__global__{target}"].astype(np.float64)
            child = arrays[f"prediction__global_plus_footprint__{target}"].astype(np.float64)
            parent_rmse = float(np.sqrt(np.mean(np.square(parent - truth))))
            child_rmse = float(np.sqrt(np.mean(np.square(child - truth))))
            increment = (parent_rmse - child_rmse) / parent_rmse
            reported = result["incremental"]["D2_object_footprint"][target]["overall"]
            assert abs(increment - reported) <= 2.0e-8


def test_oracle_target_passes_but_natural_proxy_is_blocked() -> None:
    result = _result()
    oracle = result["design_gates"]["D2_object_footprint"]
    proxy = result["design_gates"]["D4_hybrid_global_proxy"]
    assert oracle["primary_passing_targets"] == ["object_mask_difference", "visible_area_difference"]
    assert oracle["residual_passing_targets"] == ["object_mask_residual", "visible_area_residual"]
    assert oracle["deployment_compatible"] is False
    assert result["natural_proxy_recovery"]["passes"] is False
    assert proxy["primary_passing_targets"] == []
    assert proxy["passes"] is False


def test_mechanical_decision_and_forbidden_activity() -> None:
    result = _result()
    labels = result["decision"]["mechanism_labels"]
    assert "OBJECT_FOOTPRINT_TARGET_VALID" in labels
    assert "ORACLE_SPATIAL_TARGET_VALID" in labels
    assert "NATURAL_SPATIAL_EXTRACTION_BLOCK" in labels
    assert "OBJECT_SIGNAL_BEYOND_ROBOT_MOTION_CONFIRMED" in labels
    assert "SPATIAL_TARGET_DESIGN_UNRESOLVED" in labels
    assert result["decision"]["selected_design"] is None
    assert result["decision"]["next_stage_authorization"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert result["decision"]["controller_authorization"] is False
    assert not any(result["forbidden_activity_audit"].values())
