import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results/image_space_target_audit_restart/diagnosis.json"
DERIVED_PATH = ROOT / "artifacts/stage3/image_space_target_audit_restart/evaluation_derived.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result() -> dict:
    return json.loads(RESULT_PATH.read_text())


def test_restart_result_and_derived_hashes() -> None:
    result = _result()
    assert _sha256(RESULT_PATH) == "1d9e90627491cb0dace90c7e03948d3d5588913dca8700f7be326272cf244ed9"
    assert result["artifacts"]["evaluation_derived"]["sha256"] == _sha256(DERIVED_PATH)
    assert _sha256(DERIVED_PATH) == "ecd1941ad0b9fb692e0ff99d53710b4ee6e75f161780171d6f9177aa2a22ed8f"


def test_restart_result_boundary() -> None:
    boundary = _result()["data_boundary"]
    assert boundary["calibration_init_states"] == [31, 32, 33, 34]
    assert boundary["development_evaluation_init_states"] == [36, 37, 38, 39]
    assert boundary["development_evaluation_is_blind_confirmation"] is False
    assert boundary["init40_49_accessed"] is False
    assert boundary["branch_points"] == 40
    assert boundary["rows"] == 960


def test_all_evaluation_arrays_are_finite_and_have_expected_identity() -> None:
    with np.load(DERIVED_PATH, allow_pickle=False) as arrays:
        for name in arrays.files:
            value = arrays[name]
            if np.issubdtype(value.dtype, np.number):
                assert np.isfinite(value).all(), name
        assert set(np.unique(arrays["metadata__task"]).tolist()) == set(range(10))
        assert set(np.unique(arrays["metadata__init_state"]).tolist()) == {36, 37, 38, 39}
        assert set(np.unique(arrays["metadata__horizon"]).tolist()) == set(range(1, 9))
        assert set(np.unique(arrays["metadata__pair"]).tolist()) == {0, 1, 2}


def test_reported_probe_rmse_and_increment_recompute() -> None:
    result = _result()
    with np.load(DERIVED_PATH, allow_pickle=False) as arrays:
        for target in result["target_decisions"]:
            valid = arrays[f"valid__{target}"]
            truth = arrays[f"target__{target}"][valid].astype(np.float64)
            recomputed = {}
            for representation in ("global_summary", "patch_summary", "global_plus_patch"):
                prediction = arrays[f"prediction__{representation}__{target}"][valid].astype(
                    np.float64
                )
                recomputed[representation] = float(np.sqrt(np.mean((prediction - truth) ** 2)))
                reported = result["probe_metrics"][representation][target]["overall"]["rmse"]
                assert abs(recomputed[representation] - reported) <= 2.0e-9
            increment = (
                recomputed["global_summary"] - recomputed["global_plus_patch"]
            ) / recomputed["global_summary"]
            assert abs(increment - result["target_decisions"][target]["increment"]) <= 1.0e-8


def test_object_family_clears_only_the_frozen_primary_gate() -> None:
    result = _result()
    family = result["family_decisions"]["object"]
    assert family["passes"] is True
    assert family["passing_targets"] == ["object_mask_difference", "visible_area_difference"]
    for target in family["passing_targets"]:
        decision = result["target_decisions"][target]
        assert decision["increment"] >= 0.10
        assert decision["increment_ci95"][0] > 0.0
        assert decision["full_patch_shuffle_pass"] is True
        assert decision["positive_tasks"] >= 7
        assert decision["positive_horizons"] >= 4
        assert decision["positive_pairs"] == 3


def test_object_semantic_attribution_remains_nonunique() -> None:
    result = _result()
    for target in result["family_decisions"]["object"]["passing_targets"]:
        semantic = result["shuffle_controls"][target]["relevant_patch_only"]
        assert semantic["passes"] is False
        assert semantic["ci95"][0] <= 0.0
    assert "SPATIAL_SIGNAL_REMAINS_NONUNIQUE" in result["decision"]["mechanism_labels"]
    assert "SPATIAL_OBJECT_IMAGE_SIGNAL_CONFIRMED" not in result["decision"]["mechanism_labels"]


def test_authorization_and_forbidden_activity_mapping() -> None:
    result = _result()
    decision = result["decision"]
    assert decision["next_stage_authorization"] == "GO_SPATIAL_TARGET_DESIGN"
    assert decision["controller_authorization"] is False
    assert decision["repeat_similar_representation_probe_if_unresolved"] is False
    assert result["old_block_preserved"]["verdicts"] == [
        "DATA_OR_IMPLEMENTATION_BLOCK",
        "CAMERA_PROJECTION_BLOCK",
    ]
    assert not any(result["forbidden_activity_audit"].values())
