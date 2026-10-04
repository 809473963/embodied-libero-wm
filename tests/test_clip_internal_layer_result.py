from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_stage_i_extraction_hashes_shapes_and_exact_projected_parity() -> None:
    manifest = json.loads((ROOT / "results/clip_internal_layer_feature_manifest.json").read_text())
    assert manifest["complete_gate"] is True
    assert manifest["frames"] == 5044
    assert manifest["implementation_audit"]["hidden_state_count"] == 13
    assert manifest["implementation_audit"]["prelayernorm_vs_hidden0_max_abs"] == 0.0
    assert manifest["prior_L12_proj_float16_exact"] is True
    assert manifest["prior_L12_proj_max_abs"] == 0.0
    for layer, row in manifest["layers"].items():
        path = ROOT / "results/clip_internal_layer/features" / f"{layer}.npy"
        assert sha256(path) == row["sha256"]
        value = np.load(path, mmap_mode="r")
        assert list(value.shape) == row["shape"]
        assert value.dtype == np.float16
    projected = np.load(ROOT / "results/clip_internal_layer/features/L12-proj.npy", mmap_mode="r")
    with np.load(ROOT / "results/spatial_visual_representation/patch_features_calibration.npz") as archive:
        assert np.array_equal(projected, archive["patch"])


def test_stage_i_primary_metrics_gates_and_decision_recompute() -> None:
    result = json.loads((ROOT / "results/clip_internal_layer_calibration.json").read_text())
    decision = json.loads((ROOT / "results/clip_internal_layer_final_decision.json").read_text())
    with np.load(ROOT / "results/t1_semantic_teacher/calibration_sidecar.npz") as teacher:
        demo34 = teacher["pair_demo_id"] == 34
        local_mask = teacher["pair_gap"][demo34] <= 8
    with np.load(ROOT / "results/clip_internal_layer/calibration_sidecar.npz") as data:
        ordinary_label = data["ordinary_label"]
        assert len(local_mask) == len(ordinary_label)
        local_non_tie = local_mask & (ordinary_label != 0)
        cf_direction = data["cf_alternate_label"].astype(float) - data["cf_original_label"].astype(float)
        changed = cf_direction != 0
        for layer in result["layers"]:
            probability = data[f"{layer}_P1_ordinary_probability"]
            accuracy = np.mean(
                np.sign(probability[local_non_tie] - 0.5) == ordinary_label[local_non_tie]
            )
            assert np.isclose(accuracy, result["layers"][layer]["P1"]["ordinary_local_non_tie"]["accuracy"])
            original = data[f"{layer}_P1_cf_original_probability"]
            alternate = data[f"{layer}_P1_cf_alternate_probability"]
            agreement = np.mean(((alternate - original) * cf_direction > 0)[changed])
            assert np.isclose(agreement, result["layers"][layer]["P1"]["language_counterfactual"]["teacher_change_agreement"])
    baseline = result["layers"]["L12-proj"]["P1"]
    for layer, gate in result["material_gates"].items():
        value = result["layers"][layer]["P1"]
        cf = 100.0 * (value["language_counterfactual"]["teacher_change_agreement"] - baseline["language_counterfactual"]["teacher_change_agreement"])
        match = baseline["cross_demo_semantic_match"]["mean_abs_from_half"] - value["cross_demo_semantic_match"]["mean_abs_from_half"]
        difference = 100.0 * (value["cross_demo_semantic_difference"]["accuracy"] - baseline["cross_demo_semantic_difference"]["accuracy"])
        ordinary = 100.0 * (value["ordinary_local_non_tie"]["accuracy"] - baseline["ordinary_local_non_tie"]["accuracy"])
        assert np.isclose(cf, gate["teacher_change_increment_points"])
        assert np.isclose(match, gate["cross_demo_match_improvement"])
        assert np.isclose(difference, gate["cross_demo_difference_increment_points"])
        assert np.isclose(ordinary, gate["ordinary_increment_points"])
        expected = layer != "L12-proj" and cf >= 4 and (match >= 0.03 or difference >= 7) and ordinary >= -2
        assert gate["complete"] is expected
    assert result["complete_gate"] is False
    assert result["selected_layer"] is None
    assert result["labels"] == [
        "CLIP_INTERNAL_SPATIAL_REPRESENTATION_INSUFFICIENT",
        "FROZEN_CLIP_BACKBONE_LIMIT_SUPPORTED",
    ]
    assert result["authorization"] == "GO_ALTERNATE_FROZEN_VISUAL_ENCODER_AUDIT"
    assert decision["semantic_value_redistillation_authorized"] is False
    assert decision["real_candidate_value_bridge_executed"] is False


def test_stage_i_diagnostic_logits_are_finite_and_recomputed() -> None:
    probes = json.loads((ROOT / "results/clip_internal_information_probes.json").read_text())
    with np.load(ROOT / "results/clip_internal_layer/calibration_sidecar.npz") as data:
        for layer, result in probes["probes"].items():
            for name in ("referent", "M2", "M2_cross", "goal", "goal_cross"):
                logits = data[f"{layer}_{name}_logits"]
                assert np.isfinite(logits).all()
            referent_accuracy = np.mean((data[f"{layer}_referent_logits"] > 0) == data["ref_label"])
            m2_accuracy = np.mean((data[f"{layer}_M2_cross_logits"] > 0) == (data["M2_cross_label"] == 1))
            goal_accuracy = np.mean((data[f"{layer}_goal_cross_logits"] > 0) == (data["goal_cross_label"] == 1))
            assert np.isclose(referent_accuracy, result["referent_change"]["accuracy"])
            assert np.isclose(m2_accuracy, result["M2_ordering"]["cross_demo"]["accuracy"])
            assert np.isclose(goal_accuracy, result["official_goal"]["cross_demo"]["accuracy"])
    assert probes["best_layers"] == {"referent": "L6", "M2": "L6", "official_goal": "L6"}
    assert probes["representative_layer"] == "L9"
    assert probes["interpretation_case"] == "B"
