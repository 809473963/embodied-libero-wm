from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.probes.semantic_value_distillation import non_tie_metrics


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load() -> tuple[dict, dict[str, np.ndarray]]:
    result = json.loads((ROOT / "results/spatial_visual_representation_calibration.json").read_text())
    archive = np.load(ROOT / "results/spatial_visual_representation/calibration_sidecar.npz")
    return result, {name: archive[name] for name in archive.files}


def test_stage_r_artifact_hashes_and_patch_integrity() -> None:
    result, _ = load()
    artifacts = result["artifacts"]
    assert sha256(ROOT / "results/spatial_visual_representation/calibration_weights.pt") == artifacts["weights_sha256"]
    assert sha256(ROOT / "results/spatial_visual_representation/calibration_sidecar.npz") == artifacts["sidecar_sha256"]
    assert sha256(ROOT / "results/spatial_visual_representation/patch_features_calibration.npz") == artifacts["patch_features_sha256"]
    integrity = json.loads((ROOT / "results/spatial_visual_representation_patch_integrity.json").read_text())
    assert integrity["complete_gate"] is True
    assert integrity["patch_shape"] == [5044, 49, 512]
    assert integrity["repeated_float16_exact"] is True
    assert integrity["global_parity"]["maximum_absolute_error"] == 0.0
    assert integrity["clip_trainable_parameter_tensors"] == 0
    assert integrity["clip_parameter_gradients_present"] is False


def test_stage_r_recomputes_ordinary_and_language_metrics() -> None:
    result, side = load()
    local = np.abs(side["ordinary_gap"]) <= 8
    non_tie = local & (side["ordinary_label"] != 0)
    changed = side["cf_original_label"] != side["cf_alternate_label"]
    direction = side["cf_alternate_label"].astype(float) - side["cf_original_label"].astype(float)
    reversal = side["cf_original_label"] * side["cf_alternate_label"] == -1
    for family in ("R0", "R1", "R2", "R3"):
        probability = side[f"{family}_ordinary_probability"]
        ordinary = non_tie_metrics(probability[non_tie], side["ordinary_label"][non_tie])
        np.testing.assert_allclose(
            ordinary["accuracy"], result["families"][family]["ordinary_local_non_tie"]["accuracy"]
        )
        np.testing.assert_allclose(
            ordinary["auroc"], result["families"][family]["ordinary_local_non_tie"]["auroc"],
            atol=1e-6,
        )
        original = side[f"{family}_cf_original_probability"]
        alternate = side[f"{family}_cf_alternate_probability"]
        agreement = (alternate - original) * direction > 0
        np.testing.assert_allclose(
            agreement[changed].mean(),
            result["families"][family]["language_counterfactual"]["teacher_change_agreement"],
        )
        original_correct = np.sign(original - 0.5) == side["cf_original_label"]
        alternate_correct = np.sign(alternate - 0.5) == side["cf_alternate_label"]
        np.testing.assert_allclose(
            (original_correct & alternate_correct)[reversal].mean(),
            result["families"][family]["language_counterfactual"]["true_reversal"]["both_correct_rate"],
        )


def test_stage_r_recomputes_cross_demo_and_material_gates() -> None:
    result, side = load()
    baseline = result["families"]["R0"]
    for family in ("R0", "R1", "R2", "R3"):
        matched = np.abs(side[f"{family}_cross_match_probability"] - 0.5).mean()
        different = np.mean(
            np.sign(side[f"{family}_cross_difference_probability"] - 0.5)
            == side["cross_difference_label"]
        )
        np.testing.assert_allclose(matched, result["families"][family]["cross_demo_semantic_match"]["mean_abs_from_half"])
        np.testing.assert_allclose(different, result["families"][family]["cross_demo_semantic_difference"]["accuracy"])
    for family in ("R1", "R2", "R3"):
        current = result["families"][family]
        expected = {
            "teacher_change_increment_points": 100 * (
                current["language_counterfactual"]["teacher_change_agreement"]
                - baseline["language_counterfactual"]["teacher_change_agreement"]
            ),
            "cross_demo_match_mean_abs_improvement": (
                baseline["cross_demo_semantic_match"]["mean_abs_from_half"]
                - current["cross_demo_semantic_match"]["mean_abs_from_half"]
            ),
            "cross_demo_difference_increment_points": 100 * (
                current["cross_demo_semantic_difference"]["accuracy"]
                - baseline["cross_demo_semantic_difference"]["accuracy"]
            ),
            "ordinary_accuracy_increment_points": 100 * (
                current["ordinary_local_non_tie"]["accuracy"]
                - baseline["ordinary_local_non_tie"]["accuracy"]
            ),
        }
        for key, value in expected.items():
            np.testing.assert_allclose(value, result["material_gates"][family][key])
        assert result["material_gates"][family]["complete"] is False


def test_attention_diagnostics_are_finite_and_r1_is_language_invariant() -> None:
    result, side = load()
    for family, slots in (("R1", 1), ("R2", 1), ("R3", 2)):
        original = side[f"{family}_cf_original_attention"]
        alternate = side[f"{family}_cf_alternate_attention"]
        assert original.shape == (67112, slots, 49)
        assert np.isfinite(original).all() and np.isfinite(alternate).all()
        np.testing.assert_allclose(original.sum(axis=-1), 1.0, atol=2e-3)
        np.testing.assert_allclose(alternate.sum(axis=-1), 1.0, atol=2e-3)
    assert np.array_equal(side["R1_cf_original_attention"], side["R1_cf_alternate_attention"])
    assert result["families"]["R1"]["attention_diagnostics"]["language_swap_total_variation_mean"] == [0.0]


def test_stage_r_stops_chain_without_evaluation_access() -> None:
    result, _ = load()
    decision = json.loads((ROOT / "results/spatial_visual_representation_final_decision.json").read_text())
    ledger = json.loads((ROOT / "results/spatial_visual_representation/calibration_access_ledger.json").read_text())
    assert result["complete_gate"] is False
    assert result["selected_representation"] is None
    assert result["labels"] == [
        "CLIP_PATCH_REPRESENTATION_INSUFFICIENT",
        "FROZEN_CLIP_VISUAL_INFORMATION_LIMIT_SUPPORTED",
    ]
    assert result["authorization"] == "BLOCK_SEMANTIC_VALUE_REDISTILLATION"
    assert decision["stop_chain"] is True
    assert decision["world_model_authorization"] is False
    assert decision["controller_authorization"] is False
    assert ledger["stage_r_opened_demo_ids"] == [31, 32, 33, 34]
    assert ledger["stage_r_not_opened_demo_ids"] == [36, 37, 38, 39]
    assert ledger["forbidden_numeric_access"] == 0
    assert ledger["forbidden_metadata_access"] == 0


def test_stage_r_sidecar_is_finite() -> None:
    _, side = load()
    for name, value in side.items():
        assert np.isfinite(value).all(), name
