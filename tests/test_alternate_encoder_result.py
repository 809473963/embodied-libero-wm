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


def test_stage_e_primary_metrics_and_material_gates_recompute() -> None:
    protocol = json.loads((ROOT / "configs/alternate_encoder_protocol.json").read_text())
    result = json.loads((ROOT / "results/alternate_encoder_calibration.json").read_text())
    with np.load(ROOT / "results/t1_semantic_teacher/calibration_sidecar.npz") as teacher:
        demo34 = teacher["pair_demo_id"] == 34
        local = teacher["pair_gap"][demo34] <= 8
    with np.load(ROOT / "results/alternate_frozen_encoder/calibration_sidecar.npz") as sidecar:
        ordinary_label = sidecar["ordinary_label"]
        local_non_tie = local & (ordinary_label != 0)
        cf_direction = (
            sidecar["cf_alternate_label"].astype(float)
            - sidecar["cf_original_label"].astype(float)
        )
        changed = cf_direction != 0
        for family, summary in result["families"].items():
            ordinary = sidecar[f"{family}_ordinary_probability"]
            accuracy = np.mean(
                np.sign(ordinary[local_non_tie] - 0.5) == ordinary_label[local_non_tie]
            )
            assert np.isclose(accuracy, summary["ordinary_local_non_tie"]["accuracy"])
            original = sidecar[f"{family}_cf_original_probability"]
            alternate = sidecar[f"{family}_cf_alternate_probability"]
            agreement = np.mean(((alternate - original) * cf_direction > 0)[changed])
            assert np.isclose(
                agreement, summary["language_counterfactual"]["teacher_change_agreement"]
            )
            baseline = protocol["baseline"]
            gate = result["material_gates"][family]
            cf_delta = 100 * (agreement - baseline["teacher_change"])
            ordinary_delta = 100 * (accuracy - baseline["ordinary"])
            match_gain = (
                baseline["cross_match"]
                - summary["cross_demo_semantic_match"]["mean_abs_from_half"]
            )
            difference_delta = 100 * (
                summary["cross_demo_semantic_difference"]["accuracy"]
                - baseline["cross_difference"]
            )
            assert np.isclose(cf_delta, gate["teacher_change_increment_points"])
            assert np.isclose(ordinary_delta, gate["ordinary_increment_points"])
            assert np.isclose(match_gain, gate["cross_demo_match_improvement"])
            assert np.isclose(
                difference_delta, gate["cross_demo_difference_increment_points"]
            )
            expected = (
                cf_delta >= 5
                and (difference_delta >= 7 or match_gain >= 0.03)
                and ordinary_delta >= -2
            )
            assert gate["complete"] is bool(expected)


def test_stage_e_diagnostic_logits_recompute_and_remain_non_authorizing() -> None:
    probes = json.loads((ROOT / "results/alternate_encoder_information_probes.json").read_text())
    with np.load(ROOT / "results/alternate_frozen_encoder/calibration_sidecar.npz") as sidecar:
        for family, result in probes["probes"].items():
            referent_logits = sidecar[f"{family}_referent_logits"]
            m2_cross_logits = sidecar[f"{family}_M2_cross_logits"]
            goal_cross_logits = sidecar[f"{family}_goal_cross_logits"]
            assert np.isfinite(referent_logits).all()
            assert np.isfinite(m2_cross_logits).all()
            assert np.isfinite(goal_cross_logits).all()
            assert np.isclose(
                np.mean((referent_logits > 0) == sidecar["ref_label"]),
                result["referent_change"]["accuracy"],
            )
            assert np.isclose(
                np.mean((m2_cross_logits > 0) == (sidecar["M2_cross_label"] == 1)),
                result["M2_ordering"]["cross_demo"]["accuracy"],
            )
            assert np.isclose(
                np.mean((goal_cross_logits > 0) == (sidecar["goal_cross_label"] == 1)),
                result["official_goal"]["cross_demo"]["accuracy"],
            )
    assert probes["best"] == {
        "referent": "SigLIP-B16-224",
        "M2": "DINOv2-S14",
        "official_goal": "DINOv2-S14",
    }
    assert probes["selection_authority"] is False


def test_stage_e_failure_decision_hashes_and_stop_boundary() -> None:
    result_path = ROOT / "results/alternate_encoder_calibration.json"
    result = json.loads(result_path.read_text())
    decision = json.loads((ROOT / "results/alternate_encoder_final_decision.json").read_text())
    assert result["complete_gate"] is False
    assert result["selected_encoder"] is None
    assert result["partition"]["evaluation_opened"] is False
    assert result["labels"] == [
        "FROZEN_VISUAL_BACKBONE_AUDIT_BLOCKED",
        "NATURAL_VISUAL_SEMANTIC_INFORMATION_UNRESOLVED",
    ]
    assert result["authorizations"] == [
        "BLOCK_SEMANTIC_VALUE_REDISTILLATION",
        "BLOCK_REAL_CANDIDATE_VALUE_BRIDGE",
    ]
    assert result["forbidden_activity"] == {
        "demo36_39_opened": False,
        "demo40_49_numeric_access": 0,
        "demo40_49_metadata_access": 0,
        "encoder_training": False,
        "stage_v": False,
        "candidate_bridge": False,
        "world_model": False,
        "controller": False,
    }
    assert decision["source_calibration_sha256"] == sha256(result_path)
    assert decision["selected_visual_encoder"] is None
    assert decision["semantic_value_redistillation_authorized"] is False
    assert decision["real_candidate_value_bridge_authorized"] is False
    assert decision["real_candidate_value_bridge_executed"] is False
    assert decision["stop_chain"] is True
    assert decision["controller_authorization"] is False
    for name, digest in result["artifacts"].items():
        paths = {
            "weights_sha256": ROOT / "results/alternate_frozen_encoder/calibration_weights.pt",
            "sidecar_sha256": ROOT / "results/alternate_frozen_encoder/calibration_sidecar.npz",
            "information_probes_sha256": ROOT / "results/alternate_encoder_information_probes.json",
            "feature_manifest_sha256": ROOT / "results/alternate_encoder_feature_manifest.json",
            "access_ledger_sha256": ROOT / "results/alternate_frozen_encoder/access_ledger.json",
        }
        assert sha256(paths[name]) == digest
