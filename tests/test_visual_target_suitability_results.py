from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

from src.probes.missing_state import scalar_metrics


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/visual_target_suitability_protocol.json"
FREEZE = ROOT / "configs/visual_target_suitability_calibration_freeze.json"
RESULT = ROOT / "results/visual_target_suitability/diagnosis.json"
REPORT = ROOT / "docs/visual_target_suitability_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_result_scope_hashes_and_forbidden_boundaries() -> None:
    result = json.loads(RESULT.read_text())
    assert result["status"] == "COMPLETE"
    assert result["protocol"]["sha256"] == sha256(PROTOCOL)
    assert result["calibration_freeze"]["sha256"] == sha256(FREEZE)
    assert result["evaluation_scope"] == {
        "init_state_indices": [36, 37, 38, 39],
        "branch_points": 40,
        "rows": 960,
        "tasks": list(range(10)),
        "pairs": ["A0-A1", "A0-A2", "A1-A2"],
        "horizons": list(range(1, 9)),
    }
    for artifact in result["artifacts"].values():
        assert sha256(ROOT / artifact["path"]) == artifact["sha256"]
    forbidden = result["forbidden_activity_audit"]
    assert forbidden["simulator_used_only_for_masks"] is True
    assert not any(value for key, value in forbidden.items() if key != "simulator_used_only_for_masks")
    assert result["controller_authorized"] is False


def test_dense_metrics_recompute_probe_errors_and_energy_partition() -> None:
    result = json.loads(RESULT.read_text())
    metrics = np.load(ROOT / result["artifacts"]["evaluation_metrics"]["path"], allow_pickle=False)
    target = metrics["target_clip"].astype(np.float64)
    variants = metrics["variant_distances"].astype(np.float64)
    assert target.shape == (960,)
    assert variants.shape == (960, 6)
    assert np.isfinite(target).all() and np.isfinite(variants).all()
    assert np.max(np.abs(target - variants[:, 0])) <= 2e-5
    assert np.isclose(target.mean(), result["clip_target"]["overall"]["mean"], atol=1e-9)
    for name in ("constant", "V0", "V1", "V2", "V3", "V4"):
        prediction = metrics[f"prediction__{name}"].astype(np.float64)
        measured = scalar_metrics(prediction, target)
        recorded = result["probe_models"][name]
        for metric in ("rmse", "mae", "r2"):
            assert np.isclose(measured[metric], recorded[metric], atol=1e-9)
    contributions = sum(
        metrics[f"scalar__{name}_energy_contribution"].astype(np.float64)
        for name in ("robot", "relevant", "other", "background")
    )
    assert np.allclose(contributions, 1.0, atol=1e-6)


def test_masked_patch_and_action_diagnostics_recompute() -> None:
    result = json.loads(RESULT.read_text())
    metrics = np.load(ROOT / result["artifacts"]["evaluation_metrics"]["path"], allow_pickle=False)
    target = metrics["target_clip"].astype(np.float64)
    variants = metrics["variant_distances"].astype(np.float64)
    robot_removed_drop = (variants[:, 0] - variants[:, 3]) / variants[:, 0]
    relevant_removed_drop = (variants[:, 0] - variants[:, 4]) / variants[:, 0]
    background_retained = variants[:, 5] / variants[:, 0]
    decision = result["mechanical_decision"]["dominance_values"]
    assert np.isclose(np.median(robot_removed_drop), decision["robot_removed_drop_median"], atol=1e-6)
    assert np.isclose(np.median(relevant_removed_drop), decision["relevant_removed_drop_median"], atol=1e-6)
    assert np.isclose(np.median(background_retained), decision["background_only_retained_median"], atol=1e-6)

    patch_mean = metrics["scalar__patch_mean"].astype(np.float64)
    action_translation = metrics["scalar__action_translation"].astype(np.float64)
    measured_patch = float(stats.spearmanr(patch_mean, target).statistic)
    measured_action = float(stats.spearmanr(action_translation, target).statistic)
    assert np.isclose(measured_patch, result["pixel_associations"]["patch_mean"]["overall"]["spearman"], atol=1e-9)
    assert np.isclose(
        measured_action,
        result["action_geometry_associations"]["action_translation"]["D_clip"]["overall"]["spearman"],
        # The compact sidecar stores float32 scalars whereas the immutable
        # diagnosis used float64 before serialization; near-ties can alter
        # Spearman at the fifth decimal without changing the ranking result.
        atol=1e-4,
    )


def test_mechanical_decision_is_global_compression_without_redesign_authorization() -> None:
    result = json.loads(RESULT.read_text())
    assert result["final_mechanism_labels"] == ["GLOBAL_CLIP_SPATIAL_COMPRESSION"]
    assert result["next_stage_authorization"] == "BLOCK_VISUAL_TARGET_REDESIGN"
    checks = result["mechanical_decision"]["checks"]
    assert checks == {
        "robot_group_pass": False,
        "relevant_group_pass": False,
        "visibility_group_pass": False,
        "patch_group_pass": True,
        "robot_dominates": False,
        "object_dominates": False,
        "occlusion_drives": False,
        "background_sensitivity": False,
        "patch_better": False,
        "global_compression": True,
        "task_dependent": False,
        "clip_aligns": False,
        "global_unsuitable": False,
    }
    assert result["probe_models"]["increments"]["V4"]["passes"] is True
    assert result["group_shuffle_controls"]["patch"]["passes"] is True
    assert result["mechanical_decision"]["dominance_values"]["global_to_top_patch_median_ratio"] <= 0.25
    assert result["final_mechanism_labels"] != ["GLOBAL_CLIP_TARGET_UNSUITABLE_FOR_LOCAL_DYNAMICS"]


def test_report_preserves_verdict_boundaries_and_artifact_hashes() -> None:
    result = json.loads(RESULT.read_text())
    report = REPORT.read_text()
    for text in (
        "GLOBAL_CLIP_SPATIAL_COMPRESSION",
        "BLOCK_VISUAL_TARGET_REDESIGN",
        "Controller authorization: `false`",
        "do **not** clear the preregistered robot-dominance gate",
        "masking is an OOD",
        "attention attribution claim",
        "init40--49",
        sha256(PROTOCOL),
        sha256(FREEZE),
        sha256(RESULT),
        result["artifacts"]["evaluation_feature_sidecar"]["sha256"],
        result["artifacts"]["evaluation_metrics"]["sha256"],
    ):
        assert text in report
    assert "cannot claim any of the stronger conclusions" in report
    assert "do not replace it without a separately" in report
