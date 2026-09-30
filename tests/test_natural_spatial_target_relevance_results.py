from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.probes.spatial_relevance import TARGET_NAMES, safe_correlation


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/natural_spatial_target_relevance_protocol.json"
FREEZE = ROOT / "configs/natural_spatial_target_relevance_calibration_freeze.json"
AMENDMENT = ROOT / "configs/natural_spatial_target_relevance_metadata_amendment.json"
RESULT = ROOT / "results/natural_spatial_target_relevance/diagnosis.json"
REPORT = ROOT / "docs/natural_spatial_target_relevance_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_completed_result_preserves_partition_and_forbidden_boundaries() -> None:
    result = json.loads(RESULT.read_text())
    assert result["status"] == "COMPLETE"
    assert result["protocol_sha256"] == sha256(PROTOCOL)
    assert result["calibration_freeze_sha256"] == sha256(FREEZE)
    assert result["data_boundary"] == {
        "calibration_init_states": [31, 32, 33, 34],
        "development_evaluation_init_states": [36, 37, 38, 39],
        "evaluation_is_fully_blind": False,
        "historical_reference_only": [30, 35],
        "init40_49_accessed": False,
        "natural_images_only": True,
        "rows": 960,
        "branch_points": 40,
    }
    assert result["nan_inf_audit"] == "pass"
    assert not any(result["forbidden_activity_audit"].values())
    assert result["decision"] == {
        "mechanism_labels": ["NATURAL_PATCH_TASK_RELEVANCE_UNRESOLVED"],
        "next_stage_authorization": "BLOCK_VISUAL_TARGET_REDESIGN",
        "controller_authorization": False,
    }
    assert not any(record["passes"] for record in result["family_decisions"].values())
    assert not any(record["passes_primary_gate"] for record in result["target_decisions"].values())


def test_derived_sidecar_identities_shapes_hash_and_finite_values() -> None:
    result = json.loads(RESULT.read_text())
    artifact = result["artifacts"]["evaluation_derived"]
    path = ROOT / artifact["path"]
    assert sha256(path) == artifact["sha256"]
    with np.load(path, allow_pickle=False) as data:
        assert set(np.unique(data["metadata__init_state"]).tolist()) == {36, 37, 38, 39}
        assert set(np.unique(data["metadata__task"]).tolist()) == set(range(10))
        assert set(np.unique(data["metadata__horizon"]).tolist()) == set(range(1, 9))
        assert set(np.unique(data["metadata__pair"]).tolist()) == {0, 1, 2}
        assert data["representation__global_summary"].shape == (960, 10)
        assert data["representation__patch_summary"].shape == (960, 10)
        assert data["representation__global_plus_patch"].shape == (960, 20)
        assert data["representation__global_pca16"].shape == (960, 16)
        assert data["representation__patch_pca16"].shape == (960, 16)
        assert data["diagnostic__distance_map"].shape == (960, 49)
        assert data["diagnostic__pair_fractional_coverage"].shape == (960, 4, 49)
        assert np.allclose(data["diagnostic__pair_fractional_coverage"].sum(axis=1), 1.0, atol=1e-6)
        for name in data.files:
            if np.issubdtype(data[name].dtype, np.number):
                assert np.isfinite(data[name]).all(), name


def test_probe_metrics_increments_and_raw_correlations_recompute() -> None:
    result = json.loads(RESULT.read_text())
    path = ROOT / result["artifacts"]["evaluation_derived"]["path"]
    with np.load(path, allow_pickle=False) as data:
        for target_name in TARGET_NAMES:
            target = data[f"target__{target_name}"].astype(np.float64)
            global_prediction = data[f"prediction__global_summary__{target_name}"].astype(np.float64)
            patch_prediction = data[f"prediction__patch_summary__{target_name}"].astype(np.float64)
            combined_prediction = data[f"prediction__global_plus_patch__{target_name}"].astype(np.float64)
            global_rmse = float(np.sqrt(np.mean(np.square(global_prediction - target))))
            patch_rmse = float(np.sqrt(np.mean(np.square(patch_prediction - target))))
            combined_rmse = float(np.sqrt(np.mean(np.square(combined_prediction - target))))
            assert np.isclose(
                global_rmse,
                result["probe_metrics"]["global_summary"][target_name]["overall"]["rmse"],
                atol=2e-8,
            )
            assert np.isclose(
                patch_rmse,
                result["probe_metrics"]["patch_summary"][target_name]["overall"]["rmse"],
                atol=2e-8,
            )
            assert np.isclose(
                combined_rmse,
                result["probe_metrics"]["global_plus_patch"][target_name]["overall"]["rmse"],
                atol=2e-8,
            )
            improvement = (global_rmse - combined_rmse) / global_rmse
            assert np.isclose(
                improvement,
                result["incremental_global_plus_patch"][target_name]["overall"],
                atol=2e-8,
            )
            global_distance = data["diagnostic__global_distance"].astype(np.float64)
            patch_mean = data["diagnostic__patch_mean"].astype(np.float64)
            global_rho = safe_correlation(global_distance, target)["spearman"]
            patch_rho = safe_correlation(patch_mean, target)["spearman"]
            assert np.isclose(
                global_rho,
                result["raw_correlations"][target_name]["global_distance"]["overall"]["spearman"],
                atol=2e-8,
            )
            assert np.isclose(
                patch_rho,
                result["raw_correlations"][target_name]["patch_mean"]["overall"]["spearman"],
                atol=2e-8,
            )


def test_metadata_amendment_is_transparent_and_does_not_change_result() -> None:
    amendment = json.loads(AMENDMENT.read_text())
    assert amendment["status"] == "POST_EVALUATION_METADATA_CLARIFICATION_NO_METRIC_CHANGE"
    assert amendment["base_protocol_sha256"] == sha256(PROTOCOL)
    assert amendment["diagnosis_sha256_before_amendment"] == sha256(RESULT)
    assert amendment["rerun_performed"] is False
    assert "32 pixels" in amendment["clarification"]
    assert "1.75" in amendment["clarification"]


def test_report_answers_required_questions_and_matches_verdict() -> None:
    text = REPORT.read_text()
    for phrase in (
        "NATURAL_PATCH_TASK_RELEVANCE_UNRESOLVED",
        "BLOCK_VISUAL_TARGET_REDESIGN",
        "Development-data exposure statement",
        "Natural-image-only representation contract",
        "Fractional semantic coverage",
        "Equal-rank PCA",
        "Patch and semantic shuffle controls",
        "Contact is a secondary",
        "Horizon behavior",
        "Per-task behavior",
        "Candidate-pair behavior",
        "Information localization",
        "Forbidden-activity audit",
        "Q1",
        "Q2",
        "Q3",
    ):
        assert phrase in text
    assert "GO_SPATIAL_TARGET_DESIGN" not in text.split("## Mechanical decision", 1)[1]
    assert sha256(RESULT) == "a21e5371927a80b78192afb3c13d6323407907273eee879799d1d0c2d86c4c46"
