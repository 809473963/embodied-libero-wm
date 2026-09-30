from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from src.probes.spatial_relevance import (
    TARGET_NAMES,
    build_external_targets,
    contact_metrics,
    fit_logistic,
    fit_randomized_pca,
    fit_ridge,
    interaction_patch_indicator,
    predict_logistic,
    predict_ridge,
    safe_correlation,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/natural_spatial_target_relevance_protocol.json"
FREEZE = ROOT / "configs/natural_spatial_target_relevance_calibration_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_protocol_freezes_boundaries_natural_images_and_decision_gate() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["status"] == "PREREGISTERED_BEFORE_NATURAL_RELEVANCE_CALIBRATION_OR_EVALUATION"
    assert protocol["partitions"]["calibration"]["init_state_indices"] == [31, 32, 33, 34]
    assert protocol["partitions"]["development_evaluation"]["init_state_indices"] == [36, 37, 38, 39]
    assert protocol["partitions"]["development_evaluation"]["use"].startswith("one-shot")
    assert set(range(40, 50)).issubset(protocol["partitions"]["forbidden_init_state_indices"])
    assert protocol["natural_image_contract"]["rgb_modification"] == "none"
    assert protocol["natural_image_contract"]["attention_claim"] is False
    assert protocol["equal_capacity_summaries"]["patch_summary_10d"]
    assert len(protocol["equal_capacity_summaries"]["patch_summary_10d"]) == 10
    assert len(protocol["equal_capacity_summaries"]["global_summary_10d"]) == 3
    assert protocol["equal_rank_pca"]["rank"] == 16
    assert protocol["gates"]["minimum_incremental_rmse_improvement"] == 0.10
    assert protocol["gates"]["minimum_positive_tasks"] == 7
    assert protocol["decision"]["controller_authorization"] is False
    assert subprocess.check_output(
        ["git", "cat-file", "-t", protocol["frozen_upstream"]["primary_tag"]],
        cwd=ROOT,
        text=True,
    ).strip() == "tag"
    assert subprocess.check_output(
        ["git", "rev-parse", f'{protocol["frozen_upstream"]["primary_tag"]}^{{}}'],
        cwd=ROOT,
        text=True,
    ).strip() == protocol["frozen_upstream"]["primary_commit"]
    for item in protocol["frozen_upstream"]["files"]:
        assert sha256(ROOT / item["path"]) == item["sha256"]
    for item in protocol["implementation"].values():
        assert sha256(ROOT / item["path"]) == item["sha256"]


def test_interaction_region_is_geometry_defined_and_preserves_fractional_coverage() -> None:
    masks = np.zeros((4, 128, 128), dtype=bool)
    masks[0, 20:60, 20:60] = True
    masks[1, 50:90, 50:90] = True
    masks[3] = ~(masks[0] | masks[1])
    coverage = np.zeros((4, 49), dtype=np.float64)
    coverage[0, 8] = 0.25
    coverage[1, 8] = 0.10
    coverage[3] = 1.0 - coverage[0] - coverage[1]
    interaction = interaction_patch_indicator(masks, coverage, 32.0)
    assert interaction.shape == (49,)
    assert set(np.unique(interaction)).issubset({0.0, 1.0})
    assert interaction[8] == 1.0
    assert np.isclose(coverage[:, 8].sum(), 1.0)


def test_low_capacity_numeric_primitives_are_finite_and_shape_safe() -> None:
    rng = np.random.default_rng(224001)
    x = rng.normal(size=(96, 40))
    target = 0.7 * x[:, 0] - 0.2 * x[:, 1] + 0.05 * rng.normal(size=96)
    pca, score = fit_randomized_pca(x, components=16, seed=224101)
    assert pca.components.shape == (16, 40)
    assert score.shape == (96, 16)
    fit = fit_ridge(score, target, alpha=0.01)
    prediction = predict_ridge(score, fit).reshape(-1)
    assert np.isfinite(prediction).all()
    assert safe_correlation(prediction, target)["spearman"] > 0.8
    binary = (target > np.median(target)).astype(np.int64)
    logistic = fit_logistic(score, binary, alpha=1.0)
    probability = predict_logistic(score, logistic)
    metrics = contact_metrics(probability, binary)
    assert np.isfinite(probability).all()
    assert metrics["auroc"] is not None and metrics["auroc"] > 0.8


def test_external_target_contract_has_six_targets_and_960_rows() -> None:
    rng = np.random.default_rng(224002)
    branches = 40
    base = {
        "real_proprio": rng.normal(size=(branches, 3, 8, 12)),
        "task_id": np.repeat(np.arange(10), 4),
        "init_state_index": np.tile(np.arange(31, 35), 10),
    }
    identity = np.eye(3, dtype=np.float64)
    state = {
        "ee_rotmat": np.broadcast_to(identity, (branches, 3, 8, 3, 3)).copy(),
        "object_pos": rng.normal(size=(branches, 3, 8, 2, 3)),
        "object_rotmat": np.broadcast_to(identity, (branches, 3, 8, 2, 3, 3)).copy(),
        "object_mask": np.ones((branches, 2), dtype=bool),
        "contact_any": rng.integers(0, 2, size=(branches, 3, 8, 4)).astype(bool),
    }
    result = build_external_targets(base, state, np.zeros(12), np.ones(12))
    assert set(result["targets"]) == set(TARGET_NAMES)
    assert all(value.shape == (960,) for value in result["targets"].values())
    assert result["contact"].shape == (960,)
    assert result["metadata"]["horizon"].min() == 1
    assert result["metadata"]["horizon"].max() == 8
    assert np.isfinite(np.concatenate(list(result["targets"].values()))).all()


def test_calibration_is_hash_frozen_before_development_evaluation() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    freeze = json.loads(FREEZE.read_text())
    assert freeze["status"] == "FROZEN_BEFORE_INIT36_39_RELEVANCE_EVALUATION"
    assert freeze["protocol_sha256"] == sha256(PROTOCOL)
    assert freeze["partition"] == {
        "init_state_indices": [31, 32, 33, 34],
        "branch_points": 40,
        "rows": 960,
        "evaluation_features_or_state_read": False,
    }
    assert freeze["natural_image_only"] is True
    assert freeze["representation_dimensions"] == {
        "global_summary": 10,
        "patch_summary": 10,
        "global_plus_patch": 20,
        "global_pca16": 16,
        "patch_pca16": 16,
    }
    assert set(freeze["target_p75_calibration_only"]) == set(TARGET_NAMES)
    for artifact in freeze["artifacts"].values():
        assert sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert not any(freeze["forbidden_activity_audit"].values())
    assert protocol["interaction_region"] == freeze["interaction_region"]
