from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from src.probes.missing_state import (
    FEATURE_SET_GROUPS,
    fit_ridge,
    predict_ridge,
    quaternion_geodesic,
    rotation_pair_features,
    select_ridge_alpha,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/missing_state_audit_protocol.json"
AMENDMENT = ROOT / "configs/missing_state_audit_analysis_amendment.json"
FREEZE = ROOT / "configs/missing_state_audit_calibration_freeze.json"
CALIBRATION_MANIFEST = ROOT / "results/missing_state_audit/calibration_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_protocol_freeze_split_hierarchy_and_forbidden_boundaries() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["status"] == "PREREGISTERED_BEFORE_STATE_INVENTORY_OR_REPLAY"
    assert protocol["preflight_correction"]["version"] == "v1.1"
    assert protocol["frozen_upstream"]["c1_v2_config"] == "configs/c1_v2_rerank.yaml"
    assert sha256(ROOT / protocol["frozen_upstream"]["c1_v2_config"]) == protocol[
        "frozen_upstream"
    ]["c1_v2_config_sha256"]
    assert subprocess.check_output(
        ["git", "cat-file", "-t", protocol["frozen_upstream"]["primary_tag"]],
        cwd=ROOT,
        text=True,
    ).strip() == "tag"
    for item in protocol["frozen_upstream"]["files"]:
        assert sha256(ROOT / item["path"]) == item["sha256"]
    amendment = json.loads(AMENDMENT.read_text())
    assert amendment["base_protocol_sha256"] == sha256(PROTOCOL)
    assert amendment["status"] == "FROZEN_AFTER_HELD_OUT_REPLAY_AND_BEFORE_ANY_DIAGNOSIS_ARTIFACT"
    assert amendment["old_analyzer_sha256"] == protocol["implementation"]["analyzer"]["sha256"]
    assert amendment["new_analyzer_sha256"] == sha256(
        ROOT / protocol["implementation"]["analyzer"]["path"]
    )
    assert amendment["old_feature_module_sha256"] == protocol["implementation"][
        "feature_module"
    ]["sha256"]
    assert amendment["new_feature_module_sha256"] == sha256(
        ROOT / protocol["implementation"]["feature_module"]["path"]
    )
    collector = protocol["implementation"]["collector"]
    assert sha256(ROOT / collector["path"]) == collector["sha256"]
    assert protocol["partitions"]["calibration"]["init_state_indices"] == [31, 32, 33, 34]
    assert protocol["partitions"]["evaluation"]["init_state_indices"] == [36, 37, 38, 39]
    assert set(range(40, 50)).issubset(protocol["partitions"]["forbidden_init_state_indices"])
    assert protocol["scope"]["rows_per_branch"] == 24
    assert list(FEATURE_SET_GROUPS) == ["S0", "S1", "S2", "S3", "S4"]
    assert protocol["decision"]["controller_authorization"] is False
    forbidden = " ".join(protocol["forbidden"])
    for phrase in ("40-49", "future object/contact", "World Model", "controller success", "seed23/47"):
        assert phrase in forbidden


def test_quaternion_double_cover_and_rotation_representation() -> None:
    q = np.asarray([[0.2, -0.3, 0.1, 0.9]], dtype=np.float64)
    assert np.allclose(quaternion_geodesic(q, -q), 0.0, atol=1e-7)
    identity = np.eye(3)
    angle = np.pi / 4
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    rotvec, geodesic = rotation_pair_features(rotation, identity)
    assert rotvec.shape == (3,)
    assert np.isclose(geodesic, angle)


def test_ridge_standardization_cv_shapes_and_finite_values() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(96, 14))
    y = x @ rng.normal(size=(14, 1)) + 0.05 * rng.normal(size=(96, 1))
    init = np.repeat([31, 32, 33, 34], 24)
    alpha, cv = select_ridge_alpha(x, y, init, [1e-4, 1e-2, 1.0])
    assert alpha in [1e-4, 1e-2, 1.0]
    assert all(len(row["folds"]) == 4 for row in cv)
    fit = fit_ridge(x, y, alpha)
    prediction = predict_ridge(x, fit)
    assert prediction.shape == y.shape
    assert np.isfinite(prediction).all()
    assert np.all(fit.scale > 0)


def test_calibration_replay_and_freeze_precede_evaluation() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads(CALIBRATION_MANIFEST.read_text())
    freeze = json.loads(FREEZE.read_text())
    assert manifest["status"] == "COMPLETE"
    assert manifest["partition"] == "calibration"
    assert len(manifest["branch_identities"]) == 40
    assert {row["init_state_index"] for row in manifest["branch_identities"]} == {
        31,
        32,
        33,
        34,
    }
    assert all(row["prefix_audit"]["pass"] for row in manifest["branch_identities"])
    assert all(row["future_rgb_exact"] for row in manifest["branch_identities"])
    assert max(
        row["future_proprio_max_abs_difference"]
        for row in manifest["branch_identities"]
    ) == 0.0
    assert sha256(ROOT / manifest["artifact"]["path"]) == manifest["artifact"]["sha256"]

    assert freeze["status"] == "FROZEN_BEFORE_HELD_OUT_STATE_REPLAY_OR_EVALUATION"
    assert freeze["protocol_sha256"] == sha256(PROTOCOL)
    assert freeze["partition"]["evaluation_state_sidecar_read"] is False
    assert freeze["partition"]["rows"] == 960
    assert freeze["feature_group_dimensions"] == {
        "ej": 10,
        "rotation": 4,
        "object": 10,
        "contact": 13,
        "relative": 10,
    }
    assert [freeze["feature_sets"][name]["dimensions"] for name in FEATURE_SET_GROUPS] == [
        10,
        14,
        24,
        37,
        47,
    ]
    assert sha256(ROOT / freeze["weights"]["path"]) == freeze["weights"]["sha256"]
    assert freeze["object_motion_bin_thresholds_calibration_only"][0] < freeze[
        "object_motion_bin_thresholds_calibration_only"
    ][1]
    assert not any(freeze["forbidden_activity_audit"].values())
