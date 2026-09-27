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
    for item in protocol["implementation"].values():
        assert sha256(ROOT / item["path"]) == item["sha256"]
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
