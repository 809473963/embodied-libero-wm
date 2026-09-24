#!/usr/bin/env python3
"""Audit the calibration-only branch manifest and frozen scalar fit."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/wm_v2b_1_differential_calibration_protocol.json"
MANIFEST_PATH = ROOT / "results/world_model_v2b_1/branches/calibration_manifest.json"
FREEZE_PATH = ROOT / "configs/wm_v2b_1_calibration_freeze.json"
PREREGISTERED_COMMIT = "097e0571d18ce5995fda6903adf49f497449de81"
MANIFEST_SHA256 = "60faa70066f2c4a25ed093d1fed349c7db0b4a75746be1f692801e165e4532ac"
FREEZE_SHA256 = "9fa482a5ec3efe1d05142b27a373b3dc38633a2ce696b5c3ef92c82acdce39a8"
SIDECAR_SHA256 = "5e7c39e268b1696257107e2c8f068e2c4b40aa44f88ca8f6f92bc99d93d204d6"
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
GROUPS = {
    "ee_pos": np.asarray([0, 1, 2]),
    "gripper": np.asarray([3, 4]),
    "joints": np.asarray([5, 6, 7, 8, 9, 10, 11]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pairwise(value: np.ndarray) -> np.ndarray:
    return np.stack(
        [value[:, left] - value[:, right] for left, right in PAIR_INDICES], axis=1
    )


def fit_scalar(predicted: np.ndarray, real: np.ndarray) -> float:
    pred = predicted.reshape(-1).astype(np.float64)
    target = real.reshape(-1).astype(np.float64)
    return float(np.dot(pred, target) / np.dot(pred, pred))


def test_calibration_manifest_scope_prefix_identity_and_sidecar() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert sha256(MANIFEST_PATH) == MANIFEST_SHA256
    assert manifest["status"] == "COMPLETE"
    assert manifest["partition"] == "calibration"
    assert manifest["protocol"]["preregistered_commit"] == PREREGISTERED_COMMIT
    assert manifest["protocol"]["execution_source_commit"] == PREREGISTERED_COMMIT
    assert manifest["protocol"]["sha256"] == sha256(PROTOCOL_PATH)
    assert not any(manifest["forbidden_activity_audit"].values())

    without_hash = copy.deepcopy(manifest)
    expected_before_hash = without_hash["artifacts"].pop(
        "manifest_sha256_before_hash_field"
    )
    payload = (json.dumps(without_hash, indent=2, ensure_ascii=False) + "\n").encode()
    assert hashlib.sha256(payload).hexdigest() == expected_before_hash

    identities = manifest["branch_identities"]
    assert len(identities) == 40
    assert {
        (row["task_id"], row["init_state_index"], row["planning_index"])
        for row in identities
    } == {
        (task, init_state, 8)
        for task in range(10)
        for init_state in (31, 32, 33, 34)
    }
    assert all(row["prefix_steps"] == 64 for row in identities)
    assert all(row["prefix_audit"]["pass"] for row in identities)
    assert all(row["prefix_audit"]["rgb_exact"] for row in identities)
    assert all(
        row["prefix_audit"]["simulator_state_max_abs_difference"] == 0.0
        for row in identities
    )
    assert all(
        min(row["prefix_audit"]["clip_cosine_to_branch0"]) >= 0.999999
        for row in identities
    )

    sidecar_path = ROOT / protocol["artifacts"]["calibration_sidecar"]
    assert sha256(sidecar_path) == SIDECAR_SHA256
    assert manifest["artifacts"]["sidecar_sha256"] == SIDECAR_SHA256
    with np.load(sidecar_path) as sidecar:
        assert {name: list(sidecar[name].shape) for name in sidecar.files} == manifest[
            "artifacts"
        ]["sidecar_arrays"]
        assert sidecar["real_proprio"].shape == (40, 3, 8, 12)
        assert sidecar["pred_proprio_wm_v2b_standardized"].shape == (40, 3, 8, 12)
        assert np.array_equal(sidecar["candidates"], sidecar["executed_actions"])
        assert set(sidecar["task_id"].tolist()) == set(range(10))
        assert set(sidecar["init_state_index"].tolist()) == {31, 32, 33, 34}
        assert set(sidecar["planning_index"].tolist()) == {8}
        assert set(sidecar["prefix_lengths"].tolist()) == {64}
        assert all(
            np.isfinite(sidecar[name]).all()
            for name in sidecar.files
            if sidecar[name].dtype.kind in "fiu"
        )


def test_frozen_coefficients_recompute_from_calibration_only() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    freeze = json.loads(FREEZE_PATH.read_text())
    assert sha256(FREEZE_PATH) == FREEZE_SHA256
    assert freeze["status"] == "FROZEN_BEFORE_EVALUATION"
    assert freeze["protocol_sha256"] == sha256(PROTOCOL_PATH)
    assert freeze["fit_source_commit"] == PREREGISTERED_COMMIT
    assert freeze["calibration_manifest_sha256"] == MANIFEST_SHA256
    assert freeze["calibration_sidecar_sha256"] == SIDECAR_SHA256
    assert freeze["evaluation_artifacts_present_at_fit_time"] is False
    assert not any(freeze["forbidden_activity_audit"].values())

    checkpoint = torch.load(
        ROOT / protocol["frozen_upstream"]["wm_v2b_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    mean = np.asarray(checkpoint["normalization"]["proprio"]["mean"])
    std = np.asarray(checkpoint["normalization"]["proprio"]["std"])
    with np.load(ROOT / protocol["artifacts"]["calibration_sidecar"]) as sidecar:
        real = (sidecar["real_proprio"].astype(np.float64) - mean) / std
        predicted = sidecar["pred_proprio_wm_v2b_standardized"].astype(np.float64)
    real_delta = pairwise(real)
    predicted_delta = pairwise(predicted)
    coefficients = freeze["coefficients"]
    assert math.isclose(
        fit_scalar(predicted_delta, real_delta),
        coefficients["global_scalar"],
        abs_tol=1e-12,
    )
    assert np.allclose(
        [
            fit_scalar(predicted_delta[:, :, horizon], real_delta[:, :, horizon])
            for horizon in range(8)
        ],
        coefficients["per_horizon_scalar"],
        atol=1e-12,
        rtol=0,
    )
    for group, indices in GROUPS.items():
        assert math.isclose(
            fit_scalar(predicted_delta[..., indices], real_delta[..., indices]),
            coefficients["groupwise_scalar"][group],
            abs_tol=1e-12,
        )
    all_primary = [
        coefficients["global_scalar"],
        *coefficients["per_horizon_scalar"],
        *coefficients["groupwise_scalar"].values(),
    ]
    assert all(value > 0 for value in all_primary)
    # The immutable freeze records that evaluation did not exist when the fit
    # was performed.  Authorized held-out collection may happen later, so the
    # current filesystem is deliberately not used as evidence about fit time.
    assert freeze["evaluation_artifacts_present_at_fit_time"] is False
