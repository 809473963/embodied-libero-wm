#!/usr/bin/env python3
"""Audit the held-out WM-v2B.1 collection before metric analysis."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/wm_v2b_1_differential_calibration_protocol.json"
FREEZE_PATH = ROOT / "configs/wm_v2b_1_calibration_freeze.json"
MANIFEST_PATH = ROOT / "results/world_model_v2b_1/branches/evaluation_manifest.json"
PREREGISTERED_COMMIT = "097e0571d18ce5995fda6903adf49f497449de81"
CALIBRATION_FREEZE_COMMIT = "2392a42f8d5cd4d7161efd2ee230cbb1e6126b4f"
MANIFEST_SHA256 = "17cab45c59910f52c66aa50c7e635bdf2ee4db45a954256a8422891759f7a586"
SIDECAR_SHA256 = "76ff1ee009b686c5d71cd7968c37aeec02afcce9485bd3f1bd8ea232faca60fe"
FREEZE_SHA256 = "9fa482a5ec3efe1d05142b27a373b3dc38633a2ce696b5c3ef92c82acdce39a8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_evaluation_manifest_scope_prefix_identity_and_sidecar() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    freeze = json.loads(FREEZE_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())

    assert sha256(MANIFEST_PATH) == MANIFEST_SHA256
    assert sha256(FREEZE_PATH) == FREEZE_SHA256
    assert manifest["status"] == "COMPLETE"
    assert manifest["partition"] == "evaluation"
    assert manifest["protocol"]["preregistered_commit"] == PREREGISTERED_COMMIT
    assert manifest["protocol"]["execution_source_commit"] == CALIBRATION_FREEZE_COMMIT
    assert manifest["protocol"]["sha256"] == sha256(PROTOCOL_PATH)
    assert freeze["status"] == "FROZEN_BEFORE_EVALUATION"
    assert freeze["evaluation_artifacts_present_at_fit_time"] is False
    assert not any(manifest["forbidden_activity_audit"].values())
    assert manifest["nan_inf_audit"] == "pass"

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
        for init_state in (36, 37, 38, 39)
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

    sidecar_path = ROOT / protocol["artifacts"]["evaluation_sidecar"]
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
        assert set(sidecar["init_state_index"].tolist()) == {36, 37, 38, 39}
        assert set(sidecar["episode_position"].tolist()) == {6, 7, 8, 9}
        assert set(sidecar["planning_index"].tolist()) == {8}
        assert set(sidecar["prefix_lengths"].tolist()) == {64}
        assert all(
            np.isfinite(sidecar[name]).all()
            for name in sidecar.files
            if sidecar[name].dtype.kind in "fiu"
        )
