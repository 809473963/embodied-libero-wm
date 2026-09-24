#!/usr/bin/env python3
"""Static preregistration and frozen-boundary checks for WM-v2C-EJ."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/wm_v2c_ej_protocol.json"
CONFIG = ROOT / "configs/world_model_v2c_ej_seed11.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load() -> tuple[dict, dict]:
    return json.loads(PROTOCOL.read_text()), yaml.safe_load(CONFIG.read_text())


def test_upstream_freeze_hashes_and_tags() -> None:
    protocol, _ = load()
    assert protocol["status"] == "PREREGISTERED_BEFORE_GRADIENT_CALIBRATION_AND_TRAINING"
    frozen = protocol["frozen_upstream"]
    assert frozen["wm_v2b_1_verdict"] == "PARTIAL_PHYSICAL_MEDIATOR_PASS"
    assert frozen["wm_v2b_1_authorization"] == "BLOCK_WM_V2C_PHYSICAL_MEDIATOR"
    for path_key, hash_key in (
        ("wm_v2b_1_result", "wm_v2b_1_result_sha256"),
        ("candidate_validation_sidecar", "candidate_validation_sidecar_sha256"),
        ("d13_store", "d13_store_sha256"),
        ("wm_v2b_checkpoint", "wm_v2b_checkpoint_sha256"),
        ("matched_counterfactual_result", "matched_counterfactual_result_sha256"),
        ("matched_counterfactual_sidecar", "matched_counterfactual_sidecar_sha256"),
    ):
        assert sha256(ROOT / frozen[path_key]) == frozen[hash_key]
    for tag_key, commit_key in (
        ("wm_v2b_1_tag", "wm_v2b_1_commit"),
        ("c0_v2_tag", "c0_v2_commit"),
        ("candidate_diagnosis_tag", "candidate_diagnosis_commit"),
        ("matched_counterfactual_tag", "matched_counterfactual_commit"),
    ):
        assert subprocess.check_output(
            ["git", "cat-file", "-t", frozen[tag_key]], cwd=ROOT, text=True
        ).strip() == "tag"
        assert subprocess.check_output(
            ["git", "rev-parse", f"{frozen[tag_key]}^{{}}"], cwd=ROOT, text=True
        ).strip() == frozen[commit_key]


def test_contract_architecture_loss_and_gripper_exclusion_are_frozen() -> None:
    protocol, config = load()
    assert sha256(CONFIG) == protocol["experiment"]["config_sha256"]
    assert protocol["data"]["causal_contract"] == (
        "(z[k], proprio[k], language, actions[k+1:k+9]) -> z[k+1:k+9]"
    )
    assert protocol["data"]["train_samples"] == 52381
    assert protocol["data"]["validation_samples"] == 5869
    assert protocol["data"]["ej_indices"] == [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
    assert protocol["data"]["gripper_indices_excluded"] == [3, 4]
    assert config["data"]["ej_target_indices"] == protocol["data"]["ej_indices"]
    assert config["data"]["gripper_indices_excluded_from_coupling"] == [3, 4]
    assert protocol["architecture"]["adapter"] == (
        "Linear(10,64)-GELU-Linear(64,128)-GELU-Linear(128,512)"
    )
    assert protocol["architecture"]["expected_adapter_parameters"] == 75072
    assert protocol["architecture"]["expected_total_parameters"] == 901516
    assert protocol["loss"]["lambda_res"] == config["loss"]["lambda_res"] == 0.001
    assert protocol["loss"]["fake_counterfactual_separation_loss"] is False
    assert protocol["gradient_calibration"]["split"] == "train demos 0-44 only"
    assert protocol["gradient_calibration"]["clip_range"] == [0.05, 20.0]


def test_gates_and_authorization_are_preregistered() -> None:
    protocol, _ = load()
    gate = protocol["gate"]
    assert gate["A_ordinary_visual"]["relative_degradation_vs_d14_max"] == 0.05
    assert gate["C_local_predicted_real_separation"]["transfer_ratio_min"] == (
        3.0 * protocol["matched_counterfactual"]["wm_v2b_transfer_ratio"]
    )
    assert gate["C_local_predicted_real_separation"]["multiple_vs_wm_v2b_min"] == 3.0
    assert gate["D_candidate_specific_prediction"][
        "mean_relative_degradation_vs_wm_v2b_max"
    ] == 0.05
    assert gate["pass_authorization"].startswith("GO_RERANKER_REDIAGNOSIS")
    forbidden = " ".join(protocol["forbidden"])
    for phrase in (
        "gripper dimensions 3/4",
        "matched branches 30-39",
        "access init states 40-49",
        "run controller success",
        "reinterpret the full 12-D mediator as passed",
    ):
        assert phrase in forbidden

