#!/usr/bin/env python3
"""Static preregistration and frozen-boundary audit for WM-v2B."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/wm_v2b_future_proprio_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    assert protocol["protocol_version"] == "wm-v2b-future-proprio-auxiliary-v1"
    assert protocol["status"] == "PREREGISTERED_BEFORE_CALIBRATION_AND_TRAINING"
    frozen = protocol["frozen_upstream"]
    assert frozen["wm_v2a_verdict"] == "DELTA_OBJECTIVE_INSUFFICIENT"
    assert subprocess.check_output(
        ["git", "cat-file", "-t", frozen["wm_v2a_annotated_tag"]],
        cwd=ROOT,
        text=True,
    ).strip() == "tag"
    assert subprocess.check_output(
        ["git", "rev-parse", f"{frozen['wm_v2a_annotated_tag']}^{{}}"],
        cwd=ROOT,
        text=True,
    ).strip() == frozen["wm_v2a_commit"]
    for path_key, hash_key in (
        ("wm_v2a_result", "wm_v2a_result_sha256"),
        ("wm_v2a_report", "wm_v2a_report_sha256"),
        ("wm_v2a_checkpoint", "wm_v2a_checkpoint_sha256"),
        ("d14_seed11_checkpoint", "d14_seed11_checkpoint_sha256"),
    ):
        assert sha256(ROOT / frozen[path_key]) == frozen[hash_key]
    assert frozen["may_be_overwritten"] is False

    data = protocol["data"]
    assert data["horizon"] == 8
    assert data["train_samples"] == 52381
    assert data["validation_samples"] == 5869
    assert data["future_proprio_is_input"] is False
    assert "target-only" in data["auxiliary_target"]
    assert sha256(ROOT / data["store"]) == data["store_sha256"]

    experiment = protocol["experiment"]
    config_path = ROOT / experiment["config"]
    assert sha256(config_path) == experiment["config_sha256"]
    config = yaml.safe_load(config_path.read_text())
    d14 = yaml.safe_load((ROOT / "configs/world_model.yaml").read_text())
    backbone = dict(config["model"])
    assert backbone.pop("proprio_head_hidden_dim") == 128
    assert backbone == d14["model"]
    assert config["optimization"] == d14["optimization"]
    assert experiment["expected_total_parameters"] == 826444
    assert config["loss"]["lambda_p"] == "from_frozen_gradient_calibration"

    calibration = protocol["calibration"]
    assert calibration["split"] == "train only"
    assert calibration["selection_seed"] == 220501
    assert calibration["sample_count"] == 1024
    assert calibration["batch_count"] == 4
    assert calibration["batch_size"] == 256
    assert calibration["device"] == "cpu float32"
    assert calibration["shared_parameter_set"] == "model.gru.parameters() only"
    assert calibration["clip_range"] == [0.05, 20.0]
    assert calibration["validation_metrics_used"] is False

    gate = protocol["gate"]
    assert gate["visual_local_sensitivity"][
        "candidate_separation_to_new_model_error_min"
    ] == 0.01
    assert gate["visual_local_sensitivity"][
        "ratio_to_frozen_wm_v1_reference_min"
    ] == 2.0
    assert gate["proprio_action_conditioning"][
        "within_shuffle_relative_error_increase_min"
    ] == 0.05
    assert gate["proprio_local_separation"][
        "candidate_trajectory_rmse_separation_to_correct_prediction_rmse_min"
    ] == 0.05
    assert gate["seed23_seed47_authorized_only_if"] == (
        "WM_V2B_LOCAL_SENSITIVITY_PASS"
    )
    forbidden = " ".join(protocol["forbidden"])
    for phrase in (
        "future proprio as model input",
        "simulator or controller",
        "30-39 and 40-49",
        "seed23 or seed47",
        "fake wrong-action counterfactual",
    ):
        assert phrase in forbidden
    print(
        {
            "status": "PASS",
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "config_sha256": sha256(config_path),
            "frozen_commit": frozen["wm_v2a_commit"],
        }
    )


if __name__ == "__main__":
    main()
