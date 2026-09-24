#!/usr/bin/env python3
"""Static preregistration and frozen-source audit for WM-v2A."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/wm_v2a_local_dynamics_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    assert protocol["protocol_version"] == "wm-v2a-local-dynamics-resolution-v1"
    assert protocol["status"] == "PREREGISTERED_BEFORE_TRAINING"

    frozen = protocol["frozen_wm_v1"]
    assert subprocess.check_output(
        ["git", "cat-file", "-t", frozen["annotated_tag"]],
        cwd=ROOT,
        text=True,
    ).strip() == "tag"
    assert subprocess.check_output(
        ["git", "rev-parse", f"{frozen['annotated_tag']}^{{}}"],
        cwd=ROOT,
        text=True,
    ).strip() == frozen["commit"]
    for path_key, hash_key in (
        ("c1_v2_2_result", "c1_v2_2_result_sha256"),
        ("c1_v2_2_report", "c1_v2_2_report_sha256"),
        ("d14_seed11_checkpoint", "d14_seed11_checkpoint_sha256"),
        ("d15_transition_result", "d15_transition_result_sha256"),
    ):
        assert sha256(ROOT / frozen[path_key]) == frozen[hash_key]
    assert frozen["may_be_overwritten"] is False

    data = protocol["data"]
    assert data["pairing"] == (
        "(z[k], proprio[k], language, actions[k+1:k+9]) -> z[k+1:k+9]"
    )
    assert data["horizon"] == 8
    assert data["train_demo_ids"] == "0-44"
    assert data["validation_demo_ids"] == "45-49"
    assert data["train_samples"] == 52381
    assert data["validation_samples"] == 5869
    assert sha256(ROOT / data["store"]) == data["store_sha256"]

    experiment = protocol["experiment"]
    config_path = ROOT / experiment["config"]
    assert sha256(config_path) == experiment["config_sha256"]
    config = yaml.safe_load(config_path.read_text())
    d14_config = yaml.safe_load((ROOT / "configs/world_model.yaml").read_text())
    assert config["model"] == d14_config["model"]
    assert config["optimization"] == d14_config["optimization"]
    assert config["normalization"] == d14_config["normalization"]
    assert config["data"]["horizon"] == d14_config["data"]["horizon"] == 8
    assert config["run"]["seed"] == d14_config["run"]["seed"] == 11
    assert config["loss"]["lambda_delta"] == experiment["lambda_delta"] == 1.0
    assert experiment["lambda_candidates"] == [1.0]
    assert experiment["horizon_weighting"] == "equal"
    assert experiment["single_changed_factor"] == (
        "add supervised latent-delta SmoothL1 term"
    )

    diagnostics = protocol["frozen_diagnostics"]
    for path_key, hash_key in (
        ("c1_v2_2_protocol", "c1_v2_2_protocol_sha256"),
        ("c1_v2_1_sidecar", "c1_v2_1_sidecar_sha256"),
        ("c1_v2_2_sidecar", "c1_v2_2_sidecar_sha256"),
    ):
        assert sha256(ROOT / diagnostics[path_key]) == diagnostics[hash_key]
    assert diagnostics["production_noise_scale"] == 1.0
    assert diagnostics["candidate_generator_modified"] is False

    gate = protocol["gate"]
    assert gate["prediction_quality"]["same_seed_d14_relative_error_change_max"] == 0.05
    assert gate["local_sensitivity"] == {
        "candidate_separation_to_new_model_error_min": 0.01,
        "ratio_to_frozen_wm_v1_reference_min": 2.0,
    }
    assert gate["local_directional_validity"]["pairwise_ranking_accuracy_min"] == 0.53
    assert gate["local_directional_validity"]["mean_within_sample_spearman_min"] == 0.05
    assert gate["seed23_seed47_authorized_only_if"] == (
        "WM_V2A_LOCAL_SENSITIVITY_PASS"
    )
    forbidden = " ".join(protocol["forbidden"])
    for phrase in (
        "simulator or controller",
        "D20-v2 or C2-v2",
        "train seed23 or seed47",
        "future-proprio WM-v2B",
    ):
        assert phrase in forbidden
    print(
        {
            "status": "PASS",
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "config_sha256": sha256(config_path),
            "frozen_commit": frozen["commit"],
        }
    )


if __name__ == "__main__":
    main()
