#!/usr/bin/env python3
"""Static guards for WM-v2B.1 calibration preregistration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/wm_v2b_1_differential_calibration_protocol.json"
COLLECTOR = ROOT / "scripts/collect_wm_v2b_1_calibration_branches.py"
FITTER = ROOT / "scripts/fit_wm_v2b_1_differential_calibration.py"
ANALYZER = ROOT / "scripts/analyze_wm_v2b_1_differential_calibration.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_partitions_are_disjoint_preregistered_and_exclude_40_49() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["status"] == (
        "PREREGISTERED_BEFORE_INIT_31_39_BRANCH_EXECUTION"
    )
    calibration = protocol["partitions"]["calibration"]
    evaluation = protocol["partitions"]["evaluation"]
    assert calibration["init_state_indices"] == [31, 32, 33, 34]
    assert evaluation["init_state_indices"] == [36, 37, 38, 39]
    assert set(calibration["init_state_indices"]).isdisjoint(
        evaluation["init_state_indices"]
    )
    for partition in (calibration, evaluation):
        assert partition["planning_indices"] == [8]
        assert partition["prefix_steps"] == [64]
        assert partition["branch_points"] == 40
        assert partition["candidate_branches"] == 120
    assert set(protocol["partitions"]["forbidden_init_state_indices"]) == (
        set(range(30)) | set(range(40, 50))
    )
    assert protocol["partitions"]["historical_reference"][
        "init_state_indices"
    ] == [30, 35]


def test_upstream_hashes_and_annotated_freeze_are_valid() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    frozen = protocol["frozen_upstream"]
    for path_key, hash_key in (
        ("mediator_result", "mediator_result_sha256"),
        ("mediator_report", "mediator_report_sha256"),
        ("historical_branch_result", "historical_branch_result_sha256"),
        ("historical_branch_sidecar", "historical_branch_sidecar_sha256"),
        ("historical_branch_protocol", "historical_branch_protocol_sha256"),
        ("historical_branch_amendment", "historical_branch_amendment_sha256"),
        ("historical_collector", "historical_collector_sha256"),
        ("validation_sidecar", "validation_sidecar_sha256"),
        ("wm_v2b_checkpoint", "wm_v2b_checkpoint_sha256"),
        ("controller_partition_protocol", "controller_partition_protocol_sha256"),
    ):
        assert sha256(ROOT / frozen[path_key]) == frozen[hash_key]
    tag = frozen["mediator_annotated_tag"]
    assert subprocess.check_output(
        ["git", "cat-file", "-t", tag], cwd=ROOT, text=True
    ).strip() == "tag"
    assert subprocess.check_output(
        ["git", "rev-parse", f"{tag}^{{}}"], cwd=ROOT, text=True
    ).strip() == frozen["mediator_commit"]


def test_calibration_forms_gates_and_staged_evaluation_are_frozen() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    models = protocol["calibration_models"]
    assert "closed-form" in models["fit_rule"]
    assert "per_horizon_scalar" in models["evaluation_form_selection"]
    assert "forbidden" in models["evaluation_form_selection"]
    assert "never authorizes GO" in models["horizon_group_scalar"]
    gate = protocol["authorization_gate"]
    assert gate["primary_form"] == "per_horizon_scalar"
    assert ">= 0.05" in gate["aggregate_zero_improvement"]
    assert "CI lower bound > 0" in gate["aggregate_zero_improvement"]
    assert "every A0-A1/A0-A2/A1-A2" in gate[
        "candidate_pair_consistency"
    ]
    assert "full_gripper_validity" in gate
    assert "sign accuracy >= 0.55" in gate["full_gripper_validity"]
    assert "GO_WM_V2C_DESIGN" in gate["full_pass"]
    assert "PARTIAL_PHYSICAL_MEDIATOR_PASS" in gate["partial_group_gate"]
    assert "PHYSICAL_MEDIATOR_RETRAIN_REQUIRED" in gate["retrain"]

    collector = COLLECTOR.read_text()
    assert "calibration alpha freeze must be committed before evaluation" in collector
    assert "evaluation_partition_read_or_fit" in FITTER.read_text()
    assert "evaluation_used_for_fit_or_form_selection" in ANALYZER.read_text()
    for source in (collector, FITTER.read_text(), ANALYZER.read_text()):
        assert "set_state_from_flattened" not in source
        assert "check_success(" not in source
