#!/usr/bin/env python3
"""Static guards for the preregistered WM-v2B mediator-fidelity audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/wm_v2b_counterfactual_physical_mediator_protocol.json"
SCRIPT = ROOT / "scripts/audit_wm_v2b_physical_mediator.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_protocol_freezes_sources_scope_and_no_simulator_path() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["status"] == "PREREGISTERED_BEFORE_DIFFERENTIAL_METRICS"
    assert protocol["scope"]["matched_branch_points"] == 40
    assert protocol["scope"]["matched_candidate_branches"] == 120
    assert protocol["scope"]["init_state_indices"] == [30, 35]
    assert protocol["scope"]["planning_indices"] == [0, 8]
    assert protocol["scope"]["simulator_runs"] == 0
    assert protocol["scope"]["success_reward_done_used"] is False
    assert set(protocol["scope"]["init_state_indices_forbidden"]) == (
        set(range(30)) | set(range(40, 50))
    )

    frozen = protocol["frozen_upstream"]
    for path_key, hash_key in (
        ("counterfactual_result", "counterfactual_result_sha256"),
        ("counterfactual_report", "counterfactual_report_sha256"),
        ("matched_branch_sidecar", "matched_branch_sidecar_sha256"),
        ("wm_v2b_result", "wm_v2b_result_sha256"),
        ("wm_v2b_validation_sidecar", "wm_v2b_validation_sidecar_sha256"),
        ("wm_v2b_checkpoint", "wm_v2b_checkpoint_sha256"),
        ("wm_v2b_config", "wm_v2b_config_sha256"),
        ("wm_v2b_protocol", "wm_v2b_protocol_sha256"),
    ):
        assert sha256(ROOT / frozen[path_key]) == frozen[hash_key]

    for tag_key, commit_key in (
        ("counterfactual_annotated_tag", "counterfactual_commit"),
        ("wm_v2b_annotated_tag", "wm_v2b_commit"),
    ):
        tag = frozen[tag_key]
        assert subprocess.check_output(
            ["git", "cat-file", "-t", tag], cwd=ROOT, text=True
        ).strip() == "tag"
        assert subprocess.check_output(
            ["git", "rev-parse", f"{tag}^{{}}"], cwd=ROOT, text=True
        ).strip() == frozen[commit_key]

    source = SCRIPT.read_text()
    for forbidden in (
        "make_env(",
        "set_init_state(",
        "check_success(",
        "env.step(",
        "set_state_from_flattened",
    ):
        assert forbidden not in source


def test_numeric_metrics_bootstrap_and_gate_are_frozen() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    metrics = protocol["metric_contract"]
    bootstrap = protocol["bootstrap"]
    gate = protocol["authorization_gate"]
    assert metrics["minimum_real_norm_for_direction_and_ratio"] == 1e-4
    assert metrics["minimum_abs_real_delta_for_sign"] == 1e-3
    assert metrics["ranking_tie_tolerance"] == 1e-8
    assert bootstrap == {
        "unit": "matched branch point; resampling retains all three candidate pairs, all eight horizons, and all 12 dimensions",
        "replicates": 10000,
        "seed": 220601,
        "confidence": 0.95,
        "reported_intervals": [
            "aggregate differential relative improvement",
            "aggregate direction mean",
            "within-context separation Spearman",
            "pairwise ranking accuracy",
            "per-pair differential improvement",
            "per-pair direction",
            "per-horizon differential improvement",
            "per-horizon direction",
        ],
    }
    assert ">= 0.05" in gate["differential_improvement"]
    assert ">= 0.10" in gate["direction_alignment"]
    assert ">= 0.55" in gate["ranking_relation"]
    assert "every A0-A1/A0-A2/A1-A2 pair" in gate[
        "no_candidate_specific_collapse"
    ]
