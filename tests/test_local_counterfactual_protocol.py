#!/usr/bin/env python3
"""Static guards for the preregistered matched-branch experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/local_candidate_counterfactual_protocol.json"
SCRIPT = ROOT / "scripts/audit_local_candidate_counterfactual.py"
V1_1_AMENDMENT = ROOT / "configs/local_candidate_counterfactual_protocol_v1_1_amendment.json"
AMENDMENT = ROOT / "configs/local_candidate_counterfactual_protocol_v1_2_amendment.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_protocol_freezes_upstream_and_development_scope() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["status"] == "PREREGISTERED_BEFORE_SIMULATOR_BRANCHES"
    assert protocol["scope"]["task_ids"] == list(range(10))
    assert protocol["scope"]["partition"] == "controller_development"
    assert protocol["scope"]["init_state_indices_per_task"] == [30, 35]
    assert protocol["scope"]["episode_positions_in_partition"] == [0, 5]
    assert protocol["scope"]["planning_indices"] == [0, 8]
    assert protocol["scope"]["prefix_steps"] == [0, 64]
    assert protocol["scope"]["total_branch_points"] == 40
    assert protocol["scope"]["total_independent_candidate_branches"] == 120
    assert protocol["scope"]["success_reward_done_recorded"] is False
    assert set(protocol["scope"]["init_state_indices_forbidden"]) == (
        set(range(30)) | set(range(40, 50))
    )

    frozen = protocol["frozen_upstream"]
    assert subprocess.check_output(
        ["git", "cat-file", "-t", frozen["wm_v2b_annotated_tag"]],
        cwd=ROOT,
        text=True,
    ).strip() == "tag"
    assert subprocess.check_output(
        ["git", "rev-parse", f"{frozen['wm_v2b_annotated_tag']}^{{}}"],
        cwd=ROOT,
        text=True,
    ).strip() == frozen["wm_v2b_commit"]
    checks = [
        (frozen["wm_v2b_diagnosis"], frozen["wm_v2b_diagnosis_sha256"]),
        (frozen["c0_v2_checkpoint"], frozen["c0_v2_checkpoint_sha256"]),
        (frozen["c1_v2_config"], frozen["c1_v2_config_sha256"]),
        (
            frozen["controller_partition_protocol"],
            frozen["controller_partition_protocol_sha256"],
        ),
        (frozen["d15_ensemble_config"], frozen["d15_ensemble_config_sha256"]),
        (frozen["wm_v2a_checkpoint"], frozen["wm_v2a_checkpoint_sha256"]),
        (frozen["wm_v2a_config"], frozen["wm_v2a_config_sha256"]),
        (frozen["wm_v2b_checkpoint"], frozen["wm_v2b_checkpoint_sha256"]),
        (frozen["wm_v2b_config"], frozen["wm_v2b_config_sha256"]),
    ]
    checks.extend((row["path"], row["sha256"]) for row in frozen["d15_checkpoints"])
    for relative, expected in checks:
        assert sha256(ROOT / relative) == expected


def test_branch_implementation_has_no_flattened_state_restore_or_success_use() -> None:
    source = SCRIPT.read_text()
    forbidden_restore = "set_state" + "_from_flattened"
    assert forbidden_restore not in source
    assert ".check_success(" not in source
    assert "env.reward" not in source
    branch = json.loads(PROTOCOL.read_text())["branch_construction"]
    assert branch["mid_episode_flattened_state_restore"] is False
    assert branch["independent_noise_per_branch"] is False
    assert branch["replanning_inside_branch"] is False


def test_v1_2_aligns_gate_to_three_actual_branches_and_preserves_blockers() -> None:
    amendment = json.loads(AMENDMENT.read_text())
    assert amendment["amendment_version"] == (
        "local-candidate-counterfactual-dynamics-v1.2"
    )
    assert amendment["base_protocol_sha256"] == sha256(PROTOCOL)
    assert amendment["v1_1_amendment_sha256"] == sha256(V1_1_AMENDMENT)
    blocker = amendment["v1_1_preflight_blocker"]
    assert sha256(ROOT / blocker["result"]) == blocker["result_sha256"]
    assert sha256(ROOT / blocker["report"]) == blocker["report_sha256"]
    assert blocker["observed"]["three_branch_rgb_hashes_equal"] is True
    assert blocker["observed"]["three_branch_physical_states_equal"] is True
    fix = amendment["final_authorized_alignment_fix"]
    assert "three freshly created counterfactual branch environments only" in fix[
        "identity_gate"
    ]
    assert "generate A0/A1/A2 exactly once from branch 0" in fix[
        "candidate_generation"
    ]
    assert fix["clip_identity_threshold_unchanged"] == 0.999999
    assert "all real/predicted metric definitions" in amendment[
        "unchanged_from_base_and_v1_1"
    ]


def test_ratio_and_gate_definitions_are_numeric_and_preregistered() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    metrics = protocol["metric_contract"]
    rules = protocol["decision_rules"]
    assert metrics["ratio_minimum_real_cosine"] == 1e-5
    assert "ratio of group means" in metrics["transfer_ratio"]
    assert "below 1e-5" in metrics["near_zero_rule"]
    assert "0.01" in rules["real_visual_is_meaningful"]
    assert "0.05" in rules["real_physical_is_meaningful"]
    assert "0.50" in rules["visual_transfer_failure_confirmed"]
    assert "1.50" in rules["wm_local_counterfactual_prediction_fail_per_model"]
    assert rules["reference_values"]["d15_ordinary_validation_visual_error"] > 0
    assert rules["reference_values"]["wm_v2b_validation_proprio_rmse"] > 0
