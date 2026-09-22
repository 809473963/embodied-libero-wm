#!/usr/bin/env python3
"""Acceptance checks for the D22.5 diagnostic outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    path = PROJECT_ROOT / "results/control_validation/d22_5_diagnosis.json"
    result = json.loads(path.read_text())
    assert result["diagnosis_version"] == "d22.5-v1"
    assert result["verdict"] == "BASELINE_COMPETENCE_BLOCK"
    assert result["secondary_recommendation"] == "GO_D20_V2_PROTOCOL_DESIGN_ONLY"
    assert not result["d23_authorized"]
    assert not result["d20_v1_thresholds_modified"]
    assert not result["fixed_test_outcomes_used_for_analysis_or_selection"]
    assert result["inputs"]["fixed_test_artifacts"] == []
    assert result["controller_competence"]["c0"]["successes"] == 15
    assert result["controller_competence"]["c0"]["episodes"] == 100
    assert result["controller_competence"]["c0"]["nonzero_task_count"] == 3
    assert result["controller_competence"]["c0"]["zero_task_count"] == 7
    for method in ("c0", "c1"):
        shift = result["h2_distribution_shift"][method]
        assert shift["uncertainty"]["median_ratio_controller_to_expert"] > 1.5
        assert shift["uncertainty"]["p95_ratio_controller_to_expert"] > 2.0
        triggers = result["controller_validation_trigger_counterfactual"][method]
        assert triggers["d20_or"]["trigger_rate"] > triggers["d20_mismatch_only"][
            "trigger_rate"
        ]
        assert triggers["d20_or"]["projected_mean_plan_length"] < triggers[
            "d20_mismatch_only"
        ]["projected_mean_plan_length"]
        signal = result["uncertainty_signal"][method]
        assert signal["current_decile_mean_strictly_increasing"]
        assert signal["next_decile_mean_strictly_increasing"]
        assert signal["current_high_error_ranking"]["auc"] > 0.65
        assert signal["next_high_error_ranking"]["auc"] > 0.65
    assert result["controller_action_contract_audit"]["status"] == (
        "UNRESOLVED_CAUSAL_INDEX_MISMATCH"
    )
    for artifact in result["artifacts"].values():
        artifact_path = PROJECT_ROOT / artifact["path"]
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]
    print(
        {
            "verdict": result["verdict"],
            "c0_successes": result["controller_competence"]["c0"]["successes"],
            "c0_or_rate": result["controller_validation_trigger_counterfactual"]["c0"][
                "d20_or"
            ]["trigger_rate"],
            "c1_or_rate": result["controller_validation_trigger_counterfactual"]["c1"][
                "d20_or"
            ]["trigger_rate"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
