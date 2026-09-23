#!/usr/bin/env python3
"""Static audit of the pre-result confirmatory interpretation rules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/c0_v2_confirmatory_analysis_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    analysis = json.loads(PATH.read_text())
    rollout_path = ROOT / analysis["source_rollout_protocol"]
    gate_path = ROOT / analysis["development_gate_decision"]
    assert sha256(rollout_path) == analysis["source_rollout_protocol_sha256"]
    assert sha256(gate_path) == analysis["development_gate_decision_sha256"]
    assert analysis["init_state_indices"] == list(range(40, 50))
    assert analysis["episodes_per_task"] * analysis["task_count"] == 100
    assert analysis["acceptance_requires_all"] == {
        "c0_v2_total_successes_min": 20,
        "c0_v2_nonzero_tasks_min": 5,
        "success_improvement_over_c0_v1_min": 10,
        "nonzero_task_improvement_over_c0_v1_min": 2,
        "finite_actions_and_metrics": True,
        "planned_action_element_clipping_ratio_max": 0.1,
    }
    assert analysis["verdict_mapping"]["all_acceptance_checks_pass"] == (
        "CAUSAL_ALIGNMENT_IMPROVES_BASELINE"
    )
    print({"status": "PASS", "analysis_protocol_sha256": sha256(PATH)})


if __name__ == "__main__":
    main()
