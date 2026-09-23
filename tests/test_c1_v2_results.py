#!/usr/bin/env python3
"""Result-level provenance and boundary checks for the C1-v2 audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/c1_v2_rebase_protocol.json"
AUDIT_PATH = ROOT / "results/c1_v2/rebase_audit.json"
TABLE_PATH = ROOT / "results/c1_v2/wm_distribution_table.csv"
GATE_PATH = ROOT / "configs/c1_v2_gate_decision.json"
REPORT_PATH = ROOT / "docs/c1_v2_rebase_and_wm_distribution_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    audit = json.loads(AUDIT_PATH.read_text())
    gate = json.loads(GATE_PATH.read_text())

    assert audit["audit_version"] == protocol["protocol_version"]
    assert audit["protocol"]["sha256"] == sha256(PROTOCOL_PATH)
    assert gate["protocol_sha256"] == sha256(PROTOCOL_PATH)
    assert gate["audit_result_sha256"] == sha256(AUDIT_PATH)
    assert gate["distribution_table_sha256"] == sha256(TABLE_PATH)
    assert audit["verdict"] == audit["gate"]["verdict"] == gate["decision"]
    assert audit["verdict"] in {
        "GO_D20_V2",
        "BLOCK_C1_V2_RERANKER",
        "BLOCK_WM_INTERFACE",
    }

    assert audit["partition"] == {
        "name": "controller_development",
        "init_state_indices": list(range(30, 40)),
        "c1_c2_design_40_49_used": False,
        "historical_40_49_c0_v2_outcomes_known": True,
    }
    for controller in ("c0_v2", "c1_v2"):
        sources = audit["sources"][controller]
        assert len(sources) == 10
        for task_id, source in enumerate(sources):
            result_path = ROOT / source["path"]
            sidecar_path = ROOT / source["trajectory_path"]
            assert sha256(result_path) == source["sha256"]
            assert sha256(sidecar_path) == source["trajectory_sha256"]
            result = json.loads(result_path.read_text())
            assert result["controller"] == controller
            assert result["task_id"] == task_id
            assert result["episodes"] == 10 and result["complete"] is True
            identities = [
                (
                    episode["episode_index"],
                    episode["seed"],
                    episode["init_state_index"],
                )
                for episode in result["episode_results"]
            ]
            assert identities == [
                (episode, 30000 + 100 * task_id + episode, 30 + episode)
                for episode in range(10)
            ]

    distributions = audit["wm_distribution"]
    assert set(distributions) == {
        "expert_demo_validation",
        "historical_c0_v1_rollout_20_29",
        "c0_v2_rollout_30_39",
        "c1_v2_rollout_30_39",
    }
    for distribution in distributions.values():
        assert set(distribution) == {"2", "4", "6", "8"}
        for horizon in distribution.values():
            assert set(horizon) == {"mismatch", "uncertainty"}
            for metric in horizon.values():
                assert metric["count"] > 0
                assert 0.0 <= metric["d20_v1_exceed_rate"] <= 1.0

    with TABLE_PATH.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 4 * 4 * 2
    assert {int(row["horizon"]) for row in rows} == {2, 4, 6, 8}

    d20 = protocol["frozen_d20_v1_diagnostic_only"]
    assert audit["frozen_d20_v1"] == {
        "thresholds_modified": False,
        "mismatch_threshold": d20["mismatch_threshold"],
        "uncertainty_threshold": d20["uncertainty_threshold"],
        "use": "diagnostic exceedance logging only",
    }
    interface = audit["normalization_and_interface"]
    assert interface["full_sidecar_backcheck_passed"] is True
    assert interface["wm_never_reads_actor_normalized_action"] is True
    assert interface["actor"]["count"] == interface["world_model"]["count"] == 55531

    forbidden = audit["forbidden_activity_audit"]
    assert forbidden and not any(forbidden.values())
    assert gate["c1_v2_or_c2_v2_40_49_observed"] is False
    if audit["gate"]["interface_ok"] is False:
        expected = "BLOCK_WM_INTERFACE"
    elif audit["gate"]["all_pass"]:
        expected = "GO_D20_V2"
    else:
        expected = "BLOCK_C1_V2_RERANKER"
    assert audit["verdict"] == expected
    assert finite_tree(audit)

    report = REPORT_PATH.read_text()
    assert "PENDING" not in report
    assert f"Final verdict: `{audit['verdict']}`" in report
    print(
        {
            "status": "PASS",
            "verdict": audit["verdict"],
            "development_episodes": sum(
                value["episodes"]
                for value in audit["controller_development"].values()
            ),
            "distribution_rows": len(rows),
        }
    )


if __name__ == "__main__":
    main()
