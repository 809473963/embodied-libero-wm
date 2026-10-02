"""Protocol checks for the Stage-B1 semantic identifiability gate."""

from __future__ import annotations

import json
from pathlib import Path

from src.probes.semantic_goal_contract import parse_task_contract, sha256


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/semantic_goal_progress_teacher_protocol.json"


def load() -> dict:
    return json.loads(PROTOCOL.read_text())


def test_partition_and_stop_boundary() -> None:
    protocol = load()
    assert protocol["partitions"]["calibration_demo_ids"] == [31, 32, 33, 34]
    assert protocol["partitions"]["evaluation_demo_ids"] == [36, 37, 38, 39]
    assert protocol["partitions"]["forbidden_demo_ids"] == list(range(40, 50))
    assert "demo31-34 state rows" in protocol["preflight_access_boundary"]["withheld_until_identifiability_passes"]
    assert "train B2" in " ".join(protocol["forbidden"])


def test_frozen_hashes_match() -> None:
    protocol = load()
    for record in protocol["frozen_inputs"] + protocol["implementation"]:
        assert sha256(ROOT / record["path"]) == record["sha256"]
    assert protocol["frozen_upstream"]["annotated_tag"] == "candidate-value-audit-blocked"


def test_task_mapping_is_complete_and_ordered() -> None:
    sources = load()["task_semantic_sources"]
    assert [row["task_id"] for row in sources] == list(range(10))
    assert len({row["path"] for row in sources}) == 10
    for row in sources:
        contract = parse_task_contract(ROOT / row["path"], row["task_id"])
        assert contract["task_id"] == row["task_id"]
        assert contract["task_name"] == Path(row["path"]).stem


def test_counterfactual_gate_is_semantic_not_textual() -> None:
    protocol = load()
    contract = protocol["counterfactual_language_contract"]
    assert "canonical goal signature differs" in contract["admissible"]
    assert contract["must_reparse"] == ["relevant object", "target object or region", "goal predicate"]
    gate = protocol["identifiability_gate"]
    assert gate["minimum_distinct_goal_signatures"] == 2
    assert gate["minimum_tasks_with_counterfactual"] == 7


def test_downstream_thresholds_are_frozen_but_not_executed_early() -> None:
    protocol = load()
    gates = protocol["downstream_b1_gates_if_identifiable"]
    assert gates["local_le8_non_tie_min"] == 0.30
    assert gates["usable_task_min"] == 7
    assert protocol["identifiability_gate"]["failure_stop"].startswith("Do not construct")

