#!/usr/bin/env python3
"""Run the Stage-B1 source-semantic identifiability gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probes.semantic_goal_contract import inventory_contracts, sha256


VERSION = "semantic-goal-progress-teacher-v1"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != VERSION:
        raise ValueError("unexpected protocol version")
    for record in protocol["frozen_inputs"] + protocol["implementation"]:
        resolved = ROOT / record["path"]
        if sha256(resolved) != record["sha256"]:
            raise ValueError(f"frozen hash mismatch: {record['path']}")
    if protocol["partitions"]["forbidden_demo_ids"] != list(range(40, 50)):
        raise ValueError("forbidden partition changed")
    return protocol, sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/semantic_goal_progress_teacher_protocol.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/semantic_goal_progress_teacher",
    )
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.protocol)
    paths = [ROOT / entry["path"] for entry in protocol["task_semantic_sources"]]
    inventory = inventory_contracts(paths)

    rules = protocol["identifiability_gate"]
    gates = {
        "all_tasks_parsed": inventory["task_count"] == rules["required_task_count"],
        "multiple_goal_signatures": inventory["distinct_goal_signature_count"] >= rules["minimum_distinct_goal_signatures"],
        "counterfactual_task_coverage": inventory["task_count_with_goal_changing_counterfactual"] >= rules["minimum_tasks_with_counterfactual"],
    }
    pass_gate = all(gates.values())
    if pass_gate:
        labels = ["SEMANTIC_GOAL_CONTRACT_IDENTIFIABLE", "GO_SEMANTIC_TEACHER_CONSTRUCTION"]
        authorization = "GO_SEMANTIC_TEACHER_CONSTRUCTION"
    else:
        labels = [
            "SEMANTIC_TEACHER_LANGUAGE_INSUFFICIENT",
            "SEMANTIC_GOAL_PROGRESS_TEACHER_UNRESOLVED",
            "DATA_OR_IMPLEMENTATION_BLOCK",
        ]
        authorization = "BLOCK_VALUE_DISTILLATION"

    inventory_path = args.output_dir / "semantic_inventory.json"
    atomic_json(inventory_path, inventory)
    diagnosis = {
        "audit_version": VERSION,
        "created_at": protocol["created_at"],
        "source_commit": git_head(),
        "protocol": {"path": args.protocol.relative_to(ROOT).as_posix(), "sha256": protocol_hash},
        "inventory": {"path": inventory_path.relative_to(ROOT).as_posix(), "sha256": sha256(inventory_path)},
        "scope": {
            "task_definition_files_read": inventory["task_count"],
            "demo_rows_read": 0,
            "calibration_demo_rows_read": 0,
            "evaluation_demo_rows_read": 0,
            "forbidden_demo_rows_read": 0,
            "simulator_runs": 0,
            "teacher_models_trained": 0,
            "value_models_trained": 0,
        },
        "semantic_inventory_summary": {
            key: inventory[key]
            for key in (
                "task_count",
                "distinct_goal_signature_count",
                "distinct_relevant_initial_signature_count",
                "directed_real_language_swap_count",
                "directed_goal_changing_counterfactual_count",
                "task_count_with_goal_changing_counterfactual",
            )
        },
        "decision": {
            "identifiability_gate_pass": pass_gate,
            "gates": gates,
            "final_mechanism_labels": labels,
            "highest_completed_stage": "B1",
            "semantic_teacher_selected": False,
            "deployable_value_evaluator_selected": False,
            "real_candidate_value_bridge_validated": False,
            "value_distillation_authorization": authorization,
            "world_model_interface_authorization": "BLOCK_WORLD_MODEL_INTERFACE_AUDIT",
            "spatial_world_model_authorization": "BLOCK_SPATIAL_WM_PROTOTYPE",
            "controller_authorization": False,
            "stopped_before_state_teacher_construction": not pass_gate,
            "stopped_before_calibration_or_evaluation_data_access": not pass_gate,
        },
    }
    atomic_json(args.output_dir / "diagnosis.json", diagnosis)


if __name__ == "__main__":
    main()
