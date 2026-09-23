#!/usr/bin/env python3
"""Static checks for the preregistered C1-v2 rebase protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/c1_v2_rebase_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["protocol_version"] == "c1-v2-rebase-and-wm-distribution-v1"
    assert protocol["status"] == "PREREGISTERED_BEFORE_C1_V2_DEVELOPMENT_ROLLOUT"
    assert protocol["causal_mapping"]["python_half_open_count"] == 8
    assert protocol["causal_mapping"]["missing_ninth_action"] is False
    assert protocol["controller_development"]["init_state_indices"] == list(range(30, 40))
    assert "forbidden" in protocol["partition_boundaries"]["40-49"]

    frozen = protocol["c0_v2_freeze"]
    assert sha256(ROOT / frozen["checkpoint"]) == frozen["checkpoint_sha256"]
    assert sha256(ROOT / frozen["config"]) == frozen["config_sha256"]
    assert sha256(ROOT / frozen["development_result"]) == frozen[
        "development_result_sha256"
    ]
    assert sha256(ROOT / frozen["confirmatory_result"]) == frozen[
        "confirmatory_result_sha256"
    ]
    tag_commit = subprocess.check_output(
        ["git", "rev-parse", f"{frozen['annotated_tag']}^{{}}"], cwd=ROOT, text=True
    ).strip()
    assert tag_commit == frozen["commit"]

    c1 = protocol["c1_v2"]
    assert sha256(ROOT / c1["config"]) == c1["config_sha256"]
    config = yaml.safe_load((ROOT / c1["config"]).read_text())
    assert config["contract_version"] == "c1-v2"
    assert config["actor"]["checkpoint_sha256"] == frozen["checkpoint_sha256"]
    assert config["candidates"]["count"] == 3
    assert config["execution"]["horizon"] == 8

    source = protocol["controller_development"]
    assert sha256(ROOT / source["partition_source"]) == source["partition_source_sha256"]
    source_protocol = json.loads((ROOT / source["partition_source"]).read_text())
    partition = source_protocol["partitions"][source["partition_name"]]
    assert partition["init_state_indices"] == list(range(30, 40))
    for task_id in protocol["task_ids"]:
        assert partition["task_seeds"][str(task_id)] == [
            30000 + 100 * task_id + episode for episode in range(10)
        ]
    assert sha256(ROOT / source["frozen_c0_v2_baseline"]) == source[
        "frozen_c0_v2_baseline_sha256"
    ]

    gate = protocol["c1_v2_gate"]
    assert gate["decision_must_be_committed_before_any_c1_v2_or_c2_v2_rollout_on_40_49"]
    assert gate["go_verdict"] == "GO_D20_V2"
    assert gate["no_go_verdict"] == "BLOCK_C1_V2_RERANKER"
    assert gate["interface_failure_verdict"] == "BLOCK_WM_INTERFACE"
    print(
        {
            "status": "PASS",
            "protocol_sha256": sha256(PROTOCOL),
            "tag_commit": tag_commit,
            "development_indices": partition["init_state_indices"],
            "forbidden_design_indices": list(range(40, 50)),
        }
    )


if __name__ == "__main__":
    main()
