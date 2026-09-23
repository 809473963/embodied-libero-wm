#!/usr/bin/env python3
"""Static audit for the preregistered C0-v2 rollout split and frozen inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/c0_v2_causal_alignment_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    assert protocol["protocol_version"] == "c0-v2-causal-alignment-v1"
    assert protocol["status"] == "GO_C0_V2_CAUSAL_ALIGNMENT"
    partitions = protocol["partitions"]
    expected = {
        "legacy_historical_comparison": set(range(0, 20)),
        "d22_5_controller_diagnosis": set(range(20, 30)),
        "controller_development": set(range(30, 40)),
        "confirmatory_holdout": set(range(40, 50)),
    }
    for name, indices in expected.items():
        assert set(partitions[name]["init_state_indices"]) == indices
    names = list(expected)
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            assert expected[left].isdisjoint(expected[right])

    all_seeds: set[int] = set()
    for partition_name, base in (("controller_development", 30000), ("confirmatory_holdout", 40000)):
        record = partitions[partition_name]
        for task_id in protocol["task_ids"]:
            seeds = record["task_seeds"][str(task_id)]
            assert seeds == [base + 100 * task_id + episode for episode in range(10)]
            assert all_seeds.isdisjoint(seeds)
            all_seeds.update(seeds)
    assert len(all_seeds) == 200
    assert partitions["confirmatory_holdout"]["locked_until_development_gate_is_committed"]

    frozen = protocol["frozen_inputs"]
    for path_key, hash_key in (
        ("c0_v1_checkpoint", "c0_v1_checkpoint_sha256"),
        ("c0_v1_feature_manifest", "c0_v1_feature_manifest_sha256"),
        ("d20_v1_result", "d20_v1_result_sha256"),
        ("d21_result", "d21_result_sha256"),
        ("d22_result", "d22_result_sha256"),
    ):
        assert sha256(ROOT / frozen[path_key]) == frozen[hash_key]

    gate = protocol["development_gate"]
    assert gate["decision_must_be_committed_before_holdout"]
    assert gate["go_requires_all"] == {
        "c0_v2_total_successes_min": 20,
        "c0_v2_nonzero_tasks_min": 5,
        "success_improvement_over_c0_v1_min": 10,
        "nonzero_task_improvement_over_c0_v1_min": 2,
        "finite_actions_and_metrics": True,
        "planned_action_element_clipping_ratio_max": 0.1,
    }
    print(
        {
            "status": "PASS",
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "development_episodes": 100,
            "holdout_episodes_locked": 100,
            "partition_overlap": 0,
        }
    )


if __name__ == "__main__":
    main()
