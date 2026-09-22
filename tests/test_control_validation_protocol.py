#!/usr/bin/env python3
"""Validate the precommitted D22.5 controller-validation split."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark
from scripts.evaluate_c0_all import trusted_init_states


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    path = PROJECT_ROOT / "configs/control_validation_protocol.json"
    protocol = json.loads(path.read_text())
    assert protocol["protocol_version"] == "control-validation-v1"
    assert protocol["precommitted_before_rollout_collection"]
    assert protocol["task_ids"] == list(range(10))
    assert protocol["episodes_per_task"] == 10
    control_indices = protocol["controller_validation"]["init_state_indices"]
    fixed_indices = protocol["fixed_final_test_exclusion"]["init_state_indices"]
    reserved_indices = protocol["unused_reserved_init_state_indices"]
    assert control_indices == list(range(20, 30))
    assert fixed_indices == list(range(20))
    assert reserved_indices == list(range(30, 50))
    assert not (set(control_indices) & set(fixed_indices))
    assert not (set(control_indices) & set(reserved_indices))
    assert sha256(PROJECT_ROOT / "configs/eval_seeds_spatial.json") == protocol[
        "fixed_final_test_exclusion"
    ]["seed_file_sha256"]
    fixed = json.loads((PROJECT_ROOT / "configs/eval_seeds_spatial.json").read_text())
    control_seeds: set[int] = set()
    fixed_seeds = {
        int(seed) for values in fixed["task_seeds"].values() for seed in values
    }
    for task_id in range(10):
        seeds = protocol["controller_validation"]["task_seeds"][str(task_id)]
        assert seeds == [20000 + 100 * task_id + index for index in range(10)]
        assert not (set(seeds) & fixed_seeds)
        assert not (set(seeds) & control_seeds)
        control_seeds.update(seeds)
    assert len(control_seeds) == 100
    suite = benchmark.get_benchmark_dict()[protocol["suite"]]()
    counts = []
    for task_id in protocol["task_ids"]:
        count = len(trusted_init_states(suite, task_id))
        counts.append(count)
        assert count >= 50
        assert max(control_indices + fixed_indices + reserved_indices) < count
    assert sha256(PROJECT_ROOT / "results/c0/seed11/best.pt") == protocol[
        "collection_methods"
    ]["c0_rollout"]["checkpoint_sha256"]
    assert sha256(PROJECT_ROOT / "configs/c1_rerank.yaml") == protocol[
        "collection_methods"
    ]["c1_selected_rollout"]["config_sha256"]
    assert sha256(PROJECT_ROOT / "results/threshold_selection.json") == protocol[
        "frozen_d20_v1"
    ]["result_sha256"]
    print(
        {
            "status": "PASS",
            "protocol_sha256": sha256(path),
            "available_init_states_per_task": counts,
            "controller_validation_episodes": len(control_seeds),
            "fixed_test_overlap": 0,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
