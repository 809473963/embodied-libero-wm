#!/usr/bin/env python3
"""Static preregistration and frozen-source audit for C1-v2.2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/c1_v2_2_wm_action_conditioning.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PATH.read_text())
    assert protocol["protocol_version"] == "c1-v2.2-wm-action-conditioning-v1"
    assert protocol["status"] == "PREREGISTERED_BEFORE_ACTION_ABLATION"
    frozen = protocol["frozen_c1_v2_1"]
    assert frozen["blockers"] == [
        "RERANKER_SCORE_BLOCK",
        "WM_ACTION_SENSITIVITY_BLOCK",
    ]
    assert sha256(ROOT / frozen["result"]) == frozen["result_sha256"]
    assert sha256(ROOT / frozen["report"]) == frozen["report_sha256"]
    assert subprocess.check_output(
        ["git", "rev-parse", f"{frozen['annotated_tag']}^{{}}"],
        cwd=ROOT,
        text=True,
    ).strip() == frozen["commit"]

    data = protocol["data"]
    assert data["demo_ids"] == list(range(45, 50))
    assert data["task_ids"] == list(range(10))
    assert data["expected_samples"] == 5869
    assert data["expected_episodes"] == 50
    assert data["simulator_rollout"] is False
    assert data["init_state_indices_used"] == []
    assert sha256(ROOT / data["store"]) == data["store_sha256"]
    assert sha256(ROOT / data["feature_manifest"]) == data[
        "feature_manifest_sha256"
    ]

    world_model = protocol["world_model"]
    assert sha256(ROOT / world_model["ensemble_config"]) == world_model[
        "ensemble_config_sha256"
    ]
    config = yaml.safe_load((ROOT / world_model["ensemble_config"]).read_text())
    assert [sha256(ROOT / path) for path in config["member_checkpoints"]] == world_model[
        "member_checkpoint_sha256"
    ]
    assert sha256(ROOT / world_model["ordinary_validation_result"]) == world_model[
        "ordinary_validation_result_sha256"
    ]

    conditions = protocol["action_conditions"]
    assert conditions["magnitude_scales"] == [0.0, 0.25, 0.5, 1.0, 1.5]
    assert conditions["temporal"]["fixed_permutation"] == [2, 5, 0, 7, 3, 1, 6, 4]
    assert "physical-space zeros" in conditions["zero"]
    assert "do not clip" in conditions["magnitude_rule"]
    assert protocol["bootstrap"]["replicates"] == 10000
    assert protocol["optional_gradient_analysis"]["execute"] is False
    forbidden = " ".join(protocol["forbidden"])
    for text in ("controller", "init-state indices 0-49", "D20-v2", "C2-v2"):
        assert text in forbidden
    print(
        {
            "status": "PASS",
            "protocol_sha256": sha256(PATH),
            "frozen_commit": frozen["commit"],
            "validation_samples": data["expected_samples"],
        }
    )


if __name__ == "__main__":
    main()
