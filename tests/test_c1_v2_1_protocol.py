#!/usr/bin/env python3
"""Static audit of the preregistered C1-v2.1 offline diagnosis protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/c1_v2_1_candidate_diagnosis.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PATH.read_text())
    assert protocol["protocol_version"] == "c1-v2.1-candidate-discriminability-v1"
    assert protocol["status"] == "PREREGISTERED_BEFORE_OFFLINE_VALIDATION_ANALYSIS"
    frozen = protocol["frozen_c1_v2_negative"]
    assert frozen["verdict"] == "BLOCK_C1_V2_RERANKER"
    assert sha256(ROOT / frozen["audit"]) == frozen["audit_sha256"]
    assert sha256(ROOT / frozen["gate"]) == frozen["gate_sha256"]
    assert subprocess.check_output(
        ["git", "rev-parse", f"{frozen['annotated_tag']}^{{}}"], cwd=ROOT, text=True
    ).strip() == frozen["commit"]

    validation = protocol["validation"]
    assert validation["demo_ids"] == list(range(45, 50))
    assert validation["expected_samples"] == 5869
    assert validation["simulator_rollout"] is False
    assert validation["init_state_indices_used"] == []
    assert sha256(ROOT / validation["store"]) == validation["store_sha256"]

    generator = protocol["candidate_generator"]
    assert generator["noise_scales"] == [0.0, 0.25, 0.5, 1.0, 2.0]
    assert generator["production_scale"] == 1.0
    assert generator["sweep_is_diagnostic_only"] is True
    assert generator["sample_seed_rule"] == "seed_base + global_validation_sample_index"
    assert protocol["controller_development_score_scale"][
        "controller_success_may_drive_parameter_selection"
    ] is False
    forbidden = " ".join(protocol["forbidden"])
    assert "40-49" in forbidden
    assert "controller rollout" in forbidden
    assert "D20-v2" in forbidden and "C2-v2" in forbidden
    print(
        {
            "status": "PASS",
            "protocol_sha256": sha256(PATH),
            "frozen_commit": frozen["commit"],
            "validation_samples": validation["expected_samples"],
        }
    )


if __name__ == "__main__":
    main()
