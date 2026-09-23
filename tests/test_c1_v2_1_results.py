#!/usr/bin/env python3
"""Result-level provenance, boundary, and decision checks for C1-v2.1."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/c1_v2_1_candidate_diagnosis.json"
RESULT_PATH = ROOT / "results/c1_v2_1/candidate_diagnosis.json"
REPORT_PATH = ROOT / "docs/c1_v2_1_candidate_discriminability_diagnosis.md"


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
    result = json.loads(RESULT_PATH.read_text())

    assert result["diagnosis_version"] == protocol["protocol_version"]
    assert result["status"] == "PASS_OFFLINE_DIAGNOSIS_COMPLETE"
    assert result["frozen_c1_v2_verdict"] == "BLOCK_C1_V2_RERANKER"
    assert result["protocol"]["sha256"] == sha256(PROTOCOL_PATH)
    preregistered_commit = result["protocol"]["preregistered_commit"]
    assert preregistered_commit.startswith("4bed772")
    assert subprocess.check_output(
        ["git", "cat-file", "-t", preregistered_commit], cwd=ROOT, text=True
    ).strip() == "commit"

    frozen = protocol["frozen_c1_v2_negative"]
    assert result["sources"]["c1_v2_negative_commit"] == frozen["commit"]
    assert result["sources"]["c1_v2_negative_tag"] == frozen["annotated_tag"]
    assert sha256(ROOT / frozen["audit"]) == frozen["audit_sha256"]
    assert sha256(ROOT / frozen["gate"]) == frozen["gate_sha256"]
    assert subprocess.check_output(
        ["git", "rev-parse", f"{frozen['annotated_tag']}^{{}}"], cwd=ROOT, text=True
    ).strip() == frozen["commit"]

    scope = result["scope"]
    assert scope == {
        "split": "demo validation 45-49",
        "sample_count": 5869,
        "task_count": 10,
        "demo_count": 50,
        "simulator_rollout_count": 0,
        "init_state_40_49_read_or_run": False,
    }
    sample_path = ROOT / result["sample_artifact"]["path"]
    assert sha256(sample_path) == result["sample_artifact"]["sha256"]
    with np.load(sample_path) as samples:
        assert {name: list(samples[name].shape) for name in samples.files} == result[
            "sample_artifact"
        ]["arrays"]
        assert samples["task_id"].tolist() == sorted(
            samples["task_id"].tolist()
        )
        assert set(samples["task_id"].tolist()) == set(range(10))
        assert set(samples["demo_id"].tolist()) == set(range(45, 50))
        assert len(
            set(zip(samples["task_id"].tolist(), samples["demo_id"].tolist()))
        ) == 50
        assert all(np.isfinite(samples[name]).all() for name in samples.files)

        raw_mse = samples["1x_raw_action_mse"].astype(np.float64)
        scores = samples["1x_scores"].astype(np.float64)
        independent_gain = 1.0 - float(np.min(raw_mse, axis=1).mean()) / float(
            raw_mse[:, 0].mean()
        )
        independent_top_accuracy = float(
            np.mean(np.argmin(raw_mse, axis=1) == np.argmax(scores, axis=1))
        )
        production = result["production_scale_1x"]
        assert math.isclose(
            independent_gain,
            production["candidate_quality"]["best_of_3_raw_mse_relative_gain"],
            rel_tol=0.0,
            abs_tol=2e-7,
        )
        assert math.isclose(
            independent_top_accuracy,
            production["ranking_quality"][
                "top_score_equals_oracle_best_fraction"
            ],
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    assert list(result["perturbation_scale_sweep"]) == [
        "0x",
        "0.25x",
        "0.5x",
        "1x",
        "2x",
    ]
    decision = result["decision"]
    expected_blockers: list[str] = []
    if not decision["candidate_generator"]["pass"]:
        expected_blockers.append("CANDIDATE_GENERATOR_BLOCK")
    if not decision["reranker_score"]["pass"]:
        expected_blockers.append("RERANKER_SCORE_BLOCK")
    if not decision["wm_action_sensitivity"]["pass"]:
        expected_blockers.append("WM_ACTION_SENSITIVITY_BLOCK")
    if not expected_blockers:
        expected_blockers.append("NO_MECHANISM_BLOCK")
    assert result["blockers"] == decision["blockers"] == expected_blockers
    assert decision["candidate_generator"]["pass"] is True
    assert decision["reranker_score"]["pass"] is False
    assert decision["wm_action_sensitivity"]["pass"] is False
    assert decision["new_c1_controller_design_allowed"] is False

    interpretation = result["observed_mismatch_gate_interpretation"]
    assert interpretation["valid_use"] == "WM candidate-discriminability diagnostic"
    assert interpretation["may_alone_drive_controller_change"] is False
    forbidden = result["forbidden_activity_audit"]
    assert forbidden and not any(forbidden.values())
    assert result["score_component_scale"]["weights_modified"] is False
    assert finite_tree(result)

    report = REPORT_PATH.read_text()
    assert "PENDING" not in report
    for blocker in result["blockers"]:
        assert f"`{blocker}`" in report
    assert "init states 40--49" in report
    print(
        {
            "status": "PASS",
            "samples": scope["sample_count"],
            "blockers": result["blockers"],
            "sample_artifact_sha256": result["sample_artifact"]["sha256"],
        }
    )


if __name__ == "__main__":
    main()
