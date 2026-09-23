#!/usr/bin/env python3
"""Result-level provenance, boundary, and interpretation audit for C1-v2.2."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/c1_v2_2_wm_action_conditioning.json"
RESULT_PATH = ROOT / "results/c1_v2_2/wm_action_conditioning.json"
REPORT_PATH = ROOT / "docs/c1_v2_2_wm_action_conditioning_diagnosis.md"


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
    assert result["protocol"]["sha256"] == sha256(PROTOCOL_PATH)
    assert result["protocol"]["preregistered_commit"].startswith("b2e0e68")
    assert subprocess.check_output(
        ["git", "cat-file", "-t", result["protocol"]["preregistered_commit"]],
        cwd=ROOT,
        text=True,
    ).strip() == "commit"

    frozen = protocol["frozen_c1_v2_1"]
    assert subprocess.check_output(
        ["git", "rev-parse", f"{frozen['annotated_tag']}^{{}}"],
        cwd=ROOT,
        text=True,
    ).strip() == frozen["commit"]
    assert sha256(ROOT / frozen["result"]) == frozen["result_sha256"]
    assert sha256(ROOT / frozen["report"]) == frozen["report_sha256"]
    assert result["scope"] == {
        "split": "D13 validation demos 45-49",
        "sample_count": 5869,
        "episode_count": 50,
        "task_count": 10,
        "simulator_rollout_count": 0,
        "init_state_indices_read_or_run": [],
    }

    sample_path = ROOT / result["sample_artifact"]["path"]
    assert sha256(sample_path) == result["sample_artifact"]["sha256"]
    with np.load(sample_path) as samples:
        assert {name: list(samples[name].shape) for name in samples.files} == result[
            "sample_artifact"
        ]["arrays"]
        names = samples["condition_names"].tolist()
        assert names == [
            "true",
            "zero",
            "shuffle_task",
            "shuffle_cross",
            "reverse",
            "temporal_shuffle",
            "repeat_first",
            "scale_0p25",
            "scale_0p5",
            "scale_1p5",
        ]
        task = samples["task_id"].astype(np.int64)
        demo = samples["demo_id"].astype(np.int64)
        timestep = samples["start_timestep"].astype(np.int64)
        assert set(task.tolist()) == set(range(10))
        assert set(demo.tolist()) == set(range(45, 50))
        assert len(set(zip(task.tolist(), demo.tolist(), timestep.tolist()))) == 5869
        within = samples["within_task_source_index"].astype(np.int64)
        cross = samples["cross_task_source_index"].astype(np.int64)
        assert np.all(within != np.arange(5869))
        assert np.all(task[within] == task)
        assert np.all(task[cross] != task)
        assert all(
            np.isfinite(samples[name]).all()
            for name in samples.files
            if samples[name].dtype.kind in "fiu"
        )

        errors = samples["ensemble_error"].astype(np.float64)
        true_error = float(errors[0].mean())
        within_delta = float((errors[2] - errors[0]).mean())
        within_accuracy = float(np.mean(errors[0] < errors[2]))
        assert math.isclose(
            true_error,
            result["prediction_error"]["true"]["overall"],
            rel_tol=0.0,
            abs_tol=2e-9,
        )
        assert math.isclose(
            within_delta,
            result["true_action_comparisons"]["shuffle_task"]["delta"],
            rel_tol=0.0,
            abs_tol=2e-9,
        )
        assert math.isclose(
            within_accuracy,
            result["true_action_comparisons"]["shuffle_task"][
                "true_action_win_accuracy"
            ],
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    action = result["action_construction"]
    assert action["within_task_source_identity_all_different"] is True
    assert action["within_task_source_task_all_equal"] is True
    assert action["cross_task_source_task_all_different"] is True
    assert action["normalized_zero_was_not_used_as_physical_zero"] is True
    assert any(abs(value) > 0.1 for value in action["physical_zero_wm_normalized_value"])

    comparison = result["true_action_comparisons"]["shuffle_task"]
    assert comparison["relative_delta"] > 0.005
    assert comparison["delta_bootstrap"]["ci95_lower"] > 0.0
    assert comparison["true_action_win_accuracy"] > 0.525
    assert comparison["accuracy_bootstrap"]["ci95_lower"] > 0.5
    assert all(
        comparison["per_task"][str(task_id)]["delta"] > 0.0
        for task_id in range(10)
    )
    assert all(
        comparison["per_task"][str(task_id)]["true_action_win_accuracy"] > 0.5
        for task_id in range(10)
    )

    decision = result["decision"]
    assert all(decision["grouped_action_signal_checks"].values())
    assert decision["grouped_action_signal_pass_count"] == 4
    assert decision["action_signal_pass"] is True
    assert decision["task_dependent_action_signal"] is False
    assert decision["clip_state_sensitivity_checks"]["pass"] is False
    assert decision["low_local_sensitivity"] is True
    assert decision["no_action_conditioning_block"] is False
    assert result["verdicts"] == decision["verdicts"] == [
        "WM_HAS_ACTION_SIGNAL_BUT_LOW_LOCAL_SENSITIVITY"
    ]
    assert all(
        member["true_action_ranking_accuracy_overall"] > 0.5
        for member in result["ensemble_member_consistency"]["members"]
    )
    assert result["optional_gradient_analysis"]["execute"] is False
    assert result["nan_inf_audit"] == "pass"
    assert not any(result["forbidden_activity_audit"].values())
    assert finite_tree(result)

    report = REPORT_PATH.read_text()
    assert "PENDING" not in report
    assert (
        "Final verdict: `WM_HAS_ACTION_SIGNAL_BUT_LOW_LOCAL_SENSITIVITY`"
        in report
    )
    assert "init states 0--49 were not read or run" in report
    print(
        {
            "status": "PASS",
            "samples": result["scope"]["sample_count"],
            "verdicts": result["verdicts"],
            "sample_sha256": result["sample_artifact"]["sha256"],
        }
    )


if __name__ == "__main__":
    main()
