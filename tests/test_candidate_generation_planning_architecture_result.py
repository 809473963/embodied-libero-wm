"""Independent checks of the frozen planning-architecture diagnosis."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from src.probes.planning_architecture import sha256


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/candidate_generation_planning_architecture/diagnosis.json"


def load() -> tuple[dict, np.lib.npyio.NpzFile]:
    result = json.loads(RESULT.read_text())
    sidecar_path = ROOT / result["sidecar"]["path"]
    assert sha256(sidecar_path) == result["sidecar"]["sha256"]
    return result, np.load(sidecar_path)


def test_scope_and_identity_boundary() -> None:
    result, arrays = load()
    assert result["scope"]["demo_ids"] == [36, 37, 38, 39]
    assert result["scope"]["sample_count"] == 4563
    assert set(arrays["demo_id"].tolist()) == {36, 37, 38, 39}
    assert set(arrays["task_id"].tolist()) == set(range(10))
    assert result["scope"]["simulator_runs"] == 0
    assert result["scope"]["reward_done_success_reads"] == 0
    assert result["scope"]["forbidden_demo_access"] is False


def test_p0_and_family_gains_recompute_from_sidecar() -> None:
    result, arrays = load()
    for name in result["families"]:
        default = arrays[f"{name}__default_error"]
        oracle = arrays[f"{name}__oracle_error"]
        measured = (default.mean() - oracle.mean()) / default.mean()
        assert np.isclose(measured, result["families"][name]["relative_mse_gain"], atol=1e-12)
        assert np.all(oracle <= default + 1e-15)
        assert np.isfinite(arrays[f"{name}__winner_action"]).all()
    assert np.isclose(result["families"]["P0_N3"]["relative_mse_gain"], 0.016829234483728354)


def test_nested_and_alias_contracts() -> None:
    result, arrays = load()
    gains = [result["families"][name]["relative_mse_gain"] for name in ("P0_N3", "P1_N8", "P1_N16", "P1_N32", "P1_N64")]
    assert all(left <= right for left, right in zip(gains, gains[1:]))
    assert np.array_equal(
        arrays["P2_translation_rotation_N32__oracle_error"],
        arrays["P2_full_non_gripper_N32__oracle_error"],
    )
    assert np.array_equal(
        arrays["P1_N32__oracle_error"], arrays["P3_scale1_N32__oracle_error"]
    )


def test_decision_mapping_and_authorization() -> None:
    result, _ = load()
    decision = result["decision"]
    assert decision["case"] == "B"
    assert decision["final_mechanism_labels"] == [
        "CURRENT_N3_CANDIDATE_SET_UNDERPOWERED",
        "MORE_LOCAL_SAMPLES_ADD_HEADROOM",
        "GRIPPER_SEARCH_ADDS_HEADROOM",
    ]
    assert decision["planner_authorization"] == "GO_EXPANDED_CANDIDATE_PLANNER_DESIGN"
    assert decision["spatial_world_model_authorization"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert decision["controller_authorization"] is False
    assert decision["iterative_significant"] is False


def test_finite_and_validity_audit() -> None:
    result, arrays = load()
    assert all(np.isfinite(value).all() for value in arrays.values())
    for family in result["families"].values():
        assert family["invalid_candidate_count"] == 0
        assert family["oracle_worse_count"] == 0
        assert family["clip_fraction"] < 0.06
        assert 0.0 <= family["boundary_hit_fraction"] <= 1.0


def test_transitive_upstream_ledger_is_still_frozen() -> None:
    protocol_path = ROOT / "configs/relevant_object_visible_area_natural_extraction_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    upstream = protocol["frozen_upstream"]
    assert subprocess.check_output(
        ["git", "rev-parse", f"{upstream['tag']}^{{}}"], cwd=ROOT, text=True
    ).strip() == upstream["commit"]
    assert len(upstream["files"]) == 14
    for record in upstream["files"]:
        assert sha256(ROOT / record["path"]) == record["sha256"]
    assert "C0-v2" in upstream["transitive_ledger"]
    assert "WM-v2A/B/B.1/C-EJ" in upstream["transitive_ledger"]
