from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.probes.missing_state import scalar_metrics


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/missing_state_audit_protocol.json"
RESULT = ROOT / "results/missing_state_audit/diagnosis.json"
REPORT = ROOT / "docs/local_visual_counterfactual_missing_state_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_result_scope_hashes_and_forbidden_activity() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    result = json.loads(RESULT.read_text())
    assert result["status"] == "COMPLETE"
    assert result["protocol"]["sha256"] == sha256(PROTOCOL)
    assert result["evaluation_scope"] == {
        "init_state_indices": [36, 37, 38, 39],
        "branch_points": 40,
        "rows": 960,
        "tasks": list(range(10)),
        "pairs": ["A0-A1", "A0-A2", "A1-A2"],
        "horizons": list(range(1, 9)),
    }
    assert sha256(ROOT / protocol["outputs"]["state_inventory"]) == result["sources"][
        "state_inventory_sha256"
    ]
    assert sha256(ROOT / protocol["partitions"]["evaluation"]["source_sidecar"]) == result[
        "sources"
    ]["evaluation_source_sidecar_sha256"]
    assert sha256(ROOT / protocol["partitions"]["evaluation"]["state_sidecar"]) == result[
        "sources"
    ]["evaluation_state_sidecar_sha256"]
    assert sha256(ROOT / result["artifacts"]["evaluation_metrics"]) == result["artifacts"][
        "evaluation_metrics_sha256"
    ]
    assert not any(result["forbidden_activity_audit"].values())
    assert result["controller_authorized"] is False


def test_scalar_metrics_and_nested_improvements_recompute() -> None:
    result = json.loads(RESULT.read_text())
    arrays = np.load(ROOT / result["artifacts"]["evaluation_metrics"], allow_pickle=False)
    target = arrays["target_visual"]
    assert target.shape == (960,)
    for set_name, dimensions in zip(("S0", "S1", "S2", "S3", "S4"), (10, 14, 24, 37, 47)):
        prediction = arrays[f"prediction_{set_name}"]
        assert prediction.shape == (960,)
        assert arrays[f"group_{['ej', 'rotation', 'object', 'contact', 'relative'][min(int(set_name[1]), 4)]}"].shape[0] == 960
        measured = scalar_metrics(prediction, target)
        recorded = result["scalar_probes"]["sets"][set_name]
        assert recorded["dimensions"] == dimensions
        assert np.isclose(measured["rmse"], recorded["rmse"], atol=1e-14)
        assert np.isclose(measured["mae"], recorded["mae"], atol=1e-14)
    assert np.isfinite(target).all()
    assert all(np.isfinite(arrays[name]).all() for name in arrays.files)


def test_mechanical_labels_and_authorization() -> None:
    result = json.loads(RESULT.read_text())
    gates = result["gate_results"]
    assert gates == {
        "rotation_supported": False,
        "object_supported": False,
        "contact_supported": False,
        "relative_supported": False,
        "robot_kinematics_only_insufficient": True,
        "minimum_incremental_rmse_improvement": 0.05,
        "minimum_spearman": 0.1,
        "minimum_positive_tasks": 3,
    }
    assert result["final_mechanism_labels"] == [
        "ROBOT_KINEMATICS_ONLY_INSUFFICIENT",
        "NO_TESTED_STATE_EXPLAINS_VISUAL_COUNTERFACTUAL",
    ]
    assert result["next_stage_authorization"] == "BLOCK_STRUCTURED_PHYSICAL_WM_ROUTE"
    assert result["secondary_direction_probes"] == {}
    for name in ("S1_vs_S0", "S2_vs_S1", "S3_vs_S2", "S4_vs_S3"):
        record = result["scalar_probes"]["increments"][name]
        assert record["relative_rmse_improvement"] < 0.05 or record["ci95"][0] <= 0
    for record in result["group_shuffle_controls"].values():
        assert record["relative_degradation"] <= 0 or record["ci95"][0] <= 0


def test_rotation_association_exists_but_increment_gate_fails() -> None:
    result = json.loads(RESULT.read_text())
    rotation = result["rotation_chain"]
    assert rotation["action_to_real_rotation"] is True
    assert rotation["real_rotation_to_visual"] is True
    assert rotation["S1_over_S0"] is False
    assert rotation["action_to_rotation_association"]["overall"]["spearman"] > 0.7
    assert result["raw_associations"]["ee_rotation"]["overall"]["spearman"] > 0.3
    assert result["group_shuffle_controls"]["rotation"]["ci95"][0] < 0


def test_report_preserves_oracle_boundary_verdict_and_artifact_hashes() -> None:
    result = json.loads(RESULT.read_text())
    report = REPORT.read_text()
    for text in (
        "ROBOT_KINEMATICS_ONLY_INSUFFICIENT",
        "NO_TESTED_STATE_EXPLAINS_VISUAL_COUNTERFACTUAL",
        "BLOCK_STRUCTURED_PHYSICAL_WM_ROUTE",
        "Controller authorization: `false`",
        "Future use is oracle",
        result["sources"]["evaluation_state_sidecar_sha256"],
        result["artifacts"]["evaluation_metrics_sha256"],
        sha256(RESULT),
    ):
        assert text in report
    assert "40--49" in report
    assert "not a deployable World Model or controller input" in report
