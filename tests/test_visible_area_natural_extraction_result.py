import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/visible_area_natural_extraction/diagnosis.json"
DERIVED = ROOT / "artifacts/stage3/visible_area_natural_extraction/evaluation_derived.npz"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_result_hashes_and_finite_arrays():
    result = json.loads(RESULT.read_text())
    assert result["artifacts"]["evaluation_derived"]["sha256"] == digest(DERIVED)
    with np.load(DERIVED, allow_pickle=False) as data:
        assert len(data.files) >= 10
        assert all(np.isfinite(data[name]).all() for name in data.files)
        assert data["target__area"].shape == (960,)
        assert data["target__pair_delta"].shape == (960,)


def test_result_decision_matches_gates():
    result = json.loads(RESULT.read_text())
    e0 = result["gates"]["E0_visual"]["complete"]
    e1 = result["gates"]["E1_visual_language"]["complete"]
    if e0 or e1:
        assert result["next_stage_authorization"] == "GO_MINIMAL_SPATIAL_WM_PROTOTYPE"
        assert "NATURAL_VISIBLE_AREA_PROXY_VALID" in result["final_mechanism_labels"]
        assert result["selected_extractor"] is not None
    else:
        assert result["next_stage_authorization"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
        assert result["selected_extractor"] is None
    assert result["persistent_controller_authorization"] is False
    assert result["alternative_encoder_audit_authorization"] is False


def test_forbidden_activity_remains_false():
    audit = json.loads(RESULT.read_text())["forbidden_activity_audit"]
    assert not any(audit.values())

