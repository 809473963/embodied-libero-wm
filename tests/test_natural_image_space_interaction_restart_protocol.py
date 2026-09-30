import hashlib
import json
from pathlib import Path

import numpy as np

from src.probes.image_space_relevance import mask_geometry
from src.probes.image_space_relevance_restart import project_with_contract


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/natural_image_space_interaction_restart_protocol.json"
CONTRACT_PATH = ROOT / "artifacts/camera_projection_contract/frozen_contract.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_camera_contract_hash() -> None:
    protocol = _protocol()
    assert _sha256(CONTRACT_PATH) == "5eb187c5b3b7eab7a8f2503f9ee8e9dc4bb12e2e65cfb5d017b1913c2d614e0e"
    assert protocol["camera_contract"]["sha256"] == _sha256(CONTRACT_PATH)


def test_ee_projection_uses_frozen_contract() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    point = np.asarray([-0.22, 0.0, 1.16], dtype=np.float64)
    projected, depth = project_with_contract(point, contract)
    raw = np.r_[point, 1.0] @ np.asarray(contract["world_to_raw_pixel"]).T
    assert np.allclose(projected, [raw[0] / raw[2], 127.0 - raw[1] / raw[2]])
    assert depth == raw[2] and depth > 0


def test_no_projection_reselection() -> None:
    protocol = _protocol()
    assert protocol["camera_contract"]["projection_reselection"] is False
    collector = (ROOT / "scripts/collect_image_space_targets_restart.py").read_text()
    assert "candidate_conventions" not in collector
    assert "minimum_median_margin_over_alternative_pixels" not in collector


def test_object_mask_identity() -> None:
    protocol = _protocol()
    assert protocol["relevant_objects"]["selection"] == 'runtime parsed_problem["obj_of_interest"]'
    source = (ROOT / "scripts/collect_image_space_targets_restart.py").read_text()
    assert 'parsed_problem.get("obj_of_interest", [])' in source
    assert 'np.array_equal(union, semantic["relevant"])' in source


def test_object_centroid_missing_rule() -> None:
    protocol = _protocol()
    empty = np.zeros((128, 128), dtype=bool)
    assert mask_geometry(empty)["centroid"] is None
    assert protocol["image_space_targets"]["invisible_object_rule"].startswith("visibility=0")
    assert "never imputed" in protocol["image_space_targets"]["invisible_object_rule"]


def test_relative_geometry_definition() -> None:
    targets = _protocol()["image_space_targets"]
    assert targets["relative_image_displacement"] == "maximum norm of pair difference in (object centroid - projected EE)"
    assert targets["ee_object_distance_change"] == "maximum absolute change in object-centroid to projected-EE distance"


def test_target_aggregation_frozen() -> None:
    protocol = _protocol()
    assert protocol["image_space_targets"]["object_aggregation"] == (
        "retain both runtime obj_of_interest slots and use calibration-frozen maximum "
        "across valid object-wise scalar magnitudes"
    )


def test_branch_block_split() -> None:
    partitions = _protocol()["partitions"]
    assert partitions["calibration"]["init_state_indices"] == [31, 32, 33, 34]
    assert partitions["development_evaluation"]["init_state_indices"] == [36, 37, 38, 39]
    assert partitions["development_evaluation"]["use"] == (
        "frozen development evaluation, not blind confirmation"
    )
    assert _protocol()["scope"]["bootstrap_and_split_unit"].startswith("task-stratified matched branch point")


def test_no_init40_49() -> None:
    forbidden = set(_protocol()["partitions"]["forbidden_init_state_indices"])
    assert set(range(40, 50)) <= forbidden
    assert forbidden.isdisjoint({31, 32, 33, 34, 36, 37, 38, 39})


def test_no_reward_done_success() -> None:
    for relative in (
        "scripts/collect_image_space_targets_restart.py",
        "scripts/analyze_natural_image_space_interaction_restart.py",
    ):
        source = (ROOT / relative).read_text()
        for forbidden in ("get_reward", "check_success", "_check_success", '["reward"]', '["done"]'):
            assert forbidden not in source


def test_patch_summary_unchanged() -> None:
    representations = _protocol()["representations"]
    assert representations["patch_10d"] == "exact frozen patch_summary from natural-spatial audit; dimensions and broad interaction component unchanged"
    assert representations["global_plus_patch_20d"] == "exact frozen concatenation"


def test_global_summary_unchanged() -> None:
    representations = _protocol()["representations"]
    assert representations["global_10d"] == "exact frozen global_summary from natural-spatial audit"


def test_10pct_gate_unchanged() -> None:
    gates = _protocol()["gates"]
    assert gates["minimum_incremental_rmse_improvement"] == 0.10
    assert gates["minimum_positive_tasks"] == 7
    assert gates["minimum_positive_horizons"] == 4
    assert gates["minimum_positive_pairs"] == 3


def test_no_upstream_overwrite() -> None:
    protocol = _protocol()
    for group in ("frozen_upstream", "preserved_blocked_audit"):
        for artifact in protocol[group]["files"]:
            assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
