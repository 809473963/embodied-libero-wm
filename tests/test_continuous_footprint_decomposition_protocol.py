import hashlib
import json
from pathlib import Path

import numpy as np

from src.probes.continuous_footprint_decomposition import CANDIDATES, DIMENSIONS, footprint_candidates, pair_summary
from src.probes.spatial_target_design import footprint_from_object_masks, metadata_from_derived


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/continuous_footprint_information_decomposition_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def test_scope_partitions_and_authorization_are_frozen():
    protocol = load_protocol()
    assert protocol["status"] == "PREREGISTERED_BEFORE_CALIBRATION_OR_EVALUATION_RESULTS"
    assert protocol["partitions"]["calibration"]["init_state_indices"] == [31, 32, 33, 34]
    assert protocol["partitions"]["development_evaluation"]["init_state_indices"] == [36, 37, 38, 39]
    assert protocol["partitions"]["forbidden_init_state_indices"] == list(range(40, 50))
    assert protocol["current_authorization"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert protocol["controller_authorization"] is False


def test_exact_candidate_family_and_dimensions():
    protocol = load_protocol()
    assert tuple(protocol["candidate_order"]) == CANDIDATES
    assert {name: record["dimension"] for name, record in protocol["candidates"].items()} == DIMENSIONS
    assert protocol["candidates"]["F6"]["support_rule"] == "occupancy > 0.0"
    assert protocol["missing_values"]["zero_mass_rule"] == "DATA_OR_IMPLEMENTATION_BLOCK"


def test_calibration_footprints_are_finite_and_pair_summaries_are_fixed_six_dimensional():
    mask_path = ROOT / "artifacts/stage3/image_space_target_audit_restart/calibration_object_masks.npz"
    derived_path = ROOT / "artifacts/stage3/spatial_visual_target_design/calibration_derived.npz"
    with np.load(mask_path, allow_pickle=False) as masks, np.load(derived_path, allow_pickle=False) as derived:
        occupancy, area = footprint_from_object_masks(masks["per_object_masks"], 7)
        metadata = metadata_from_derived(derived)
    candidates = footprint_candidates(occupancy, area)
    assert tuple(candidates) == CANDIDATES
    for name, candidate in candidates.items():
        assert candidate.state.shape == (40, 3, 8, DIMENSIONS[name])
        assert np.isfinite(candidate.state).all()
        summary = pair_summary(candidate, metadata)
        assert summary.shape == (960, 6)
        assert np.isfinite(summary).all()


def test_gates_and_minimum_dimension_selection_are_narrow():
    protocol = load_protocol()
    gates = protocol["gates"]
    assert gates["minimum_primary_increment"] == 0.10
    assert gates["minimum_residual_increment"] == 0.05
    assert gates["minimum_retention_vs_F0"] == 0.80
    assert gates["minimum_positive_tasks"] == 7
    assert gates["minimum_positive_horizons"] == 4
    assert gates["minimum_positive_pairs"] == 3
    assert protocol["decision"]["minimum_rule"] == "lowest dimension among candidates with a complete target gate; candidate order breaks exact dimensional ties"
    assert protocol["decision"]["world_model_block"] == "BLOCK_SPATIAL_WM_PROTOTYPE"


def test_protocol_hashes_match_frozen_inputs_and_implementation():
    protocol = load_protocol()
    for record in protocol["frozen_upstream"]["files"]:
        assert sha256(ROOT / record["path"]) == record["sha256"]
    for record in protocol["implementation"].values():
        assert sha256(ROOT / record["path"]) == record["sha256"]


def test_no_forbidden_activity_is_authorized():
    protocol = load_protocol()
    text = json.dumps(protocol).lower()
    for phrase in ("do not train a world model", "do not train a natural decoder", "do not access init40-49", "seed23/47"):
        assert phrase in text
    assert protocol["decision"]["alternative_encoder_authorized"] is False
