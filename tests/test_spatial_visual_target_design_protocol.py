import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/spatial_visual_target_design_protocol.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def test_protocol_boundary_and_candidate_set() -> None:
    protocol = _protocol()
    assert protocol["protocol_version"] == "spatial-visual-target-design-study-v1"
    assert protocol["status"] == "PREREGISTERED_BEFORE_CALIBRATION_OR_EVALUATION_RESULTS"
    assert protocol["partitions"]["calibration"]["init_state_indices"] == [31, 32, 33, 34]
    assert protocol["partitions"]["development_evaluation"]["init_state_indices"] == [36, 37, 38, 39]
    assert protocol["partitions"]["forbidden_init_state_indices"] == list(range(40, 50))
    assert set(protocol["candidate_definitions"]) == {"D0", "D1", "D2", "D3", "D4", "D5"}


def test_protocol_freezes_choices_and_gates() -> None:
    protocol = _protocol()
    assert protocol["t2_generic"]["pca_ranks"] == [1, 2, 4]
    assert protocol["footprint"]["resolutions"] == [4, 7]
    assert protocol["t4_proxy"]["hidden_widths"] == [0, 32]
    gates = protocol["gates"]
    assert gates["minimum_primary_increment"] == 0.10
    assert gates["minimum_residual_increment"] == 0.05
    assert gates["minimum_positive_tasks"] == 7
    assert gates["minimum_positive_horizons"] == 4


def test_protocol_hashes_frozen_inputs_and_implementation() -> None:
    protocol = _protocol()
    for record in protocol["frozen_upstream"]["files"]:
        assert _sha256(ROOT / record["path"]) == record["sha256"]
    for record in protocol["implementation"].values():
        assert _sha256(ROOT / record["path"]) == record["sha256"]


def test_protocol_preserves_authorization_boundary() -> None:
    protocol = _protocol()
    assert protocol["current_authorization"] == "GO_SPATIAL_TARGET_DESIGN"
    assert protocol["controller_authorization"] is False
    forbidden = " ".join(protocol["forbidden"])
    for phrase in ("World Model", "reranker", "controller", "init40-49", "seed23/47"):
        assert phrase in forbidden
