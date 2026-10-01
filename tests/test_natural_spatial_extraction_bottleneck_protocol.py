import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.probes.natural_spatial_extraction import CAPACITIES, DESIGNS, PatchDecoder, binary_metrics


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/natural_spatial_extraction_bottleneck_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def test_scope_and_partitions_are_frozen():
    protocol = load_protocol()
    assert protocol["status"] == "PREREGISTERED_BEFORE_CALIBRATION_OR_EVALUATION_RESULTS"
    assert protocol["current_authorization"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert protocol["partitions"]["calibration"]["init_state_indices"] == [31, 32, 33, 34]
    assert protocol["partitions"]["development_evaluation"]["init_state_indices"] == [36, 37, 38, 39]
    assert protocol["partitions"]["forbidden_init_state_indices"] == list(range(40, 50))
    assert protocol["controller_authorization"] is False


def test_exact_teacher_and_capacity_ladders():
    protocol = load_protocol()
    assert tuple(protocol["teachers"]["natural_candidates"]) == DESIGNS
    assert protocol["teachers"]["resolution"] == 7
    assert protocol["teachers"]["B1-binary"]["occupancy_rule"] == "soft_cell_occupancy > 0.0"
    assert protocol["teachers"]["B3-ordinal"]["positive_split"] == "calibration positive-cell median"
    assert protocol["teachers"]["B4-binary-coarse-area"]["area_bins"] == "calibration visible-area tertiles"
    assert tuple(protocol["decoders"]["capacities"]) == CAPACITIES
    assert protocol["decoders"]["C3-local-context"]["spatial_layers"] == "one 3x3 depthwise convolution plus one 1x1 pointwise convolution"


def test_decoder_shapes_and_capacity_are_small():
    patch = torch.zeros(2, 49, 512)
    for capacity in CAPACITIES:
        model = PatchDecoder(capacity, 2, "continuous")
        occupancy, area = model(patch)
        assert occupancy.shape == (2, 7, 7)
        assert area.shape == (2,)
        assert sum(parameter.numel() for parameter in model.parameters()) < 200_000
        ordinal = PatchDecoder(capacity, 3, "coarse")
        occupancy, area = ordinal(patch)
        assert occupancy.shape == (2, 7, 7, 3)
        assert area.shape == (2, 3)


def test_binary_metrics_do_not_use_overall_accuracy():
    record = binary_metrics(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 0, 1, 0]), 0.5)
    assert set(record) == {"iou", "dice_f1", "balanced_accuracy", "positive_recall", "positive_precision", "auroc"}
    assert "accuracy" not in record


def test_protocol_hashes_and_upstream_files():
    protocol = load_protocol()
    for record in protocol["frozen_upstream"]["files"]:
        assert sha256(ROOT / record["path"]) == record["sha256"]
    for record in protocol["implementation"].values():
        assert sha256(ROOT / record["path"]) == record["sha256"]


def test_gate_and_decision_mapping_are_narrow():
    protocol = load_protocol()
    gates = protocol["gates"]
    assert gates["minimum_primary_increment"] == 0.10
    assert gates["minimum_residual_increment"] == 0.05
    assert gates["minimum_positive_tasks"] == 7
    assert gates["minimum_positive_horizons"] == 4
    assert gates["minimum_positive_pairs"] == 3
    assert gates["natural_recovery"]["binary_minimum_iou"] == 0.60
    assert protocol["decision"]["go"] == "GO_SPATIAL_WM_PROTOTYPE"
    assert protocol["decision"]["block"] == "BLOCK_SPATIAL_WM_PROTOTYPE"
    assert "SOFT_FOOTPRINT_INFORMATION_REQUIRED" in protocol["decision"]["allowed_labels"]
    assert protocol["decision"]["go_scope"] == "seed11 single-model offline spatial World Model mechanism prototype only"


def test_no_forbidden_architecture_or_activity_is_authorized():
    protocol = load_protocol()
    text = json.dumps(protocol).lower()
    assert "sam" not in protocol["decoders"]["allowed_inputs"]
    assert protocol["decoders"]["allowed_inputs"] == "frozen natural CLIP ViT-B/32 patch features [49,512] only"
    assert all(term in text for term in ("do not train a world model", "do not access init40-49", "seed23/47"))
