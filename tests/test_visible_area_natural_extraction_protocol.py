import json
from pathlib import Path

import numpy as np

from src.probes.visible_area_extraction import (
    AreaHead,
    candidate_ordering,
    image_features,
    pair_rows,
    parameter_count,
    scalar_pair_summary,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/relevant_object_visible_area_natural_extraction_protocol.json"


def test_protocol_boundary_and_gates():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["status"] == "PREREGISTERED_BEFORE_CALIBRATION_OR_EVALUATION_RESULTS"
    assert protocol["partitions"]["calibration"]["init_state_indices"] == [31, 32, 33, 34]
    assert protocol["partitions"]["development_evaluation"]["init_state_indices"] == [36, 37, 38, 39]
    assert protocol["partitions"]["forbidden_init_state_indices"] == list(range(40, 50))
    assert protocol["gates"]["minimum_mae_improvement"] == 0.20
    assert protocol["gates"]["minimum_area_increment"] == 0.10
    assert protocol["gates"]["minimum_residual_increment"] == 0.05
    assert protocol["gates"]["minimum_dynamic_range_ratio"] == 0.25
    assert protocol["decision"]["persistent_controller_authorization"] is False


def test_model_capacity_is_small():
    assert parameter_count(512, "M1_linear") == 513
    assert parameter_count(1024, "M2_mlp") < 200_000
    assert isinstance(AreaHead(512, "M2_mlp"), AreaHead)


def test_feature_and_pair_contracts():
    rng = np.random.default_rng(1)
    global_feature = rng.normal(size=(40, 3, 8, 512)).astype(np.float32)
    patch = rng.normal(size=(40, 3, 8, 49, 512)).astype(np.float32)
    language = rng.normal(size=(10, 512)).astype(np.float32)
    task = np.repeat(np.arange(10), 4)
    features = image_features(global_feature, patch, language, task)
    assert features["E0_visual__V0_global"].shape == (960, 512)
    assert features["E1_visual_language__V1_patch_mean"].shape == (960, 1024)
    state = rng.uniform(size=(40, 3, 8))
    delta, metadata = pair_rows(state)
    assert delta.shape == (960,)
    assert metadata["branch"].shape == (960,)
    assert scalar_pair_summary(state).shape == (960, 6)
    ordering = candidate_ordering(state, state)
    assert ordering["mean_three_way_spearman"] == 1.0
    assert ordering["pairwise_ordering_accuracy"] == 1.0

