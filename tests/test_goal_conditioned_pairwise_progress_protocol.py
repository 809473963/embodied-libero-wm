"""Protocol and construction checks for the Stage-B progress audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.probes.goal_conditioned_progress import (
    AntisymmetricPairMLP,
    GAP_NAMES,
    ScalarProgressMLP,
    load_trajectories,
    make_cross_demo_controls,
    make_same_demo_pairs,
    parameter_count,
    sigmoid,
)
from src.probes.planning_architecture import sha256


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/goal_conditioned_pairwise_progress_protocol.json"


def load() -> dict:
    return json.loads(PROTOCOL.read_text())


def test_partition_and_forbidden_input_boundary() -> None:
    protocol = load()
    assert protocol["partitions"] == {
        "calibration_demo_ids": [31, 32, 33, 34],
        "evaluation_demo_ids": [36, 37, 38, 39],
        "historical_reference_demo_ids": [30, 35],
        "forbidden_demo_ids": list(range(40, 50)),
    }
    forbidden = protocol["input_contract"]["forbidden_inputs"]
    assert "expert action" in forbidden
    assert "reward" in forbidden and "success" in forbidden
    assert protocol["decision_rules"]["controller_always"] is False


def test_frozen_hashes_match() -> None:
    protocol = load()
    for record in protocol["frozen_inputs"] + protocol["implementation"]:
        assert sha256(ROOT / record["path"]) == record["sha256"]
    assert protocol["frozen_upstream"]["annotated_tag"] == "expanded-candidate-interface-valid"


def test_pair_bins_are_balanced_and_reverse_complete_on_calibration() -> None:
    store = ROOT / load()["frozen_inputs"][0]["path"]
    data = load_trajectories(store, [31, 32, 33, 34])
    pairs = make_same_demo_pairs(data, 32)
    assert len(data.trajectories) == 40
    assert len(pairs.a) == 15360
    assert {int(np.sum(pairs.gap_bin == index)) for index in range(len(GAP_NAMES))} == {2560}
    assert np.array_equal(pairs.a[0::2], pairs.b[1::2])
    assert np.array_equal(pairs.b[0::2], pairs.a[1::2])
    assert np.array_equal(pairs.label[0::2], 1 - pairs.label[1::2])


def test_cross_demo_controls_are_fixed_and_task_matched() -> None:
    store = ROOT / load()["frozen_inputs"][0]["path"]
    data = load_trajectories(store, [31, 32, 33, 34])
    controls = make_cross_demo_controls(data, 16)
    assert len(controls["matched"].a) == 640
    assert len(controls["mismatched"].a) == 1280
    for pairs in controls.values():
        assert np.all(data.task_id[pairs.a] == data.task_id[pairs.b])
        assert np.all(data.demo_id[pairs.a] != data.demo_id[pairs.b])


def test_model_capacity_and_structural_symmetry() -> None:
    models = [AntisymmetricPairMLP(512), AntisymmetricPairMLP(12), AntisymmetricPairMLP(524)]
    assert [parameter_count(model) for model in models] == [131201, 35201, 133505]
    assert parameter_count(ScalarProgressMLP(512)) == 65665
    generator = torch.Generator().manual_seed(11)
    a = torch.randn((7, 512), generator=generator)
    b = torch.randn((7, 512), generator=generator)
    language = torch.randn((7, 512), generator=generator)
    with torch.inference_mode():
        forward = models[0](a, b, language).numpy()
        reverse = models[0](b, a, language).numpy()
    assert np.allclose(forward, -reverse, atol=1e-7)
    assert np.allclose(sigmoid(forward) + sigmoid(reverse), 1.0, atol=1e-12)


def test_decision_thresholds_are_explicit() -> None:
    gate = load()["decision_rules"]
    assert gate["local_accuracy_min"] == 0.65
    assert gate["local_ci_lower_min"] == 0.55
    assert gate["positive_task_min"] == 7
    assert gate["wrong_language_drop_min"] == 0.05
    assert gate["robot_shortcut_required_visual_margin"] == 0.02
    assert gate["v2_noninferiority_margin"] == 0.02

