import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.allowed_demo_guard import DemoAccessViolation
from src.data.semantic_feature_guard import AllowedDemoFeatureGuard
from src.probes.semantic_value_distillation import (
    DirectSemanticPair,
    S1AbsoluteSemantic,
    ScalarSemanticPotential,
    parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]


def test_feature_guard_rejects_forbidden_demo_before_open() -> None:
    guard = AllowedDemoFeatureGuard("calibration", ROOT / "does-not-need-to-exist.hdf5")
    with pytest.raises(DemoAccessViolation):
        guard.read_vision(0, 40)
    assert guard.ledger.as_dict()["forbidden_access_count"] == 1


def test_feature_guard_rejects_enumeration() -> None:
    guard = AllowedDemoFeatureGuard("evaluation", ROOT / "does-not-need-to-exist.hdf5")
    with pytest.raises(DemoAccessViolation):
        guard.keys()


def test_student_capacity_and_pair_antisymmetry() -> None:
    language = torch.randn(7, 512)
    a, b = torch.randn(7, 512), torch.randn(7, 512)
    pair = DirectSemanticPair(True)
    forward, reverse = pair(a, b, language), pair(b, a, language)
    assert torch.equal(forward, -reverse)
    assert parameter_count(pair) <= 200_000
    assert parameter_count(DirectSemanticPair(False)) <= 200_000
    assert parameter_count(S1AbsoluteSemantic()) <= 200_000
    assert parameter_count(ScalarSemanticPotential()) <= 200_000


def test_frozen_architecture_and_sampler_contracts() -> None:
    architectures = json.loads((ROOT / "configs/semantic_student_architectures.json").read_text())
    sampler = json.loads((ROOT / "configs/semantic_pair_sampler.json").read_text())
    selection = json.loads((ROOT / "configs/semantic_student_selection.json").read_text())
    assert architectures["parameter_ceiling"] == 200_000
    assert sampler["primary_gap_max"] == 8
    assert sampler["reverse_rule"].startswith("add the exact reverse")
    assert selection["noninferiority_tolerance_points"] == 2.0
    assert selection["post_evaluation_reselection"] is False


def test_t1_definition_is_exactly_m2_t1() -> None:
    definition = json.loads((ROOT / "configs/t1_teacher_frozen_definition.json").read_text())
    assert definition["selected_margin"] == "M2"
    assert definition["selected_teacher"] == "T1"
    assert np.isclose(definition["epsilon_goal"], 0.004065508647683736)
