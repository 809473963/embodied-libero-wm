from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.probes.clip_internal_layer import (
    LAYERS, READERS, InternalLayerPairReader, ReferentChangeProbe,
    fixed_patch_coordinates, parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_i_protocol_freezes_teacher_split_and_finite_layers() -> None:
    protocol = json.loads((ROOT / "configs/clip_internal_layer_protocol.json").read_text())
    contract = json.loads((ROOT / "configs/clip_layer_extraction_contract.json").read_text())
    assert protocol["status"] == "PREREGISTERED_BEFORE_STAGE_I_FEATURE_EXTRACTION_OR_TRAINING"
    assert tuple(protocol["visual"]["features"]) == LAYERS
    assert tuple(protocol["readers"]["families"]) == READERS
    assert protocol["partitions"]["fit_demo_ids"] == [31, 32, 33]
    assert protocol["partitions"]["selection_demo_ids"] == [34]
    assert protocol["partitions"]["stage_i_not_opened_demo_ids"] == [36, 37, 38, 39]
    assert protocol["partitions"]["forbidden_including_metadata"] == list(range(40, 50))
    assert protocol["frozen_teacher"]["immutable"] is True
    assert protocol["selection"]["fair_material_baseline"] == "P1/L12-proj from this protocol"
    assert contract["implementation"]["hidden_state_indexing"].startswith("encoder hidden_states[0]")
    assert tuple(contract["fixed_extraction_points"]) == LAYERS
    for row in protocol["frozen_files"]:
        assert sha256(ROOT / row["path"]) == row["sha256"]


def test_stage_i_reader_capacity_shape_antisymmetry_and_coordinates() -> None:
    coordinates = torch.from_numpy(fixed_patch_coordinates())
    expected_axis = 2.0 * (np.arange(7) + 0.5) / 7.0 - 1.0
    assert coordinates.shape == (49, 2)
    assert np.allclose(np.unique(coordinates[:, 0]), expected_axis)
    assert np.allclose(np.unique(coordinates[:, 1]), expected_axis)
    for token_dim in (512, 768):
        tokens_a = torch.randn(3, 49, token_dim)
        tokens_b = torch.randn(3, 49, token_dim)
        language = torch.randn(3, 512)
        for family in READERS:
            model = InternalLayerPairReader(token_dim, family).eval()
            with torch.no_grad():
                forward = model(tokens_a, tokens_b, language, coordinates)
                reverse = model(tokens_b, tokens_a, language, coordinates)
                state, attention = model.encoder(tokens_a, language, coordinates, True)
            assert parameter_count(model) <= 150_000
            assert state.shape == (3, 51)
            assert attention is not None and attention.shape == (3, 49)
            assert torch.allclose(attention.sum(dim=1), torch.ones(3), atol=1e-6)
            assert torch.allclose(forward, -reverse, atol=1e-6)


def test_referent_probe_is_invariant_to_language_pair_order() -> None:
    model = ReferentChangeProbe(768).eval()
    tokens = torch.randn(4, 49, 768)
    language_a = torch.randn(4, 512)
    language_b = torch.randn(4, 512)
    coordinates = torch.from_numpy(fixed_patch_coordinates())
    with torch.no_grad():
        ab = model(tokens, language_a, language_b, coordinates)
        ba = model(tokens, language_b, language_a, coordinates)
    assert torch.allclose(ab, ba, atol=1e-6)
