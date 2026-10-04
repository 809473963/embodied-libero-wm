from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.probes.clip_internal_layer import (
    InternalLayerPairReader,
    ReferentChangeProbe,
    coordinates_for_token_count,
    parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_stage_e_protocol_is_bounded_frozen_and_calibration_only() -> None:
    protocol = json.loads((ROOT / "configs/alternate_encoder_protocol.json").read_text())
    contracts = json.loads((ROOT / "configs/alternate_encoder_contracts.json").read_text())
    status = "PREREGISTERED_BEFORE_STAGE_E_WEIGHT_DOWNLOAD_FEATURE_EXTRACTION_OR_TRAINING"
    assert protocol["status"] == status
    assert contracts["status"] == status
    assert protocol["authorization_basis"]["stage_i_authorization"] == (
        "GO_ALTERNATE_FROZEN_VISUAL_ENCODER_AUDIT"
    )
    assert protocol["partitions"] == {
        "fit_demo_ids": [31, 32, 33],
        "selection_demo_ids": [34],
        "stage_e_not_opened_demo_ids": [36, 37, 38, 39],
        "historical_disabled": [30, 35],
        "forbidden_including_metadata": list(range(40, 50)),
    }
    assert protocol["visual"]["families"] == ["DINOv2-S14", "SigLIP-B16-224"]
    assert protocol["visual"]["third_family_disabled"] is True
    assert set(contracts["families"]) == {"DINOv2-S14", "SigLIP-B16-224"}
    assert contracts["freeze"]["weights_trainable"] is False
    assert protocol["frozen_teacher"]["immutable"] is True
    assert protocol["reader"]["parameter_max"] == 150_000
    assert protocol["material_gate"] == {
        "teacher_change_increment_min_points": 5.0,
        "and_at_least_one": {
            "cross_demo_difference_increment_min_points": 7.0,
            "cross_demo_match_mean_abs_improvement_min": 0.03,
        },
        "ordinary_accuracy_max_regression_points": 2.0,
        "all_required": True,
    }
    for row in protocol["frozen_files"]:
        assert sha256(ROOT / row["path"]) == row["sha256"]


def test_stage_e_exact_encoder_contracts_and_preprocessing() -> None:
    contracts = json.loads((ROOT / "configs/alternate_encoder_contracts.json").read_text())
    dino = contracts["families"]["DINOv2-S14"]
    siglip = contracts["families"]["SigLIP-B16-224"]
    assert dino["repository"] == "facebook/dinov2-small"
    assert dino["revision"] == "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    assert dino["spatial_shape"] == [16, 16]
    assert (dino["token_count"], dino["dimension"]) == (256, 384)
    assert "center crop 224x224" in dino["preprocessing"]
    assert siglip["repository"] == "google/siglip-base-patch16-224"
    assert siglip["revision"] == "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
    assert siglip["spatial_shape"] == [14, 14]
    assert (siglip["token_count"], siglip["dimension"]) == (196, 768)
    assert "no crop" in siglip["preprocessing"]
    for family in (dino, siglip):
        assert family["license"] == "Apache-2.0"
        assert len(family["config_sha256"]) == 64
        assert len(family["preprocessor_sha256"]) == 64


def test_stage_e_reader_generalizes_to_native_grids_with_equal_capacity() -> None:
    for token_count, token_dim in ((256, 384), (196, 768)):
        coordinates = torch.from_numpy(coordinates_for_token_count(token_count))
        tokens_a = torch.randn(2, token_count, token_dim)
        tokens_b = torch.randn(2, token_count, token_dim)
        language = torch.randn(2, 512)
        model = InternalLayerPairReader(token_dim, "P1").eval()
        with torch.no_grad():
            forward = model(tokens_a, tokens_b, language, coordinates)
            reverse = model(tokens_b, tokens_a, language, coordinates)
            state, attention = model.encoder(tokens_a, language, coordinates, True)
        assert coordinates.shape == (token_count, 2)
        assert np.isclose(float(coordinates[:, 0].min()), -1 + 1 / np.sqrt(token_count))
        assert parameter_count(model) <= 150_000
        assert state.shape == (2, 51)
        assert attention is not None and attention.shape == (2, token_count)
        assert torch.allclose(attention.sum(dim=1), torch.ones(2), atol=1e-6)
        assert torch.allclose(forward, -reverse, atol=1e-6)
        probe = ReferentChangeProbe(token_dim).eval()
        assert parameter_count(probe) <= 150_000


def test_stage_e_decision_tree_stops_on_failure_and_never_runs_bridge() -> None:
    protocol = json.loads((ROOT / "configs/alternate_encoder_protocol.json").read_text())
    mapping = protocol["decision_mapping"]
    assert mapping["pass_labels"] == ["ALTERNATE_FROZEN_VISUAL_ENCODER_VALID"]
    assert mapping["pass_authorization"] == "GO_SEMANTIC_VALUE_REDISTILLATION"
    assert mapping["fail_labels"] == [
        "FROZEN_VISUAL_BACKBONE_AUDIT_BLOCKED",
        "NATURAL_VISUAL_SEMANTIC_INFORMATION_UNRESOLVED",
    ]
    assert mapping["fail_authorizations"] == [
        "BLOCK_SEMANTIC_VALUE_REDISTILLATION",
        "BLOCK_REAL_CANDIDATE_VALUE_BRIDGE",
    ]
    assert "run real candidates, World Model, reranker, controller, or task success" in (
        protocol["forbidden"]
    )
