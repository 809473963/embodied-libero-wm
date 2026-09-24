#!/usr/bin/env python3
"""WM-v2C-EJ architecture, initialization, target, and gradient tests."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.wm_v2c_ej_common import load_initialized_model
from src.world_models import (
    EJ_PROPRIO_INDICES,
    FutureProprioWorldModelDataset,
    apply_world_model_normalization,
    cosine_distance,
    ej_delta_error,
    physical_visual_residual_loss,
    standardized_proprio_delta_target,
)


PROTOCOL = ROOT / "configs/wm_v2c_ej_protocol.json"
CONFIG = ROOT / "configs/world_model_v2c_ej_seed11.yaml"
STORE = ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5"


def load() -> tuple[dict, dict, torch.nn.Module, dict]:
    protocol = json.loads(PROTOCOL.read_text())
    config = yaml.safe_load(CONFIG.read_text())
    model, normalization, _ = load_initialized_model(config, protocol)
    return protocol, config, model, normalization


def test_parameter_count_zero_initialization_and_exact_v2b_parity() -> None:
    protocol, _, model, _ = load()
    assert sum(parameter.numel() for parameter in model.parameters()) == 901516
    assert sum(parameter.numel() for parameter in model.adapter_parameters()) == 75072
    assert protocol["architecture"]["expected_total_parameters"] == 901516
    final = model.physical_visual_adapter[-1]
    assert torch.count_nonzero(final.weight) == 0
    assert torch.count_nonzero(final.bias) == 0

    batch = 2
    z0 = torch.nn.functional.normalize(torch.randn(batch, 512), dim=-1)
    proprio = torch.randn(batch, 12)
    language = torch.nn.functional.normalize(torch.randn(batch, 512), dim=-1)
    actions = torch.randn(batch, 8, 7)
    components = model.forward_with_components(z0, proprio, language, actions)
    legacy_prediction = torch.nn.functional.normalize(
        z0[:, None] + components["base_visual_delta"], dim=-1
    )
    assert torch.equal(components["physical_visual_residual"], torch.zeros(batch, 8, 512))
    assert torch.equal(components["pred_z"], legacy_prediction)
    assert torch.allclose(
        torch.linalg.vector_norm(components["pred_z"], dim=-1),
        torch.ones(batch, 8),
        atol=1e-6,
    )


def test_gripper_is_structurally_excluded_and_ablation_modes_are_defined() -> None:
    _, _, model, _ = load()
    full = torch.randn(2, 8, 12)
    changed = full.clone()
    changed[..., 3:5] += 10000.0
    assert EJ_PROPRIO_INDICES == (0, 1, 2, 5, 6, 7, 8, 9, 10, 11)
    assert torch.equal(model.select_ej(full), model.select_ej(changed))
    selected = model.select_ej(full)
    assert selected.shape == (2, 8, 10)
    assert torch.count_nonzero(model.ablate_adapter_input(selected, "ee_only")[..., 3:]) == 0
    assert torch.count_nonzero(model.ablate_adapter_input(selected, "joints_only")[..., :3]) == 0
    assert torch.equal(model.ablate_adapter_input(selected, "full_ej"), selected)


def test_real_d13_batch_shapes_losses_backward_and_optimizer_step() -> None:
    _, _, model, normalization = load()
    dataset = FutureProprioWorldModelDataset(STORE, "train", 8)
    try:
        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    finally:
        dataset.close()
    normalized_proprio, normalized_actions = apply_world_model_normalization(
        batch["proprio0"], batch["actions"], normalization
    )
    target = standardized_proprio_delta_target(
        batch["proprio0"], batch["proprio_future"], normalization
    )
    components = model.forward_with_components(
        batch["z0"], normalized_proprio, batch["language"], normalized_actions
    )
    assert components["pred_z"].shape == (2, 8, 512)
    assert components["pred_proprio_delta"].shape == target.shape == (2, 8, 12)
    assert components["pred_ej_delta"].shape == (2, 8, 10)
    assert components["physical_visual_residual"].shape == (2, 8, 512)
    visual = cosine_distance(components["pred_z"], batch["z_future"]).mean()
    ej = ej_delta_error(components["pred_proprio_delta"], target).mean()
    residual = physical_visual_residual_loss(
        components["physical_visual_residual"]
    ).mean()
    loss = visual + 0.5 * ej + 0.001 * residual
    assert torch.isfinite(loss)
    optimizer = AdamW(model.parameters(), lr=3e-4)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.physical_visual_adapter[-1].weight.grad is not None
    assert torch.count_nonzero(model.physical_visual_adapter[-1].weight.grad) > 0
    optimizer.step()


def test_deployment_signature_has_no_future_target() -> None:
    _, _, model, _ = load()
    assert list(inspect.signature(model.forward).parameters) == [
        "z0",
        "proprio0",
        "language",
        "actions",
    ]

