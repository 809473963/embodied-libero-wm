#!/usr/bin/env python3
"""WM-v2B data-index, architecture, loss, and gradient tests."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_world_model_v2b import model_from_config
from src.world_models import (
    FutureProprioWorldModelDataset,
    LatentDynamicsModel,
    apply_world_model_normalization,
    cosine_distance,
    proprio_delta_error,
    standardized_proprio_delta_target,
)


STORE = ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5"
D14_CHECKPOINT = ROOT / "results/world_model/d14_single_seed11/best.pt"


def data_test(normalization: dict) -> None:
    train = FutureProprioWorldModelDataset(STORE, "train", 8)
    validation = FutureProprioWorldModelDataset(STORE, "validation", 8)
    assert len(train) == 52381
    assert len(validation) == 5869
    try:
        probes = (
            (train, 0),
            (train, len(train) // 2),
            (validation, 0),
            (validation, len(validation) // 2),
            (validation, len(validation) - 1),
        )
        with h5py.File(STORE, "r") as store:
            for dataset, index in probes:
                sample = dataset[index]
                record = dataset.record_at(index)
                group = store[
                    f"tasks/task_{record.task_id}/demos/demo_{record.demo_id}"
                ]
                k = record.start_timestep
                assert sample["proprio_future"].shape == (8, 12)
                assert np.array_equal(
                    sample["proprio_future"].numpy(), group["proprio"][k + 1 : k + 9]
                )
                assert np.array_equal(sample["proprio0"].numpy(), group["proprio"][k])
                assert np.array_equal(sample["actions"].numpy(), group["actions"][k + 1 : k + 9])
        batch = next(iter(DataLoader(validation, batch_size=4, shuffle=False)))
        target = standardized_proprio_delta_target(
            batch["proprio0"], batch["proprio_future"], normalization
        )
        mean = torch.tensor(normalization["proprio"]["mean"]).view(1, 1, 12)
        std = torch.tensor(normalization["proprio"]["std"]).view(1, 1, 12)
        independent = (batch["proprio_future"] - mean) / std - (
            batch["proprio0"].unsqueeze(1) - mean
        ) / std
        assert target.shape == (4, 8, 12)
        assert torch.allclose(target, independent)
    finally:
        train.close()
        validation.close()


def model_test(config: dict, normalization: dict) -> None:
    torch.manual_seed(11)
    base = LatentDynamicsModel(**{
        key: int(value)
        for key, value in config["model"].items()
        if key != "proprio_head_hidden_dim"
    })
    torch.manual_seed(11)
    model = model_from_config(config)
    assert sum(parameter.numel() for parameter in base.parameters()) == 792000
    assert sum(parameter.numel() for parameter in model.parameters()) == 826444
    base_state = base.state_dict()
    aux_state = model.state_dict()
    for name, value in base_state.items():
        assert torch.equal(value, aux_state[name]), name
    assert list(inspect.signature(model.forward).parameters) == [
        "z0",
        "proprio0",
        "language",
        "actions",
    ]

    dataset = FutureProprioWorldModelDataset(STORE, "train", 8)
    try:
        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    finally:
        dataset.close()
    p0, actions = apply_world_model_normalization(
        batch["proprio0"], batch["actions"], normalization
    )
    target_p = standardized_proprio_delta_target(
        batch["proprio0"], batch["proprio_future"], normalization
    )
    base_z = base(batch["z0"], p0, batch["language"], actions)
    pred_z, pred_p = model(batch["z0"], p0, batch["language"], actions)
    assert pred_z.shape == (2, 8, 512)
    assert pred_p.shape == target_p.shape == (2, 8, 12)
    assert torch.equal(base_z, pred_z)
    assert torch.allclose(torch.linalg.vector_norm(pred_z, dim=-1), torch.ones(2, 8), atol=1e-5)

    visual = cosine_distance(pred_z, batch["z_future"]).mean()
    proprio = proprio_delta_error(pred_p, target_p).mean()
    assert torch.isfinite(visual) and torch.isfinite(proprio)
    optimizer = AdamW(model.parameters(), lr=3e-4)
    (visual + proprio).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(value is not None for value in gradients)
    assert all(torch.isfinite(value).all() for value in gradients if value is not None)
    optimizer.step()

    # Independence proof: fixed visual prediction/target leaves L_z unchanged,
    # while changing a real proprio target changes L_p.
    fixed_visual = cosine_distance(pred_z.detach(), batch["z_future"]).mean()
    p_loss_a = proprio_delta_error(pred_p.detach(), target_p).mean()
    p_loss_b = proprio_delta_error(pred_p.detach(), target_p + 0.5).mean()
    assert torch.equal(
        fixed_visual, cosine_distance(pred_z.detach(), batch["z_future"]).mean()
    )
    assert not torch.isclose(p_loss_a, p_loss_b)


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/world_model_v2b_seed11.yaml").read_text())
    checkpoint = torch.load(D14_CHECKPOINT, map_location="cpu", weights_only=False)
    normalization = checkpoint["normalization"]
    assert normalization["proprio"]["count"] == 55981
    assert normalization["action"]["count"] == 55531
    data_test(normalization)
    model_test(config, normalization)
    print(
        {
            "status": "PASS",
            "train_samples": 52381,
            "validation_samples": 5869,
            "parameters": 826444,
            "future_proprio_is_input": False,
        }
    )


if __name__ == "__main__":
    main()
