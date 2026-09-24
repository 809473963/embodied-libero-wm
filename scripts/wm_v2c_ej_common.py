#!/usr/bin/env python3
"""Shared provenance and initialization helpers for WM-v2C-EJ."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]

from src.world_models import LatentDynamicsEJCoupledModel
from src.world_models.normalization import file_sha256


FROZEN_FILE_PAIRS = (
    ("wm_v2b_1_result", "wm_v2b_1_result_sha256"),
    ("wm_v2b_1_report", "wm_v2b_1_report_sha256"),
    ("c0_v2_checkpoint", "c0_v2_checkpoint_sha256"),
    ("candidate_protocol", "candidate_protocol_sha256"),
    ("candidate_result", "candidate_result_sha256"),
    ("candidate_validation_sidecar", "candidate_validation_sidecar_sha256"),
    ("candidate_generator_code", "candidate_generator_code_sha256"),
    ("candidate_policy_code", "candidate_policy_code_sha256"),
    ("action_conditioning_protocol", "action_conditioning_protocol_sha256"),
    ("reranker_negative_result", "reranker_negative_result_sha256"),
    ("d13_data_config", "d13_data_config_sha256"),
    ("d13_data_contract", "d13_data_contract_sha256"),
    ("d13_store", "d13_store_sha256"),
    ("d14_seed11_checkpoint", "d14_seed11_checkpoint_sha256"),
    ("d15_manifest", "d15_manifest_sha256"),
    ("wm_v2a_checkpoint", "wm_v2a_checkpoint_sha256"),
    ("wm_v2a_result", "wm_v2a_result_sha256"),
    ("wm_v2b_checkpoint", "wm_v2b_checkpoint_sha256"),
    ("wm_v2b_result", "wm_v2b_result_sha256"),
    ("matched_counterfactual_result", "matched_counterfactual_result_sha256"),
    ("matched_counterfactual_sidecar", "matched_counterfactual_sidecar_sha256"),
)

FROZEN_TAG_PAIRS = (
    ("wm_v2b_1_tag", "wm_v2b_1_commit"),
    ("c0_v2_tag", "c0_v2_commit"),
    ("candidate_diagnosis_tag", "candidate_diagnosis_commit"),
    ("matched_counterfactual_tag", "matched_counterfactual_commit"),
)


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def identity_sha256(records: list[dict[str, int]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_protocol_and_config(
    protocol_path: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text())
    config = yaml.safe_load(config_path.read_text())
    if (
        protocol["status"]
        != "PREREGISTERED_BEFORE_GRADIENT_CALIBRATION_AND_TRAINING"
    ):
        raise ValueError("WM-v2C-EJ protocol is not preregistered")
    if config["run"]["method"] != "wm_v2c_ej_physical_visual_coupling":
        raise ValueError("unexpected WM-v2C-EJ method")
    if file_sha256(config_path) != protocol["experiment"]["config_sha256"]:
        raise ValueError("WM-v2C-EJ config differs from preregistration")
    if int(config["run"]["seed"]) != 11:
        raise ValueError("WM-v2C-EJ first experiment is seed11 only")
    frozen = protocol["frozen_upstream"]
    for path_key, hash_key in FROZEN_FILE_PAIRS:
        path = ROOT / frozen[path_key]
        if file_sha256(path) != frozen[hash_key]:
            raise ValueError(f"frozen upstream changed: {path_key}")
    for tag_key, commit_key in FROZEN_TAG_PAIRS:
        commit = subprocess.check_output(
            ["git", "rev-parse", f"{frozen[tag_key]}^{{}}"], cwd=ROOT, text=True
        ).strip()
        if commit != frozen[commit_key]:
            raise ValueError(f"frozen tag changed: {frozen[tag_key]}")
    if frozen["wm_v2b_1_authorization"] != "BLOCK_WM_V2C_PHYSICAL_MEDIATOR":
        raise ValueError("upstream 12-D mediator block was reinterpreted")
    if protocol["data"]["gripper_indices_excluded"] != [3, 4]:
        raise ValueError("gripper exclusion changed")
    if config["data"]["gripper_indices_excluded_from_coupling"] != [3, 4]:
        raise ValueError("config gripper exclusion changed")
    return protocol, config


def model_from_config(config: dict[str, Any]) -> LatentDynamicsEJCoupledModel:
    values = {key: int(value) for key, value in config["model"].items()}
    adapter_hidden = values.pop("physical_adapter_hidden_dim")
    adapter_bottleneck = values.pop("physical_adapter_bottleneck_dim")
    proprio_head = values.pop("proprio_head_hidden_dim")
    return LatentDynamicsEJCoupledModel(
        physical_adapter_hidden_dim=adapter_hidden,
        physical_adapter_bottleneck_dim=adapter_bottleneck,
        proprio_head_hidden_dim=proprio_head,
        **values,
    )


def load_initialized_model(
    config: dict[str, Any], protocol: dict[str, Any]
) -> tuple[LatentDynamicsEJCoupledModel, dict[str, Any], dict[str, Any]]:
    checkpoint_path = ROOT / config["initialization"]["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["method"] != "wm_v2b_future_proprio_auxiliary":
        raise ValueError("initialization checkpoint is not WM-v2B")
    if file_sha256(checkpoint_path) != protocol["frozen_upstream"][
        "wm_v2b_checkpoint_sha256"
    ]:
        raise ValueError("WM-v2B initialization checkpoint changed")
    model = model_from_config(config)
    incompatibility = model.load_state_dict(checkpoint["model_state"], strict=False)
    expected_missing = {
        name
        for name in model.state_dict()
        if name.startswith("physical_visual_adapter.")
    }
    if set(incompatibility.missing_keys) != expected_missing:
        raise ValueError(
            f"unexpected missing initialization keys: {incompatibility.missing_keys}"
        )
    if incompatibility.unexpected_keys:
        raise ValueError(
            f"unexpected initialization keys: {incompatibility.unexpected_keys}"
        )
    for name, value in checkpoint["model_state"].items():
        if not torch.equal(model.state_dict()[name], value):
            raise ValueError(f"WM-v2B weight did not load exactly: {name}")
    final = model.physical_visual_adapter[-1]
    if not isinstance(final, torch.nn.Linear):
        raise TypeError("adapter final layer changed")
    if torch.count_nonzero(final.weight) or torch.count_nonzero(final.bias):
        raise ValueError("adapter final layer is not exact zero initialization")
    expected_parameters = int(protocol["architecture"]["expected_total_parameters"])
    if sum(parameter.numel() for parameter in model.parameters()) != expected_parameters:
        raise ValueError("WM-v2C-EJ parameter count changed")
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError("WM-v2C-EJ initialization contains NaN/Inf")
    return model, checkpoint["normalization"], checkpoint


def assert_initial_visual_parity(components: dict[str, torch.Tensor]) -> float:
    residual = components["physical_visual_residual"]
    if torch.count_nonzero(residual):
        raise ValueError("zero-initialized adapter changed visual prediction")
    return float(residual.abs().max())


def finite_numpy_tree(values: dict[str, np.ndarray]) -> bool:
    return all(np.isfinite(value).all() for value in values.values())
