#!/usr/bin/env python3
"""Preregistration and unit tests for the physical-to-visual sufficiency probe."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/counterfactual_physical_visual_probe_protocol.json"
FREEZE_PATH = ROOT / "configs/counterfactual_physical_visual_probe_freeze.json"

from src.probes import (
    ContextConditionedPhysicalVisualProbe,
    fit_ridge,
    predict_ridge,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def test_upstream_freeze_tags_hashes_and_tree() -> None:
    protocol = load_protocol()
    assert protocol["status"] == "PREREGISTERED_BEFORE_PROBE_CALIBRATION"
    frozen = protocol["frozen_upstream"]
    for tag, commit in frozen["annotated_tags"].items():
        assert subprocess.check_output(
            ["git", "cat-file", "-t", tag], cwd=ROOT, text=True
        ).strip() == "tag"
        assert subprocess.check_output(
            ["git", "rev-parse", f"{tag}^{{}}"], cwd=ROOT, text=True
        ).strip() == commit
    for record in frozen["files"]:
        assert sha256(ROOT / record["path"]) == record["sha256"]

    paths = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            frozen["reports_results_tree"]["tag"],
            "--",
            "docs",
            "results",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    paths = [
        path
        for path in paths
        if Path(path).suffix in {".md", ".json", ".csv", ".yaml", ".yml"}
    ]
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256(ROOT / relative).encode())
        digest.update(b"\n")
    assert len(paths) == frozen["reports_results_tree"]["file_count"]
    assert digest.hexdigest() == frozen["reports_results_tree"]["sha256"]


def test_split_model_metrics_and_forbidden_contract_are_preregistered() -> None:
    protocol = load_protocol()
    partitions = protocol["partitions"]
    assert partitions["calibration_init_states"] == [31, 32, 33, 34]
    assert partitions["evaluation_init_states"] == [36, 37, 38, 39]
    assert partitions["historical_reference_only"] == [30, 35]
    assert set(range(40, 50)).issubset(partitions["forbidden_init_states"])
    assert partitions["samples_per_partition"] == 40 * 3 * 8
    assert partitions["simulator_rerun"] is False

    contract = protocol["sample_contract"]
    assert contract["gripper_indices_excluded"] == [3, 4]
    assert contract["input_dimensions"] == {
        "physical": 10,
        "context": 256,
        "visual_target": 512,
    }
    assert protocol["probes"]["ridge_grid"] == [0.0001, 0.001, 0.01, 0.1, 1.0]
    assert protocol["probes"]["p3"]["hidden_dim"] == 64
    assert protocol["probes"]["p3"]["context_output_shortcut"] is False
    assert protocol["bootstrap"]["unit"].startswith("task-stratified held-out branch")
    assert protocol["gates"]["minimum_incremental_improvement"] == 0.05
    forbidden = " ".join(protocol["forbidden"])
    for phrase in (
        "init 40-49",
        "gripper",
        "future RGB",
        "World Model",
        "controller",
        "seed23/47",
    ):
        assert phrase in forbidden


def test_recorded_implementation_hashes() -> None:
    protocol = load_protocol()
    implementation = protocol["implementation"]
    assert sha256(ROOT / implementation["script"]) == implementation["script_sha256"]
    assert sha256(ROOT / implementation["module"]) == implementation["module_sha256"]


def test_p3_shapes_finite_gradients_and_exact_antisymmetry() -> None:
    torch.manual_seed(17)
    model = ContextConditionedPhysicalVisualProbe()
    context = torch.randn(7, 256)
    physical = torch.randn(7, 10)
    prediction = model(context, physical)
    reverse = model(context, -physical)
    assert prediction.shape == (7, 512)
    assert torch.isfinite(prediction).all()
    assert torch.allclose(prediction, -reverse, atol=1.0e-7, rtol=0)
    loss = prediction.square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 49856


def test_ridge_no_bias_preserves_antisymmetry_and_bias_variant_fits_intercept() -> None:
    rng = np.random.default_rng(19)
    x = rng.normal(size=(200, 10))
    weight = rng.normal(size=(10, 12))
    y = x @ weight
    no_bias = fit_ridge(x, y, 1.0e-4, bias=False)
    prediction = predict_ridge(x[:20], no_bias)
    reverse = predict_ridge(-x[:20], no_bias)
    assert np.allclose(prediction, -reverse, atol=1.0e-12, rtol=0)

    context = rng.normal(size=(200, 16))
    target = context @ rng.normal(size=(16, 5)) + 3.0
    with_bias = fit_ridge(context, target, 1.0e-4, bias=True)
    assert np.isfinite(predict_ridge(context, with_bias)).all()
    assert np.linalg.norm(with_bias.bias) > 0


def test_calibration_freeze_precedes_held_out_evaluation() -> None:
    protocol = load_protocol()
    freeze = json.loads(FREEZE_PATH.read_text())
    assert freeze["status"] == "FROZEN_BEFORE_HELD_OUT_EVALUATION"
    assert freeze["protocol_sha256"] == sha256(PROTOCOL_PATH)
    assert freeze["calibration_execution_source_commit"] == subprocess.check_output(
        ["git", "rev-parse", "b59cef7^{}"], cwd=ROOT, text=True
    ).strip()
    assert freeze["partition"]["init_state_indices"] == [31, 32, 33, 34]
    assert freeze["partition"]["sample_count"] == 960
    assert freeze["source_sidecar_sha256"] == protocol["partitions"][
        "calibration_sha256"
    ]
    assert freeze["context"]["candidate_copy_max_abs_difference"] == 0.0
    assert freeze["context"]["actions_or_futures_used"] is False
    assert freeze["evaluation_sidecar_read"] is False
    assert not any(freeze["forbidden_activity_audit"].values())

    grid = protocol["probes"]["ridge_grid"]
    assert freeze["ridge"]["grid"] == grid
    for name in ("p1", "p2"):
        records = freeze["ridge"][f"{name}_cv"]
        assert [row["alpha"] for row in records] == grid
        assert all(len(row["folds"]) == 4 for row in records)
        selected = min(records, key=lambda row: (row["mean_fold_rmse"], -row["alpha"]))
        assert freeze["ridge"][f"{name}_selected_alpha"] == selected["alpha"]
    assert freeze["p3"]["parameters"] == 49856
    assert len(freeze["p3"]["folds"]) == 4
    best_epochs = [row["best_epoch"] for row in freeze["p3"]["folds"]]
    assert freeze["p3"]["final_epoch"] == int(np.floor(np.median(best_epochs) + 0.5))

    weight_path = ROOT / freeze["weights"]["path"]
    assert sha256(weight_path) == freeze["weights"]["sha256"]
    with np.load(weight_path, allow_pickle=False) as weights:
        assert weights["p1_weight"].shape == (10, 512)
        assert weights["p1_bias"].shape == (512,)
        assert np.count_nonzero(weights["p1_bias"]) == 0
        assert weights["p2_weight"].shape == (256, 512)
        assert weights["p2_bias"].shape == (512,)
        assert weights["p3__physical__weight"].shape == (64, 10)
        assert weights["p3__context_gate__weight"].shape == (64, 256)
        assert weights["p3__visual__weight"].shape == (512, 64)
        assert all(np.isfinite(weights[name]).all() for name in weights.files)
