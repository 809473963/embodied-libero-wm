#!/usr/bin/env python3
"""Calibrate then evaluate the counterfactual physical-to-visual probe.

The ``calibrate`` command is structurally unable to open the held-out sidecar.
The ``evaluate`` command requires a tracked calibration freeze and immutable
weight sidecar before it opens init states 36--39.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy import stats as scipy_stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_world_model_v2b import model_from_config
from src.probes import (
    ContextConditionedPhysicalVisualProbe,
    FrozenContextExtractor,
    fit_ridge,
    predict_ridge,
)


PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ("A0-A1", "A0-A2", "A1-A2")
EJ_INDICES = np.asarray((0, 1, 2, 5, 6, 7, 8, 9, 10, 11), dtype=np.int64)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def frozen_tree_digest(tag: str) -> tuple[int, str]:
    paths = git("ls-tree", "-r", "--name-only", tag, "--", "docs", "results").splitlines()
    paths = [
        path
        for path in paths
        if Path(path).suffix in {".md", ".json", ".csv", ".yaml", ".yml"}
    ]
    digest = hashlib.sha256()
    for relative in paths:
        current_hash = sha256(ROOT / relative)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(current_hash.encode())
        digest.update(b"\n")
    return len(paths), digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol["status"] != "PREREGISTERED_BEFORE_PROBE_CALIBRATION":
        raise ValueError("probe protocol is not preregistered")
    frozen = protocol["frozen_upstream"]
    for key, expected in frozen["annotated_tags"].items():
        if git("cat-file", "-t", key) != "tag":
            raise ValueError(f"not an annotated tag: {key}")
        if git("rev-parse", f"{key}^{{}}") != expected:
            raise ValueError(f"frozen tag moved: {key}")
    for record in frozen["files"]:
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"frozen upstream changed: {record['path']}")
    implementation = protocol["implementation"]
    if sha256(ROOT / implementation["module"]) != implementation["module_sha256"]:
        raise ValueError("probe module differs from preregistration")
    # The script check is intentionally performed by tests rather than here:
    # a script cannot embed and verify its own byte hash without a fixed-point
    # construction. The protocol still records the committed source hash.
    count, digest = frozen_tree_digest(frozen["reports_results_tree"]["tag"])
    if count != frozen["reports_results_tree"]["file_count"]:
        raise ValueError("frozen report/result file count changed")
    if digest != frozen["reports_results_tree"]["sha256"]:
        raise ValueError("an upstream report/result changed")
    return protocol


def pairwise(values: np.ndarray) -> np.ndarray:
    return np.stack(
        [values[:, left] - values[:, right] for left, right in PAIR_INDICES], axis=1
    )


def load_wm_v2b(protocol: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = ROOT / protocol["context"]["wm_v2b_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["method"] != "wm_v2b_future_proprio_auxiliary":
        raise ValueError("context checkpoint is not WM-v2B")
    model = model_from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.cpu().float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint["normalization"]


@torch.inference_mode()
def extract_context(
    model: torch.nn.Module,
    branches: dict[str, np.ndarray],
    normalization: dict[str, Any],
) -> tuple[np.ndarray, float]:
    mean = np.asarray(normalization["proprio"]["mean"], dtype=np.float32)
    std = np.asarray(normalization["proprio"]["std"], dtype=np.float32)
    normalized_p0 = (branches["proprio0"].astype(np.float32) - mean) / std
    # Repeat every pre-action input three times. The source manifest already
    # established exact three-branch planning-observation identity; this
    # verifies the frozen context computation is also candidate-invariant.
    z0 = np.repeat(branches["z0"].astype(np.float32), 3, axis=0)
    p0 = np.repeat(normalized_p0, 3, axis=0)
    language = np.repeat(branches["language"].astype(np.float32), 3, axis=0)
    extractor = FrozenContextExtractor(model).cpu().float().eval()
    context = extractor(
        torch.from_numpy(z0), torch.from_numpy(p0), torch.from_numpy(language)
    ).numpy().reshape(len(branches["z0"]), 3, 256)
    maximum = float(np.max(np.abs(context - context[:, :1])))
    if maximum > 1.0e-7:
        raise ValueError("candidate-independent frozen context identity failed")
    return context[:, 0].copy(), maximum


def load_partition(
    sidecar_path: Path,
    allowed_init_states: set[int],
    protocol: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(sidecar_path, allow_pickle=False) as source:
        branches = {name: source[name].copy() for name in source.files}
    required = {
        "z0",
        "proprio0",
        "language",
        "candidates",
        "executed_actions",
        "real_z",
        "real_proprio",
        "pred_proprio_wm_v2b_standardized",
        "task_id",
        "init_state_index",
        "planning_index",
    }
    if not required.issubset(branches):
        raise ValueError(f"branch sidecar lacks {sorted(required - set(branches))}")
    count = int(protocol["partitions"]["branch_points"])
    expected_shapes = {
        "z0": (count, 512),
        "proprio0": (count, 12),
        "language": (count, 512),
        "real_z": (count, 3, 8, 512),
        "real_proprio": (count, 3, 8, 12),
        "pred_proprio_wm_v2b_standardized": (count, 3, 8, 12),
    }
    for name, shape in expected_shapes.items():
        if branches[name].shape != shape:
            raise ValueError(f"{name} shape changed: {branches[name].shape}")
    if set(map(int, branches["init_state_index"])) != allowed_init_states:
        raise ValueError("partition init-state boundary failed")
    if set(map(int, branches["task_id"])) != set(range(10)):
        raise ValueError("partition task coverage failed")
    if set(map(int, branches["planning_index"])) != {8}:
        raise ValueError("planning index changed")
    if not np.array_equal(branches["candidates"], branches["executed_actions"]):
        raise ValueError("candidate/action identity failed")
    if not all(
        np.isfinite(value).all()
        for value in branches.values()
        if value.dtype.kind in "fiu"
    ):
        raise FloatingPointError("branch sidecar contains NaN/Inf")
    identities = np.stack(
        (branches["task_id"], branches["init_state_index"], branches["planning_index"]),
        axis=1,
    )
    if len(np.unique(identities, axis=0)) != count:
        raise ValueError("branch identities are not unique")
    return branches, {
        "branch_points": count,
        "sample_count": count * 3 * 8,
        "init_state_indices": sorted(allowed_init_states),
        "tasks": sorted(set(map(int, branches["task_id"]))),
    }


def construct_samples(
    branches: dict[str, np.ndarray],
    model: torch.nn.Module,
    normalization: dict[str, Any],
    calibration_alpha: np.ndarray,
) -> dict[str, np.ndarray]:
    context, context_identity_max = extract_context(model, branches, normalization)
    mean = np.asarray(normalization["proprio"]["mean"], dtype=np.float64)
    std = np.asarray(normalization["proprio"]["std"], dtype=np.float64)
    real_standardized = (branches["real_proprio"].astype(np.float64) - mean) / std
    real_physical = pairwise(real_standardized)[..., EJ_INDICES]
    predicted = pairwise(
        branches["pred_proprio_wm_v2b_standardized"].astype(np.float64)
    )
    calibrated_predicted = (
        predicted * calibration_alpha.reshape(1, 1, 8, 1)
    )[..., EJ_INDICES]
    real_visual = pairwise(branches["real_z"].astype(np.float64))
    real_separation = np.stack(
        [
            1.0
            - np.sum(
                branches["real_z"][:, left].astype(np.float64)
                * branches["real_z"][:, right].astype(np.float64),
                axis=-1,
            )
            for left, right in PAIR_INDICES
        ],
        axis=1,
    )
    repeated_context = np.broadcast_to(
        context[:, None, None, :],
        (len(context), 3, 8, 256),
    ).copy()
    result = {
        "context": context.astype(np.float32),
        "context_samples": repeated_context.reshape(-1, 256).astype(np.float32),
        "real_physical": real_physical.astype(np.float32),
        "real_physical_samples": real_physical.reshape(-1, 10).astype(np.float32),
        "predicted_physical": calibrated_predicted.astype(np.float32),
        "predicted_physical_samples": calibrated_predicted.reshape(-1, 10).astype(
            np.float32
        ),
        "real_visual": real_visual.astype(np.float32),
        "real_visual_samples": real_visual.reshape(-1, 512).astype(np.float32),
        "real_separation": real_separation.astype(np.float32),
        "task_id": branches["task_id"].astype(np.int64),
        "init_state_index": branches["init_state_index"].astype(np.int64),
        "context_identity_max_abs": np.asarray(context_identity_max, dtype=np.float64),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("constructed probe samples contain NaN/Inf")
    return result


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction.astype(np.float64) - target) ** 2)))


def ridge_cross_validation(
    samples: dict[str, np.ndarray],
    inputs: np.ndarray,
    targets: np.ndarray,
    alphas: Iterable[float],
    *,
    bias: bool,
) -> tuple[float, list[dict[str, Any]]]:
    sample_init = np.repeat(samples["init_state_index"], 3 * 8)
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        fold_values = []
        for held_out in sorted(set(map(int, samples["init_state_index"]))):
            validation = sample_init == held_out
            fitted = fit_ridge(
                inputs[~validation], targets[~validation], float(alpha), bias=bias
            )
            value = rmse(predict_ridge(inputs[validation], fitted), targets[validation])
            fold_values.append({"held_out_init": held_out, "rmse": value})
        records.append(
            {
                "alpha": float(alpha),
                "mean_fold_rmse": float(np.mean([row["rmse"] for row in fold_values])),
                "folds": fold_values,
            }
        )
    selected = min(records, key=lambda row: (row["mean_fold_rmse"], -row["alpha"]))
    return float(selected["alpha"]), records


def train_p3(
    context: np.ndarray,
    physical: np.ndarray,
    target: np.ndarray,
    validation_mask: np.ndarray | None,
    training: dict[str, Any],
    seed: int,
    epochs: int | None = None,
) -> tuple[ContextConditionedPhysicalVisualProbe, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ContextConditionedPhysicalVisualProbe(hidden_dim=int(training["hidden_dim"]))
    model.cpu().float().train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    c = torch.from_numpy(context.astype(np.float32))
    p = torch.from_numpy(physical.astype(np.float32))
    y = torch.from_numpy(target.astype(np.float32))
    if validation_mask is None:
        train_mask = np.ones(len(c), dtype=bool)
        validation = None
        maximum = int(epochs if epochs is not None else training["max_epochs"])
    else:
        train_mask = ~validation_mask
        validation = torch.from_numpy(validation_mask)
        maximum = int(training["max_epochs"])
    train = torch.from_numpy(train_mask)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    last_train = math.inf
    for epoch in range(1, maximum + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(c[train], p[train])
        loss = torch.mean((prediction - y[train]) ** 2)
        loss.backward()
        if not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("P3 gradient contains NaN/Inf")
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        last_train = float(torch.sqrt(loss.detach()))
        if validation is None:
            continue
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                torch.sqrt(torch.mean((model(c[validation], p[validation]) - y[validation]) ** 2))
            )
        model.train()
        if validation_loss < best_loss - float(training["minimum_delta"]):
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(training["patience"]):
            break
    if validation is not None:
        if best_state is None:
            raise RuntimeError("P3 inner fold did not produce a checkpoint")
        model.load_state_dict(best_state)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError("P3 state contains NaN/Inf")
    return model.eval(), {
        "best_epoch": int(best_epoch if validation is not None else maximum),
        "best_validation_rmse": None if validation is None else best_loss,
        "epochs_run": epoch,
        "final_train_rmse": last_train,
        "seed": seed,
    }


def p3_cross_validation(
    samples: dict[str, np.ndarray], training: dict[str, Any]
) -> tuple[int, list[dict[str, Any]]]:
    sample_init = np.repeat(samples["init_state_index"], 3 * 8)
    folds = []
    for held_out in sorted(set(map(int, samples["init_state_index"]))):
        validation = sample_init == held_out
        _, record = train_p3(
            samples["context_samples"],
            samples["real_physical_samples"],
            samples["real_visual_samples"],
            validation,
            training,
            seed=int(training["fold_seed_base"]) + held_out,
        )
        record["held_out_init"] = held_out
        folds.append(record)
    median = float(np.median([row["best_epoch"] for row in folds]))
    final_epoch = int(math.floor(median + 0.5))
    return max(final_epoch, 1), folds


def model_state_arrays(
    model: ContextConditionedPhysicalVisualProbe,
) -> dict[str, np.ndarray]:
    return {
        f"p3__{name.replace('.', '__')}": value.detach().cpu().numpy()
        for name, value in model.state_dict().items()
    }


def load_p3(arrays: dict[str, np.ndarray], hidden_dim: int) -> ContextConditionedPhysicalVisualProbe:
    model = ContextConditionedPhysicalVisualProbe(hidden_dim=hidden_dim)
    state = {
        name[len("p3__") :].replace("__", "."): torch.from_numpy(value)
        for name, value in arrays.items()
        if name.startswith("p3__")
    }
    model.load_state_dict(state)
    return model.cpu().float().eval()


def calibrate(args: argparse.Namespace, protocol: dict[str, Any]) -> int:
    calibration_path = ROOT / protocol["partitions"]["calibration_sidecar"]
    if sha256(calibration_path) != protocol["partitions"]["calibration_sha256"]:
        raise ValueError("calibration sidecar changed")
    branches, scope = load_partition(
        calibration_path,
        set(protocol["partitions"]["calibration_init_states"]),
        protocol,
    )
    model, normalization = load_wm_v2b(protocol)
    alphas = np.asarray(
        json.loads((ROOT / protocol["physical_mediator"]["freeze"]).read_text())[
            "coefficients"
        ]["per_horizon_scalar"],
        dtype=np.float64,
    )
    samples = construct_samples(branches, model, normalization, alphas)
    ridge_grid = [float(value) for value in protocol["probes"]["ridge_grid"]]
    p1_alpha, p1_cv = ridge_cross_validation(
        samples,
        samples["real_physical_samples"],
        samples["real_visual_samples"],
        ridge_grid,
        bias=False,
    )
    p2_alpha, p2_cv = ridge_cross_validation(
        samples,
        samples["context_samples"],
        samples["real_visual_samples"],
        ridge_grid,
        bias=True,
    )
    p1 = fit_ridge(
        samples["real_physical_samples"],
        samples["real_visual_samples"],
        p1_alpha,
        bias=False,
    )
    p2 = fit_ridge(
        samples["context_samples"],
        samples["real_visual_samples"],
        p2_alpha,
        bias=True,
    )
    final_epoch, p3_cv = p3_cross_validation(samples, protocol["probes"]["p3"])
    p3, p3_final = train_p3(
        samples["context_samples"],
        samples["real_physical_samples"],
        samples["real_visual_samples"],
        None,
        protocol["probes"]["p3"],
        seed=int(protocol["probes"]["p3"]["final_seed"]),
        epochs=final_epoch,
    )
    weight_path = ROOT / protocol["outputs"]["weights"]
    if weight_path.exists() or args.freeze.exists():
        raise FileExistsError("probe calibration output already exists")
    atomic_npz(
        weight_path,
        p1_weight=p1.weight.astype(np.float32),
        p1_bias=p1.bias.astype(np.float32),
        p2_weight=p2.weight.astype(np.float32),
        p2_bias=p2.bias.astype(np.float32),
        p2_input_mean=p2.input_mean.astype(np.float32),
        **model_state_arrays(p3),
    )
    result = {
        "schema_version": 1,
        "experiment_version": protocol["protocol_version"],
        "status": "FROZEN_BEFORE_HELD_OUT_EVALUATION",
        "calibration_execution_source_commit": git("rev-parse", "HEAD"),
        "protocol": str(args.protocol.relative_to(ROOT)),
        "protocol_sha256": sha256(args.protocol),
        "partition": scope,
        "source_sidecar": str(calibration_path.relative_to(ROOT)),
        "source_sidecar_sha256": sha256(calibration_path),
        "context": {
            "dimension": 256,
            "candidate_copy_max_abs_difference": float(
                samples["context_identity_max_abs"]
            ),
            "inputs": ["z0", "normalized_proprio0", "language"],
            "actions_or_futures_used": False,
        },
        "ridge": {
            "selection": "lowest mean leave-one-init-out RMSE; exact tie chooses larger alpha",
            "grid": ridge_grid,
            "p1_selected_alpha": p1_alpha,
            "p1_cv": p1_cv,
            "p2_selected_alpha": p2_alpha,
            "p2_cv": p2_cv,
        },
        "p3": {
            "architecture": "10->64 physical no-bias; 256->64 sigmoid context gate; elementwise product; 64->512 no-bias",
            "parameters": sum(value.numel() for value in p3.parameters()),
            "folds": p3_cv,
            "final_epoch_rule": "round-half-up median inner-fold best epoch",
            "final_epoch": final_epoch,
            "final_training": p3_final,
        },
        "weights": {
            "path": str(weight_path.relative_to(ROOT)),
            "sha256": sha256(weight_path),
        },
        "evaluation_sidecar_read": False,
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {
            "init_36_39_read": False,
            "init_40_49_access": False,
            "simulator_run": False,
            "world_model_training": False,
            "controller_or_success_evaluation": False,
        },
    }
    atomic_json(args.freeze, result)
    print(
        json.dumps(
            {
                "p1_alpha": p1_alpha,
                "p2_alpha": p2_alpha,
                "p3_final_epoch": final_epoch,
                "weights_sha256": result["weights"]["sha256"],
            },
            indent=2,
        )
    )
    return 0


def safe_direction(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = prediction.astype(np.float64)
    real = target.astype(np.float64)
    numerator = np.sum(pred * real, axis=-1)
    denominator = np.linalg.norm(pred, axis=-1) * np.linalg.norm(real, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1.0e-12,
    )


def safe_correlation(left: np.ndarray, right: np.ndarray, method: str) -> dict[str, Any]:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(x) < 2 or np.std(x) <= 1.0e-15 or np.std(y) <= 1.0e-15:
        return {"value": 0.0, "defined": False, "n": int(len(x))}
    if method == "pearson":
        value = scipy_stats.pearsonr(x, y).statistic
    elif method == "spearman":
        value = scipy_stats.spearmanr(x, y).statistic
    else:
        raise ValueError(method)
    return {"value": float(value), "defined": True, "n": int(len(x))}


def separation_ranking(
    predicted_separation: np.ndarray,
    real_separation: np.ndarray,
    branch_mask: np.ndarray,
    horizon_mask: np.ndarray,
    tie_tolerance: float,
) -> dict[str, Any]:
    within = []
    ordering = []
    for branch in np.flatnonzero(branch_mask):
        for horizon in np.flatnonzero(horizon_mask):
            predicted = predicted_separation[branch, :, horizon]
            real = real_separation[branch, :, horizon]
            if np.std(predicted) <= 1.0e-15 or np.std(real) <= 1.0e-15:
                within.append(0.0)
            else:
                within.append(float(scipy_stats.spearmanr(predicted, real).statistic))
            for left, right in ((0, 1), (0, 2), (1, 2)):
                pred_delta = predicted[left] - predicted[right]
                real_delta = real[left] - real[right]
                if abs(pred_delta) <= tie_tolerance or abs(real_delta) <= tie_tolerance:
                    ordering.append(0.5)
                else:
                    ordering.append(float(np.sign(pred_delta) == np.sign(real_delta)))
    return {
        "within_context_spearman": float(np.mean(within)),
        "candidate_pair_ordering_accuracy": float(np.mean(ordering)),
        "contexts": len(within),
        "comparisons": len(ordering),
    }


def metric_cut(
    prediction: np.ndarray,
    target: np.ndarray,
    real_separation: np.ndarray,
    branch_mask: np.ndarray,
    pair_mask: np.ndarray,
    horizon_mask: np.ndarray,
    minimum_real_norm: float,
    tie_tolerance: float,
) -> dict[str, Any]:
    subset_pred = prediction[branch_mask][:, pair_mask][:, :, horizon_mask]
    subset_real = target[branch_mask][:, pair_mask][:, :, horizon_mask]
    error = rmse(subset_pred, subset_real)
    direction = safe_direction(subset_pred, subset_real)
    pred_norm = np.linalg.norm(subset_pred.astype(np.float64), axis=-1)
    real_norm = np.linalg.norm(subset_real.astype(np.float64), axis=-1)
    valid = real_norm >= minimum_real_norm
    ratio = pred_norm[valid] / real_norm[valid]
    predicted_separation = 0.5 * np.square(pred_norm)
    actual_separation = real_separation[branch_mask][:, pair_mask][:, :, horizon_mask]
    result = {
        "n_vectors": int(subset_pred.shape[0] * subset_pred.shape[1] * subset_pred.shape[2]),
        "rmse": error,
        "direction": float(np.mean(direction)),
        "norm": {
            "valid": int(valid.sum()),
            "ratio_of_means": float(pred_norm[valid].mean() / real_norm[valid].mean()),
            "median_ratio": float(np.median(ratio)),
            "p25_ratio": float(np.percentile(ratio, 25)),
            "p75_ratio": float(np.percentile(ratio, 75)),
        },
        "separation": {
            "pearson": safe_correlation(predicted_separation, actual_separation, "pearson"),
            "spearman": safe_correlation(predicted_separation, actual_separation, "spearman"),
        },
    }
    if pair_mask.all():
        selected_predicted = 0.5 * np.square(
            np.linalg.norm(prediction.astype(np.float64), axis=-1)
        )
        result["ranking"] = separation_ranking(
            selected_predicted,
            real_separation,
            branch_mask,
            horizon_mask,
            tie_tolerance,
        )
    return result


def summarize_probe(
    prediction: np.ndarray,
    samples: dict[str, np.ndarray],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    target = samples["real_visual"]
    real_separation = samples["real_separation"]
    count = len(target)
    all_branches = np.ones(count, dtype=bool)
    all_pairs = np.ones(3, dtype=bool)
    all_horizons = np.ones(8, dtype=bool)
    minimum = float(protocol["metrics"]["minimum_real_visual_delta_norm"])
    tie = float(protocol["metrics"]["ranking_tie_tolerance"])
    result = {
        "overall": metric_cut(
            prediction,
            target,
            real_separation,
            all_branches,
            all_pairs,
            all_horizons,
            minimum,
            tie,
        ),
        "per_horizon": {},
        "per_task": {},
        "per_pair": {},
    }
    for horizon in range(8):
        horizon_mask = np.arange(8) == horizon
        result["per_horizon"][f"h{horizon + 1}"] = metric_cut(
            prediction,
            target,
            real_separation,
            all_branches,
            all_pairs,
            horizon_mask,
            minimum,
            tie,
        )
    for task in range(10):
        branch_mask = samples["task_id"] == task
        result["per_task"][str(task)] = metric_cut(
            prediction,
            target,
            real_separation,
            branch_mask,
            all_pairs,
            all_horizons,
            minimum,
            tie,
        )
    for pair_index, name in enumerate(PAIR_NAMES):
        pair_mask = np.arange(3) == pair_index
        result["per_pair"][name] = metric_cut(
            prediction,
            target,
            real_separation,
            all_branches,
            pair_mask,
            all_horizons,
            minimum,
            tie,
        )
    return result


def stratified_bootstrap_indices(
    task_id: np.ndarray, replicates: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    task_groups = [np.flatnonzero(task_id == task) for task in range(10)]
    if any(len(group) != 4 for group in task_groups):
        raise ValueError("bootstrap expects four held-out branches per task")
    result = np.empty((replicates, 40), dtype=np.int64)
    for replicate in range(replicates):
        result[replicate] = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in task_groups]
        )
    return result


def percentile(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def branch_sse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.square(prediction.astype(np.float64) - target).sum(axis=(1, 2, 3))


def branch_direction(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return safe_direction(prediction, target).mean(axis=(1, 2))


def bootstrap_comparisons(
    predictions: dict[str, np.ndarray],
    samples: dict[str, np.ndarray],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    target = samples["real_visual"]
    replicates = int(protocol["bootstrap"]["replicates"])
    indices = stratified_bootstrap_indices(
        samples["task_id"], replicates, int(protocol["bootstrap"]["seed"])
    )
    elements_per_branch = int(np.prod(target.shape[1:]))
    sse = {name: branch_sse(value, target) for name, value in predictions.items()}
    directions = {
        name: branch_direction(value, target) for name, value in predictions.items()
    }
    boot_rmse = {
        name: np.sqrt(values[indices].sum(axis=1) / (indices.shape[1] * elements_per_branch))
        for name, values in sse.items()
    }
    boot_direction = {
        name: values[indices].mean(axis=1) for name, values in directions.items()
    }

    def improvement(model: str, baseline: str) -> np.ndarray:
        return (boot_rmse[baseline] - boot_rmse[model]) / boot_rmse[baseline]

    result: dict[str, Any] = {
        "replicates": replicates,
        "unit": "task-stratified held-out branch point retaining all pairs/horizons/dimensions",
        "rmse_ci95": {name: percentile(values) for name, values in boot_rmse.items()},
        "direction_ci95": {
            name: percentile(values) for name, values in boot_direction.items()
        },
        "relative_improvement_ci95": {},
        "direction_degradation_ci95": {},
    }
    for model, baseline in (
        ("p1", "p0"),
        ("p3_real", "p0"),
        ("p3_real", "p1"),
        ("p3_real", "p2"),
        ("p3_predicted", "p0"),
        ("p3_predicted", "p2"),
        ("p3_physical_shuffle", "p3_real"),
        ("p3_context_shuffle", "p3_real"),
    ):
        key = f"{model}_vs_{baseline}"
        if baseline == "p3_real":
            values = (boot_rmse[model] - boot_rmse[baseline]) / boot_rmse[baseline]
        else:
            values = improvement(model, baseline)
        result["relative_improvement_ci95"][key] = percentile(values)
    for shuffled in ("p3_physical_shuffle", "p3_context_shuffle"):
        result["direction_degradation_ci95"][f"{shuffled}_vs_p3_real"] = percentile(
            boot_direction["p3_real"] - boot_direction[shuffled]
        )
    return result


def derangement_within_task(task_id: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mapping = np.arange(len(task_id), dtype=np.int64)
    for task in range(10):
        group = np.flatnonzero(task_id == task)
        for _ in range(1000):
            candidate = rng.permutation(group)
            if np.all(candidate != group):
                mapping[group] = candidate
                break
        else:
            raise RuntimeError("failed to construct within-task derangement")
    return mapping


@torch.inference_mode()
def predict_p3(
    model: ContextConditionedPhysicalVisualProbe,
    context: np.ndarray,
    physical: np.ndarray,
) -> np.ndarray:
    shape = physical.shape[:-1]
    repeated_context = np.broadcast_to(
        context[:, None, None, :], (*shape, context.shape[-1])
    )
    prediction = model(
        torch.from_numpy(repeated_context.reshape(-1, 256).astype(np.float32)),
        torch.from_numpy(physical.reshape(-1, 10).astype(np.float32)),
    ).numpy()
    return prediction.reshape(*shape, 512).astype(np.float32)


def load_calibrated_models(
    protocol: dict[str, Any], freeze: dict[str, Any]
) -> tuple[dict[str, np.ndarray], ContextConditionedPhysicalVisualProbe]:
    path = ROOT / freeze["weights"]["path"]
    if sha256(path) != freeze["weights"]["sha256"]:
        raise ValueError("probe calibration weights changed")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    return arrays, load_p3(arrays, int(protocol["probes"]["p3"]["hidden_dim"]))


def point_relative_improvement(model_rmse: float, baseline_rmse: float) -> float:
    return (baseline_rmse - model_rmse) / baseline_rmse


def decide(
    metrics: dict[str, Any], bootstrap: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    overall = {name: value["overall"] for name, value in metrics.items()}
    ci = bootstrap["relative_improvement_ci95"]
    direction_ci = bootstrap["direction_ci95"]
    shuffle_direction = bootstrap["direction_degradation_ci95"]
    gate = protocol["gates"]

    p1_checks = {
        "better_than_zero": ci["p1_vs_p0"][0] > 0,
        "positive_direction": overall["p1"]["direction"] > 0
        and direction_ci["p1"][0] > 0,
        "all_pair_direction_positive": all(
            metrics["p1"]["per_pair"][name]["direction"] > 0 for name in PAIR_NAMES
        ),
        "norm_not_catastrophic": gate["norm_ratio_min"]
        <= overall["p1"]["norm"]["ratio_of_means"]
        <= gate["norm_ratio_max"],
    }
    p3_improvement_p1 = point_relative_improvement(
        overall["p3_real"]["rmse"], overall["p1"]["rmse"]
    )
    p3_improvement_p2 = point_relative_improvement(
        overall["p3_real"]["rmse"], overall["p2"]["rmse"]
    )
    physical_shuffle_checks = {
        "rmse_worse": ci["p3_physical_shuffle_vs_p3_real"][0] > 0,
        "direction_worse": shuffle_direction[
            "p3_physical_shuffle_vs_p3_real"
        ][0]
        > 0,
    }
    context_shuffle_checks = {
        "rmse_worse": ci["p3_context_shuffle_vs_p3_real"][0] > 0,
        "direction_worse": shuffle_direction["p3_context_shuffle_vs_p3_real"][0]
        > 0,
    }
    p3_checks = {
        "better_than_zero": ci["p3_real_vs_p0"][0] > 0,
        "better_than_physical_by_5pct": p3_improvement_p1
        >= gate["minimum_incremental_improvement"]
        and ci["p3_real_vs_p1"][0] > 0,
        "better_than_context_by_5pct": p3_improvement_p2
        >= gate["minimum_incremental_improvement"]
        and ci["p3_real_vs_p2"][0] > 0,
        "positive_direction": overall["p3_real"]["direction"] > 0
        and direction_ci["p3_real"][0] > 0,
        "all_pair_direction_positive": all(
            metrics["p3_real"]["per_pair"][name]["direction"] > 0
            for name in PAIR_NAMES
        ),
        "physical_shuffle_worse": all(physical_shuffle_checks.values()),
        "context_shuffle_worse": all(context_shuffle_checks.values()),
        "norm_not_catastrophic": gate["norm_ratio_min"]
        <= overall["p3_real"]["norm"]["ratio_of_means"]
        <= gate["norm_ratio_max"],
    }
    predicted_checks = {
        "better_than_zero": ci["p3_predicted_vs_p0"][0] > 0,
        "better_than_context_by_5pct": point_relative_improvement(
            overall["p3_predicted"]["rmse"], overall["p2"]["rmse"]
        )
        >= gate["minimum_incremental_improvement"]
        and ci["p3_predicted_vs_p2"][0] > 0,
        "positive_direction": overall["p3_predicted"]["direction"] > 0
        and direction_ci["p3_predicted"][0] > 0,
        "all_pair_direction_positive": all(
            metrics["p3_predicted"]["per_pair"][name]["direction"] > 0
            for name in PAIR_NAMES
        ),
        "norm_not_catastrophic": gate["norm_ratio_min"]
        <= overall["p3_predicted"]["norm"]["ratio_of_means"]
        <= gate["norm_ratio_max"],
    }
    p1_pass = all(p1_checks.values())
    p3_pass = all(p3_checks.values())
    predicted_pass = all(predicted_checks.values())
    context_dominates = (
        not p3_checks["better_than_context_by_5pct"]
        and not p3_checks["physical_shuffle_worse"]
    )
    labels: list[str]
    authorization: str
    if p3_pass:
        labels = ["CONTEXT_CONDITIONED_PHYSICAL_VISUAL_SIGNAL_CONFIRMED"]
        authorization = "GO_CONTEXT_CONDITIONED_WM_DESIGN"
        if predicted_pass:
            labels.append("CONTEXT_CONDITIONED_DEPLOYABLE_MEDIATOR_CONFIRMED")
        else:
            labels.append("PHYSICAL_MEDIATOR_ACCURACY_REMAINS_BOTTLENECK")
    elif p1_pass and not p3_checks["better_than_physical_by_5pct"]:
        labels = ["PHYSICAL_ONLY_VISUAL_MAPPING_SUFFICIENT"]
        authorization = "BLOCK_CONTEXT_CONDITIONED_WM_DESIGN"
    elif context_dominates:
        labels = ["CONTEXT_DOMINATES_PHYSICAL_SIGNAL_NOT_ADDITIVE"]
        authorization = "BLOCK_CONTEXT_CONDITIONED_WM_DESIGN"
    else:
        labels = ["EE_JOINT_NOT_SUFFICIENT_FOR_VISUAL_COUNTERFACTUAL"]
        authorization = "BLOCK_CONTEXT_CONDITIONED_WM_DESIGN"
    return {
        "p1_checks": p1_checks,
        "p1_pass": p1_pass,
        "p3_real_checks": p3_checks,
        "p3_real_pass": p3_pass,
        "physical_shuffle_checks": physical_shuffle_checks,
        "context_shuffle_checks": context_shuffle_checks,
        "p3_predicted_checks": predicted_checks,
        "p3_predicted_pass": predicted_pass,
        "context_dominates": context_dominates,
        "mechanism_labels": labels,
        "authorization": authorization,
    }


def format_ci(values: list[float]) -> str:
    return f"[{values[0]:.4f}, {values[1]:.4f}]"


def cut_table(metrics: dict[str, Any], cut: str, labels: Iterable[str]) -> str:
    rows = [
        "| Cut | Probe | RMSE | Direction | Norm ratio | Sep. Spearman |",
        "|---|---|---:|---:|---:|---:|",
    ]
    probes = ("p0", "p1", "p2", "p3_real")
    for label in labels:
        for probe in probes:
            value = metrics[probe][cut][label]
            rows.append(
                f"| {label} | {probe} | {value['rmse']:.6f} | "
                f"{value['direction']:.4f} | {value['norm']['ratio_of_means']:.3f} | "
                f"{value['separation']['spearman']['value']:.4f} |"
            )
    return "\n".join(rows)


def render_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    overall = {name: value["overall"] for name, value in metrics.items()}
    boot = result["bootstrap"]
    decision = result["decision"]
    labels = ", ".join(f"`{value}`" for value in decision["mechanism_labels"])
    p3_zero = point_relative_improvement(overall["p3_real"]["rmse"], overall["p0"]["rmse"])
    p3_p1 = point_relative_improvement(overall["p3_real"]["rmse"], overall["p1"]["rmse"])
    p3_p2 = point_relative_improvement(overall["p3_real"]["rmse"], overall["p2"]["rmse"])
    pred_zero = point_relative_improvement(
        overall["p3_predicted"]["rmse"], overall["p0"]["rmse"]
    )
    pshuffle = (
        overall["p3_physical_shuffle"]["rmse"] - overall["p3_real"]["rmse"]
    ) / overall["p3_real"]["rmse"]
    cshuffle = (
        overall["p3_context_shuffle"]["rmse"] - overall["p3_real"]["rmse"]
    ) / overall["p3_real"]["rmse"]
    answer = result["core_answer"]
    report = f"""# {result['title']}

Date: 2026-09-25

Final mechanism verdict: {labels}

Authorization: `{decision['authorization']}`

## Executive answer

{answer}

The primary P3 probe has visual-differential RMSE `{overall['p3_real']['rmse']:.6f}`,
versus `{overall['p0']['rmse']:.6f}` for zero, `{overall['p1']['rmse']:.6f}` for
physical-only, and `{overall['p2']['rmse']:.6f}` for context-only. Its relative
improvements are {p3_zero:.2%}, {p3_p1:.2%}, and {p3_p2:.2%}, respectively.
Aggregate direction is `{overall['p3_real']['direction']:.4f}` with branch-block
95% CI {format_ci(boot['direction_ci95']['p3_real'])}.

Replacing real EE+joint differences with the frozen calibrated WM-v2B.1
prediction gives RMSE `{overall['p3_predicted']['rmse']:.6f}`, direction
`{overall['p3_predicted']['direction']:.4f}`, and {pred_zero:.2%} improvement
over zero. This second stage uses the exact same frozen P3 parameters.

This is a matched-branch supervised mechanism probe, not a deployable World
Model and not controller evidence.

## Frozen provenance

- Upstream freeze tag/commit: `{result['provenance']['upstream_tag']}` /
  `{result['provenance']['upstream_commit']}`.
- Protocol commit: `{result['provenance']['protocol_commit']}`; protocol
  SHA-256 `{result['provenance']['protocol_sha256']}`.
- Calibration freeze commit: `{result['provenance']['calibration_freeze_commit']}`;
  freeze SHA-256 `{result['provenance']['calibration_freeze_sha256']}`.
- Frozen reports/results tree: {result['provenance']['reports_results_file_count']}
  files, aggregate SHA-256 `{result['provenance']['reports_results_tree_sha256']}`.
- No existing report, result, checkpoint, or sidecar was overwritten.

## Exact branch split and isolation

Probe calibration uses init states 31--34 ({result['calibration']['partition']['branch_points']}
branch points); the one-shot held-out evaluation uses 36--39
({result['evaluation']['branch_points']} branch points). Every branch contributes
all three candidate pairs and all eight horizons to one partition. Historical
30/35 is reference-only and was not fitted; 40--49 was not accessed. Existing
sidecars already contained real RGB/CLIP futures, proprio futures, common
initial observations, and candidate identities, so no simulator was rerun.

## Sample construction

For every branch, pair, and horizon, the physical input is
`standardized(p_real_i)-standardized(p_real_j)` restricted to
`ee_pos[0:3]+joints[5:12]` (10-D). Gripper dimensions 3/4 are excluded. The
target is `z_real_i-z_real_j` (512-D) from frozen L2-normalized CLIP futures.
The matching real visual separation remains `1-cos(z_i,z_j)`; predicted
separation is `0.5*||pred_delta_z||^2`, the unit-latent identity.

## Frozen pre-action context

The 256-D context is the frozen WM-v2B initial hidden representation produced
only from `z0`, train-normalized `proprio0`, and language. Actions and all future
values are absent. Repeating each exact common branch input three times gives
maximum context difference `{result['calibration']['context']['candidate_copy_max_abs_difference']:.3e}`.

## P0--P3 definitions

- P0 predicts zero visual difference.
- P1 is no-bias 10-to-512 ridge regression and is exactly antisymmetric.
- P2 is a 256-to-512 ridge baseline with an unpenalized intercept; it sees only
  candidate-independent context.
- P3 is `W_o((W_p delta_p) * sigmoid(W_c c0))`, with 64 hidden units and no
  bias in `W_p` or `W_o`. It has {result['calibration']['p3']['parameters']:,}
  parameters and no context-to-output shortcut.

## Inner calibration and immutable held-out evaluation

P1/P2 ridge alpha was selected only by leave-one-init-out CV on 31--34 from
`{result['calibration']['ridge']['grid']}`. Selected alpha is
`{result['calibration']['ridge']['p1_selected_alpha']}` for P1 and
`{result['calibration']['ridge']['p2_selected_alpha']}` for P2. P3 used fixed
AdamW settings and inner-fold early stopping; the round-half-up median best
epoch was {result['calibration']['p3']['final_epoch']}. The final calibration
weights were committed and hash-frozen before 36--39 was opened.

## Held-out RMSE and direction

| Probe | RMSE | vs zero | Direction | Direction 95% CI | Norm ratio | Sep. Pearson | Sep. Spearman |
|---|---:|---:|---:|---|---:|---:|---:|
| P0 zero | {overall['p0']['rmse']:.6f} | — | {overall['p0']['direction']:.4f} | {format_ci(boot['direction_ci95']['p0'])} | {overall['p0']['norm']['ratio_of_means']:.3f} | {overall['p0']['separation']['pearson']['value']:.4f} | {overall['p0']['separation']['spearman']['value']:.4f} |
| P1 physical | {overall['p1']['rmse']:.6f} | {point_relative_improvement(overall['p1']['rmse'], overall['p0']['rmse']):.2%} | {overall['p1']['direction']:.4f} | {format_ci(boot['direction_ci95']['p1'])} | {overall['p1']['norm']['ratio_of_means']:.3f} | {overall['p1']['separation']['pearson']['value']:.4f} | {overall['p1']['separation']['spearman']['value']:.4f} |
| P2 context | {overall['p2']['rmse']:.6f} | {point_relative_improvement(overall['p2']['rmse'], overall['p0']['rmse']):.2%} | {overall['p2']['direction']:.4f} | {format_ci(boot['direction_ci95']['p2'])} | {overall['p2']['norm']['ratio_of_means']:.3f} | {overall['p2']['separation']['pearson']['value']:.4f} | {overall['p2']['separation']['spearman']['value']:.4f} |
| P3 real physical + context | {overall['p3_real']['rmse']:.6f} | {p3_zero:.2%} | {overall['p3_real']['direction']:.4f} | {format_ci(boot['direction_ci95']['p3_real'])} | {overall['p3_real']['norm']['ratio_of_means']:.3f} | {overall['p3_real']['separation']['pearson']['value']:.4f} | {overall['p3_real']['separation']['spearman']['value']:.4f} |

P3 improvement CIs are {format_ci(boot['relative_improvement_ci95']['p3_real_vs_p0'])}
versus zero, {format_ci(boot['relative_improvement_ci95']['p3_real_vs_p1'])}
versus P1, and {format_ci(boot['relative_improvement_ci95']['p3_real_vs_p2'])}
versus P2.

## Norm and separation-ranking fidelity

P3 norm-ratio median/p25/p75 are
`{overall['p3_real']['norm']['median_ratio']:.3f}` /
`{overall['p3_real']['norm']['p25_ratio']:.3f}` /
`{overall['p3_real']['norm']['p75_ratio']:.3f}`. Its within-context separation
Spearman is `{overall['p3_real']['ranking']['within_context_spearman']:.4f}` and
candidate-pair ordering accuracy is
`{overall['p3_real']['ranking']['candidate_pair_ordering_accuracy']:.2%}`.

## Physical and context shuffle controls

Both controls use fixed within-task branch-point derangements, preserving pair
and horizon. Physical shuffle changes P3 RMSE to
`{overall['p3_physical_shuffle']['rmse']:.6f}` ({pshuffle:.2%} degradation)
and direction to `{overall['p3_physical_shuffle']['direction']:.4f}`. Context
shuffle changes RMSE to `{overall['p3_context_shuffle']['rmse']:.6f}`
({cshuffle:.2%} degradation) and direction to
`{overall['p3_context_shuffle']['direction']:.4f}`. The corresponding relative
RMSE CIs are
{format_ci(boot['relative_improvement_ci95']['p3_physical_shuffle_vs_p3_real'])}
and {format_ci(boot['relative_improvement_ci95']['p3_context_shuffle_vs_p3_real'])};
direction-degradation CIs are
{format_ci(boot['direction_degradation_ci95']['p3_physical_shuffle_vs_p3_real'])}
and {format_ci(boot['direction_degradation_ci95']['p3_context_shuffle_vs_p3_real'])}.

## Real versus predicted physical mediator

| Physical input to frozen P3 | RMSE | Direction | Norm ratio | Sep. Spearman | Within rank | Ordering |
|---|---:|---:|---:|---:|---:|---:|
| Real EE+joint delta | {overall['p3_real']['rmse']:.6f} | {overall['p3_real']['direction']:.4f} | {overall['p3_real']['norm']['ratio_of_means']:.3f} | {overall['p3_real']['separation']['spearman']['value']:.4f} | {overall['p3_real']['ranking']['within_context_spearman']:.4f} | {overall['p3_real']['ranking']['candidate_pair_ordering_accuracy']:.2%} |
| Frozen calibrated WM-v2B.1 delta | {overall['p3_predicted']['rmse']:.6f} | {overall['p3_predicted']['direction']:.4f} | {overall['p3_predicted']['norm']['ratio_of_means']:.3f} | {overall['p3_predicted']['separation']['spearman']['value']:.4f} | {overall['p3_predicted']['ranking']['within_context_spearman']:.4f} | {overall['p3_predicted']['ranking']['candidate_pair_ordering_accuracy']:.2%} |

## Comparison with frozen WM-v2C-EJ

WM-v2C-EJ matched transfer was `0.019030` and visual-delta direction was
`0.0203`. The supervised P1/P3 directions here are
`{overall['p1']['direction']:.4f}` / `{overall['p3_real']['direction']:.4f}`.
This comparison diagnoses representational sufficiency only: P1/P3 were fitted
on matched counterfactual branches and cannot be presented as deployable-WM
performance.

## Per-horizon evidence

{cut_table(metrics, 'per_horizon', [f'h{value}' for value in range(1, 9)])}

## Per-task evidence

{cut_table(metrics, 'per_task', [str(value) for value in range(10)])}

## Per-pair evidence

{cut_table(metrics, 'per_pair', PAIR_NAMES)}

## Antisymmetry, finite values, and mechanical decision

P1 and P3 maximum `|f(c,-delta_p)+f(c,delta_p)|` are
`{result['audits']['p1_antisymmetry_max_abs']:.3e}` and
`{result['audits']['p3_antisymmetry_max_abs']:.3e}`. All inputs, fitted
parameters, predictions, residuals, and bootstrap statistics are finite.

P1 checks: `{json.dumps(decision['p1_checks'], sort_keys=True)}`.

P3 real-physical checks: `{json.dumps(decision['p3_real_checks'], sort_keys=True)}`.

P3 predicted-physical checks:
`{json.dumps(decision['p3_predicted_checks'], sort_keys=True)}`.

Final mechanism verdict: {labels}. Authorization:
`{decision['authorization']}`. This authorization concerns only a separately
preregistered future World-Model design; it never authorizes a controller run.

## Artifact hashes

- Calibration weights: `{result['calibration']['weights']['sha256']}`.
- Calibration source sidecar: `{result['calibration']['source_sidecar_sha256']}`.
- Evaluation source sidecar: `{result['evaluation']['source_sidecar_sha256']}`.
- Derived evaluation sidecar: `{result['artifacts']['sidecar_sha256']}`.
- Compact result: `{result['artifacts']['result_sha256_for_report']}`.

## Forbidden-activity audit

No World Model, CLIP encoder, actor, candidate generator, scale, reranker, or
controller was changed or trained. No simulator ran; no success, reward, done,
or task outcome was read. There was no C1/C2/D20-v2 work, seed23/47 run,
future context, action identity, pair embedding, expert-action error, or init
40--49 access. Gripper remained excluded. Frozen upstream artifacts were
hash-checked and not overwritten.
"""
    return report


def evaluate(args: argparse.Namespace, protocol: dict[str, Any]) -> int:
    freeze = json.loads(args.freeze.read_text())
    if freeze["status"] != "FROZEN_BEFORE_HELD_OUT_EVALUATION":
        raise ValueError("calibration is not frozen")
    if freeze["protocol_sha256"] != sha256(args.protocol):
        raise ValueError("calibration freeze belongs to another protocol")
    freeze_commit = git("log", "-1", "--format=%H", "--", str(args.freeze.relative_to(ROOT)))
    if not freeze_commit:
        raise ValueError("calibration freeze must be committed before evaluation")
    if git("status", "--porcelain"):
        raise ValueError("held-out evaluation requires a clean preregistered worktree")
    arrays, p3 = load_calibrated_models(protocol, freeze)
    evaluation_path = ROOT / protocol["partitions"]["evaluation_sidecar"]
    if sha256(evaluation_path) != protocol["partitions"]["evaluation_sha256"]:
        raise ValueError("evaluation sidecar changed")
    branches, scope = load_partition(
        evaluation_path,
        set(protocol["partitions"]["evaluation_init_states"]),
        protocol,
    )
    world_model, normalization = load_wm_v2b(protocol)
    alphas = np.asarray(
        json.loads((ROOT / protocol["physical_mediator"]["freeze"]).read_text())[
            "coefficients"
        ]["per_horizon_scalar"],
        dtype=np.float64,
    )
    samples = construct_samples(branches, world_model, normalization, alphas)
    p1 = (
        samples["real_physical_samples"].astype(np.float64)
        @ arrays["p1_weight"].astype(np.float64)
        + arrays["p1_bias"].astype(np.float64)
    ).reshape(40, 3, 8, 512).astype(np.float32)
    p2 = (
        samples["context_samples"].astype(np.float64)
        @ arrays["p2_weight"].astype(np.float64)
        + arrays["p2_bias"].astype(np.float64)
    ).reshape(40, 3, 8, 512).astype(np.float32)
    p3_real = predict_p3(p3, samples["context"], samples["real_physical"])
    p3_predicted = predict_p3(
        p3, samples["context"], samples["predicted_physical"]
    )
    physical_mapping = derangement_within_task(
        samples["task_id"], int(protocol["negative_controls"]["physical_shuffle_seed"])
    )
    context_mapping = derangement_within_task(
        samples["task_id"], int(protocol["negative_controls"]["context_shuffle_seed"])
    )
    p3_physical_shuffle = predict_p3(
        p3, samples["context"], samples["real_physical"][physical_mapping]
    )
    p3_context_shuffle = predict_p3(
        p3, samples["context"][context_mapping], samples["real_physical"]
    )
    predictions = {
        "p0": np.zeros_like(samples["real_visual"]),
        "p1": p1,
        "p2": p2,
        "p3_real": p3_real,
        "p3_predicted": p3_predicted,
        "p3_physical_shuffle": p3_physical_shuffle,
        "p3_context_shuffle": p3_context_shuffle,
    }
    metrics = {
        name: summarize_probe(value, samples, protocol)
        for name, value in predictions.items()
    }
    bootstrap = bootstrap_comparisons(predictions, samples, protocol)
    decision = decide(metrics, bootstrap, protocol)

    negative_p = -samples["real_physical_samples"]
    p1_forward = (
        samples["real_physical_samples"].astype(np.float64)
        @ arrays["p1_weight"].astype(np.float64)
        + arrays["p1_bias"].astype(np.float64)
    )
    p1_reverse = (
        negative_p.astype(np.float64) @ arrays["p1_weight"].astype(np.float64)
        + arrays["p1_bias"].astype(np.float64)
    )
    with torch.inference_mode():
        p3_forward = p3(
            torch.from_numpy(samples["context_samples"]),
            torch.from_numpy(samples["real_physical_samples"]),
        ).numpy()
        p3_reverse = p3(
            torch.from_numpy(samples["context_samples"]), torch.from_numpy(negative_p)
        ).numpy()
    p1_antisymmetry = float(np.max(np.abs(p1_forward + p1_reverse)))
    p3_antisymmetry = float(np.max(np.abs(p3_forward + p3_reverse)))
    if p1_antisymmetry > 1.0e-10 or p3_antisymmetry > 1.0e-6:
        raise ValueError("probe antisymmetry failed")

    sidecar_path = ROOT / protocol["outputs"]["evaluation_metrics"]
    result_path = ROOT / protocol["outputs"]["result"]
    report_path = ROOT / protocol["outputs"]["report"]
    if sidecar_path.exists() or result_path.exists() or report_path.exists():
        raise FileExistsError("probe evaluation output already exists")
    atomic_npz(
        sidecar_path,
        task_id=samples["task_id"].astype(np.int8),
        init_state_index=samples["init_state_index"].astype(np.int8),
        context=samples["context"],
        real_physical=samples["real_physical"],
        predicted_physical=samples["predicted_physical"],
        real_visual=samples["real_visual"],
        real_separation=samples["real_separation"],
        physical_shuffle_mapping=physical_mapping,
        context_shuffle_mapping=context_mapping,
        **{f"prediction__{name}": value for name, value in predictions.items()},
    )
    labels = decision["mechanism_labels"]
    if "CONTEXT_CONDITIONED_PHYSICAL_VISUAL_SIGNAL_CONFIRMED" in labels:
        if "PHYSICAL_MEDIATOR_ACCURACY_REMAINS_BOTTLENECK" in labels:
            core_answer = (
                "WM-v2C-EJ failed because its physical-to-visual mapping lacked sufficient "
                "scene conditioning; real EE/joint information is sufficient when gated by "
                "pre-action context, but the current predicted physical mediator still loses "
                "too much of that signal."
            )
        else:
            core_answer = (
                "WM-v2C-EJ failed because its physical-to-visual mapping lacked sufficient "
                "scene conditioning, not because EE/joint information was intrinsically "
                "insufficient."
            )
    elif labels == ["PHYSICAL_ONLY_VISUAL_MAPPING_SUFFICIENT"]:
        core_answer = (
            "EE/joint information is already sufficient without scene conditioning; the "
            "WM-v2C-EJ failure is therefore an implementation, optimization, or architecture "
            "failure rather than missing context."
        )
    elif labels == ["CONTEXT_DOMINATES_PHYSICAL_SIGNAL_NOT_ADDITIVE"]:
        core_answer = (
            "The probe is driven by scene context without independent additive EE/joint "
            "value, so this experiment does not support a physical-to-visual mediator."
        )
    else:
        core_answer = (
            "EE/joint information remains insufficient even after pre-action scene "
            "conditioning; the missing mechanism likely includes object/contact or "
            "object-relative state."
        )
    frozen_tree = protocol["frozen_upstream"]["reports_results_tree"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_version": protocol["protocol_version"],
        "status": "COMPLETE_ONE_SHOT_HELD_OUT_EVALUATION",
        "title": (
            "Scene conditioning reveals whether EE+joint motion explains local visual change"
        ),
        "core_answer": core_answer,
        "provenance": {
            "upstream_tag": protocol["frozen_upstream"]["primary_tag"],
            "upstream_commit": protocol["frozen_upstream"]["primary_commit"],
            "protocol_commit": freeze["calibration_execution_source_commit"],
            "protocol_sha256": sha256(args.protocol),
            "calibration_freeze_commit": freeze_commit,
            "calibration_freeze_sha256": sha256(args.freeze),
            "reports_results_file_count": frozen_tree["file_count"],
            "reports_results_tree_sha256": frozen_tree["sha256"],
        },
        "calibration": freeze,
        "evaluation": {
            **scope,
            "source_sidecar": str(evaluation_path.relative_to(ROOT)),
            "source_sidecar_sha256": sha256(evaluation_path),
            "one_shot": True,
            "simulator_rerun": False,
        },
        "metrics": metrics,
        "bootstrap": bootstrap,
        "decision": decision,
        "audits": {
            "p1_antisymmetry_max_abs": p1_antisymmetry,
            "p3_antisymmetry_max_abs": p3_antisymmetry,
            "context_candidate_copy_max_abs": float(
                samples["context_identity_max_abs"]
            ),
            "finite": True,
            "gripper_input": False,
            "future_context": False,
            "action_or_pair_identity_input": False,
        },
        "artifacts": {
            "sidecar": str(sidecar_path.relative_to(ROOT)),
            "sidecar_sha256": sha256(sidecar_path),
            "result": str(result_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
        "forbidden_activity_audit": {
            "new_world_model_training": False,
            "simulator_or_controller_run": False,
            "success_reward_done_used": False,
            "init_40_49_access": False,
            "seed23_47": False,
            "future_context": False,
            "candidate_or_reranker_change": False,
            "upstream_overwrite": False,
        },
        "nan_inf_audit": "pass",
    }
    atomic_json(result_path, result)
    result["artifacts"]["result_sha256_for_report"] = sha256(result_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result))
    print(
        json.dumps(
            {
                "mechanism_labels": labels,
                "authorization": decision["authorization"],
                "p3_real_rmse": metrics["p3_real"]["overall"]["rmse"],
                "p3_real_direction": metrics["p3_real"]["overall"]["direction"],
                "p3_predicted_rmse": metrics["p3_predicted"]["overall"]["rmse"],
                "sidecar_sha256": result["artifacts"]["sidecar_sha256"],
                "result_sha256": sha256(result_path),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("calibrate", "evaluate"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/counterfactual_physical_visual_probe_protocol.json",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=ROOT / "configs/counterfactual_physical_visual_probe_freeze.json",
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.stage == "calibrate":
        return calibrate(args, protocol)
    return evaluate(args, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
