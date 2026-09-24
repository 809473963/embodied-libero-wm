#!/usr/bin/env python3
"""Fit frozen closed-form proprio-differential calibration coefficients."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
GROUPS = {
    "ee_pos": np.asarray([0, 1, 2]),
    "gripper": np.asarray([3, 4]),
    "joints": np.asarray([5, 6, 7, 8, 9, 10, 11]),
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def pairwise_delta(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[1:] != (3, 8, 12):
        raise ValueError(f"expected [N,3,8,12], got {array.shape}")
    return np.stack(
        [array[:, left] - array[:, right] for left, right in PAIR_INDICES], axis=1
    )


def fit_scalar(predicted: np.ndarray, real: np.ndarray) -> float:
    pred = np.asarray(predicted, dtype=np.float64).reshape(-1)
    target = np.asarray(real, dtype=np.float64).reshape(-1)
    denominator = float(np.dot(pred, pred))
    if denominator <= 1e-20:
        raise ValueError("calibration predictor has zero least-squares denominator")
    alpha = float(np.dot(pred, target) / denominator)
    if not math.isfinite(alpha):
        raise FloatingPointError("fitted alpha is NaN/Inf")
    return alpha


def apply_form(predicted: np.ndarray, form: str, coefficients: Any) -> np.ndarray:
    result = np.asarray(predicted, dtype=np.float64).copy()
    if form == "global_scalar":
        return result * float(coefficients)
    if form == "per_horizon_scalar":
        return result * np.asarray(coefficients, dtype=np.float64)[None, None, :, None]
    if form == "groupwise_scalar":
        for group, indices in GROUPS.items():
            result[..., indices] *= float(coefficients[group])
        return result
    if form == "horizon_group_scalar":
        for group, indices in GROUPS.items():
            result[..., indices] *= np.asarray(
                coefficients[group], dtype=np.float64
            )[None, None, :, None]
        return result
    raise KeyError(form)


def fit_all(predicted: np.ndarray, real: np.ndarray) -> dict[str, Any]:
    coefficients: dict[str, Any] = {
        "global_scalar": fit_scalar(predicted, real),
        "per_horizon_scalar": [
            fit_scalar(predicted[:, :, horizon], real[:, :, horizon])
            for horizon in range(8)
        ],
        "groupwise_scalar": {
            group: fit_scalar(predicted[..., indices], real[..., indices])
            for group, indices in GROUPS.items()
        },
        "horizon_group_scalar": {
            group: [
                fit_scalar(
                    predicted[:, :, horizon][..., indices],
                    real[:, :, horizon][..., indices],
                )
                for horizon in range(8)
            ]
            for group, indices in GROUPS.items()
        },
    }
    return coefficients


def rmse(predicted: np.ndarray, real: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - real) ** 2)))


def calibration_fit_diagnostics(
    predicted: np.ndarray, real: np.ndarray, coefficients: dict[str, Any]
) -> dict[str, Any]:
    zero = float(np.sqrt(np.mean(real**2)))
    raw = rmse(predicted, real)
    diagnostics: dict[str, Any] = {
        "zero_difference_rmse": zero,
        "raw_rmse": raw,
        "forms": {},
    }
    for form, values in coefficients.items():
        calibrated = apply_form(predicted, form, values)
        error = rmse(calibrated, real)
        diagnostics["forms"][form] = {
            "calibration_partition_rmse": error,
            "improvement_over_zero": float((zero - error) / zero),
            "improvement_over_raw": float((raw - error) / raw),
        }
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/wm_v2b_1_differential_calibration_protocol.json",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "PREREGISTERED_BEFORE_INIT_31_39_BRANCH_EXECUTION":
        raise ValueError("protocol is not preregistered")
    artifacts = protocol["artifacts"]
    output = ROOT / artifacts["calibration_freeze"]
    calibration_manifest_path = ROOT / artifacts["calibration_manifest"]
    calibration_sidecar_path = ROOT / artifacts["calibration_sidecar"]
    evaluation_manifest_path = ROOT / artifacts["evaluation_manifest"]
    evaluation_sidecar_path = ROOT / artifacts["evaluation_sidecar"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration freeze: {output}")
    if evaluation_manifest_path.exists() or evaluation_sidecar_path.exists():
        raise RuntimeError("refusing to fit after evaluation partition exists")
    if not calibration_manifest_path.exists() or not calibration_sidecar_path.exists():
        raise FileNotFoundError("calibration partition artifacts are incomplete")
    manifest = json.loads(calibration_manifest_path.read_text())
    if manifest["status"] != "COMPLETE" or manifest["partition"] != "calibration":
        raise ValueError("calibration branch manifest is not complete")
    if manifest["protocol"]["sha256"] != sha256(args.protocol):
        raise ValueError("calibration manifest protocol mismatch")
    if manifest["artifacts"]["sidecar_sha256"] != sha256(calibration_sidecar_path):
        raise ValueError("calibration sidecar hash mismatch")

    checkpoint_path = ROOT / protocol["frozen_upstream"]["wm_v2b_checkpoint"]
    if sha256(checkpoint_path) != protocol["frozen_upstream"]["wm_v2b_checkpoint_sha256"]:
        raise ValueError("WM-v2B checkpoint changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    p_mean = np.asarray(
        checkpoint["normalization"]["proprio"]["mean"], dtype=np.float64
    )
    p_std = np.asarray(
        checkpoint["normalization"]["proprio"]["std"], dtype=np.float64
    )
    with np.load(calibration_sidecar_path) as sidecar:
        if set(sidecar["init_state_index"].tolist()) != {31, 32, 33, 34}:
            raise ValueError("calibration sidecar contains wrong init states")
        real = (sidecar["real_proprio"].astype(np.float64) - p_mean) / p_std
        predicted = sidecar["pred_proprio_wm_v2b_standardized"].astype(np.float64)
        task_id = sidecar["task_id"].astype(np.int64)
        init_state = sidecar["init_state_index"].astype(np.int64)
        planning_index = sidecar["planning_index"].astype(np.int64)
    if set(task_id.tolist()) != set(range(10)) or set(planning_index.tolist()) != {8}:
        raise ValueError("calibration sidecar identity scope changed")
    real_delta = pairwise_delta(real)
    predicted_delta = pairwise_delta(predicted)
    coefficients = fit_all(predicted_delta, real_delta)
    diagnostics = calibration_fit_diagnostics(
        predicted_delta, real_delta, coefficients
    )
    all_coefficients = [
        coefficients["global_scalar"],
        *coefficients["per_horizon_scalar"],
        *coefficients["groupwise_scalar"].values(),
        *[
            value
            for values in coefficients["horizon_group_scalar"].values()
            for value in values
        ],
    ]
    if not np.isfinite(np.asarray(all_coefficients, dtype=np.float64)).all():
        raise FloatingPointError("calibration coefficients contain NaN/Inf")

    freeze: dict[str, Any] = {
        "schema_version": 1,
        "experiment_version": protocol["protocol_version"],
        "status": "FROZEN_BEFORE_EVALUATION",
        "protocol": str(args.protocol.relative_to(ROOT)),
        "protocol_sha256": sha256(args.protocol),
        "fit_source_commit": source_commit(),
        "fit_partition": {
            "name": "calibration",
            "init_state_indices": [31, 32, 33, 34],
            "branch_points": 40,
            "candidate_pairs_per_horizon": 120,
            "horizon": 8,
        },
        "calibration_manifest": str(calibration_manifest_path.relative_to(ROOT)),
        "calibration_manifest_sha256": sha256(calibration_manifest_path),
        "calibration_sidecar": str(calibration_sidecar_path.relative_to(ROOT)),
        "calibration_sidecar_sha256": sha256(calibration_sidecar_path),
        "fit_rule": protocol["calibration_models"]["fit_rule"],
        "coefficients": coefficients,
        "negative_alpha": {
            "global_scalar": bool(coefficients["global_scalar"] < 0),
            "per_horizon_scalar": [
                bool(value < 0) for value in coefficients["per_horizon_scalar"]
            ],
            "groupwise_scalar": {
                key: bool(value < 0)
                for key, value in coefficients["groupwise_scalar"].items()
            },
            "horizon_group_scalar": {
                key: [bool(value < 0) for value in values]
                for key, values in coefficients["horizon_group_scalar"].items()
            },
        },
        "calibration_partition_diagnostics_not_held_out": diagnostics,
        "evaluation_artifacts_present_at_fit_time": False,
        "forbidden_activity_audit": {
            "evaluation_partition_read_or_fit": False,
            "iterative_tuning": False,
            "alpha_clipped_or_sign_corrected": False,
            "neural_model_trained_or_modified": False,
            "calibration_form_selected_from_evaluation": False,
        },
    }
    atomic_json(output, freeze)
    print(
        json.dumps(
            {
                "status": freeze["status"],
                "global_alpha": coefficients["global_scalar"],
                "per_horizon_alpha": coefficients["per_horizon_scalar"],
                "groupwise_alpha": coefficients["groupwise_scalar"],
                "output": str(output),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
