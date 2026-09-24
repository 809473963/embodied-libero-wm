#!/usr/bin/env python3
"""Held-out analysis for WM-v2B.1 differential physical calibration."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_wm_v2b_physical_mediator import (
    DIMENSION_NAMES,
    GROUPS,
    PAIR_NAMES,
    absolute_breakdown,
    absolute_from_smooth_l1,
    correlation,
    differential_stats,
    finite_tree,
    pairwise_delta,
    rank_contexts,
    ranking_summary,
    sign_fidelity,
    summary,
    vector_cosine,
)
from scripts.fit_wm_v2b_1_differential_calibration import apply_form


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


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content)
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def path_last_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()


def percentile_interval(value: np.ndarray) -> list[float]:
    return [float(np.percentile(value, 2.5)), float(np.percentile(value, 97.5))]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: source[name].copy() for name in source.files}


def load_and_validate(
    protocol_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text())
    if protocol["status"] != "PREREGISTERED_BEFORE_INIT_31_39_BRANCH_EXECUTION":
        raise ValueError("protocol is not preregistered")
    frozen = protocol["frozen_upstream"]
    for path_key, hash_key in (
        ("mediator_result", "mediator_result_sha256"),
        ("mediator_report", "mediator_report_sha256"),
        ("historical_branch_result", "historical_branch_result_sha256"),
        ("historical_branch_sidecar", "historical_branch_sidecar_sha256"),
        ("historical_branch_protocol", "historical_branch_protocol_sha256"),
        ("historical_branch_amendment", "historical_branch_amendment_sha256"),
        ("historical_collector", "historical_collector_sha256"),
        ("validation_sidecar", "validation_sidecar_sha256"),
        ("wm_v2b_checkpoint", "wm_v2b_checkpoint_sha256"),
        ("controller_partition_protocol", "controller_partition_protocol_sha256"),
    ):
        if sha256(ROOT / frozen[path_key]) != frozen[hash_key]:
            raise ValueError(f"frozen source changed: {frozen[path_key]}")
    artifacts = protocol["artifacts"]
    freeze_path = ROOT / artifacts["calibration_freeze"]
    freeze = json.loads(freeze_path.read_text())
    if freeze["status"] != "FROZEN_BEFORE_EVALUATION":
        raise ValueError("calibration coefficients are not frozen")
    if freeze["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("calibration freeze protocol mismatch")
    freeze_commit = path_last_commit(freeze_path)
    if not freeze_commit:
        raise ValueError("calibration freeze was not committed before evaluation")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(freeze_path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip():
        raise ValueError("calibration freeze has uncommitted changes")

    manifests: dict[str, Any] = {}
    for partition in ("calibration", "evaluation"):
        manifest_path = ROOT / artifacts[f"{partition}_manifest"]
        sidecar_path = ROOT / artifacts[f"{partition}_sidecar"]
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "COMPLETE" or manifest["partition"] != partition:
            raise ValueError(f"{partition} manifest incomplete")
        if manifest["protocol"]["sha256"] != sha256(protocol_path):
            raise ValueError(f"{partition} protocol mismatch")
        if manifest["artifacts"]["sidecar_sha256"] != sha256(sidecar_path):
            raise ValueError(f"{partition} sidecar hash mismatch")
        if len(manifest["branch_identities"]) != 40:
            raise ValueError(f"{partition} branch count changed")
        if not all(row["prefix_audit"]["pass"] for row in manifest["branch_identities"]):
            raise ValueError(f"{partition} contains failed prefix identity")
        manifests[partition] = manifest
    calibration_manifest_path = ROOT / artifacts["calibration_manifest"]
    calibration_sidecar_path = ROOT / artifacts["calibration_sidecar"]
    if freeze["calibration_manifest_sha256"] != sha256(calibration_manifest_path):
        raise ValueError("calibration manifest changed after alpha freeze")
    if freeze["calibration_sidecar_sha256"] != sha256(calibration_sidecar_path):
        raise ValueError("calibration sidecar changed after alpha freeze")
    return protocol, freeze, manifests


def standardize_real(
    arrays: dict[str, np.ndarray], p_mean: np.ndarray, p_std: np.ndarray
) -> np.ndarray:
    return (arrays["real_proprio"].astype(np.float64) - p_mean) / p_std


def branch_absolute(
    arrays: dict[str, np.ndarray], p_mean: np.ndarray, p_std: np.ndarray
) -> np.ndarray:
    real = standardize_real(arrays, p_mean, p_std)
    predicted = arrays["pred_proprio_wm_v2b_standardized"].astype(np.float64)
    return np.abs(predicted - real)


def block_bootstrap(
    calibrated: np.ndarray,
    raw: np.ndarray,
    real: np.ndarray,
    within: np.ndarray,
    ordering: np.ndarray,
    sign_threshold: float,
    replicates: int,
    seed: int,
    minimum_real_norm: float,
) -> dict[str, Any]:
    pred = np.asarray(calibrated, dtype=np.float64)
    raw_pred = np.asarray(raw, dtype=np.float64)
    target = np.asarray(real, dtype=np.float64)
    count = target.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(replicates, count), endpoint=False)

    def block_sum(value: np.ndarray) -> np.ndarray:
        return value.reshape(count, -1).sum(axis=1)

    eligible = np.linalg.norm(target, axis=-1) >= minimum_real_norm
    element_count = eligible.reshape(count, -1).sum(axis=1) * target.shape[-1]
    sampled_elements = element_count[indices].sum(axis=1)
    calibrated_ss = block_sum((pred - target) ** 2 * eligible[..., None])
    raw_ss = block_sum((raw_pred - target) ** 2 * eligible[..., None])
    zero_ss = block_sum(target**2 * eligible[..., None])
    calibrated_rmse = np.sqrt(calibrated_ss[indices].sum(axis=1) / sampled_elements)
    raw_rmse = np.sqrt(raw_ss[indices].sum(axis=1) / sampled_elements)
    zero_rmse = np.sqrt(zero_ss[indices].sum(axis=1) / sampled_elements)

    direction = vector_cosine(pred, target)
    direction_sum = block_sum(direction * eligible)
    direction_count = eligible.reshape(count, -1).sum(axis=1)
    sampled_direction = direction_sum[indices].sum(axis=1) / direction_count[
        indices
    ].sum(axis=1)

    rank_mean = block_sum(within)[indices].sum(axis=1) / (within[0].size * count)
    ordering_mean = block_sum(ordering)[indices].sum(axis=1) / (
        ordering[0].size * count
    )
    sign_eligible = np.abs(target) >= sign_threshold
    sign_correct = np.sign(pred) == np.sign(target)
    sign_count = block_sum(sign_eligible)
    sign_hits = block_sum(sign_correct * sign_eligible)
    sampled_sign_count = sign_count[indices].sum(axis=1)
    sampled_sign = np.divide(
        sign_hits[indices].sum(axis=1),
        sampled_sign_count,
        out=np.zeros(replicates, dtype=np.float64),
        where=sampled_sign_count > 0,
    )
    return {
        "replicates": replicates,
        "seed": seed,
        "unit": "branch_point",
        "improvement_zero_ci95": percentile_interval(
            (zero_rmse - calibrated_rmse) / zero_rmse
        ),
        "improvement_raw_ci95": percentile_interval(
            (raw_rmse - calibrated_rmse) / raw_rmse
        ),
        "direction_mean_ci95": percentile_interval(sampled_direction),
        "within_context_spearman_ci95": percentile_interval(rank_mean),
        "ordering_accuracy_ci95": percentile_interval(ordering_mean),
        "sign_accuracy_ci95": percentile_interval(sampled_sign),
    }


def differential_with_baselines(
    calibrated: np.ndarray,
    raw: np.ndarray,
    real: np.ndarray,
    minimum_norm: float,
) -> dict[str, Any]:
    result = differential_stats(calibrated, real, minimum_norm)
    eligible = np.linalg.norm(real, axis=-1) >= minimum_norm
    raw_rmse = float(np.sqrt(np.mean((raw[eligible] - real[eligible]) ** 2)))
    result["raw_rmse"] = raw_rmse
    result["improvement_over_raw"] = float(
        (raw_rmse - result["differential_rmse"]) / raw_rmse
    )
    return result


def form_analysis(
    calibrated: np.ndarray,
    raw: np.ndarray,
    real: np.ndarray,
    task_id: np.ndarray,
    protocol: dict[str, Any],
    seed_offset: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics = protocol["metrics"]
    minimum_norm = float(metrics["minimum_real_norm"])
    sign_threshold = float(metrics["minimum_abs_real_delta_for_sign"])
    pred_norm = np.linalg.norm(calibrated, axis=-1)
    real_norm = np.linalg.norm(real, axis=-1)
    within, ordering = rank_contexts(
        pred_norm, real_norm, float(metrics["ranking_tie_tolerance"])
    )
    ranking = ranking_summary(pred_norm, real_norm, within, ordering, task_id)
    sign = sign_fidelity(calibrated, real, sign_threshold, task_id)
    bootstrap_cfg = protocol["bootstrap"]

    def cut(pred: np.ndarray, raw_cut: np.ndarray, real_cut: np.ndarray) -> dict[str, Any]:
        return differential_with_baselines(pred, raw_cut, real_cut, minimum_norm)

    differential = {
        "overall": cut(calibrated, raw, real),
        "per_horizon": [
            {
                "horizon": horizon + 1,
                **cut(
                    calibrated[:, :, horizon],
                    raw[:, :, horizon],
                    real[:, :, horizon],
                ),
            }
            for horizon in range(8)
        ],
        "per_task": {
            str(task): cut(
                calibrated[task_id == task],
                raw[task_id == task],
                real[task_id == task],
            )
            for task in range(10)
        },
        "per_pair": {
            name: cut(calibrated[:, pair], raw[:, pair], real[:, pair])
            for pair, name in enumerate(PAIR_NAMES)
        },
        "groups": {
            name: cut(
                calibrated[..., indices], raw[..., indices], real[..., indices]
            )
            for name, indices in GROUPS.items()
        },
    }
    bootstrap = {
        "overall": block_bootstrap(
            calibrated,
            raw,
            real,
            within,
            ordering,
            sign_threshold,
            int(bootstrap_cfg["replicates"]),
            int(bootstrap_cfg["seed"]) + seed_offset,
            minimum_norm,
        ),
        "per_pair": {},
        "per_horizon": [],
        "groups": {},
    }
    for pair, name in enumerate(PAIR_NAMES):
        pair_pred = calibrated[:, pair : pair + 1]
        pair_raw = raw[:, pair : pair + 1]
        pair_real = real[:, pair : pair + 1]
        dummy_within = np.ones((len(task_id), 1), dtype=np.float64)
        dummy_ordering = np.ones((len(task_id), 1), dtype=np.float64)
        bootstrap["per_pair"][name] = block_bootstrap(
            pair_pred,
            pair_raw,
            pair_real,
            dummy_within,
            dummy_ordering,
            sign_threshold,
            int(bootstrap_cfg["replicates"]),
            int(bootstrap_cfg["seed"]) + seed_offset + 10 + pair,
            minimum_norm,
        )
    for horizon in range(8):
        bootstrap["per_horizon"].append(
            {
                "horizon": horizon + 1,
                **block_bootstrap(
                    calibrated[:, :, horizon : horizon + 1],
                    raw[:, :, horizon : horizon + 1],
                    real[:, :, horizon : horizon + 1],
                    within[:, horizon : horizon + 1],
                    ordering[:, horizon : horizon + 1],
                    sign_threshold,
                    int(bootstrap_cfg["replicates"]),
                    int(bootstrap_cfg["seed"]) + seed_offset + 100 + horizon,
                    minimum_norm,
                ),
            }
        )
    for group_index, (name, indices) in enumerate(GROUPS.items()):
        group_pred = calibrated[..., indices]
        group_raw = raw[..., indices]
        group_real = real[..., indices]
        group_norm = np.linalg.norm(group_pred, axis=-1)
        group_real_norm = np.linalg.norm(group_real, axis=-1)
        group_within, group_ordering = rank_contexts(
            group_norm,
            group_real_norm,
            float(metrics["ranking_tie_tolerance"]),
        )
        bootstrap["groups"][name] = block_bootstrap(
            group_pred,
            group_raw,
            group_real,
            group_within,
            group_ordering,
            sign_threshold,
            int(bootstrap_cfg["replicates"]),
            int(bootstrap_cfg["seed"]) + seed_offset + 200 + group_index,
            minimum_norm,
        )
    return {
        "differential": differential,
        "ranking": ranking,
        "sign": sign,
        "bootstrap": bootstrap,
    }, {
        "calibrated_delta": calibrated.astype(np.float32),
        "predicted_norm": pred_norm.astype(np.float32),
        "within_context_spearman": within.astype(np.float32),
        "ordering_score": ordering.astype(np.float32),
    }


def aggregate_gate(form: dict[str, Any]) -> bool:
    overall = form["differential"]["overall"]
    bootstrap = form["bootstrap"]["overall"]
    return bool(
        overall["relative_improvement_over_zero"] >= 0.05
        and bootstrap["improvement_zero_ci95"][0] > 0
    )


def group_gate(form: dict[str, Any], group: str) -> dict[str, bool]:
    values = form["differential"]["groups"][group]
    sign = form["sign"]["groups"][group]
    bootstrap = form["bootstrap"]["groups"][group]
    return {
        "zero_improvement": bool(
            values["relative_improvement_over_zero"] >= 0.05
            and bootstrap["improvement_zero_ci95"][0] > 0
        ),
        "direction": bool(
            values["direction_cosine"]["mean"] >= 0.10
            and bootstrap["direction_mean_ci95"][0] > 0
        ),
        "sign": bool(
            sign["accuracy"] is not None
            and sign["accuracy"] >= 0.55
            and bootstrap["sign_accuracy_ci95"][0] > 0.50
        ),
    }


def make_decision(
    forms: dict[str, Any], freeze: dict[str, Any]
) -> dict[str, Any]:
    primary = forms["per_horizon_scalar"]
    overall = primary["differential"]["overall"]
    ranking = primary["ranking"]["overall"]
    bootstrap = primary["bootstrap"]["overall"]
    pair_checks = {}
    for name in PAIR_NAMES:
        values = primary["differential"]["per_pair"][name]
        pair_checks[name] = {
            "improvement_zero_positive": bool(
                values["relative_improvement_over_zero"] > 0
            ),
            "direction_positive": bool(values["direction_cosine"]["mean"] > 0),
            "magnitude_ratio_in_0_5_1_5": bool(
                0.5 <= values["magnitude_ratio"]["ratio_of_means"] <= 1.5
            ),
        }
    gripper = group_gate(primary, "gripper")
    gripper_sign = primary["sign"]["groups"]["gripper"]
    gripper["coverage"] = bool(
        gripper_sign["eligible_fraction"] is not None
        and gripper_sign["eligible_fraction"] >= 0.10
    )
    gate = {
        "aggregate_zero_improvement": bool(
            overall["relative_improvement_over_zero"] >= 0.05
            and bootstrap["improvement_zero_ci95"][0] > 0
        ),
        "aggregate_raw_improvement": bool(
            overall["improvement_over_raw"] > 0
            and bootstrap["improvement_raw_ci95"][0] > 0
        ),
        "direction": bool(
            overall["direction_cosine"]["mean"] >= 0.10
            and bootstrap["direction_mean_ci95"][0] > 0
        ),
        "ranking": bool(
            ranking["within_context_spearman"]["mean"] >= 0.10
            and bootstrap["within_context_spearman_ci95"][0] > 0
            and ranking["pairwise_ordering_accuracy"]["mean"] >= 0.55
            and bootstrap["ordering_accuracy_ci95"][0] > 0.50
        ),
        "candidate_pair_consistency": bool(
            all(all(checks.values()) for checks in pair_checks.values())
        ),
        "full_gripper_validity": bool(all(gripper.values())),
        "positive_alpha": bool(
            all(value > 0 for value in freeze["coefficients"]["per_horizon_scalar"])
        ),
    }
    full = bool(all(gate.values()))
    partial_checks = {
        group: group_gate(forms["groupwise_scalar"], group)
        for group in ("ee_pos", "joints", "gripper")
    }
    partial_valid_groups = [
        group
        for group in ("ee_pos", "joints")
        if all(partial_checks[group].values())
    ]
    gripper_partial_pass = all(partial_checks["gripper"].values())
    partial = bool(not full and partial_valid_groups and not gripper_partial_pass)
    if full:
        verdict = "PHYSICAL_MEDIATOR_CALIBRATION_PASS"
        authorization = "GO_WM_V2C_DESIGN"
        mechanisms = [
            "PHYSICAL_MEDIATOR_RELATIVE_SIGNAL_VALID",
            "PHYSICAL_MEDIATOR_VALID_VISUAL_TRANSFER_BLOCK_CONFIRMED",
        ]
    elif partial:
        verdict = "PARTIAL_PHYSICAL_MEDIATOR_PASS"
        authorization = "BLOCK_WM_V2C_PHYSICAL_MEDIATOR"
        mechanisms = [
            "EE_JOINT_MEDIATOR_VALID_GRIPPER_INVALID"
            if set(partial_valid_groups) == {"ee_pos", "joints"}
            else "PARTIAL_COORDINATE_PHYSICAL_MEDIATOR_SIGNAL"
        ]
    else:
        verdict = "PHYSICAL_MEDIATOR_RETRAIN_REQUIRED"
        authorization = "BLOCK_WM_V2C_PHYSICAL_MEDIATOR"
        mechanisms = ["SIMPLE_DIFFERENTIAL_CALIBRATION_INSUFFICIENT"]
    return {
        "primary_form": "per_horizon_scalar",
        "primary_gate_pass": gate,
        "candidate_pair_checks": pair_checks,
        "gripper_checks": gripper,
        "partial_group_checks": partial_checks,
        "partial_valid_groups": partial_valid_groups,
        "verdict": verdict,
        "authorization": authorization,
        "mechanism_labels": mechanisms,
    }


def markdown_table(headers: list[str], rows: Iterable[Iterable[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(result: dict[str, Any]) -> str:
    analysis = result["analysis"]
    absolute = analysis["absolute_error"]
    forms = analysis["forms"]
    primary = forms["per_horizon_scalar"]
    decision = analysis["decision"]
    freeze = result["calibration_freeze"]
    result_hash = hashlib.sha256(
        (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode()
    ).hexdigest()

    absolute_rows = []
    for name in ("validation", "historical_30_35", "calibration_31_34", "evaluation_36_39"):
        values = absolute[name]["overall"]
        ratio = absolute["ratios_to_validation"].get(name)
        absolute_rows.append(
            [
                name,
                f"{values['rmse']:.6f}",
                f"{values['mae']:.6f}",
                f"{values['smooth_l1_beta1']:.6f}",
                "1.000x" if ratio is None else f"{ratio['rmse']:.3f}x",
            ]
        )

    form_rows = []
    for name, values in forms.items():
        overall = values["differential"]["overall"]
        boot = values["bootstrap"]["overall"]
        ranking = values["ranking"]["overall"]
        form_rows.append(
            [
                name,
                f"{overall['differential_rmse']:.6f}",
                f"{overall['zero_difference_rmse']:.6f}",
                f"{overall['raw_rmse']:.6f}",
                f"{100*overall['relative_improvement_over_zero']:.2f}%",
                f"[{boot['improvement_zero_ci95'][0]:.3f},{boot['improvement_zero_ci95'][1]:.3f}]",
                f"{100*overall['improvement_over_raw']:.2f}%",
                f"{overall['magnitude_ratio']['ratio_of_means']:.3f}",
                f"{overall['direction_cosine']['mean']:.3f}",
                f"{ranking['within_context_spearman']['mean']:.3f}",
                f"{ranking['pairwise_ordering_accuracy']['mean']:.2%}",
            ]
        )

    horizon_rows = []
    for row, boot, alpha in zip(
        primary["differential"]["per_horizon"],
        primary["bootstrap"]["per_horizon"],
        freeze["coefficients"]["per_horizon_scalar"],
    ):
        horizon_rows.append(
            [
                str(row["horizon"]),
                f"{alpha:.6f}",
                f"{row['differential_rmse']:.6f}",
                f"{row['zero_difference_rmse']:.6f}",
                f"{100*row['relative_improvement_over_zero']:.2f}%",
                f"[{boot['improvement_zero_ci95'][0]:.3f},{boot['improvement_zero_ci95'][1]:.3f}]",
                f"{row['magnitude_ratio']['ratio_of_means']:.3f}",
                f"{row['direction_cosine']['mean']:.3f}",
            ]
        )

    pair_rows = []
    for name in PAIR_NAMES:
        row = primary["differential"]["per_pair"][name]
        boot = primary["bootstrap"]["per_pair"][name]
        pair_rows.append(
            [
                name,
                f"{row['differential_rmse']:.6f}",
                f"{row['zero_difference_rmse']:.6f}",
                f"{100*row['relative_improvement_over_zero']:.2f}%",
                f"[{boot['improvement_zero_ci95'][0]:.3f},{boot['improvement_zero_ci95'][1]:.3f}]",
                f"{row['direction_cosine']['mean']:.3f}",
                f"{row['magnitude_ratio']['ratio_of_means']:.3f}",
                "pass" if all(decision["candidate_pair_checks"][name].values()) else "FAIL",
            ]
        )

    group_rows = []
    groupwise = forms["groupwise_scalar"]
    for name in GROUPS:
        row = groupwise["differential"]["groups"][name]
        sign = groupwise["sign"]["groups"][name]
        boot = groupwise["bootstrap"]["groups"][name]
        group_rows.append(
            [
                name,
                f"{freeze['coefficients']['groupwise_scalar'][name]:.6f}",
                f"{100*row['relative_improvement_over_zero']:.2f}%",
                f"[{boot['improvement_zero_ci95'][0]:.3f},{boot['improvement_zero_ci95'][1]:.3f}]",
                f"{row['direction_cosine']['mean']:.3f}",
                f"{row['magnitude_ratio']['ratio_of_means']:.3f}",
                "n/a" if sign["accuracy"] is None else f"{sign['accuracy']:.2%}",
                "n/a" if sign["eligible_fraction"] is None else f"{sign['eligible_fraction']:.2%}",
                "pass" if all(decision["partial_group_checks"][name].values()) else "FAIL",
            ]
        )

    task_rows = []
    for task in range(10):
        row = primary["differential"]["per_task"][str(task)]
        task_rows.append(
            [
                str(task),
                f"{row['differential_rmse']:.6f}",
                f"{row['zero_difference_rmse']:.6f}",
                f"{100*row['relative_improvement_over_zero']:.2f}%",
                f"{row['magnitude_ratio']['ratio_of_means']:.3f}",
                f"{row['direction_cosine']['mean']:.3f}",
            ]
        )

    gate_rows = [
        [name, "pass" if passed else "FAIL"]
        for name, passed in decision["primary_gate_pass"].items()
    ]
    overall = primary["differential"]["overall"]
    boot = primary["bootstrap"]["overall"]
    ranking = primary["ranking"]["overall"]
    historical_ratio = absolute["ratios_to_validation"]["historical_30_35"]["rmse"]
    calibration_ratio = absolute["ratios_to_validation"]["calibration_31_34"]["rmse"]
    evaluation_ratio = absolute["ratios_to_validation"]["evaluation_36_39"]["rmse"]
    return f"""# {decision['verdict'].replace('_', ' ').title()}

Date: 2026-09-24

Final verdict: `{decision['verdict']}`

WM-v2C authorization: `{decision['authorization']}`

Mechanism label(s): {', '.join(f'`{value}`' for value in decision['mechanism_labels'])}

## Executive summary

The frozen WM-v2B physical head was calibrated on init states 31--34 and
evaluated once on disjoint init states 36--39. The primary form was fixed
before evaluation as eight positive/negative-unconstrained per-horizon
closed-form scalars; no neural model was trained and no evaluation result was
used to choose a form.

On held-out evaluation, primary calibrated RMSE is
`{overall['differential_rmse']:.6f}` versus zero-difference
`{overall['zero_difference_rmse']:.6f}` and raw frozen prediction
`{overall['raw_rmse']:.6f}`. Improvement over zero is
`{100*overall['relative_improvement_over_zero']:.2f}%` with branch-block 95% CI
`[{boot['improvement_zero_ci95'][0]:.3f}, {boot['improvement_zero_ci95'][1]:.3f}]`;
improvement over raw is `{100*overall['improvement_over_raw']:.2f}%`.

Direction remains `{overall['direction_cosine']['mean']:.3f}` and separation
ranking remains Spearman `{ranking['within_context_spearman']['mean']:.3f}` /
ordering accuracy `{ranking['pairwise_ordering_accuracy']['mean']:.2%}`. The
mechanical gate therefore assigns `{decision['verdict']}` and keeps
`{decision['authorization']}`.

Policy-induced absolute physical-model shift replicates beyond historical
30/35: branch RMSE ratios to D13 validation are `{historical_ratio:.3f}x`
(historical), `{calibration_ratio:.3f}x` (31--34), and `{evaluation_ratio:.3f}x`
(36--39). Historical includes plan 0 and 8, while both new partitions use
plan 8 only; the new calibration/evaluation comparison is directly matched.

## Preregistration and isolation

- Protocol commit: `{result['protocol']['preregistered_commit']}`; SHA-256 `{result['protocol']['sha256']}`.
- Calibration alpha freeze commit: `{result['protocol']['calibration_freeze_commit']}`; freeze SHA-256 `{result['protocol']['calibration_freeze_sha256']}`.
- Calibration: init 31--34, 40 branch points, planning index 8, 64-step prefixes.
- Evaluation: init 36--39, 40 branch points, planning index 8, 64-step prefixes.
- Historical init 30/35 is reference-only. Init 40--49 was not accessed.
- Every branch used a fresh environment and exact prefix replay; no mid-episode restore, reward, done, success, or task completion signal was used.

All 80 new branch points passed bitwise RGB/observable/prefix identity,
zero-tolerance proprio identity, simulator-state tolerance `1e-10`, and CLIP
cosine `>=0.999999` before candidate execution.

## Frozen coefficients

Global alpha: `{freeze['coefficients']['global_scalar']:.9f}`.

Per-horizon alphas:

```text
{json.dumps(freeze['coefficients']['per_horizon_scalar'])}
```

Groupwise alphas:

```json
{json.dumps(freeze['coefficients']['groupwise_scalar'], indent=2)}
```

No alpha was clipped, sign-corrected, or iteratively tuned. Horizon×group
coefficients were also frozen from calibration data but are eligible only for
the preregistered secondary diagnostic and cannot authorize GO.

## Absolute error and distribution shift

{markdown_table(['Source', 'RMSE', 'MAE', 'SmoothL1', 'RMSE / validation'], absolute_rows)}

## Held-out calibration-form comparison

{markdown_table(['Form', 'RMSE', 'Zero', 'Raw', 'vs zero', 'zero CI', 'vs raw', 'norm ratio', 'direction', 'rank rho', 'ordering'], form_rows)}

`per_horizon_scalar` is the only full-authorization form. The other rows are
diagnostics and were not used to select a winner from held-out data.

## Primary per-horizon behavior

{markdown_table(['h', 'alpha', 'RMSE', 'zero', 'vs zero', '95% CI', 'norm ratio', 'direction'], horizon_rows)}

## Candidate-pair consistency

{markdown_table(['Pair', 'RMSE', 'zero', 'vs zero', '95% CI', 'direction', 'norm ratio', 'gate'], pair_rows)}

## Coordinate groups and gripper

The table uses the preregistered groupwise form for the partial-mediator gate.
Gripper is never hidden inside the aggregate.

{markdown_table(['Group', 'alpha', 'vs zero', '95% CI', 'direction', 'norm ratio', 'sign', 'sign coverage', 'partial gate'], group_rows)}

Primary-form gripper gate details:

```json
{json.dumps(decision['gripper_checks'], indent=2)}
```

## Per-task held-out primary results

{markdown_table(['Task', 'RMSE', 'zero', 'vs zero', 'norm ratio', 'direction'], task_rows)}

## Ranking fidelity

Primary-form within-context Spearman is
`{ranking['within_context_spearman']['mean']:.4f}` with 95% CI
`[{boot['within_context_spearman_ci95'][0]:.4f}, {boot['within_context_spearman_ci95'][1]:.4f}]`.
Pair-of-pair ordering accuracy is
`{ranking['pairwise_ordering_accuracy']['mean']:.2%}` with 95% CI
`[{boot['ordering_accuracy_ci95'][0]:.4f}, {boot['ordering_accuracy_ci95'][1]:.4f}]`.
Pooled predicted/real separation Pearson/Spearman are
`{ranking['pooled']['pearson']:.4f}` / `{ranking['pooled']['spearman']:.4f}`.

## Mechanical decision

{markdown_table(['Primary gate', 'Result'], gate_rows)}

Candidate-pair checks:

```json
{json.dumps(decision['candidate_pair_checks'], indent=2)}
```

Partial group checks:

```json
{json.dumps(decision['partial_group_checks'], indent=2)}
```

Final verdict: `{decision['verdict']}`. WM-v2C authorization remains
`{decision['authorization']}`. This experiment does not design or train WM-v2C.

## Integrity and forbidden-activity audit

- Result: `{result['artifacts']['result']}`, SHA-256 `{result_hash}`.
- Derived sidecar: `{result['artifacts']['derived_sidecar']}`, SHA-256 `{result['artifacts']['derived_sidecar_sha256']}`.
- Calibration sidecar SHA-256: `{result['artifacts']['calibration_sidecar_sha256']}`.
- Evaluation sidecar SHA-256: `{result['artifacts']['evaluation_sidecar_sha256']}`.
- NaN/Inf audit: `{result['nan_inf_audit']}`.
- No WM-v2B retraining/modification, neural calibration, evaluation fitting,
  controller/candidate/reranker change, C1/C2 success run, D20-v2, reward/done,
  task-success selection, fake target, WM-v2C training, or init 40--49 access occurred.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/wm_v2b_1_differential_calibration_protocol.json",
    )
    args = parser.parse_args()
    protocol, freeze, manifests = load_and_validate(args.protocol)
    artifacts = protocol["artifacts"]
    output = ROOT / artifacts["result"]
    derived_path = ROOT / artifacts["derived_sidecar"]
    report_path = ROOT / artifacts["report"]
    for path in (output, derived_path, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    checkpoint = torch.load(
        ROOT / protocol["frozen_upstream"]["wm_v2b_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    p_mean = np.asarray(checkpoint["normalization"]["proprio"]["mean"], dtype=np.float64)
    p_std = np.asarray(checkpoint["normalization"]["proprio"]["std"], dtype=np.float64)
    validation = load_npz(ROOT / protocol["frozen_upstream"]["validation_sidecar"])
    historical = load_npz(ROOT / protocol["frozen_upstream"]["historical_branch_sidecar"])
    calibration = load_npz(ROOT / artifacts["calibration_sidecar"])
    evaluation = load_npz(ROOT / artifacts["evaluation_sidecar"])
    if set(calibration["init_state_index"].tolist()) != {31, 32, 33, 34}:
        raise ValueError("calibration identities changed")
    if set(evaluation["init_state_index"].tolist()) != {36, 37, 38, 39}:
        raise ValueError("evaluation identities changed")
    if set(evaluation["planning_index"].tolist()) != {8}:
        raise ValueError("evaluation planning index changed")

    validation_absolute = absolute_from_smooth_l1(
        validation["correct_p_element_error"].astype(np.float64)
    )
    if not np.allclose(
        np.sqrt(np.mean(validation_absolute**2, axis=-1)),
        validation["correct_p_rmse"],
        atol=2e-7,
        rtol=0,
    ):
        raise ValueError("validation SmoothL1 inversion failed")
    absolute_sources = {
        "validation": absolute_breakdown(
            validation_absolute, validation["task_id"].astype(np.int64)
        ),
        "historical_30_35": absolute_breakdown(
            branch_absolute(historical, p_mean, p_std).reshape(-1, 8, 12),
            np.repeat(historical["task_id"].astype(np.int64), 3),
        ),
        "calibration_31_34": absolute_breakdown(
            branch_absolute(calibration, p_mean, p_std).reshape(-1, 8, 12),
            np.repeat(calibration["task_id"].astype(np.int64), 3),
        ),
        "evaluation_36_39": absolute_breakdown(
            branch_absolute(evaluation, p_mean, p_std).reshape(-1, 8, 12),
            np.repeat(evaluation["task_id"].astype(np.int64), 3),
        ),
    }
    validation_metrics = absolute_sources["validation"]["overall"]
    ratios = {
        name: {
            "rmse": values["overall"]["rmse"] / validation_metrics["rmse"],
            "mae": values["overall"]["mae"] / validation_metrics["mae"],
            "smooth_l1": values["overall"]["smooth_l1_beta1"]
            / validation_metrics["smooth_l1_beta1"],
        }
        for name, values in absolute_sources.items()
        if name != "validation"
    }
    absolute_sources["ratios_to_validation"] = ratios

    real_standardized = standardize_real(evaluation, p_mean, p_std)
    raw_future = evaluation["pred_proprio_wm_v2b_standardized"].astype(np.float64)
    real_delta = pairwise_delta(real_standardized)
    raw_delta = pairwise_delta(raw_future)
    task_id = evaluation["task_id"].astype(np.int64)
    forms: dict[str, Any] = {}
    derived: dict[str, np.ndarray] = {
        "task_id": task_id.astype(np.int8),
        "init_state_index": evaluation["init_state_index"].astype(np.int8),
        "planning_index": evaluation["planning_index"].astype(np.int8),
        "real_delta": real_delta.astype(np.float32),
        "raw_delta": raw_delta.astype(np.float32),
    }
    simple_forms = ("global_scalar", "per_horizon_scalar", "groupwise_scalar")
    for index, form_name in enumerate(simple_forms):
        calibrated = apply_form(raw_delta, form_name, freeze["coefficients"][form_name])
        form_result, form_arrays = form_analysis(
            calibrated, raw_delta, real_delta, task_id, protocol, index * 1000
        )
        forms[form_name] = form_result
        for name, value in form_arrays.items():
            derived[f"{form_name}_{name}"] = value

    simple_aggregate_pass = {
        name: aggregate_gate(forms[name]) for name in simple_forms
    }
    secondary_evaluated = not any(simple_aggregate_pass.values())
    if secondary_evaluated:
        calibrated = apply_form(
            raw_delta,
            "horizon_group_scalar",
            freeze["coefficients"]["horizon_group_scalar"],
        )
        form_result, form_arrays = form_analysis(
            calibrated, raw_delta, real_delta, task_id, protocol, 3000
        )
        forms["horizon_group_scalar_secondary"] = form_result
        for name, value in form_arrays.items():
            derived[f"horizon_group_scalar_secondary_{name}"] = value

    decision = make_decision(forms, freeze)
    analysis = {
        "absolute_error": absolute_sources,
        "forms": forms,
        "simple_aggregate_zero_gate_pass": simple_aggregate_pass,
        "secondary_horizon_group_evaluated": secondary_evaluated,
        "decision": decision,
    }
    if not all(np.isfinite(value).all() for value in derived.values()):
        raise FloatingPointError("derived arrays contain NaN/Inf")
    atomic_npz(derived_path, derived)

    protocol_path = args.protocol
    freeze_path = ROOT / artifacts["calibration_freeze"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_version": protocol["protocol_version"],
        "status": "COMPLETE",
        "verdict": decision["verdict"],
        "wm_v2c_authorization": decision["authorization"],
        "mechanism_labels": decision["mechanism_labels"],
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": sha256(protocol_path),
            "preregistered_commit": path_last_commit(protocol_path),
            "execution_source_commit": source_commit(),
            "calibration_freeze": str(freeze_path.relative_to(ROOT)),
            "calibration_freeze_sha256": sha256(freeze_path),
            "calibration_freeze_commit": path_last_commit(freeze_path),
        },
        "scope": {
            "calibration_init_states": [31, 32, 33, 34],
            "evaluation_init_states": [36, 37, 38, 39],
            "historical_reference_init_states": [30, 35],
            "planning_index": 8,
            "prefix_steps": 64,
            "calibration_branch_points": 40,
            "evaluation_branch_points": 40,
            "candidate_branches_per_partition": 120,
            "horizon": 8,
            "init_state_40_49_accessed": False,
            "success_reward_done_used": False,
        },
        "calibration_freeze": freeze,
        "analysis": analysis,
        "artifacts": {
            "result": str(output.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
            "calibration_manifest_sha256": sha256(
                ROOT / artifacts["calibration_manifest"]
            ),
            "calibration_sidecar_sha256": sha256(
                ROOT / artifacts["calibration_sidecar"]
            ),
            "evaluation_manifest_sha256": sha256(
                ROOT / artifacts["evaluation_manifest"]
            ),
            "evaluation_sidecar_sha256": sha256(
                ROOT / artifacts["evaluation_sidecar"]
            ),
            "derived_sidecar": str(derived_path.relative_to(ROOT)),
            "derived_sidecar_sha256": sha256(derived_path),
            "derived_sidecar_arrays": {
                name: list(value.shape) for name, value in derived.items()
            },
        },
        "nan_inf_audit": "pass",
        "forbidden_activity_audit": {
            "wm_v2b_retrained_or_modified": False,
            "neural_calibration_model": False,
            "evaluation_used_for_fit_or_form_selection": False,
            "alpha_clipped_sign_corrected_or_iteratively_tuned": False,
            "candidate_scale_generator_or_reranker_modified": False,
            "controller_success_c1_c2_or_d20_v2_run": False,
            "task_success_reward_done_or_fake_target_used": False,
            "wm_v2c_designed_or_trained": False,
            "init_state_40_49_access": False,
            "upstream_artifact_overwritten": False,
        },
    }
    if not finite_tree(result):
        raise FloatingPointError("result contains NaN/Inf")
    atomic_json(output, result)
    result["artifacts"]["result_sha256_before_hash_field"] = sha256(output)
    atomic_json(output, result)
    atomic_text(report_path, render_report(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["verdict"],
                "authorization": result["wm_v2c_authorization"],
                "mechanism_labels": result["mechanism_labels"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
