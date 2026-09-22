#!/usr/bin/env python3
"""D20 one-time validation-only mismatch/uncertainty threshold selection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.world_models.normalization import file_sha256


def evaluate_pair(
    mismatch: np.ndarray,
    uncertainty: np.ndarray,
    mismatch_threshold: float,
    uncertainty_threshold: float,
) -> dict[str, float | int]:
    mismatch_trigger = mismatch > mismatch_threshold
    uncertainty_trigger = uncertainty > uncertainty_threshold
    combined = mismatch_trigger | uncertainty_trigger
    return {
        "check_count": int(len(mismatch)),
        "trigger_count": int(combined.sum()),
        "nominal_or_trigger_rate": float(combined.mean()),
        "mismatch_only_rate": float((mismatch_trigger & ~uncertainty_trigger).mean()),
        "uncertainty_only_rate": float((uncertainty_trigger & ~mismatch_trigger).mean()),
        "both_rate": float((mismatch_trigger & uncertainty_trigger).mean()),
    }


def choose_grid_row(rows: list[dict[str, Any]], target_rate: float) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if 0.0 < row["nominal_or_trigger_rate"] <= target_rate + 1.0e-12
    ]
    if not eligible:
        raise ValueError("no nonzero grid point satisfies the nominal trigger-rate target")
    return max(
        eligible,
        key=lambda row: (
            row["nominal_or_trigger_rate"],
            -(row["mismatch_quantile"] + row["uncertainty_quantile"]),
            -row["mismatch_quantile"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/d20_threshold_selection.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/threshold_selection.json"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["contract_version"] != "d20-v1":
        raise ValueError("expected D20-v1 threshold configuration")
    horizons = [int(value) for value in config["check_horizons"]]
    if horizons != [2, 4, 6, 8]:
        raise ValueError("D20-v1 freezes verification horizons h2/h4/h6/h8")
    transition_path = PROJECT_ROOT / config["source_transition_eval"]
    transition = json.loads(transition_path.read_text())
    sample_path = PROJECT_ROOT / config["source_sample_records"]
    if transition["status"] != "PASS" or file_sha256(sample_path) != transition["sample_records"]["sha256"]:
        raise ValueError("D16 source provenance mismatch")
    source_rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
    points: list[dict[str, Any]] = []
    for row in source_rows:
        if row["demo_id"] not in range(45, 50):
            raise ValueError("threshold source includes a non-validation demo")
        for horizon in horizons:
            index = horizon - 1
            points.append(
                {
                    "task_id": int(row["task_id"]),
                    "demo_id": int(row["demo_id"]),
                    "start_timestep": int(row["start_timestep"]),
                    "horizon": horizon,
                    "mismatch": float(row["learned_cosine_error"][index]),
                    "uncertainty": float(row["uncertainty"][index]),
                }
            )
    mismatch = np.asarray([point["mismatch"] for point in points], dtype=np.float64)
    uncertainty = np.asarray([point["uncertainty"] for point in points], dtype=np.float64)
    if len(points) != 5869 * 4 or not np.isfinite(mismatch).all() or not np.isfinite(uncertainty).all():
        raise ValueError("invalid D20 validation check points")
    quantiles = [float(value) for value in config["threshold_quantiles"]]
    mismatch_values = {quantile: float(np.quantile(mismatch, quantile)) for quantile in quantiles}
    uncertainty_values = {
        quantile: float(np.quantile(uncertainty, quantile)) for quantile in quantiles
    }
    grid = []
    for mismatch_quantile in quantiles:
        for uncertainty_quantile in quantiles:
            metrics = evaluate_pair(
                mismatch,
                uncertainty,
                mismatch_values[mismatch_quantile],
                uncertainty_values[uncertainty_quantile],
            )
            grid.append(
                {
                    "mismatch_quantile": mismatch_quantile,
                    "uncertainty_quantile": uncertainty_quantile,
                    "mismatch_threshold": mismatch_values[mismatch_quantile],
                    "uncertainty_threshold": uncertainty_values[uncertainty_quantile],
                    **metrics,
                }
            )
    target_rate = float(config["target_nominal_or_trigger_rate"])
    selected = choose_grid_row(grid, target_rate)
    trigger = (mismatch > selected["mismatch_threshold"]) | (
        uncertainty > selected["uncertainty_threshold"]
    )
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    task_trigger: dict[int, list[int]] = defaultdict(list)
    horizon_trigger: dict[int, list[int]] = defaultdict(list)
    for point, triggered in zip(points, trigger.tolist()):
        grouped[(point["task_id"], point["demo_id"])].append(int(triggered))
        task_trigger[point["task_id"]].append(int(triggered))
        horizon_trigger[point["horizon"]].append(int(triggered))
    episode_records = [
        {
            "task_id": task_id,
            "demo_id": demo_id,
            "check_count": len(values),
            "trigger_count": sum(values),
            "trigger_rate": float(np.mean(values)),
            "any_trigger": bool(any(values)),
        }
        for (task_id, demo_id), values in sorted(grouped.items())
    ]

    # D19 fixed-test events are never used above. This post-selection diagnostic
    # only estimates how often the chosen uncertainty threshold is already
    # exceeded by C1's selected OOD candidates before D21 is implemented.
    d19_path = PROJECT_ROOT / "results/c1/seed11/task0_d19.json"
    d19 = json.loads(d19_path.read_text())
    d19_uncertainty = []
    for episode in d19["episode_results"]:
        for event in episode["plan_events"]:
            selected_index = int(event["selected_candidate"])
            # D19 stores mean h1-h8 uncertainty per candidate, not each horizon.
            d19_uncertainty.append(float(event["mean_uncertainty"][selected_index]))
    d19_rate = float(
        np.mean(np.asarray(d19_uncertainty) > selected["uncertainty_threshold"])
    )
    result = {
        "schema_version": 1,
        "contract_version": "d20-v1",
        "status": "PASS",
        "config": str(args.config.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(args.config),
        "d16_result_sha256": file_sha256(transition_path),
        "d16_samples_sha256": file_sha256(sample_path),
        "selection_data": "D13 validation demos 45-49 only; no fixed simulator evaluation episode",
        "check_horizons": horizons,
        "source_sample_count": len(source_rows),
        "check_point_count": len(points),
        "episode_count": len(episode_records),
        "task_count": len(task_trigger),
        "target_nominal_or_trigger_rate": target_rate,
        "trigger_rule": config["trigger_rule"],
        "selection_rule": config["selection_rule"],
        "label_semantics": "nominal expert-transition trigger calibration; no failure/deviation labels",
        "mismatch_quantile_thresholds": {str(key): value for key, value in mismatch_values.items()},
        "uncertainty_quantile_thresholds": {
            str(key): value for key, value in uncertainty_values.items()
        },
        "grid": grid,
        "selected": selected,
        "selected_breakdown": {
            "episode_any_trigger_fraction": float(
                np.mean([record["any_trigger"] for record in episode_records])
            ),
            "mean_triggers_per_episode": float(
                np.mean([record["trigger_count"] for record in episode_records])
            ),
            "per_horizon_trigger_rate": {
                str(horizon): float(np.mean(horizon_trigger[horizon])) for horizon in horizons
            },
            "per_task_trigger_rate": {
                str(task_id): float(np.mean(task_trigger[task_id])) for task_id in range(10)
            },
            "episodes": episode_records,
        },
        "post_selection_excluded_d19_diagnostic": {
            "used_for_selection": False,
            "source": str(d19_path.relative_to(PROJECT_ROOT)),
            "source_sha256": file_sha256(d19_path),
            "available_statistic": "selected-candidate mean h1-h8 uncertainty per plan",
            "plan_count": len(d19_uncertainty),
            "fraction_above_selected_uncertainty_threshold": d19_rate,
            "warning": "test-rollout distribution shift diagnostic; must not alter selected thresholds",
        },
        "nan_inf_audit": "pass",
        "d21_authorized": True,
    }
    if selected["nominal_or_trigger_rate"] > target_rate + 1.0e-12:
        raise RuntimeError("selected threshold violates the frozen nominal trigger-rate target")
    if not all(math.isfinite(float(value)) for value in (selected["mismatch_threshold"], selected["uncertainty_threshold"])):
        raise FloatingPointError("selected threshold is NaN/Inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "check_point_count", "target_nominal_or_trigger_rate", "selected", "selected_breakdown", "post_selection_excluded_d19_diagnostic")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
