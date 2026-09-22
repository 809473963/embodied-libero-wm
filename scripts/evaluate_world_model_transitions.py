#!/usr/bin/env python3
"""D16 formal held-out copy-current versus learned-predictor comparison."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_world_model_ensemble import load_ensemble
from src.d13 import WorldModelDataset
from src.world_models import apply_world_model_normalization, cosine_distance
from src.world_models.normalization import file_sha256


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def paired_episode_bootstrap(
    episode_deltas: dict[int, list[float]], replicates: int, seed: int
) -> dict[str, float | int | str]:
    """Task-stratified bootstrap of paired episode-level improvements."""

    if sorted(episode_deltas) != list(range(10)):
        raise ValueError("bootstrap requires all ten tasks")
    if any(len(values) != 5 for values in episode_deltas.values()):
        raise ValueError("bootstrap requires five held-out episodes per task")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    arrays = {task_id: np.asarray(values, dtype=np.float64) for task_id, values in episode_deltas.items()}
    for index in range(replicates):
        task_means = []
        for task_id in range(10):
            values = arrays[task_id]
            selected = rng.integers(0, len(values), size=len(values))
            task_means.append(float(values[selected].mean()))
        draws[index] = np.mean(task_means)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "unit": "episode",
        "stratified_by": "task",
        "replicates": replicates,
        "seed": seed,
        "mean": float(draws.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def decide_gate(
    overall_improvement: float,
    per_horizon_improvement: list[float],
    bootstrap_ci_lower: float,
) -> dict[str, Any]:
    checks = {
        "positive_overall_improvement": overall_improvement > 0.0,
        "positive_episode_bootstrap_ci_lower": bootstrap_ci_lower > 0.0,
        "every_horizon_improves": all(value > 0.0 for value in per_horizon_improvement),
        "h8_improvement_greater_than_h1": per_horizon_improvement[-1]
        > per_horizon_improvement[0],
    }
    return {"checks": checks, "pass": all(checks.values())}


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/d16_transition_eval.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/wm/transition_eval.json"
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        default=PROJECT_ROOT / "results/wm/transition_eval_samples.jsonl",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["contract_version"] != "d16-v1" or int(config["horizon"]) != 8:
        raise ValueError("expected frozen D16-v1 H=8 evaluation")
    ensemble_config_path = PROJECT_ROOT / config["ensemble_config"]
    ensemble_config = yaml.safe_load(ensemble_config_path.read_text())
    checkpoint_paths = [PROJECT_ROOT / path for path in ensemble_config["member_checkpoints"]]
    device = torch.device(args.device)
    ensemble, contract, members = load_ensemble(checkpoint_paths, device)
    normalization = contract["normalization"]
    store_path = PROJECT_ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5"
    if file_sha256(store_path) != contract["d13_store_sha256"]:
        raise ValueError("D13 store hash differs from ensemble checkpoint contract")
    dataset = WorldModelDataset(store_path, split="validation", horizon=8)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    learned_sum = torch.zeros(8, dtype=torch.float64)
    copy_sum = torch.zeros(8, dtype=torch.float64)
    rows: list[dict[str, Any]] = []
    try:
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            proprio, actions = apply_world_model_normalization(
                batch["proprio0"], batch["actions"], normalization
            )
            learned_prediction, uncertainty, _ = ensemble.predict(
                batch["z0"], proprio, batch["language"], actions
            )
            copy_prediction = batch["z0"].unsqueeze(1).expand(-1, 8, -1)
            learned_error = cosine_distance(learned_prediction, batch["z_future"])
            copy_error = cosine_distance(copy_prediction, batch["z_future"])
            if not all(
                torch.isfinite(value).all()
                for value in (learned_prediction, uncertainty, learned_error, copy_error)
            ):
                raise FloatingPointError("D16 evaluation contains NaN/Inf")
            learned_sum += learned_error.double().sum(dim=0).cpu()
            copy_sum += copy_error.double().sum(dim=0).cpu()
            for index in range(learned_error.shape[0]):
                learned_values = learned_error[index].cpu().tolist()
                copy_values = copy_error[index].cpu().tolist()
                rows.append(
                    {
                        "task_id": int(batch["task_id"][index]),
                        "demo_id": int(batch["demo_id"][index]),
                        "start_timestep": int(batch["start_timestep"][index]),
                        "learned_cosine_error": learned_values,
                        "copy_cosine_error": copy_values,
                        "copy_minus_learned": [
                            copy_value - learned_value
                            for copy_value, learned_value in zip(copy_values, learned_values)
                        ],
                        "uncertainty": uncertainty[index].cpu().tolist(),
                    }
                )
    finally:
        dataset.close()
    if len(rows) != 5869:
        raise RuntimeError(f"expected 5869 validation samples, got {len(rows)}")

    args.samples_output.parent.mkdir(parents=True, exist_ok=True)
    with args.samples_output.open("w") as destination:
        for row in rows:
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")

    count = len(rows)
    learned_per_horizon = (learned_sum / count).tolist()
    copy_per_horizon = (copy_sum / count).tolist()
    improvement_per_horizon = [
        copy_value - learned_value
        for copy_value, learned_value in zip(copy_per_horizon, learned_per_horizon)
    ]
    learned_overall = float(np.mean(learned_per_horizon))
    copy_overall = float(np.mean(copy_per_horizon))
    overall_improvement = copy_overall - learned_overall

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], row["demo_id"])].append(row)
    if len(grouped) != 50:
        raise RuntimeError(f"expected 50 held-out episodes, got {len(grouped)}")
    episode_records: list[dict[str, Any]] = []
    episode_deltas: dict[int, list[float]] = defaultdict(list)
    for (task_id, demo_id), episode_rows in sorted(grouped.items()):
        learned = float(
            np.mean([value for row in episode_rows for value in row["learned_cosine_error"]])
        )
        copy = float(
            np.mean([value for row in episode_rows for value in row["copy_cosine_error"]])
        )
        delta = copy - learned
        episode_deltas[task_id].append(delta)
        episode_records.append(
            {
                "task_id": task_id,
                "demo_id": demo_id,
                "sample_count": len(episode_rows),
                "learned_cosine_error": learned,
                "copy_cosine_error": copy,
                "copy_minus_learned": delta,
            }
        )
    task_records = []
    for task_id in range(10):
        task_rows = [row for row in rows if row["task_id"] == task_id]
        task_learned = float(
            np.mean([value for row in task_rows for value in row["learned_cosine_error"]])
        )
        task_copy = float(
            np.mean([value for row in task_rows for value in row["copy_cosine_error"]])
        )
        task_records.append(
            {
                "task_id": task_id,
                "sample_count": len(task_rows),
                "learned_cosine_error": task_learned,
                "copy_cosine_error": task_copy,
                "copy_minus_learned": task_copy - task_learned,
            }
        )
    bootstrap_config = config["bootstrap"]
    bootstrap = paired_episode_bootstrap(
        dict(episode_deltas),
        int(bootstrap_config["replicates"]),
        int(bootstrap_config["seed"]),
    )
    gate = decide_gate(
        overall_improvement, improvement_per_horizon, float(bootstrap["ci95_lower"])
    )
    result = {
        "schema_version": 1,
        "contract_version": "d16-v1",
        "status": "PASS" if gate["pass"] else "BLOCK_D17",
        "config": str(args.config.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(args.config),
        "ensemble_config_sha256": file_sha256(ensemble_config_path),
        "d13_data_contract_version": contract["d13_data_contract_version"],
        "d13_store_sha256": contract["d13_store_sha256"],
        "split": "validation demos 45-49; no simulator evaluation/test episodes",
        "sample_count": count,
        "episode_count": len(episode_records),
        "member_seeds": [member["seed"] for member in members],
        "learned": {
            "predictor": "D15 normalized ensemble mean",
            "overall_cosine_error": learned_overall,
            "per_horizon_cosine_error": learned_per_horizon,
        },
        "copy_current": {
            "predictor": "z0 repeated for all eight horizons",
            "overall_cosine_error": copy_overall,
            "per_horizon_cosine_error": copy_per_horizon,
        },
        "paired_improvement": {
            "definition": "copy cosine error minus learned cosine error; positive favors learned",
            "absolute_overall": overall_improvement,
            "relative_overall": overall_improvement / copy_overall,
            "per_horizon": improvement_per_horizon,
            "sample_fraction_improved": float(
                np.mean(
                    [
                        np.mean(row["copy_cosine_error"])
                        > np.mean(row["learned_cosine_error"])
                        for row in rows
                    ]
                )
            ),
            "episode_fraction_improved": float(
                np.mean([row["copy_minus_learned"] > 0.0 for row in episode_records])
            ),
            "task_fraction_improved": float(
                np.mean([row["copy_minus_learned"] > 0.0 for row in task_records])
            ),
            "episode_block_bootstrap": bootstrap,
        },
        "tasks": task_records,
        "episodes": episode_records,
        "gate": gate,
        "sample_records": {
            "path": str(args.samples_output.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(args.samples_output),
            "count": count,
        },
        "nan_inf_audit": "pass",
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not gate["pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
