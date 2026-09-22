#!/usr/bin/env python3
"""D18 end-to-end audit of the offline world-model foundation."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.d13.dataset import TRAIN_DEMO_IDS, VALIDATION_DEMO_IDS
from src.world_models.normalization import ACTION_RULE, PROPRIO_RULE, file_sha256


STORE = PROJECT_ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5"
CHECKPOINTS = [
    PROJECT_ROOT / "results/world_model/d14_single_seed11/best.pt",
    PROJECT_ROOT / "results/world_model/d15_member_seed23/best.pt",
    PROJECT_ROOT / "results/world_model/d15_member_seed47/best.pt",
]


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def main() -> int:
    checks: dict[str, dict[str, Any]] = {}
    with h5py.File(STORE, "r") as store:
        fields = set()
        train_episodes = set()
        validation_episodes = set()
        sample_counts = {"train": 0, "validation": 0}
        observation_counts = {"train": 0, "validation": 0}
        for task_group in store["tasks"].values():
            task_id = int(task_group.attrs["task_id"])
            for demo_group in task_group["demos"].values():
                demo_id = int(demo_group.attrs["demo_id"])
                fields.update(demo_group.keys())
                count = int(demo_group.attrs["observation_count"])
                split = "train" if demo_id in TRAIN_DEMO_IDS else "validation"
                target = train_episodes if split == "train" else validation_episodes
                target.add((task_id, demo_id))
                observation_counts[split] += count
                sample_counts[split] += count - 8
        checks["d13_store_contract"] = {
            "pass": str(store.attrs["world_model_data_contract"]) == "d13-v1"
            and int(store.attrs["horizon"]) == 8
            and fields == {"vision", "proprio", "actions", "timestep"},
            "store_sha256": file_sha256(STORE),
            "fields": sorted(fields),
            "observations": observation_counts,
            "samples": sample_counts,
        }
        checks["episode_split"] = {
            "pass": not (train_episodes & validation_episodes)
            and len(train_episodes) == 450
            and len(validation_episodes) == 50,
            "train_episodes": len(train_episodes),
            "validation_episodes": len(validation_episodes),
            "overlap": len(train_episodes & validation_episodes),
        }

    checkpoint_records = []
    reference = None
    for path in CHECKPOINTS:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model_finite = all(torch.isfinite(value).all() for value in checkpoint["model_state"].values())
        optimizer_tensors = [
            value
            for state in checkpoint["optimizer_state"]["state"].values()
            for value in state.values()
            if torch.is_tensor(value)
        ]
        optimizer_finite = all(torch.isfinite(value).all() for value in optimizer_tensors)
        comparable = {
            "contract": checkpoint["d13_data_contract_version"],
            "store": checkpoint["d13_store_sha256"],
            "contract_hash": checkpoint["d13_data_contract_sha256"],
            "normalization": checkpoint["normalization"],
            "model": checkpoint["config"]["model"],
        }
        reference = comparable if reference is None else reference
        checkpoint_records.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "seed": int(checkpoint["config"]["run"]["seed"]),
                "sha256": file_sha256(path),
                "best_validation": float(checkpoint["best_validation"]),
                "model_finite": bool(model_finite),
                "optimizer_finite": bool(optimizer_finite),
                "contract_matches": comparable == reference,
            }
        )
    normalization = reference["normalization"]
    checks["normalization"] = {
        "pass": normalization["proprio"]["count"] == 55981
        and normalization["action"]["count"] == 55531
        and normalization["proprio"]["rule"] == PROPRIO_RULE
        and normalization["action"]["rule"] == ACTION_RULE,
        "proprio_count": normalization["proprio"]["count"],
        "action_count": normalization["action"]["count"],
        "action0_excluded": True,
        "overlapping_windows_expanded": False,
    }
    checks["ensemble_checkpoints"] = {
        "pass": [record["seed"] for record in checkpoint_records] == [11, 23, 47]
        and all(record["model_finite"] and record["optimizer_finite"] for record in checkpoint_records)
        and all(record["contract_matches"] for record in checkpoint_records),
        "members": checkpoint_records,
    }

    transition_path = PROJECT_ROOT / "results/wm/transition_eval.json"
    transition = json.loads(transition_path.read_text())
    sample_path = PROJECT_ROOT / transition["sample_records"]["path"]
    checks["held_out_predictive_gate"] = {
        "pass": transition["status"] == "PASS"
        and transition["gate"]["pass"]
        and file_sha256(sample_path) == transition["sample_records"]["sha256"],
        "learned_cosine_error": transition["learned"]["overall_cosine_error"],
        "copy_cosine_error": transition["copy_current"]["overall_cosine_error"],
        "relative_improvement": transition["paired_improvement"]["relative_overall"],
        "episode_bootstrap_ci95": [
            transition["paired_improvement"]["episode_block_bootstrap"]["ci95_lower"],
            transition["paired_improvement"]["episode_block_bootstrap"]["ci95_upper"],
        ],
        "all_horizons_improve": transition["gate"]["checks"]["every_horizon_improves"],
        "all_tasks_improve": transition["paired_improvement"]["task_fraction_improved"] == 1.0,
    }

    review_path = PROJECT_ROOT / "artifacts/stage2/d17_review/review_samples.json"
    review = json.loads(review_path.read_text())
    correct = review["samples"]["correct"]
    incorrect = review["samples"]["incorrect"]
    finite_review = all(
        math.isfinite(float(sample["learned_mean_error"]))
        and all(math.isfinite(float(value)) for value in sample["learned_cosine_error"])
        for sample in correct + incorrect
    )
    checks["qualitative_review"] = {
        "pass": len(correct) == 20
        and len(incorrect) == 20
        and {sample["task_id"] for sample in correct} == set(range(10))
        and {sample["task_id"] for sample in incorrect} == set(range(10))
        and finite_review,
        "low_error_samples": len(correct),
        "high_error_samples": len(incorrect),
        "review_sha256": file_sha256(review_path),
        "interpretation": "mismatch relates to transition difficulty but is not a control-failure label",
    }

    c0_diff = git_output(
        "diff",
        "--name-only",
        "pre-d13-foundation..HEAD",
        "--",
        "src/c0",
        "src/policies/c0_actor_policy.py",
        "scripts/train_c0.py",
        "configs/c0_actor.yaml",
    )
    checks["c0_unchanged"] = {"pass": c0_diff == "", "changed_paths": c0_diff.splitlines()}
    all_offline_checks_pass = all(check["pass"] for check in checks.values())
    result = {
        "schema_version": 1,
        "day": "D18",
        "source_commit": git_output("rev-parse", "HEAD"),
        "offline_world_model_gate": "PASS" if all_offline_checks_pass else "BLOCK",
        "formal_project_g2": "PENDING_C1_OUTPUT_D19",
        "ordering_note": (
            "The stage table includes C1 output in final G2, while the day plan implements C1 in D19. "
            "D18 can authorize D19 but cannot claim final project G2 or authorize C2."
        ),
        "proceed_to_d19": all_offline_checks_pass,
        "c2_authorized": False,
        "checks": checks,
        "known_limits": [
            "uncertainty/error Pearson correlation is diagnostic, not calibrated",
            "D17 found static high-error and dynamic low-error counterexamples",
            "no online threshold has been selected",
            "no C1 or C2 control result exists yet",
        ],
    }
    output = PROJECT_ROOT / "configs/d18_g2_checklist.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if all_offline_checks_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
