#!/usr/bin/env python3
"""Run the preregistered robust same-scene referent grounding audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
sys.path.insert(0, str(ROOT))

from scripts.evaluate_c0_all import make_env  # noqa: E402
from scripts.run_semantic_teacher_clean import disable_cameras  # noqa: E402
from src.data.allowed_demo_guard import AllowedDemoGuard  # noqa: E402
from src.probes.robust_grounding import (  # noqa: E402
    BOWLS,
    FAMILIES,
    compatibility,
    exact_resolve,
    exact_source_match,
    maximum_score,
    perturbation_vectors,
    resolve_scores,
    threshold_grid,
)
from src.probes.semantic_teacher_clean import atomic_json, atomic_npz, sha256  # noqa: E402


VERSION = "robust-grounding-audit-v1"


def last_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip()


def validate_protocol(path: Path, partition: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != VERSION or protocol["status"] != "PREREGISTERED":
        raise RuntimeError("invalid or unfrozen grounding protocol")
    if not last_commit(path):
        raise RuntimeError("protocol must be committed before data access")
    for item in protocol["frozen_files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen file changed: {item['path']}")
    access = json.loads((ROOT / protocol["inputs"]["allowed_demo_access"]).read_text())
    contracts = json.loads((ROOT / protocol["inputs"]["task_grounding_contract"]).read_text())["tasks"]
    if partition == "evaluation":
        for key in ("calibration", "frozen_resolver"):
            target = ROOT / protocol["outputs"][key]
            if not target.exists() or not last_commit(target):
                raise RuntimeError("Stage G calibration and resolver must be committed before evaluation")
            if subprocess.check_output(["git", "status", "--porcelain", "--", str(target.relative_to(ROOT))], cwd=ROOT, text=True).strip():
                raise RuntimeError("Stage G calibration freeze is dirty")
    return protocol, access, contracts


def collect(protocol: dict[str, Any], access: dict[str, Any], contracts: list[dict[str, Any]], partition: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    guard = AllowedDemoGuard(partition, {int(k): ROOT / v for k, v in access["task_files"].items()}, ["states"])
    shifts = perturbation_vectors()
    task_ids, demo_ids = [], []
    exact_rows, score_rows, detail_rows, perturb_rows = [], [], [], []
    for task_id, row in enumerate(contracts):
        env = make_env(ROOT / row["bddl_path"])
        env.seed(92000 + task_id)
        env.reset()
        disable_cameras(env)
        try:
            for demo_id in guard.allowed_demo_ids:
                states = guard.read_dataset(task_id, demo_id, "states")
                if states.ndim != 2 or not np.isfinite(states).all():
                    raise ValueError("invalid allowed state array")
                env.regenerate_obs_from_state(states[0])
                exact = np.zeros((10, 2), dtype=np.bool_)
                scores = np.zeros((10, len(FAMILIES), 2), dtype=np.float64)
                details = np.zeros((10, 2, 2), dtype=np.float64)
                perturbed = np.zeros((10, len(FAMILIES), len(shifts), 2), dtype=np.float64)
                for instruction, contract in enumerate(contracts):
                    for bowl_index, bowl in enumerate(BOWLS):
                        exact[instruction, bowl_index] = exact_source_match(env.env, contract, bowl)
                        for family_index, family in enumerate(FAMILIES):
                            score, detail = compatibility(env.env, contract, bowl, family)
                            scores[instruction, family_index, bowl_index] = score
                            if family_index == 1:
                                details[instruction, bowl_index] = [detail["horizontal_distance"], detail["vertical_mismatch"]]
                            for shift_index, shift in enumerate(shifts):
                                perturbed[instruction, family_index, shift_index, bowl_index] = compatibility(env.env, contract, bowl, family, shift)[0]
                task_ids.append(task_id); demo_ids.append(int(demo_id))
                exact_rows.append(exact); score_rows.append(scores); detail_rows.append(details); perturb_rows.append(perturbed)
        finally:
            env.close()
    guard.assert_clean()
    arrays = {
        "task_id": np.asarray(task_ids, dtype=np.int8),
        "demo_id": np.asarray(demo_ids, dtype=np.int8),
        "exact": np.asarray(exact_rows, dtype=np.bool_),
        "scores": np.asarray(score_rows, dtype=np.float64),
        "detail_horizontal_vertical": np.asarray(detail_rows, dtype=np.float64),
        "perturbed_scores": np.asarray(perturb_rows, dtype=np.float64),
        "perturbation_vectors_m": shifts,
    }
    if not all(np.isfinite(v).all() for k, v in arrays.items() if k != "exact"):
        raise FloatingPointError("non-finite grounding sidecar")
    return arrays, guard.ledger.as_dict()


def decision_metrics(arrays: dict[str, np.ndarray], family: str, threshold: float, score_array: np.ndarray | None = None) -> dict[str, Any]:
    fi = FAMILIES.index(family)
    scores = arrays["scores"][:, :, fi, :] if score_array is None else score_array
    statuses, refs, gaps = [], [], []
    correct = arrays["task_id"]
    for episode in range(len(correct)):
        status, ref, gap = resolve_scores(scores[episode, int(correct[episode])], threshold, maximum_score(family))
        statuses.append(status); refs.append(ref); gaps.append(gap)
    statuses = np.asarray(statuses); refs = np.asarray(refs); gaps = np.asarray(gaps)
    unique = statuses == "UNIQUE"
    right = unique & (refs == 0)
    per_task = {}
    for task in range(10):
        mask = correct == task
        per_task[str(task)] = {
            "coverage": float(np.mean(unique[mask])),
            "accuracy": float(np.mean(right[mask])),
            "unique_precision": float(np.sum(right[mask]) / np.sum(unique[mask])) if np.any(unique[mask]) else 0.0,
        }
    exact_unique = np.asarray([exact_resolve(arrays["exact"][e, int(correct[e])])[0] == "UNIQUE" for e in range(len(correct))])
    agreement_mask = exact_unique
    exact_agreement = float(np.mean(unique[agreement_mask] & (refs[agreement_mask] == 0))) if np.any(agreement_mask) else 0.0
    return {
        "family": family, "threshold": threshold,
        "coverage": float(np.mean(unique)),
        "unique_precision": float(np.sum(right) / np.sum(unique)) if np.any(unique) else 0.0,
        "wrong_unique": int(np.sum(unique & (refs != 0))),
        "ambiguous": int(np.sum(statuses == "AMBIGUOUS")),
        "not_found": int(np.sum(statuses == "NOT_FOUND")),
        "task_coverage_ge_75_count": int(sum(v["coverage"] >= 0.75 for v in per_task.values())),
        "exact_success_agreement": exact_agreement,
        "mean_unique_gap": float(np.mean(gaps[unique])) if np.any(unique) else 0.0,
        "median_unique_gap": float(np.median(gaps[unique])) if np.any(unique) else 0.0,
        "per_task": per_task,
        "statuses": statuses.tolist(), "referents": refs.tolist(), "gaps": gaps.tolist(),
    }


def choose(protocol: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    screens = []
    for family in FAMILIES:
        for threshold in threshold_grid(family):
            row = decision_metrics(arrays, family, threshold)
            row["screen_pass"] = bool(row["unique_precision"] >= protocol["gates"]["precision_min"])
            screens.append(row)
    eligible = [r for r in screens if r["screen_pass"]]
    if not eligible:
        eligible = screens
    selected = sorted(eligible, key=lambda r: (-r["unique_precision"], -r["coverage"], -r["task_coverage_ge_75_count"], -r["exact_success_agreement"], -r["mean_unique_gap"], FAMILIES.index(r["family"]), r["threshold"]))[0]
    return selected, screens


def counterfactual_metrics(arrays: dict[str, np.ndarray], family: str, threshold: float) -> dict[str, Any]:
    fi = FAMILIES.index(family); eligible = switched = 0; task_counts = np.zeros(10, dtype=np.int64)
    for e, task in enumerate(arrays["task_id"]):
        cs, cr, _ = resolve_scores(arrays["scores"][e, int(task), fi], threshold, maximum_score(family))
        if cs != "UNIQUE": continue
        for alt in range(10):
            if alt == int(task): continue
            status, ref, _ = resolve_scores(arrays["scores"][e, alt, fi], threshold, maximum_score(family))
            if status == "UNIQUE":
                eligible += 1; switched += int(ref != cr); task_counts[int(task)] += 1
    return {"eligible_cases": eligible, "referent_switch_rate": switched / eligible if eligible else 0.0,
            "tasks_with_eligible_alternate": int(np.sum(task_counts > 0)), "per_source_task_eligible": task_counts.tolist()}


def perturbation_metrics(arrays: dict[str, np.ndarray], family: str, threshold: float, base: dict[str, Any]) -> dict[str, Any]:
    fi = FAMILIES.index(family); stable = total = 0; by_mm = {}
    base_status, base_refs = base["statuses"], base["referents"]
    shifts = arrays["perturbation_vectors_m"]
    for magnitude in (0.001, 0.002, 0.005):
        keep = count = 0
        for shift_index in np.flatnonzero(np.isclose(np.max(np.abs(shifts), axis=1), magnitude)):
            for e, task in enumerate(arrays["task_id"]):
                status, ref, _ = resolve_scores(arrays["perturbed_scores"][e, int(task), fi, shift_index], threshold, maximum_score(family))
                keep += int(status == base_status[e] and ref == base_refs[e]); count += 1
        by_mm[str(int(magnitude * 1000))] = keep / count
        stable += keep; total += count
    return {"overall_decision_stability": stable / total, "by_millimeter": by_mm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/robust_grounding_protocol.json")
    args = parser.parse_args()
    protocol, access, contracts = validate_protocol(args.protocol, args.partition)
    arrays, ledger = collect(protocol, access, contracts, args.partition)
    if args.partition == "calibration":
        selected, screens = choose(protocol, arrays)
        resolver = {"resolver_version": "robust-grounding-resolver-v1", "family": selected["family"], "threshold": selected["threshold"], "maximum_score": maximum_score(selected["family"]), "selection": "precision, coverage, task coverage, exact-subset agreement, confidence, simplicity", "calibration_sidecar_sha256": "PENDING"}
    else:
        resolver = json.loads((ROOT / protocol["outputs"]["frozen_resolver"]).read_text())
        selected = decision_metrics(arrays, resolver["family"], float(resolver["threshold"])); screens = []
    counterfactual = counterfactual_metrics(arrays, selected["family"], float(selected["threshold"]))
    perturb = perturbation_metrics(arrays, selected["family"], float(selected["threshold"]), selected)
    gates = {
        "precision": selected["unique_precision"] >= protocol["gates"]["precision_min"],
        "coverage": selected["coverage"] >= protocol["gates"]["coverage_min"],
        "task_coverage": selected["task_coverage_ge_75_count"] >= protocol["gates"]["task_coverage_count_min"],
        "exact_subset_agreement": selected["exact_success_agreement"] >= protocol["gates"]["exact_agreement_min"],
        "counterfactual_task_coverage": counterfactual["tasks_with_eligible_alternate"] >= protocol["gates"]["counterfactual_task_min"],
        "finite": True, "forbidden_access_zero": ledger["forbidden_access_count"] == 0,
    }
    result = {"result_version": "robust-grounding-result-v1", "partition": args.partition, "episode_count": len(arrays["task_id"]), "selected": selected, "screen": screens, "counterfactual": counterfactual, "perturbation": perturb, "gates": gates, "complete_gate": bool(all(gates.values())), "access_ledger": ledger}
    sidecar_key = f"{args.partition}_sidecar"; output_key = "calibration" if args.partition == "calibration" else "evaluation"
    atomic_npz(ROOT / protocol["outputs"][sidecar_key], arrays)
    result["sidecar_sha256"] = sha256(ROOT / protocol["outputs"][sidecar_key])
    atomic_json(ROOT / protocol["outputs"][output_key], result)
    atomic_json(ROOT / protocol["outputs"][f"{args.partition}_ledger"], ledger)
    if args.partition == "calibration":
        resolver["calibration_sidecar_sha256"] = result["sidecar_sha256"]
        resolver["calibration_result_sha256"] = sha256(ROOT / protocol["outputs"]["calibration"])
        atomic_json(ROOT / protocol["outputs"]["frozen_resolver"], resolver)
    print(json.dumps({"partition": args.partition, "selected": [selected["family"], selected["threshold"]], "complete_gate": result["complete_gate"], "forbidden_access_count": ledger["forbidden_access_count"]}, indent=2))


if __name__ == "__main__":
    main()
