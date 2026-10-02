#!/usr/bin/env python3
"""Run the conditional T1 semantic-teacher evaluation after Stage G passes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_semantic_teacher_clean import analyze, build_pairs, collect_partition  # noqa: E402
from src.probes.robust_grounding import BOWLS, FAMILIES, maximum_score, resolve_scores  # noqa: E402
from src.probes.semantic_teacher_clean import atomic_json, atomic_npz, sha256, summarize_labels  # noqa: E402


VERSION = "t1-semantic-teacher-full-v1"


def grounding_arrays(grounding: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    status = np.full((len(grounding), 10), -1, dtype=np.int8)
    referent = np.full((len(grounding), 10), -1, dtype=np.int8)
    for episode in grounding:
        eid = int(episode["episode_id"])
        for item in episode["instructions"]:
            instruction = int(item["instruction_task_id"])
            status[eid, instruction] = {"NOT_FOUND": 0, "UNIQUE": 1, "AMBIGUOUS": 2}[item["status"]]
            if item["referent"] in BOWLS: referent[eid, instruction] = BOWLS.index(item["referent"])
    return status, referent


def last_commit(path: Path) -> str:
    return subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))], cwd=ROOT, text=True).strip()


def validate(path: Path, partition: str) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != VERSION or protocol["status"] != "PREREGISTERED_AFTER_STAGE_G_PASS":
        raise RuntimeError("invalid T1 protocol")
    if not last_commit(path): raise RuntimeError("T1 protocol must be committed")
    for item in protocol["frozen_files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]: raise RuntimeError(f"frozen file changed: {item['path']}")
    stage_g = json.loads((ROOT / protocol["inputs"][f"grounding_{'calibration' if partition == 'calibration' else 'evaluation'}"]).read_text())
    if not stage_g["complete_gate"]: raise RuntimeError("Stage G did not authorize T1")
    if partition == "evaluation":
        for key in ("teacher_definition", "calibration"):
            target = ROOT / protocol["outputs"][key]
            if not target.exists() or not last_commit(target): raise RuntimeError("T1 calibration must be committed")
            if subprocess.check_output(["git", "status", "--porcelain", "--", str(target.relative_to(ROOT))], cwd=ROOT, text=True).strip(): raise RuntimeError("T1 calibration freeze is dirty")
    access = json.loads((ROOT / protocol["inputs"]["allowed_demo_access"]).read_text())
    return protocol, access


def robust_grounding(protocol: dict[str, Any], partition: str) -> list[dict[str, Any]]:
    resolver = json.loads((ROOT / protocol["inputs"]["frozen_resolver"]).read_text())
    data = np.load(ROOT / protocol["inputs"][f"grounding_{partition}_sidecar"])
    fi = FAMILIES.index(resolver["family"])
    rows = []
    for episode in range(len(data["task_id"])):
        instructions = []
        for instruction in range(10):
            status, ref, gap = resolve_scores(data["scores"][episode, instruction, fi], float(resolver["threshold"]), maximum_score(resolver["family"]))
            instructions.append({"instruction_task_id": instruction, "status": status, "referent": BOWLS[ref] if ref >= 0 else None, "confidence_gap": gap})
        rows.append({"episode_id": episode, "task_id": int(data["task_id"][episode]), "demo_id": int(data["demo_id"][episode]), "instructions": instructions})
    return rows


def margin_candidates(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    geometry = arrays["geometry"]
    xy_clear, z_order, surface_gap = geometry[:, :, 1], geometry[:, :, 2], geometry[:, :, 3]
    return {
        "M0": np.minimum(np.minimum(xy_clear / 0.03, z_order / 0.03), -np.abs(surface_gap) / 0.03),
        "M1": np.minimum(np.minimum(xy_clear / 0.03, z_order / 0.03), -np.abs(surface_gap) / 0.06),
        "M2": np.minimum(xy_clear / 0.03, z_order / 0.03),
    }


def rank_auc(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values).reshape(-1); labels = np.asarray(labels).reshape(-1).astype(bool)
    order = np.argsort(values, kind="stable"); ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]: end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0; start = end
    n1, n0 = int(np.sum(labels)), int(np.sum(~labels))
    return float((np.sum(ranks[labels]) - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else 0.5


def select_margin(arrays: dict[str, np.ndarray]) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for family, values in margin_candidates(arrays).items():
        goal = arrays["goal"]
        pos, neg = values[goal], values[~goal]
        rows.append({"family": family, "auc": rank_auc(values, goal), "positive_median": float(np.median(pos)), "negative_median": float(np.median(neg)), "median_separation": float(np.median(pos) - np.median(neg)), "minimum": float(np.min(values)), "maximum": float(np.max(values)), "finite": bool(np.isfinite(values).all())})
    selected = sorted(rows, key=lambda r: (-r["auc"], -r["median_separation"], ("M0", "M1", "M2").index(r["family"])))[0]["family"]
    return selected, rows


def t0_summary(arrays: dict[str, np.ndarray], grounding: list[dict[str, Any]]) -> dict[str, Any]:
    _, refs = grounding_arrays(grounding); pairs = build_pairs(arrays, refs); labels = []
    for a, b, ref in zip(pairs["left"], pairs["right"], pairs["referent"]):
        ga, gb = bool(arrays["goal"][a, ref]), bool(arrays["goal"][b, ref])
        labels.append(1 if gb and not ga else -1 if ga and not gb else 0)
    labels = np.asarray(labels, dtype=np.int8); local = pairs["gap"] <= 8
    per_task = {str(t): summarize_labels(labels[local & (pairs["task_id"] == t)]) for t in range(10)}
    result = summarize_labels(labels[local]); result["usable_task_count"] = int(sum(v["non_tie_fraction"] >= 0.30 for v in per_task.values())); result["per_task"] = per_task
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--partition", choices=("calibration", "evaluation"), required=True); parser.add_argument("--protocol", type=Path, default=ROOT / "configs/t1_teacher_full_protocol.json"); args = parser.parse_args()
    protocol, access = validate(args.protocol, args.partition)
    arrays, _, ledger = collect_partition(protocol, access, args.partition)
    grounding = robust_grounding(protocol, args.partition)
    if args.partition == "calibration":
        selected_margin, margin_screen = select_margin(arrays); definition = None
    else:
        definition = json.loads((ROOT / protocol["outputs"]["teacher_definition"]).read_text()); selected_margin = definition["selected_margin_family"]; margin_screen = []
    arrays["margin"] = margin_candidates(arrays)[selected_margin]
    t0 = t0_summary(arrays, grounding)
    result, definition_out = analyze(arrays, grounding, args.partition, protocol, definition)
    result["selected_margin_family"] = selected_margin; result["margin_screen"] = margin_screen; result["t0_local_signal"] = t0; result["access_ledger"] = ledger
    result["stage_t_complete_gate"] = bool(result["complete_gate"] and result["grounding"]["correct_instruction_unique_rate"] == 1.0 and ledger["forbidden_access_count"] == 0)
    sidecar_key = f"{args.partition}_sidecar"; output_key = "calibration" if args.partition == "calibration" else "evaluation"
    sidecar = dict(arrays); status, refs = grounding_arrays(grounding); sidecar["ground_status"] = status; sidecar["ground_referent"] = refs
    atomic_npz(ROOT / protocol["outputs"][sidecar_key], sidecar); result["sidecar_sha256"] = sha256(ROOT / protocol["outputs"][sidecar_key])
    atomic_json(ROOT / protocol["outputs"][f"{args.partition}_ledger"], ledger); atomic_json(ROOT / protocol["outputs"][output_key], result)
    if definition_out is not None:
        definition_out["definition_version"] = "t1-semantic-teacher-full-definition-v1"; definition_out["selected_margin_family"] = selected_margin; definition_out["margin_selection"] = margin_screen; definition_out["grounding_resolver"] = json.loads((ROOT / protocol["inputs"]["frozen_resolver"]).read_text()); definition_out["calibration_result_sha256"] = sha256(ROOT / protocol["outputs"]["calibration"]); atomic_json(ROOT / protocol["outputs"]["teacher_definition"], definition_out)
    print(json.dumps({"partition": args.partition, "selected_margin": selected_margin, "selected_teacher": result["selected_family"], "complete_gate": result["stage_t_complete_gate"], "forbidden_access_count": ledger["forbidden_access_count"]}, indent=2))


if __name__ == "__main__": main()
