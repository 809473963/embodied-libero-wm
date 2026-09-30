#!/usr/bin/env python3
"""Calibrate and evaluate the camera-contract-backed image-space restart."""

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
sys.path.insert(0, str(ROOT))

from scripts import analyze_natural_image_space_interaction as legacy
from src.probes.image_space_relevance_restart import (
    EXPECTED_CONTRACT_SHA256,
    build_dataset_from_contract,
    validate_contract,
)


PROTOCOL_VERSION = "natural-image-space-interaction-target-audit-restart-v1"
PROTOCOL_PATH = ROOT / "configs/natural_image_space_interaction_restart_protocol.json"


def last_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()


def load_protocol(path: Path, phase: str) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected restart protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_RESTART_TARGET_COLLECTION_OR_RESULTS":
        raise ValueError("restart protocol is not preregistered")
    commit = last_commit(path)
    if not commit:
        raise RuntimeError("restart protocol must be committed")
    tag = protocol["frozen_upstream"]["primary_tag"]
    if subprocess.check_output(["git", "cat-file", "-t", tag], cwd=ROOT, text=True).strip() != "tag":
        raise RuntimeError("camera-contract tag is not annotated")
    if subprocess.check_output(["git", "rev-parse", f"{tag}^{{}}"], cwd=ROOT, text=True).strip() != protocol["frozen_upstream"]["primary_commit"]:
        raise RuntimeError("camera-contract tag moved")
    for group in ("frozen_upstream", "preserved_blocked_audit"):
        for item in protocol[group]["files"]:
            if legacy.sha256(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"frozen input changed: {item['path']}")
    for item in protocol["implementation"].values():
        if legacy.sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"restart implementation changed: {item['path']}")
    contract_path = ROOT / protocol["camera_contract"]["path"]
    if legacy.sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("DATA_OR_IMPLEMENTATION_BLOCK: camera contract hash")
    validate_contract(json.loads(contract_path.read_text()))
    for partition in ("calibration",) if phase == "calibrate" else ("calibration", "development_evaluation"):
        record = protocol["partitions"][partition]
        manifest_path = ROOT / record["per_object_mask_manifest"]
        sidecar_path = ROOT / record["per_object_mask_sidecar"]
        if not manifest_path.exists() or not last_commit(manifest_path) or not sidecar_path.exists():
            raise RuntimeError(f"{partition} per-object masks are not frozen")
        manifest = json.loads(manifest_path.read_text())
        if manifest["artifact"]["sha256"] != legacy.sha256(sidecar_path):
            raise RuntimeError(f"{partition} per-object mask hash changed")
    if phase == "calibrate":
        for key in ("calibration_weights", "calibration_derived", "calibration_freeze", "diagnosis"):
            if (ROOT / protocol["outputs"][key]).exists():
                raise FileExistsError(protocol["outputs"][key])
    else:
        freeze_path = ROOT / protocol["outputs"]["calibration_freeze"]
        if not freeze_path.exists() or not last_commit(freeze_path):
            raise RuntimeError("restart calibration freeze must be committed")
        freeze = json.loads(freeze_path.read_text())
        if freeze["status"] != "FROZEN_BEFORE_INIT36_39_IMAGE_SPACE_RESTART_EVALUATION":
            raise RuntimeError("restart calibration status changed")
        if freeze["protocol_sha256"] != legacy.sha256(path):
            raise RuntimeError("restart calibration references another protocol")
        for item in freeze["artifacts"].values():
            if legacy.sha256(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"restart calibration artifact changed: {item['path']}")
    return protocol, commit


def load_partition(protocol: dict[str, Any], partition: str) -> dict[str, Any]:
    record = protocol["partitions"][partition]
    spatial = legacy._load_npz(ROOT / record["natural_spatial_derived"])
    feature = legacy._load_npz(ROOT / record["visual_features"])
    masks = legacy._load_npz(ROOT / record["frozen_masks"])
    per_object = legacy._load_npz(ROOT / record["per_object_mask_sidecar"])
    state = legacy._load_npz(ROOT / record["state_sidecar"])
    combined_masks = dict(masks)
    combined_masks["patch_coverage"] = feature["patch_coverage"]
    contract = json.loads((ROOT / protocol["camera_contract"]["path"]).read_text())
    dataset = build_dataset_from_contract(
        spatial,
        combined_masks,
        per_object,
        state,
        contract,
        float(protocol["image_space_targets"]["strict_patch_sigma_pixels"]),
        int(protocol["image_space_targets"]["adjacency_radius_pixels"]),
        float(protocol["image_space_targets"]["near_contact_pixels"]),
    )
    expected_inits = set(record["init_state_indices"])
    if set(np.unique(state["init_state_index"]).tolist()) != expected_inits:
        raise ValueError("partition init-state boundary changed")
    return {"dataset": dataset, "spatial": spatial, "state": state, "masks": combined_masks}


def implementation_validation(
    dataset: dict[str, Any], masks: dict[str, np.ndarray], protocol: dict[str, Any]
) -> dict[str, Any]:
    del masks
    projected = np.asarray(dataset["projected_ee"], dtype=np.float64)
    depth = np.asarray(dataset["projection_depth"], dtype=np.float64)
    finite = np.isfinite(projected).all(axis=-1) & np.isfinite(depth)
    positive = depth > 0.0
    inside = (
        (projected[..., 0] >= 0.0)
        & (projected[..., 0] < 128.0)
        & (projected[..., 1] >= 0.0)
        & (projected[..., 1] < 128.0)
    )
    record = {
        "role": "frozen-contract implementation validity only; no convention comparison or camera-science decision",
        "n": int(depth.size),
        "finite_fraction": float(np.mean(finite)),
        "positive_depth_fraction": float(np.mean(positive)),
        "inside_image_fraction": float(np.mean(inside)),
        "minimum_depth": float(np.min(depth)),
        "maximum_depth": float(np.max(depth)),
    }
    record["pass"] = bool(finite.all() and positive.all() and inside.all())
    return record


def _rewrite_calibration_freeze(protocol: dict[str, Any], protocol_commit: str) -> None:
    path = ROOT / protocol["outputs"]["calibration_freeze"]
    freeze = json.loads(path.read_text())
    freeze.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "status": "FROZEN_BEFORE_INIT36_39_IMAGE_SPACE_RESTART_EVALUATION",
            "protocol_commit": protocol_commit,
            "protocol_sha256": legacy.sha256(PROTOCOL_PATH),
            "camera_contract": {
                "path": protocol["camera_contract"]["path"],
                "sha256": EXPECTED_CONTRACT_SHA256,
                "status": "CAMERA_PROJECTION_CONTRACT_VALIDATED",
                "authorization": "GO_IMAGE_SPACE_TARGET_AUDIT_RESTART",
                "projection_reselection": False,
            },
            "old_block_preserved": protocol["preserved_blocked_audit"]["verdicts"],
        }
    )
    legacy.atomic_json(path, freeze)


def _semantic_attribution(result: dict[str, Any], family: str) -> bool:
    target_names = result["family_decisions"][family]["passing_targets"]
    if family == "interaction":
        return any(
            result["shuffle_controls"][name]["strict_interaction_only"]["passes"]
            for name in target_names
        )
    return any(
        result["shuffle_controls"][name]["relevant_patch_only"]["passes"]
        for name in target_names
    )


def _rewrite_evaluation_result(protocol: dict[str, Any], protocol_commit: str) -> None:
    path = ROOT / protocol["outputs"]["diagnosis"]
    result = json.loads(path.read_text())
    families = result["family_decisions"]
    authorized_pass = any(families[name]["passes"] for name in ("object", "relative", "interaction"))
    labels: list[str] = []
    if families["robot"]["passes"]:
        labels.append("SPATIAL_EE_SIGNAL_CONFIRMED")
    attributed = False
    for family, label in (
        ("object", "SPATIAL_OBJECT_IMAGE_SIGNAL_CONFIRMED"),
        ("relative", "SPATIAL_RELATIVE_IMAGE_GEOMETRY_SIGNAL_CONFIRMED"),
        ("interaction", "SPATIAL_INTERACTION_IMAGE_SIGNAL_CONFIRMED"),
    ):
        if families[family]["passes"] and _semantic_attribution(result, family):
            labels.append(label)
            attributed = True
    mismatch = any(
        record["supports_mismatch"]
        for record in result["comparison_to_frozen_3d_targets"].values()
    )
    if mismatch:
        labels.append("IMAGE_SPACE_TARGET_MISMATCH_SUPPORTED")
    if authorized_pass:
        if attributed:
            labels.append("GLOBAL_POOLING_LOSES_IMAGE_SPACE_INTERACTION_SIGNAL")
        else:
            labels.append("SPATIAL_SIGNAL_REMAINS_NONUNIQUE")
        authorization = "GO_SPATIAL_TARGET_DESIGN"
    else:
        authorization = "BLOCK_VISUAL_TARGET_REDESIGN"
        authorized_names = sum(
            (list(families[name]["targets"]) for name in ("object", "relative", "interaction")),
            [],
        )
        maximum_increment = max(
            float(result["target_decisions"][name]["increment"])
            for name in authorized_names
        )
        if maximum_increment <= 0.0:
            labels.append("IMAGE_SPACE_TARGETS_DO_NOT_RESCUE_PATCH_RELEVANCE")
        else:
            labels.append("NATURAL_IMAGE_SPACE_RELEVANCE_UNRESOLVED")

    result.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_commit": protocol_commit,
            "protocol_sha256": legacy.sha256(PROTOCOL_PATH),
            "calibration_freeze_sha256": legacy.sha256(ROOT / protocol["outputs"]["calibration_freeze"]),
            "old_block_preserved": {
                "verdicts": protocol["preserved_blocked_audit"]["verdicts"],
                "artifacts_changed": False,
            },
            "camera_projection": {
                "contract_path": protocol["camera_contract"]["path"],
                "contract_sha256": EXPECTED_CONTRACT_SHA256,
                "contract_status": "CAMERA_PROJECTION_CONTRACT_VALIDATED",
                "contract_authorization": "GO_IMAGE_SPACE_TARGET_AUDIT_RESTART",
                "projection_reselection": False,
                "calibration_implementation_validity": result["camera_projection"]["calibration_validation"],
                "evaluation_implementation_validity": result["camera_projection"]["evaluation_validation"],
            },
            "decision": {
                "mechanism_labels": labels,
                "next_stage_authorization": authorization,
                "controller_authorization": False,
                "representation_audit_stop_recommended": bool(
                    "NATURAL_IMAGE_SPACE_RELEVANCE_UNRESOLVED" in labels
                ),
                "repeat_similar_representation_probe_if_unresolved": False,
            },
        }
    )
    result["data_boundary"]["development_evaluation_is_blind_confirmation"] = False
    result["data_boundary"]["development_evaluation_description"] = (
        "frozen development evaluation, not blind confirmation"
    )
    result["forbidden_activity_audit"].update(
        {
            "projection_reselection": False,
            "old_blocked_audit_overwrite": False,
            "frozen_camera_contract_overwrite": False,
        }
    )
    calibration_manifest = ROOT / protocol["partitions"]["calibration"]["per_object_mask_manifest"]
    evaluation_manifest = ROOT / protocol["partitions"]["development_evaluation"]["per_object_mask_manifest"]
    result["artifacts"].update(
        {
            "calibration_freeze": {
                "path": protocol["outputs"]["calibration_freeze"],
                "sha256": legacy.sha256(ROOT / protocol["outputs"]["calibration_freeze"]),
            },
            "calibration_object_mask_manifest": {
                "path": str(calibration_manifest.relative_to(ROOT)),
                "sha256": legacy.sha256(calibration_manifest),
            },
            "evaluation_object_mask_manifest": {
                "path": str(evaluation_manifest.relative_to(ROOT)),
                "sha256": legacy.sha256(evaluation_manifest),
            },
        }
    )
    if not legacy.finite_tree(result):
        raise FloatingPointError("restart result contains NaN/Inf")
    legacy.atomic_json(path, result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("calibrate", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    protocol, commit = load_protocol(args.protocol, args.phase)
    legacy.PROTOCOL_VERSION = PROTOCOL_VERSION
    legacy.load_partition = load_partition
    legacy.projection_validation = implementation_validation
    if args.phase == "calibrate":
        legacy.calibrate(protocol, commit)
        _rewrite_calibration_freeze(protocol, commit)
    else:
        legacy.evaluate(protocol, commit)
        _rewrite_evaluation_result(protocol, commit)
        result = json.loads((ROOT / protocol["outputs"]["diagnosis"]).read_text())
        print(json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
