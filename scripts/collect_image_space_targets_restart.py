#!/usr/bin/env python3
"""Collect only missing per-object natural masks for the image-space restart."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from robosuite.utils.camera_utils import get_camera_transform_matrix

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
sys.path.insert(0, str(ROOT))

from libero.libero import benchmark, get_libero_path  # noqa: E402

from scripts import audit_local_candidate_counterfactual as base  # noqa: E402
from scripts.evaluate_c0_all import refresh_policy_observation, trusted_init_states  # noqa: E402
from scripts.inventory_visual_target_masks import (  # noqa: E402
    mask_id_sets,
    masks_from_segmentation,
    render_rgb,
    render_segmentation,
)
from src.probes.image_space_relevance import IMAGE_HEIGHT, IMAGE_WIDTH  # noqa: E402
from src.probes.image_space_relevance_restart import (  # noqa: E402
    EXPECTED_CONTRACT_SHA256,
    implementation_sanity_distance,
    validate_contract,
)
from src.probes.visual_target import MASK_NAMES  # noqa: E402


PROTOCOL_VERSION = "natural-image-space-interaction-target-audit-restart-v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def last_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()


def load_protocol(path: Path, partition: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected restart protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_RESTART_TARGET_COLLECTION_OR_RESULTS":
        raise ValueError("restart protocol is not preregistered")
    commit = last_commit(path)
    if not commit:
        raise RuntimeError("restart protocol must be committed before replay")
    tag = protocol["frozen_upstream"]["primary_tag"]
    if subprocess.check_output(["git", "cat-file", "-t", tag], cwd=ROOT, text=True).strip() != "tag":
        raise RuntimeError("camera-contract tag is not annotated")
    if subprocess.check_output(["git", "rev-parse", f"{tag}^{{}}"], cwd=ROOT, text=True).strip() != protocol["frozen_upstream"]["primary_commit"]:
        raise RuntimeError("camera-contract tag moved")
    for item in protocol["frozen_upstream"]["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen upstream changed: {item['path']}")
    for item in protocol["implementation"].values():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"restart implementation changed: {item['path']}")
    contract_path = ROOT / protocol["camera_contract"]["path"]
    if sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("DATA_OR_IMPLEMENTATION_BLOCK: camera contract hash")
    contract = json.loads(contract_path.read_text())
    validate_contract(contract)
    if partition == "development_evaluation":
        freeze = ROOT / protocol["outputs"]["calibration_freeze"]
        if not freeze.exists() or not last_commit(freeze):
            raise RuntimeError("calibration freeze must be committed before evaluation replay")
    record = protocol["partitions"][partition]
    for key in ("per_object_mask_sidecar", "per_object_mask_manifest"):
        if (ROOT / record[key]).exists():
            raise FileExistsError(f"refusing to overwrite {record[key]}")
    return protocol, commit, contract


def _per_object_masks(
    segmentation: np.ndarray, ids: dict[str, Any], names: list[str]
) -> np.ndarray:
    is_geom = segmentation[..., 0] == 5
    geom = segmentation[..., 1]
    result = np.zeros((2, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.bool_)
    for index, name in enumerate(names[:2]):
        result[index] = is_geom & np.isin(geom, list(ids["per_object"][name]))
    return result


def collect_partition(
    protocol: dict[str, Any],
    partition: str,
    protocol_commit: str,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    record = protocol["partitions"][partition]
    with np.load(ROOT / record["branch_sidecar"], allow_pickle=False) as handle:
        source = {name: np.asarray(handle[name]) for name in handle.files}
    with np.load(ROOT / record["frozen_masks"], allow_pickle=False) as handle:
        frozen_masks = {name: np.asarray(handle[name]) for name in handle.files}
    with np.load(ROOT / record["state_sidecar"], allow_pickle=False) as handle:
        state = {name: np.asarray(handle[name]) for name in handle.files}
    for key in ("task_id", "init_state_index", "seed", "planning_index"):
        if not np.array_equal(source[key], frozen_masks[key]) or not np.array_equal(source[key], state[key]):
            raise RuntimeError(f"frozen partition identity mismatch: {key}")

    expected_inits = set(protocol["partitions"][partition]["init_state_indices"])
    if set(np.unique(source["init_state_index"]).tolist()) != expected_inits:
        raise RuntimeError("partition init-state boundary changed")
    suite = benchmark.get_benchmark_dict()[protocol["scope"]["suite"]]()
    all_masks: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    sanity: list[dict[str, Any]] = []
    frozen_matrix = np.asarray(contract["world_to_raw_pixel"], dtype=np.float64)

    for branch in range(source["task_id"].shape[0]):
        task_id = int(source["task_id"][branch])
        init_index = int(source["init_state_index"][branch])
        seed = int(source["seed"][branch])
        prefix_length = int(source["prefix_lengths"][branch])
        prefix = np.asarray(source["prefix_actions"][branch, :prefix_length], dtype=np.float32)
        candidates = np.asarray(source["candidates"][branch], dtype=np.float32)
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_state = trusted_init_states(suite, task_id)[init_index]
        envs = []
        try:
            observations, simulator_states = [], []
            for _ in range(3):
                env, observation, sim_state = base.replay_prefix(bddl, seed, init_state, prefix)
                envs.append(env)
                observations.append(observation)
                simulator_states.append(sim_state)
            reference_rgb = np.asarray(observations[0]["agentview_image"], dtype=np.uint8)
            if not all(np.array_equal(reference_rgb, np.asarray(obs["agentview_image"], dtype=np.uint8)) for obs in observations[1:]):
                raise RuntimeError("fresh branch planning RGB identity failed")
            if not all(np.array_equal(simulator_states[0], value) for value in simulator_states[1:]):
                raise RuntimeError("fresh branch simulator-state identity failed")

            inner = envs[0].env
            runtime_matrix = np.asarray(
                get_camera_transform_matrix(
                    inner.sim,
                    contract["camera_name"],
                    IMAGE_HEIGHT,
                    IMAGE_WIDTH,
                ),
                dtype=np.float64,
            )
            matrix_error = float(np.max(np.abs(runtime_matrix - frozen_matrix)))
            if matrix_error > float(protocol["camera_contract"]["maximum_runtime_matrix_difference"]):
                raise RuntimeError("DATA_OR_IMPLEMENTATION_BLOCK: runtime camera matrix drift")

            if partition == "calibration" and init_index == 31:
                # The exact site marker can be occluded after the plan-8 prefix.
                # Check contract invocation at the official zero-step init31
                # planning observation, matching the frozen camera experiment.
                sanity_env, _ = base.create_episode_env(bddl, seed, init_state)
                try:
                    sanity_observation = refresh_policy_observation(sanity_env)
                    sanity_inner = sanity_env.env
                    sanity_rgb = render_rgb(sanity_inner)
                    segmentation = render_segmentation(sanity_inner)
                    site_id = int(sanity_inner.robots[0].eef_site_id)
                    site_mask = (segmentation[..., 0] == 6) & (segmentation[..., 1] == site_id)
                    world = np.asarray(sanity_inner.sim.data.site_xpos[site_id], dtype=np.float64)
                    check = implementation_sanity_distance(world, site_mask, contract)
                    check.update(
                        {
                            "task_id": task_id,
                            "init_state_index": init_index,
                            "site_id": site_id,
                            "zero_step_planning_observation": True,
                            "rgb_matches_observation_bitwise": bool(
                                np.array_equal(
                                    sanity_rgb,
                                    np.asarray(
                                        sanity_observation["agentview_image"], dtype=np.uint8
                                    ),
                                )
                            ),
                        }
                    )
                    if not (
                        check["finite"]
                        and check["positive_depth"]
                        and check["in_frame"]
                        and check["rgb_matches_observation_bitwise"]
                        and check["distance_pixels"]
                        <= float(protocol["camera_contract"]["implementation_sanity_max_pixels"])
                    ):
                        raise RuntimeError("DATA_OR_IMPLEMENTATION_BLOCK: frozen-contract sanity")
                    sanity.append(check)
                finally:
                    sanity_env.close()

            per_candidate_ids = [mask_id_sets(env.env) for env in envs]
            object_names = [str(name) for name in state["object_names"][branch] if str(name)]
            if object_names != list(inner.parsed_problem.get("obj_of_interest", [])):
                raise RuntimeError("relevant-object identity changed")
            masks = np.zeros((3, 8, 2, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.bool_)
            rgb_exact = True
            merged_exact = True
            state_exact = True
            camera_static = True
            first_mismatch: dict[str, Any] | None = None
            for horizon in range(8):
                for candidate, env in enumerate(envs):
                    env.step(candidates[candidate, horizon])
                    observation = refresh_policy_observation(env)
                    expected_rgb = np.asarray(source["real_rgb"][branch, candidate, horizon])
                    actual_rgb = np.asarray(observation["agentview_image"], dtype=np.uint8)
                    direct_rgb = render_rgb(env.env)
                    observation_rgb_exact = bool(np.array_equal(actual_rgb, expected_rgb))
                    direct_rgb_exact = bool(np.array_equal(direct_rgb, expected_rgb))
                    rgb_exact &= observation_rgb_exact
                    rgb_exact &= direct_rgb_exact
                    segmentation = render_segmentation(env.env)
                    ids = per_candidate_ids[candidate]
                    semantic = masks_from_segmentation(segmentation, ids)
                    stacked = np.stack([semantic[name] for name in MASK_NAMES])
                    frozen_stacked = np.asarray(
                        frozen_masks["masks"][branch, candidate, horizon], dtype=np.bool_
                    )
                    stacked_exact = bool(np.array_equal(stacked, frozen_stacked))
                    merged_exact &= stacked_exact
                    per_object = _per_object_masks(segmentation, ids, object_names)
                    union = np.any(per_object[np.asarray(state["object_mask"][branch], dtype=bool)], axis=0)
                    union_exact = bool(np.array_equal(union, semantic["relevant"]))
                    merged_exact &= union_exact
                    masks[candidate, horizon] = per_object
                    ee_site = int(env.env.robots[0].eef_site_id)
                    actual_ee = np.asarray(env.env.sim.data.site_xpos[ee_site], dtype=np.float64)
                    expected_ee = np.asarray(state["ee_pos"][branch, candidate, horizon], dtype=np.float64)
                    state_exact &= bool(np.max(np.abs(actual_ee - expected_ee)) <= 1.0e-12)
                    current_matrix = np.asarray(
                        get_camera_transform_matrix(
                            env.env.sim,
                            contract["camera_name"],
                            IMAGE_HEIGHT,
                            IMAGE_WIDTH,
                        ),
                        dtype=np.float64,
                    )
                    camera_static &= bool(np.max(np.abs(current_matrix - frozen_matrix)) <= 1.0e-12)
                    if first_mismatch is None and not (
                        observation_rgb_exact and direct_rgb_exact and stacked_exact and union_exact
                    ):
                        expected_i16 = expected_rgb.astype(np.int16)
                        actual_i16 = actual_rgb.astype(np.int16)
                        direct_i16 = direct_rgb.astype(np.int16)
                        first_mismatch = {
                            "branch": branch,
                            "task_id": task_id,
                            "init_state_index": init_index,
                            "candidate": candidate,
                            "horizon": horizon,
                            "observation_rgb_exact": observation_rgb_exact,
                            "direct_rgb_exact": direct_rgb_exact,
                            "observation_max_abs_difference": int(
                                np.max(np.abs(actual_i16 - expected_i16))
                            ),
                            "direct_max_abs_difference": int(
                                np.max(np.abs(direct_i16 - expected_i16))
                            ),
                            "observation_different_values": int(
                                np.count_nonzero(actual_rgb != expected_rgb)
                            ),
                            "direct_different_values": int(
                                np.count_nonzero(direct_rgb != expected_rgb)
                            ),
                            "observation_equals_direct": bool(
                                np.array_equal(actual_rgb, direct_rgb)
                            ),
                            "merged_mask_exact": stacked_exact,
                            "merged_mask_different_pixels": int(
                                np.count_nonzero(stacked != frozen_stacked)
                            ),
                            "per_object_union_exact": union_exact,
                        }
            if not rgb_exact or not merged_exact or not state_exact or not camera_static:
                raise RuntimeError(
                    "frozen replay/per-object mask gate failed: "
                    f"rgb_exact={rgb_exact}, merged_exact={merged_exact}, "
                    f"state_exact={state_exact}, camera_static={camera_static}, "
                    f"first_mismatch={json.dumps(first_mismatch, sort_keys=True)}"
                )
            all_masks.append(masks)
            records.append(
                {
                    "branch_index": branch,
                    "task_id": task_id,
                    "init_state_index": init_index,
                    "seed": seed,
                    "planning_index": int(source["planning_index"][branch]),
                    "prefix_steps": prefix_length,
                    "candidate_sha256": base.array_sha256(candidates),
                    "relevant_objects": object_names,
                    "planning_rgb_identical_across_fresh_branches": True,
                    "simulator_state_identical_across_fresh_branches": True,
                    "future_rgb_matches_frozen": True,
                    "merged_masks_match_frozen": True,
                    "ee_world_state_matches_frozen": True,
                    "camera_contract_matrix_static": True,
                }
            )
        finally:
            for env in envs:
                env.close()
        print(json.dumps({"phase": "restart_object_masks", "partition": partition, "branch": branch + 1, "task": task_id, "init": init_index}), flush=True)

    arrays = {"per_object_masks": np.stack(all_masks)}
    for key in ("task_id", "init_state_index", "seed", "planning_index"):
        arrays[key] = np.asarray(source[key])
    manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": "COMPLETE",
        "partition": partition,
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256(ROOT / "configs/natural_image_space_interaction_restart_protocol.json"),
        "camera_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "reason_for_replay": "existing sidecars lack per-object natural future masks; all RGB, merged-mask, EE-state, and branch identities are checked against frozen data",
        "branch_identities": records,
        "implementation_sanity": sanity,
        "forbidden_activity_audit": {
            "masked_clip_inference": False,
            "reward_done_success_or_task_outcome_read": False,
            "model_or_controller_change": False,
            "init40_49_access": False,
            "projection_reselection": False,
        },
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("per-object sidecar contains NaN/Inf")
    return manifest, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", required=True, choices=("calibration", "development_evaluation"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/natural_image_space_interaction_restart_protocol.json",
    )
    args = parser.parse_args()
    protocol, commit, contract = load_protocol(args.protocol, args.partition)
    manifest, arrays = collect_partition(protocol, args.partition, commit, contract)
    record = protocol["partitions"][args.partition]
    sidecar = ROOT / record["per_object_mask_sidecar"]
    manifest_path = ROOT / record["per_object_mask_manifest"]
    atomic_npz(sidecar, arrays)
    manifest["artifact"] = {
        "path": str(sidecar.relative_to(ROOT)),
        "sha256": sha256(sidecar),
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "COMPLETE", "sha256": manifest["artifact"]["sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
