#!/usr/bin/env python3
"""Generate D17 stratified low/high-error transition review sheets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.world_models.normalization import file_sha256


def select_stratified_tails(
    rows: list[dict[str, Any]], per_task: int = 2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose low/high learned-error samples, with distinct demos per task."""

    enriched = [
        {**row, "learned_mean_error": float(np.mean(row["learned_cosine_error"]))}
        for row in rows
    ]

    def choose(task_rows: list[dict[str, Any]], reverse: bool) -> list[dict[str, Any]]:
        ordered = sorted(task_rows, key=lambda row: row["learned_mean_error"], reverse=reverse)
        selected: list[dict[str, Any]] = []
        used_demos: set[int] = set()
        for row in ordered:
            if row["demo_id"] in used_demos:
                continue
            selected.append(row)
            used_demos.add(row["demo_id"])
            if len(selected) == per_task:
                return selected
        raise ValueError("not enough distinct held-out demos for stratified selection")

    correct: list[dict[str, Any]] = []
    incorrect: list[dict[str, Any]] = []
    for task_id in range(10):
        task_rows = [row for row in enriched if row["task_id"] == task_id]
        if not task_rows:
            raise ValueError(f"missing task {task_id}")
        correct.extend(choose(task_rows, reverse=False))
        incorrect.extend(choose(task_rows, reverse=True))
    return correct, incorrect


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(path, size=size) if path.is_file() else ImageFont.load_default()


def source_episode(store: h5py.File, task_id: int, demo_id: int) -> tuple[Path, str]:
    task = store[f"tasks/task_{task_id}"]
    return PROJECT_ROOT / str(task.attrs["source_hdf5_relative_path"]), str(
        task.attrs["task_language"]
    )


def make_panel(
    category: str,
    ordinal: int,
    row: dict[str, Any],
    frames: np.ndarray,
    automatic: dict[str, Any],
) -> Image.Image:
    panel = Image.new("RGB", (536, 190), "white")
    draw = ImageDraw.Draw(panel)
    header = font(13)
    label = font(12)
    draw.text(
        (4, 3),
        (
            f"{category.upper()} #{ordinal:02d}  task={row['task_id']} demo={row['demo_id']} "
            f"k={row['start_timestep']}  err={row['learned_mean_error']:.5f} "
            f"copy={automatic['copy_mean_error']:.5f}"
        ),
        fill="black",
        font=header,
    )
    draw.text(
        (4, 22),
        (
            f"unc={automatic['uncertainty_mean']:.5f}  pixelMAE(k,h8)={automatic['pixel_mae_h8']:.1f} "
            f"eef_disp_h8={automatic['eef_displacement_h8']:.3f}  grip_changes={automatic['gripper_sign_changes']}"
        ),
        fill="black",
        font=label,
    )
    labels = ["obs[k]", "obs[k+1]", "obs[k+4]", "obs[k+8]"]
    for index, (frame, frame_label) in enumerate(zip(frames, labels)):
        x = 4 + index * 133
        panel.paste(Image.fromarray(frame), (x, 57))
        draw.rectangle((x, 42, x + 127, 57), fill=(0, 0, 0))
        draw.text((x + 3, 42), frame_label, fill="white", font=label)
    return panel


def make_sheets(category: str, panels: list[Image.Image], output_dir: Path) -> list[str]:
    names: list[str] = []
    for sheet_index in range(2):
        subset = panels[sheet_index * 10 : (sheet_index + 1) * 10]
        sheet = Image.new("RGB", (1076, 962), (224, 224, 224))
        for index, panel in enumerate(subset):
            x = 2 + (index % 2) * 538
            y = 2 + (index // 2) * 192
            sheet.paste(panel, (x, y))
        name = f"{category}_{sheet_index + 1}.png"
        sheet.save(output_dir / name, optimize=True)
        names.append(name)
    return names


def pearson(first: list[float], second: list[float]) -> float | None:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.std() == 0.0 or second_array.std() == 0.0:
        return None
    return float(np.corrcoef(first_array, second_array)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/d17_sample_review.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/stage2/d17_review"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["contract_version"] != "d17-v1" or config["frames"] != [0, 1, 4, 8]:
        raise ValueError("expected frozen D17-v1 review configuration")
    transition_path = PROJECT_ROOT / config["source_transition_eval"]
    sample_path = PROJECT_ROOT / config["source_sample_records"]
    transition = json.loads(transition_path.read_text())
    if transition["status"] != "PASS" or file_sha256(sample_path) != transition["sample_records"]["sha256"]:
        raise ValueError("D16 result/sample provenance mismatch")
    rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
    correct, incorrect = select_stratified_tails(
        rows, int(config["selection"]["per_task_per_category"])
    )
    if len(correct) != 20 or len(incorrect) != 20:
        raise RuntimeError("D17 requires exactly 20 samples per category")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store_path = PROJECT_ROOT / "artifacts/stage2/world_model_clip_vit_b32_h8_obs_aligned/observations.hdf5"
    all_records: dict[str, list[dict[str, Any]]] = {"correct": [], "incorrect": []}
    all_panels: dict[str, list[Image.Image]] = {"correct": [], "incorrect": []}
    with h5py.File(store_path, "r") as store:
        for category, selected in (("correct", correct), ("incorrect", incorrect)):
            for ordinal, row in enumerate(selected):
                task_id = int(row["task_id"])
                demo_id = int(row["demo_id"])
                k = int(row["start_timestep"])
                path, language = source_episode(store, task_id, demo_id)
                with h5py.File(path, "r") as source:
                    demo = source[f"data/demo_{demo_id}"]
                    rgb = demo["obs/agentview_rgb"]
                    frames = rgb[[k, k + 1, k + 4, k + 8]].astype(np.uint8)
                    ee = demo["obs/ee_pos"][[k, k + 1, k + 4, k + 8]].astype(np.float32)
                    actions = demo["actions"][k + 1 : k + 9].astype(np.float32)
                gripper = actions[:, 6]
                automatic = {
                    "copy_mean_error": float(np.mean(row["copy_cosine_error"])),
                    "copy_minus_learned_mean": float(np.mean(row["copy_minus_learned"])),
                    "uncertainty_mean": float(np.mean(row["uncertainty"])),
                    "pixel_mae_h1": float(np.abs(frames[1].astype(np.float32) - frames[0]).mean()),
                    "pixel_mae_h4": float(np.abs(frames[2].astype(np.float32) - frames[0]).mean()),
                    "pixel_mae_h8": float(np.abs(frames[3].astype(np.float32) - frames[0]).mean()),
                    "eef_displacement_h1": float(np.linalg.norm(ee[1] - ee[0])),
                    "eef_displacement_h4": float(np.linalg.norm(ee[2] - ee[0])),
                    "eef_displacement_h8": float(np.linalg.norm(ee[3] - ee[0])),
                    "mean_translation_action_l2": float(np.linalg.norm(actions[:, :3], axis=1).mean()),
                    "mean_rotation_action_l2": float(np.linalg.norm(actions[:, 3:6], axis=1).mean()),
                    "gripper_sign_changes": int(np.sum(np.signbit(gripper[1:]) != np.signbit(gripper[:-1]))),
                    "action_saturation_fraction": float(np.mean(np.abs(actions) >= 0.99)),
                }
                if not all(
                    math.isfinite(value)
                    for value in automatic.values()
                    if isinstance(value, float)
                ):
                    raise FloatingPointError("D17 automatic review metrics contain NaN/Inf")
                record = {
                    "category": category,
                    "ordinal": ordinal,
                    "task_id": task_id,
                    "task_language": language,
                    "demo_id": demo_id,
                    "start_timestep": k,
                    "learned_mean_error": row["learned_mean_error"],
                    "learned_cosine_error": row["learned_cosine_error"],
                    "copy_cosine_error": row["copy_cosine_error"],
                    "uncertainty": row["uncertainty"],
                    "automatic": automatic,
                }
                all_records[category].append(record)
                all_panels[category].append(make_panel(category, ordinal, row, frames, automatic))
    sheets = {
        category: make_sheets(category, panels, args.output_dir)
        for category, panels in all_panels.items()
    }
    summaries = {}
    for category, records in all_records.items():
        summaries[category] = {
            "count": len(records),
            "mean_learned_error": float(np.mean([row["learned_mean_error"] for row in records])),
            **{
                f"mean_{name}": float(np.mean([row["automatic"][name] for row in records]))
                for name in (
                    "copy_mean_error",
                    "uncertainty_mean",
                    "pixel_mae_h8",
                    "eef_displacement_h8",
                    "mean_translation_action_l2",
                    "mean_rotation_action_l2",
                    "gripper_sign_changes",
                    "action_saturation_fraction",
                )
            },
        }
    all_selected = all_records["correct"] + all_records["incorrect"]
    correlations = {
        name: pearson(
            [row["learned_mean_error"] for row in all_selected],
            [row["automatic"][name] for row in all_selected],
        )
        for name in (
            "uncertainty_mean",
            "pixel_mae_h8",
            "eef_displacement_h8",
            "mean_translation_action_l2",
            "mean_rotation_action_l2",
            "action_saturation_fraction",
        )
    }
    output = {
        "schema_version": 1,
        "contract_version": "d17-v1",
        "config_sha256": file_sha256(args.config),
        "d16_result_sha256": file_sha256(transition_path),
        "d16_samples_sha256": file_sha256(sample_path),
        "selection_definition": config["selection"],
        "relative_tail_warning": "correct/incorrect are low/high model-error tails, not semantic truth labels",
        "frames": config["frames"],
        "summaries": summaries,
        "selected_tail_correlations": correlations,
        "correlation_null_semantics": "null means undefined because one selected-tail variable is constant",
        "sheets": sheets,
        "samples": all_records,
    }
    output_path = args.output_dir / "review_samples.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"summaries": summaries, "correlations": correlations, "sheets": sheets}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
