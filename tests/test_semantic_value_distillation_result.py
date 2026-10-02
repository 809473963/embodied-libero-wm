import json
from pathlib import Path

import numpy as np

from src.probes.semantic_value_distillation import (
    block_draws,
    bootstrap_mean,
    ci95,
    correctness,
)


ROOT = Path(__file__).resolve().parents[1]


def load_arrays(path: Path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def test_selected_representative_metrics_recompute_exactly() -> None:
    result = json.loads((ROOT / "results/semantic_value_evaluation.json").read_text())
    derived = load_arrays(ROOT / "results/semantic_value_distillation/evaluation_sidecar.npz")
    teacher = load_arrays(ROOT / "results/t1_semantic_teacher/evaluation_sidecar.npz")
    label = derived["ordinary_label"]
    probability = derived["S2__ordinary_probability"].astype(np.float64)
    local = teacher["pair_gap"] <= 8
    non_tie = label != 0
    mask = local & non_tie
    accuracy = correctness(probability[mask], label[mask]).mean()
    assert np.isclose(accuracy, result["metrics"]["S2"]["ordinary"]["local_non_tie"]["accuracy"])
    draws = block_draws(10_000, 20261003)
    bootstrap = bootstrap_mean(
        correctness(probability[mask], label[mask]),
        derived["ordinary_task"][mask], derived["ordinary_demo"][mask],
        (36, 37, 38, 39), draws,
    )
    assert np.allclose(ci95(bootstrap), result["metrics"]["S2"]["ordinary"]["local_accuracy_ci95"])


def test_language_counterfactual_and_image_difference_recompute() -> None:
    result = json.loads((ROOT / "results/semantic_value_evaluation.json").read_text())
    arrays = load_arrays(ROOT / "results/semantic_value_distillation/evaluation_sidecar.npz")
    original, alternate = arrays["cf_original_label"], arrays["cf_alternate_label"]
    changed = original != alternate
    direction = alternate.astype(float) - original.astype(float)
    def agreement(name):
        p0 = arrays[f"{name}__cf_original_probability"].astype(float)
        p1 = arrays[f"{name}__cf_alternate_probability"].astype(float)
        return ((p1 - p0) * direction > 0).astype(float)
    primary, image = agreement("S2"), agreement("IMAGE_ONLY")
    assert np.isclose(primary[changed].mean(), result["metrics"]["S2"]["language_counterfactual"]["teacher_change_agreement"])
    assert np.isclose((primary[changed] - image[changed]).mean(), result["primary_vs_image_only"]["accuracy_difference"])


def test_tie_calibration_recomputes_and_all_probabilities_are_finite() -> None:
    result = json.loads((ROOT / "results/semantic_value_evaluation.json").read_text())
    arrays = load_arrays(ROOT / "results/semantic_value_distillation/evaluation_sidecar.npz")
    teacher = load_arrays(ROOT / "results/t1_semantic_teacher/evaluation_sidecar.npz")
    tie = (teacher["pair_gap"] <= 8) & (arrays["ordinary_label"] == 0)
    probability = arrays["S2__ordinary_probability"].astype(float)
    assert np.isclose(np.abs(probability[tie] - 0.5).mean(),
                      result["metrics"]["S2"]["tie"]["mean_abs_from_half"])
    for key, value in arrays.items():
        if "probability" in key:
            assert np.isfinite(value).all()
            assert np.all((value >= 0) & (value <= 1))


def test_mechanical_block_and_authorization() -> None:
    result = json.loads((ROOT / "results/semantic_value_evaluation.json").read_text())
    assert result["frozen_representative"] == "S2"
    assert result["selected_semantic_value_student"] == "none"
    assert result["complete_gate"] is False
    assert result["authorization"] == "BLOCK_REAL_CANDIDATE_VALUE_BRIDGE"
    assert result["continuing_blocks"] == [
        "BLOCK_WORLD_MODEL_INTERFACE_AUDIT", "BLOCK_SPATIAL_WM_PROTOTYPE"
    ]
    assert result["controller_authorization"] is False
    assert result["real_candidate_bridge_executed"] is False
    assert result["final_labels"] == [
        "SEMANTIC_VALUE_DISTILLATION_BLOCKED",
        "VALUE_STUDENT_LANGUAGE_INSENSITIVE",
        "VALUE_STUDENT_VISUAL_INFORMATION_INSUFFICIENT",
        "FROZEN_VISUAL_VALUE_INFORMATION_INSUFFICIENT",
        "VALUE_STUDENT_SCENE_SHORTCUT",
    ]


def test_evaluation_access_is_exactly_allowlisted() -> None:
    result = json.loads((ROOT / "results/semantic_value_evaluation.json").read_text())
    ledger = result["access_ledger"]
    assert ledger["opened_demo_ids"] == [36, 37, 38, 39]
    assert ledger["metadata_operations"] == []
    assert ledger["forbidden_access_count"] == 0
    assert len(ledger["numeric_datasets_accessed"]) == 41
    assert all("/vision" in path or "language_features" in path
               for path in ledger["numeric_datasets_accessed"])
    assert not any(f"demo_{demo}" in path for demo in range(40, 50)
                   for path in ledger["numeric_datasets_accessed"])
