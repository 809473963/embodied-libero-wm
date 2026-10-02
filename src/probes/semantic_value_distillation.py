"""Frozen-feature semantic value distillation models and audit utilities.

The privileged arrays in this module are used only to construct labels and
strata.  Every deployable model consumes frozen natural-image features and,
except for the explicit image-only ablation, frozen task-language features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.semantic_feature_guard import AllowedDemoFeatureGuard


GAP_NAMES = ("1-2", "3-4", "5-8", "9-16", ">16")
LOCAL_GAP_CODES = frozenset((0, 1, 2))


@dataclass(frozen=True)
class SemanticData:
    vision: np.ndarray
    language: np.ndarray
    task_id: np.ndarray
    demo_id: np.ndarray
    episode_id: np.ndarray
    timestep: np.ndarray
    goal: np.ndarray
    margin: np.ndarray
    proprio: np.ndarray
    ground_status: np.ndarray
    ground_referent: np.ndarray
    pair_left: np.ndarray
    pair_right: np.ndarray
    pair_gap: np.ndarray
    pair_task: np.ndarray
    pair_demo: np.ndarray
    pair_label: np.ndarray


@dataclass(frozen=True)
class PairSet:
    a: np.ndarray
    b: np.ndarray
    language_id: np.ndarray
    label: np.ndarray
    task_id: np.ndarray
    demo_id: np.ndarray
    gap: np.ndarray
    gap_code: np.ndarray
    pair_type: np.ndarray


@dataclass(frozen=True)
class CounterfactualSet:
    a: np.ndarray
    b: np.ndarray
    original_language: np.ndarray
    alternate_language: np.ndarray
    original_label: np.ndarray
    alternate_label: np.ndarray
    same_referent: np.ndarray
    task_id: np.ndarray
    demo_id: np.ndarray


class S1AbsoluteSemantic(nn.Module):
    """Two-head state model; heads retain Boolean-before-margin semantics."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(1024, hidden), nn.GELU())
        self.goal = nn.Linear(hidden, 1)
        self.margin = nn.Linear(hidden, 1)

    def forward(self, vision: torch.Tensor, language: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(torch.cat((vision, language), dim=-1))
        return self.goal(hidden).squeeze(-1), self.margin(hidden).squeeze(-1)


class ScalarSemanticPotential(nn.Module):
    """State scalar trained only through pairwise Bradley--Terry supervision."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, vision: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((vision, language), dim=-1)).squeeze(-1)


class DirectSemanticPair(nn.Module):
    """Small antisymmetric pair model with optional language input."""

    def __init__(self, language: bool = True, hidden: int = 64) -> None:
        super().__init__()
        self.language = bool(language)
        input_dim = 3 * 512 + (512 if self.language else 0)
        self.raw = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def _raw(self, a: torch.Tensor, b: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        values = [a, b, b - a]
        if self.language:
            values.append(language)
        return self.raw(torch.cat(values, dim=-1)).squeeze(-1)

    def forward(self, a: torch.Tensor, b: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        return self._raw(a, b, language) - self._raw(b, a, language)


class LanguageOnlyDiagnostic(nn.Module):
    """Low-capacity task-language prior; reverse-balanced training drives neutrality."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(512, 1)

    def forward(self, language: torch.Tensor) -> torch.Tensor:
        return self.net(language).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters() if value.requires_grad))


def load_partition(
    sidecar_path: str | Path,
    store_path: str | Path,
    partition: str,
) -> tuple[SemanticData, dict[str, Any]]:
    """Load one admitted partition and prove exact row identity by task/demo/time."""

    with np.load(sidecar_path) as archive:
        side = {key: archive[key] for key in archive.files}
    guard = AllowedDemoFeatureGuard(partition, store_path)
    language = guard.read_language().astype(np.float32)
    if language.shape != (10, 512) or not np.isfinite(language).all():
        raise ValueError("invalid frozen language feature matrix")
    vision_parts = []
    cursor = 0
    for task in range(10):
        for demo in guard.allowed_demo_ids:
            image = guard.read_vision(task, demo).astype(np.float32)
            if image.ndim != 2 or image.shape[1] != 512 or not np.isfinite(image).all():
                raise ValueError("invalid frozen visual feature array")
            count = len(image)
            sl = slice(cursor, cursor + count)
            if cursor + count > len(side["task_id"]):
                raise ValueError("feature rows exceed teacher rows")
            if not (
                np.all(side["task_id"][sl] == task)
                and np.all(side["demo_id"][sl] == demo)
                and np.array_equal(side["timestep"][sl], np.arange(count))
            ):
                raise ValueError("D13/T1 row identity mismatch")
            vision_parts.append(image)
            cursor += count
    if cursor != len(side["task_id"]):
        raise ValueError("D13/T1 frame totals differ")
    guard.assert_clean()
    data = SemanticData(
        vision=np.concatenate(vision_parts),
        language=language,
        task_id=side["task_id"].astype(np.int64),
        demo_id=side["demo_id"].astype(np.int64),
        episode_id=side["episode_id"].astype(np.int64),
        timestep=side["timestep"].astype(np.int64),
        goal=side["goal"].astype(np.bool_),
        margin=side["selected_margin_values"].astype(np.float64),
        proprio=side["proprio"].astype(np.float64),
        ground_status=side["ground_status"].astype(np.int8),
        ground_referent=side["ground_referent"].astype(np.int8),
        pair_left=side["pair_left"].astype(np.int64),
        pair_right=side["pair_right"].astype(np.int64),
        pair_gap=side["pair_gap"].astype(np.int64),
        pair_task=side["pair_task_id"].astype(np.int64),
        pair_demo=side["pair_demo_id"].astype(np.int64),
        pair_label=side["pair_label"].astype(np.int8),
    )
    return data, guard.ledger.as_dict()


def fit_standardizer(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = value.mean(axis=0).astype(np.float32)
    std = value.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def standardize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((value - mean) / std).astype(np.float32)


def gap_code(gap: np.ndarray | int) -> np.ndarray:
    value = np.asarray(gap)
    return np.select(
        [value <= 2, value <= 4, value <= 8, value <= 16],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.int8)


def compare_pairs(
    data: SemanticData,
    a: np.ndarray,
    b: np.ndarray,
    language_id: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    language_id = np.asarray(language_id, dtype=np.int64)
    episode = data.episode_id[a]
    if not np.all(data.ground_status[episode, language_id] == 1):
        raise ValueError("pair requested a non-UNIQUE language grounding")
    ref = data.ground_referent[episode, language_id]
    ga, gb = data.goal[a, ref], data.goal[b, ref]
    delta = data.margin[b, ref] - data.margin[a, ref]
    result = np.zeros(len(a), dtype=np.int8)
    result[(~ga) & gb] = 1
    result[ga & (~gb)] = -1
    same = ga == gb
    result[same & (delta > epsilon)] = 1
    result[same & (delta < -epsilon)] = -1
    return result


def ordinary_pairs(data: SemanticData, demos: Iterable[int] | None = None) -> PairSet:
    mask = np.ones(len(data.pair_left), dtype=np.bool_)
    if demos is not None:
        mask &= np.isin(data.pair_demo, tuple(int(value) for value in demos))
    index = np.flatnonzero(mask)
    return PairSet(
        data.pair_left[index], data.pair_right[index], data.pair_task[index],
        data.pair_label[index], data.pair_task[index], data.pair_demo[index],
        data.pair_gap[index], gap_code(data.pair_gap[index]),
        np.full(len(index), "ordinary", dtype="U24"),
    )


def _long_pairs(data: SemanticData, demos: tuple[int, ...], epsilon: float) -> PairSet:
    rows: list[tuple[int, int, int, int, int, int]] = []
    for episode in np.unique(data.episode_id):
        ids = np.flatnonzero(data.episode_id == episode)
        if int(data.demo_id[ids[0]]) not in demos:
            continue
        task, demo = int(data.task_id[ids[0]]), int(data.demo_id[ids[0]])
        candidates = [(int(ids[left]), int(ids[left + gap]), gap)
                      for gap in range(17, min(33, len(ids)))
                      for left in range(len(ids) - gap)]
        if not candidates:
            continue
        chosen = np.linspace(0, len(candidates) - 1, min(64, len(candidates)), dtype=np.int64)
        for pos in chosen:
            a, b, gap = candidates[int(pos)]
            rows.append((a, b, task, demo, gap, task))
    if not rows:
        empty = np.empty(0, dtype=np.int64)
        return PairSet(empty, empty, empty, empty.astype(np.int8), empty, empty, empty,
                       empty.astype(np.int8), np.empty(0, dtype="U24"))
    raw = np.asarray(rows, dtype=np.int64)
    label = compare_pairs(data, raw[:, 0], raw[:, 1], raw[:, 5], epsilon)
    return PairSet(raw[:, 0], raw[:, 1], raw[:, 5], label, raw[:, 2], raw[:, 3],
                   raw[:, 4], np.full(len(raw), 4, dtype=np.int8),
                   np.full(len(raw), "ordinary", dtype="U24"))


def _even_take(indices: list[int], limit: int) -> list[int]:
    if len(indices) <= limit:
        return indices
    positions = np.linspace(0, len(indices) - 1, limit, dtype=np.int64)
    return [indices[int(position)] for position in positions]


def _append_record(
    records: dict[tuple[int, int, int, int], tuple[int, int, int, int, int, int, int, int, str]],
    a: int, b: int, language: int, label: int, task: int, demo: int,
    gap: int, code: int, kind: str,
) -> None:
    records[(a, b, language, label)] = (a, b, language, label, task, demo, gap, code, kind)


def make_training_pairs(
    data: SemanticData,
    demos: Iterable[int],
    epsilon: float,
    sampler: dict[str, Any],
    robot_match_threshold: float,
    robot_difference_threshold: float,
    proprio_mean: np.ndarray,
    proprio_std: np.ndarray,
) -> PairSet:
    """Build the frozen balanced mix and add an exact reverse for every pair."""

    demos = tuple(int(value) for value in demos)
    base = ordinary_pairs(data, demos)
    long = _long_pairs(data, demos, epsilon)
    arrays = {}
    for field in ("a", "b", "language_id", "label", "task_id", "demo_id", "gap", "gap_code"):
        arrays[field] = np.concatenate((getattr(base, field), getattr(long, field)))
    records: dict[tuple[int, int, int, int], tuple[int, int, int, int, int, int, int, int, str]] = {}

    ordinary_groups: dict[tuple[int, ...], list[int]] = {}
    for index in range(len(arrays["a"])):
        key = (int(arrays["task_id"][index]), int(arrays["demo_id"][index]),
               int(arrays["gap_code"][index]), int(arrays["label"][index]))
        ordinary_groups.setdefault(key, []).append(index)
    for indices in ordinary_groups.values():
        for index in _even_take(indices, int(sampler["ordinary_per_stratum"])):
            _append_record(records, *(int(arrays[name][index]) for name in
                ("a", "b", "language_id", "label", "task_id", "demo_id", "gap", "gap_code")), "ordinary")

    cf_candidates: dict[tuple[int, ...], list[tuple[int, int, int, int, int, int, int, int]]] = {}
    for index in range(len(arrays["a"])):
        a, b = int(arrays["a"][index]), int(arrays["b"][index])
        episode, task = int(data.episode_id[a]), int(arrays["task_id"][index])
        original = int(arrays["label"][index])
        correct_ref = int(data.ground_referent[episode, task])
        for alternate in range(10):
            if alternate == task or data.ground_status[episode, alternate] != 1:
                continue
            alternate_label = int(compare_pairs(
                data, np.asarray([a]), np.asarray([b]), np.asarray([alternate]), epsilon
            )[0])
            same_ref = int(data.ground_referent[episode, alternate] == correct_ref)
            key = (task, int(arrays["demo_id"][index]), int(arrays["gap_code"][index]),
                   same_ref, original, alternate_label)
            cf_candidates.setdefault(key, []).append(
                (a, b, alternate, alternate_label, task, int(arrays["demo_id"][index]),
                 int(arrays["gap"][index]), int(arrays["gap_code"][index]))
            )
    for candidates in cf_candidates.values():
        for row_index in _even_take(list(range(len(candidates))), int(sampler["counterfactual_per_stratum"])):
            row = candidates[row_index]
            _append_record(records, *row, "counterfactual")
            # The same visual pair under its correct language is retained so
            # ignoring language creates irreducible error whenever labels differ.
            task = row[4]
            original_label = int(compare_pairs(
                data, np.asarray([row[0]]), np.asarray([row[1]]), np.asarray([task]), epsilon
            )[0])
            _append_record(records, row[0], row[1], task, original_label,
                           row[4], row[5], row[6], row[7], "counterfactual_anchor")

    z = (data.proprio - np.asarray(proprio_mean)) / np.asarray(proprio_std)
    distances = np.linalg.norm(z[arrays["b"]] - z[arrays["a"]], axis=1)
    control_groups: dict[tuple[int, ...], list[int]] = {}
    for index in range(len(arrays["a"])):
        if arrays["gap"][index] > 8:
            continue
        label = int(arrays["label"][index])
        control = 0 if label == 0 and distances[index] >= robot_difference_threshold else (
            1 if label != 0 and distances[index] <= robot_match_threshold else -1
        )
        if control >= 0:
            key = (int(arrays["task_id"][index]), int(arrays["demo_id"][index]), control)
            control_groups.setdefault(key, []).append(index)
    for key, indices in control_groups.items():
        for index in _even_take(indices, int(sampler["robot_control_per_stratum"])):
            _append_record(records, *(int(arrays[name][index]) for name in
                ("a", "b", "language_id", "label", "task_id", "demo_id", "gap", "gap_code")),
                "semantic_tie_robot_diff" if key[-1] == 0 else "robot_match_semantic_diff")

    # Self-pairs under every uniquely grounded real instruction explicitly
    # retain the P=0.5 target without inventing a binary direction.
    for episode in np.unique(data.episode_id):
        ids = np.flatnonzero(data.episode_id == episode)
        if int(data.demo_id[ids[0]]) not in demos:
            continue
        anchors = np.linspace(0, len(ids) - 1, int(sampler["self_pair_frames_per_demo"]), dtype=np.int64)
        task, demo = int(data.task_id[ids[0]]), int(data.demo_id[ids[0]])
        for position in anchors:
            frame = int(ids[int(position)])
            for language in range(10):
                if data.ground_status[episode, language] == 1:
                    _append_record(records, frame, frame, language, 0, task, demo, 0, 0, "same_state_tie")

    forward = list(records.values())
    for row in forward:
        a, b, language, label, task, demo, gap, code, kind = row
        if a != b:
            _append_record(records, b, a, language, -label, task, demo, -gap, code, f"reverse_{kind}")
    rows = list(records.values())
    rows.sort(key=lambda row: (row[4], row[5], row[7], row[8], row[0], row[1], row[2], row[3]))
    matrix = np.asarray([row[:8] for row in rows], dtype=np.int64)
    kinds = np.asarray([row[8] for row in rows], dtype="U32")
    return PairSet(matrix[:, 0], matrix[:, 1], matrix[:, 2], matrix[:, 3].astype(np.int8),
                   matrix[:, 4], matrix[:, 5], matrix[:, 6], matrix[:, 7].astype(np.int8), kinds)


def make_state_targets(
    data: SemanticData, demos: Iterable[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames, languages, goals, margins = [], [], [], []
    admitted = tuple(int(value) for value in demos)
    for frame in np.flatnonzero(np.isin(data.demo_id, admitted)):
        episode = int(data.episode_id[frame])
        for language in range(10):
            if data.ground_status[episode, language] != 1:
                continue
            ref = int(data.ground_referent[episode, language])
            frames.append(frame); languages.append(language)
            goals.append(float(data.goal[frame, ref])); margins.append(float(data.margin[frame, ref]))
    return (np.asarray(frames, dtype=np.int64), np.asarray(languages, dtype=np.int64),
            np.asarray(goals, dtype=np.float32), np.asarray(margins, dtype=np.float32))


def make_counterfactual_set(data: SemanticData, epsilon: float, local_only: bool = True) -> CounterfactualSet:
    indices = np.flatnonzero(data.pair_gap <= 8) if local_only else np.arange(len(data.pair_left))
    rows = []
    for index in indices:
        a, b = int(data.pair_left[index]), int(data.pair_right[index])
        task, demo, original = int(data.pair_task[index]), int(data.pair_demo[index]), int(data.pair_label[index])
        episode = int(data.episode_id[a]); correct_ref = int(data.ground_referent[episode, task])
        for alternate in range(10):
            if alternate == task or data.ground_status[episode, alternate] != 1:
                continue
            alternate_label = int(compare_pairs(
                data, np.asarray([a]), np.asarray([b]), np.asarray([alternate]), epsilon
            )[0])
            same = int(data.ground_referent[episode, alternate] == correct_ref)
            rows.append((a, b, task, alternate, original, alternate_label, same, task, demo))
    matrix = np.asarray(rows, dtype=np.int64)
    return CounterfactualSet(
        matrix[:, 0], matrix[:, 1], matrix[:, 2], matrix[:, 3],
        matrix[:, 4].astype(np.int8), matrix[:, 5].astype(np.int8),
        matrix[:, 6].astype(np.bool_), matrix[:, 7], matrix[:, 8],
    )


def make_cross_demo_controls(
    data: SemanticData,
    epsilon: float,
    points: int,
    clear_margin_min: float,
) -> dict[str, PairSet]:
    episodes: dict[tuple[int, int], np.ndarray] = {}
    for episode in np.unique(data.episode_id):
        ids = np.flatnonzero(data.episode_id == episode)
        episodes[(int(data.task_id[ids[0]]), int(data.demo_id[ids[0]]))] = ids
    demos = sorted(np.unique(data.demo_id).tolist())
    output: dict[str, list[tuple[int, int, int, int, int, int, int, int, str]]] = {"matched": [], "different": []}
    for task in range(10):
        for slot, demo in enumerate(demos):
            other = demos[(slot + 1) % len(demos)]
            source, target = episodes[(task, demo)], episodes[(task, other)]
            ref_source = int(data.ground_referent[int(data.episode_id[source[0]]), task])
            ref_target = int(data.ground_referent[int(data.episode_id[target[0]]), task])
            anchors = np.linspace(0, len(source) - 1, points, dtype=np.int64)
            for anchor in anchors:
                a = int(source[int(anchor)])
                ga, ma = bool(data.goal[a, ref_source]), float(data.margin[a, ref_source])
                same_goal = data.goal[target, ref_target] == ga
                delta = np.abs(data.margin[target, ref_target] - ma)
                matched_candidates = np.flatnonzero(same_goal & (delta <= epsilon))
                if len(matched_candidates):
                    local = int(matched_candidates[np.argmin(delta[matched_candidates])])
                    b = int(target[local])
                    for x, y in ((a, b), (b, a)):
                        output["matched"].append((x, y, task, 0, task, demo, 0, -1, "cross_demo_match"))
                labels = compare_pairs(
                    data, np.full(len(target), a), target,
                    np.full(len(target), task), epsilon,
                )
                goal_diff = data.goal[target, ref_target] != ga
                clear = (labels != 0) & (goal_diff | (delta >= clear_margin_min))
                candidates = np.flatnonzero(clear)
                if len(candidates):
                    # Choose the least extreme state that still clears the frozen margin.
                    local = int(candidates[np.argmin(delta[candidates])])
                    b, label = int(target[local]), int(labels[local])
                    output["different"].append((a, b, task, label, task, demo,
                                                  int(data.timestep[b]) - int(data.timestep[a]), -1,
                                                  "cross_demo_difference"))
                    output["different"].append((b, a, task, -label, task, demo,
                                                  int(data.timestep[a]) - int(data.timestep[b]), -1,
                                                  "cross_demo_difference_reverse"))
    result = {}
    for name, rows in output.items():
        matrix = np.asarray([row[:8] for row in rows], dtype=np.int64)
        result[name] = PairSet(
            matrix[:, 0], matrix[:, 1], matrix[:, 2], matrix[:, 3].astype(np.int8),
            matrix[:, 4], matrix[:, 5], matrix[:, 6], matrix[:, 7].astype(np.int8),
            np.asarray([row[8] for row in rows], dtype="U32"),
        )
    return result


def set_deterministic(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def _loader(*arrays: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    dataset = TensorDataset(*(torch.from_numpy(value) for value in arrays))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def train_s1(
    vision: np.ndarray, language: np.ndarray,
    frames: np.ndarray, language_ids: np.ndarray,
    goal: np.ndarray, margin: np.ndarray,
    epochs: int, batch_size: int, seed: int,
) -> tuple[S1AbsoluteSemantic, dict[str, float]]:
    set_deterministic(seed)
    model = S1AbsoluteSemantic()
    margin_mean, margin_std = float(margin.mean()), float(margin.std())
    if margin_std < 1e-6: margin_std = 1.0
    margin_target = ((margin - margin_mean) / margin_std).astype(np.float32)
    positives = max(float(goal.sum()), 1.0)
    pos_weight = min(10.0, float((len(goal) - goal.sum()) / positives))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = _loader(frames, language_ids, goal, margin_target,
                     batch_size=batch_size, seed=seed)
    vt, lt = torch.from_numpy(vision), torch.from_numpy(language)
    model.train()
    for _ in range(epochs):
        for fi, li, gy, my in loader:
            optimizer.zero_grad(set_to_none=True)
            gl, mp = model(vt[fi], lt[li])
            goal_loss = nn.functional.binary_cross_entropy_with_logits(
                gl, gy, pos_weight=torch.tensor(pos_weight)
            )
            loss = goal_loss + nn.functional.mse_loss(mp, my)
            loss.backward(); optimizer.step()
    return model.eval(), {"margin_mean": margin_mean, "margin_std": margin_std,
                          "goal_positive_weight": pos_weight}


def train_scalar_potential(
    vision: np.ndarray, language: np.ndarray, pairs: PairSet,
    epochs: int, batch_size: int, seed: int,
) -> ScalarSemanticPotential:
    set_deterministic(seed)
    model = ScalarSemanticPotential()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    target = ((pairs.label.astype(np.float32) + 1.0) / 2.0).astype(np.float32)
    loader = _loader(pairs.a, pairs.b, pairs.language_id, target,
                     batch_size=batch_size, seed=seed)
    vt, lt = torch.from_numpy(vision), torch.from_numpy(language)
    model.train()
    for _ in range(epochs):
        for a, b, li, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(vt[b], lt[li]) - model(vt[a], lt[li])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward(); optimizer.step()
    return model.eval()


def train_direct(
    vision: np.ndarray, language: np.ndarray, pairs: PairSet,
    epochs: int, batch_size: int, seed: int, use_language: bool,
) -> DirectSemanticPair:
    set_deterministic(seed)
    model = DirectSemanticPair(language=use_language)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    target = ((pairs.label.astype(np.float32) + 1.0) / 2.0).astype(np.float32)
    loader = _loader(pairs.a, pairs.b, pairs.language_id, target,
                     batch_size=batch_size, seed=seed)
    vt, lt = torch.from_numpy(vision), torch.from_numpy(language)
    model.train()
    for _ in range(epochs):
        for a, b, li, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(vt[a], vt[b], lt[li])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward(); optimizer.step()
    return model.eval()


def train_language_only(
    language: np.ndarray, pairs: PairSet, epochs: int, batch_size: int, seed: int,
) -> LanguageOnlyDiagnostic:
    set_deterministic(seed)
    model = LanguageOnlyDiagnostic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    target = ((pairs.label.astype(np.float32) + 1.0) / 2.0).astype(np.float32)
    loader = _loader(pairs.language_id, target, batch_size=batch_size, seed=seed)
    lt = torch.from_numpy(language)
    model.train()
    for li, y in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(lt[li])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward(); optimizer.step()
    return model.eval()


def terminal_prototypes(data: SemanticData, demos: Iterable[int], tail: int) -> np.ndarray:
    prototypes = []
    for task in range(10):
        values = []
        for demo in demos:
            ids = np.flatnonzero((data.task_id == task) & (data.demo_id == int(demo)))
            values.append(data.vision[ids[-tail:]])
        value = np.concatenate(values).mean(axis=0).astype(np.float64)
        value /= max(float(np.linalg.norm(value)), 1e-12)
        prototypes.append(value)
    return np.asarray(prototypes, dtype=np.float32)


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def predict_logits(
    family: str,
    model: nn.Module | None,
    vision: np.ndarray,
    language: np.ndarray,
    pairs: PairSet,
    prototypes: np.ndarray | None = None,
    batch_size: int = 2048,
) -> np.ndarray:
    if family == "S0":
        if prototypes is None: raise ValueError("S0 prototypes missing")
        image = vision.astype(np.float64)
        image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
        a = np.sum(image[pairs.a] * prototypes[pairs.language_id], axis=1)
        b = np.sum(image[pairs.b] * prototypes[pairs.language_id], axis=1)
        return b - a
    values = []
    vt, lt = torch.from_numpy(vision), torch.from_numpy(language)
    assert model is not None
    with torch.inference_mode():
        for start in range(0, len(pairs.a), batch_size):
            stop = min(start + batch_size, len(pairs.a))
            a = torch.from_numpy(pairs.a[start:stop]); b = torch.from_numpy(pairs.b[start:stop])
            li = torch.from_numpy(pairs.language_id[start:stop])
            if family == "S1":
                assert isinstance(model, S1AbsoluteSemantic)
                ga, ma = model(vt[a], lt[li]); gb, mb = model(vt[b], lt[li])
                # Each coordinate is bounded in [0,1].  Adding an exact 2-unit
                # Boolean class offset guarantees predicted goal priority.
                sa = 2.0 * (ga >= 0).to(ma.dtype) + torch.sigmoid(ma)
                sb = 2.0 * (gb >= 0).to(mb.dtype) + torch.sigmoid(mb)
                values.append((sb - sa).numpy())
            elif family == "S1.5":
                assert isinstance(model, ScalarSemanticPotential)
                values.append((model(vt[b], lt[li]) - model(vt[a], lt[li])).numpy())
            elif family in ("S2", "IMAGE_ONLY"):
                assert isinstance(model, DirectSemanticPair)
                values.append(model(vt[a], vt[b], lt[li]).numpy())
            elif family == "LANGUAGE_ONLY":
                assert isinstance(model, LanguageOnlyDiagnostic)
                values.append(model(lt[li]).numpy())
            else:
                raise ValueError(f"unknown family {family}")
    return np.concatenate(values).astype(np.float64)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, grid: Iterable[float]) -> float:
    target = (np.asarray(labels, dtype=np.float64) + 1.0) / 2.0
    trials = []
    for value in grid:
        probability = np.clip(sigmoid(logits * float(value)), 1e-7, 1 - 1e-7)
        loss = -np.mean(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
        trials.append((float(loss), float(value)))
    return min(trials)[1]


def rank_auc(score: np.ndarray, positive: np.ndarray) -> float:
    score, positive = np.asarray(score), np.asarray(positive, dtype=np.bool_)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if not n1 or not n0: return float("nan")
    ranks = rankdata(score, method="average")
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def correctness(probability: np.ndarray, label: np.ndarray) -> np.ndarray:
    probability, label = np.asarray(probability), np.asarray(label)
    prediction = np.sign(probability - 0.5)
    result = (prediction == label).astype(np.float64)
    result[prediction == 0] = 0.5
    return result


def non_tie_metrics(probability: np.ndarray, label: np.ndarray) -> dict[str, float]:
    keep = np.asarray(label) != 0
    probability = np.clip(np.asarray(probability, dtype=np.float64)[keep], 1e-7, 1 - 1e-7)
    signed = np.asarray(label)[keep]
    target = (signed > 0).astype(np.float64)
    return {
        "rows": int(len(target)),
        "accuracy": float(correctness(probability, signed).mean()),
        "auroc": rank_auc(probability, target > 0),
        "log_loss": float(-np.mean(target * np.log(probability) + (1 - target) * np.log(1 - probability))),
        "brier": float(np.mean(np.square(probability - target))),
    }


def block_draws(replicates: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 4, size=(replicates, 10, 4), dtype=np.int8)


def bootstrap_mean(
    value: np.ndarray, task: np.ndarray, demo: np.ndarray,
    demo_values: tuple[int, ...], draws: np.ndarray,
) -> np.ndarray:
    if not demo_values:
        raise ValueError("bootstrap needs at least one demo value")
    sums = np.zeros((10, len(demo_values)), dtype=np.float64)
    counts = np.zeros((10, len(demo_values)), dtype=np.float64)
    for task_id in range(10):
        for slot, demo_id in enumerate(demo_values):
            mask = (task == task_id) & (demo == demo_id)
            sums[task_id, slot] = np.asarray(value)[mask].sum(); counts[task_id, slot] = mask.sum()
    sampled_sum = np.zeros(len(draws)); sampled_count = np.zeros(len(draws))
    for task_id in range(10):
        indices = draws[:, task_id] % len(demo_values)
        sampled_sum += sums[task_id][indices].sum(axis=1)
        sampled_count += counts[task_id][indices].sum(axis=1)
    if np.any(sampled_count == 0): raise ValueError("empty bootstrap draw")
    return sampled_sum / sampled_count


def ci95(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def summarize_ordinary(
    probability: np.ndarray, pairs: PairSet,
    demos: tuple[int, ...], draws: np.ndarray,
) -> dict[str, Any]:
    local = np.isin(pairs.gap_code, tuple(LOCAL_GAP_CODES))
    non_tie = pairs.label != 0
    primary = local & non_tie
    credit = correctness(probability, pairs.label)
    boot = bootstrap_mean(credit[primary], pairs.task_id[primary], pairs.demo_id[primary], demos, draws)
    result: dict[str, Any] = {
        "local_non_tie": non_tie_metrics(probability[local], pairs.label[local]),
        "local_accuracy_ci95": ci95(boot),
        "tasks_above_60pct": int(sum(
            correctness(probability[primary & (pairs.task_id == task)], pairs.label[primary & (pairs.task_id == task)]).mean() > 0.60
            for task in range(10)
        )),
        "per_task": {}, "per_gap": {},
    }
    for task in range(10):
        mask = primary & (pairs.task_id == task)
        result["per_task"][str(task)] = non_tie_metrics(probability[mask], pairs.label[mask])
    for code, name in enumerate(GAP_NAMES):
        mask = (pairs.gap_code == code) & non_tie
        result["per_gap"][name] = non_tie_metrics(probability[mask], pairs.label[mask])
    return result


def tie_metrics(
    probability: np.ndarray, pairs: PairSet,
    demos: tuple[int, ...], draws: np.ndarray,
) -> dict[str, Any]:
    mask = np.isin(pairs.gap_code, tuple(LOCAL_GAP_CODES)) & (pairs.label == 0)
    deviation = np.abs(probability[mask] - 0.5)
    boot = bootstrap_mean(deviation, pairs.task_id[mask], pairs.demo_id[mask], demos, draws)
    return {
        "rows": int(mask.sum()), "mean_abs_from_half": float(deviation.mean()),
        "mean_ci95": ci95(boot), "median_abs_from_half": float(np.median(deviation)),
        "p90_abs_from_half": float(np.quantile(deviation, 0.90)),
    }


def summarize_counterfactual(
    original_probability: np.ndarray,
    alternate_probability: np.ndarray,
    cf: CounterfactualSet,
    demos: tuple[int, ...], draws: np.ndarray,
) -> dict[str, Any]:
    changed = cf.original_label != cf.alternate_label
    direction = cf.alternate_label.astype(np.float64) - cf.original_label.astype(np.float64)
    agreement = ((alternate_probability - original_probability) * direction > 0).astype(np.float64)
    changed_agreement = agreement[changed]
    boot = bootstrap_mean(changed_agreement, cf.task_id[changed], cf.demo_id[changed], demos, draws)
    per_task = {}
    for task in range(10):
        mask = changed & (cf.task_id == task)
        per_task[str(task)] = {
            "rows": int(mask.sum()),
            "agreement": float(agreement[mask].mean()) if mask.any() else 0.0,
        }
    reversal = (cf.original_label * cf.alternate_label) == -1
    reversal_correct = (
        ((original_probability > 0.5) & (cf.original_label == 1)) |
        ((original_probability < 0.5) & (cf.original_label == -1))
    ) & (
        ((alternate_probability > 0.5) & (cf.alternate_label == 1)) |
        ((alternate_probability < 0.5) & (cf.alternate_label == -1))
    )
    reversal_boot = bootstrap_mean(reversal_correct[reversal].astype(float), cf.task_id[reversal],
                                    cf.demo_id[reversal], demos, draws)
    non_tie_to_tie = (cf.original_label != 0) & (cf.alternate_label == 0)
    reduction = np.abs(original_probability[non_tie_to_tie] - 0.5) - np.abs(alternate_probability[non_tie_to_tie] - 0.5)
    same = cf.same_referent
    same_change = np.abs(alternate_probability[same] - original_probability[same])
    same_flip = (np.sign(alternate_probability[same] - 0.5) != np.sign(original_probability[same] - 0.5)).astype(float)
    different = ~same
    different_changed = different & changed
    different_agreement = agreement[different_changed]
    return {
        "eligible_rows": int(len(changed)),
        "teacher_changed_rows": int(changed.sum()),
        "teacher_change_agreement": float(changed_agreement.mean()),
        "teacher_change_agreement_ci95": ci95(boot),
        "positive_task_count": int(sum(row["agreement"] > 0.5 for row in per_task.values())),
        "per_task": per_task,
        "true_reversal": {
            "rows": int(reversal.sum()), "accuracy": float(reversal_correct[reversal].mean()),
            "accuracy_ci95": ci95(reversal_boot),
        },
        "non_tie_to_tie": {
            "rows": int(non_tie_to_tie.sum()),
            "mean_distance_reduction": float(reduction.mean()),
            "median_distance_reduction": float(np.median(reduction)),
            "fraction_improved": float(np.mean(reduction > 0)),
        },
        "same_referent": {
            "rows": int(same.sum()), "mean_abs_probability_change": float(same_change.mean()),
            "median_abs_probability_change": float(np.median(same_change)),
            "p90_abs_probability_change": float(np.quantile(same_change, 0.90)),
            "false_preference_flip_rate": float(same_flip.mean()),
        },
        "different_referent": {
            "rows": int(different.sum()), "teacher_changed_rows": int(different_changed.sum()),
            "teacher_change_agreement": float(different_agreement.mean()) if len(different_agreement) else 0.0,
        },
        "row_agreement": agreement,
        "changed_mask": changed,
    }


def cross_demo_metrics(probability: np.ndarray, pairs: PairSet) -> dict[str, Any]:
    if np.all(pairs.label == 0):
        deviation = np.abs(probability - 0.5)
        return {"rows": int(len(probability)), "mean_abs_from_half": float(deviation.mean()),
                "median_abs_from_half": float(np.median(deviation)),
                "p90_abs_from_half": float(np.quantile(deviation, 0.90))}
    return non_tie_metrics(probability, pairs.label)


def triplet_pairs(
    data: SemanticData, points: int = 8, epsilon: float = 0.004065508647683736
) -> tuple[PairSet, np.ndarray]:
    rows=[]; groups=[]; group=0
    for episode in np.unique(data.episode_id):
        ids=np.flatnonzero(data.episode_id==episode)
        chosen=np.unique(np.linspace(0,len(ids)-1,points,dtype=np.int64))
        selected=ids[chosen]; task=int(data.task_id[ids[0]]); demo=int(data.demo_id[ids[0]])
        for i in range(len(selected)-2):
            a,b,c=(int(selected[i]),int(selected[i+1]),int(selected[i+2]))
            for x,y in ((a,b),(b,c),(c,a)):
                label=int(compare_pairs(data,np.asarray([x]),np.asarray([y]),np.asarray([task]),epsilon)[0])
                rows.append((x,y,task,label,task,demo,0,-1,"triplet"));groups.append(group)
            group+=1
    matrix=np.asarray([row[:8] for row in rows],dtype=np.int64)
    return PairSet(matrix[:,0],matrix[:,1],matrix[:,2],matrix[:,3].astype(np.int8),matrix[:,4],matrix[:,5],matrix[:,6],matrix[:,7].astype(np.int8),np.asarray([r[8] for r in rows],dtype="U16")),np.asarray(groups,dtype=np.int64)


def cycle_rate(probability: np.ndarray, group: np.ndarray) -> dict[str, Any]:
    cycles=[]
    for value in np.unique(group):
        p=probability[group==value]
        if len(p)!=3: raise ValueError("triplet group must contain three edges")
        cycles.append(bool(np.all(p>0.5) or np.all(p<0.5)))
    return {"triplets":len(cycles),"cycle_rate":float(np.mean(cycles))}
