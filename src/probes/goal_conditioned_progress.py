"""Frozen-feature demonstration-progress models and audit utilities.

Labels in this module come only from temporal order in expert demonstrations.
They are not rewards, success labels, or task-optimal values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


GAP_NAMES = ("1-2", "3-4", "5-8", "9-16", ">16", "terminal-anchor")
LOCAL_GAPS = frozenset(("1-2", "3-4", "5-8"))


@dataclass(frozen=True)
class TrajectoryData:
    vision: np.ndarray
    proprio: np.ndarray
    language_by_task: np.ndarray
    task_id: np.ndarray
    demo_id: np.ndarray
    timestep: np.ndarray
    progress: np.ndarray
    trajectories: tuple[dict[str, int], ...]


@dataclass(frozen=True)
class PairData:
    a: np.ndarray
    b: np.ndarray
    label: np.ndarray
    task_id: np.ndarray
    demo_id: np.ndarray
    signed_gap: np.ndarray
    gap_bin: np.ndarray


class ScalarProgressMLP(nn.Module):
    def __init__(self, state_dim: int, language_dim: int = 512, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + language_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state, language), dim=-1)).squeeze(-1)


class AntisymmetricPairMLP(nn.Module):
    """Pair model whose reverse probability is exactly complementary."""

    def __init__(self, state_dim: int, language_dim: int = 512, hidden: int = 64) -> None:
        super().__init__()
        input_dim = 3 * state_dim + language_dim
        self.raw = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def _raw(self, a: torch.Tensor, b: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        return self.raw(torch.cat((a, b, b - a, language), dim=-1)).squeeze(-1)

    def forward(self, a: torch.Tensor, b: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        return self._raw(a, b, language) - self._raw(b, a, language)


def parameter_count(model: nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters() if value.requires_grad))


def load_trajectories(store_path: str | Path, demo_ids: Iterable[int]) -> TrajectoryData:
    admitted = tuple(int(value) for value in demo_ids)
    if set(admitted) not in ({31, 32, 33, 34}, {36, 37, 38, 39}):
        raise ValueError("only the frozen calibration or evaluation partition is allowed")
    vision, proprio, task_id, demo_id, timestep, progress = [], [], [], [], [], []
    trajectories: list[dict[str, int]] = []
    cursor = 0
    with h5py.File(store_path, "r") as store:
        if str(store.attrs.get("world_model_data_contract")) != "d13-v1":
            raise ValueError("unexpected feature-store contract")
        language = store["task_metadata/language_features"][:].astype(np.float32)
        if language.shape != (10, 512):
            raise ValueError("unexpected language feature shape")
        for task in range(10):
            for demo in admitted:
                group = store[f"tasks/task_{task}/demos/demo_{demo}"]
                if int(group.attrs["task_id"]) != task or int(group.attrs["demo_id"]) != demo:
                    raise ValueError("HDF5 identity mismatch")
                count = int(group.attrs["observation_count"])
                image = group["vision"][:].astype(np.float32)
                prop = group["proprio"][:].astype(np.float32)
                if image.shape != (count, 512) or prop.shape != (count, 12):
                    raise ValueError("unexpected trajectory shape")
                vision.append(image)
                proprio.append(prop)
                task_id.append(np.full(count, task, dtype=np.int64))
                demo_id.append(np.full(count, demo, dtype=np.int64))
                timestep.append(np.arange(count, dtype=np.int64))
                progress.append(np.arange(count, dtype=np.float32) / np.float32(count - 1))
                trajectories.append({"task": task, "demo": demo, "start": cursor, "length": count})
                cursor += count
    arrays = tuple(np.concatenate(values) for values in (vision, proprio, task_id, demo_id, timestep, progress))
    if not all(np.isfinite(value).all() for value in (*arrays, language)):
        raise FloatingPointError("trajectory store contains NaN/Inf")
    return TrajectoryData(*arrays[:2], language, *arrays[2:], tuple(trajectories))


def _even_sample(items: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if len(items) < count:
        raise ValueError(f"pair population {len(items)} is smaller than frozen sample count {count}")
    positions = np.linspace(0, len(items) - 1, count, dtype=np.int64)
    return [items[int(index)] for index in positions]


def make_same_demo_pairs(data: TrajectoryData, per_bin: int = 32) -> PairData:
    a, b, label, task_id, demo_id, signed_gap, gap_bin = [], [], [], [], [], [], []
    for trajectory in data.trajectories:
        start, length = trajectory["start"], trajectory["length"]
        definitions: list[tuple[str, list[int]]] = [
            ("1-2", [1, 2]),
            ("3-4", [3, 4]),
            ("5-8", [5, 6, 7, 8]),
            ("9-16", list(range(9, 17))),
            (">16", list(range(17, min(33, length)))),
        ]
        populations: list[tuple[str, list[tuple[int, int]]]] = []
        for name, gaps in definitions:
            population = [(left, left + gap) for gap in gaps for left in range(length - gap)]
            populations.append((name, population))
        populations.append(("terminal-anchor", [(left, length - 1) for left in range(length - 1)]))
        for name, population in populations:
            for left, right in _even_sample(population, per_bin):
                ga, gb = start + left, start + right
                gap = right - left
                for first, second, target, sign in ((ga, gb, 1.0, 1), (gb, ga, 0.0, -1)):
                    a.append(first); b.append(second); label.append(target)
                    task_id.append(trajectory["task"]); demo_id.append(trajectory["demo"])
                    signed_gap.append(sign * gap); gap_bin.append(GAP_NAMES.index(name))
    result = PairData(
        np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64),
        np.asarray(label, dtype=np.float32), np.asarray(task_id, dtype=np.int64),
        np.asarray(demo_id, dtype=np.int64), np.asarray(signed_gap, dtype=np.int64),
        np.asarray(gap_bin, dtype=np.int8),
    )
    if len(result.a) != len(data.trajectories) * len(GAP_NAMES) * per_bin * 2:
        raise AssertionError("unexpected pair count")
    return result


def make_cross_demo_controls(data: TrajectoryData, points: int = 16) -> dict[str, PairData]:
    records: dict[tuple[int, int], dict[str, int]] = {
        (item["task"], item["demo"]): item for item in data.trajectories
    }
    demos_by_task = {
        task: sorted(item["demo"] for item in data.trajectories if item["task"] == task)
        for task in range(10)
    }
    output: dict[str, list[Any]] = {name: [[] for _ in range(7)] for name in ("matched", "mismatched")}
    matched_progress = np.linspace(0.1, 0.9, points)
    low_progress = np.linspace(0.1, 0.55, points)
    for task, demos in demos_by_task.items():
        for slot, demo_a in enumerate(demos):
            demo_b = demos[(slot + 1) % len(demos)]
            ta, tb = records[(task, demo_a)], records[(task, demo_b)]
            for p in matched_progress:
                ia = ta["start"] + int(round(p * (ta["length"] - 1)))
                ib = tb["start"] + int(round(p * (tb["length"] - 1)))
                values = (ia, ib, 0.5, task, demo_a, 0, -1)
                for target, value in zip(output["matched"], values): target.append(value)
            for p in low_progress:
                q = p + 0.25
                ia = ta["start"] + int(round(p * (ta["length"] - 1)))
                ib = tb["start"] + int(round(q * (tb["length"] - 1)))
                for first, second, target_value, gap in ((ia, ib, 1.0, 25), (ib, ia, 0.0, -25)):
                    values = (first, second, target_value, task, demo_a, gap, -1)
                    for target, value in zip(output["mismatched"], values): target.append(value)
    return {
        name: PairData(
            np.asarray(values[0], dtype=np.int64), np.asarray(values[1], dtype=np.int64),
            np.asarray(values[2], dtype=np.float32), np.asarray(values[3], dtype=np.int64),
            np.asarray(values[4], dtype=np.int64), np.asarray(values[5], dtype=np.int64),
            np.asarray(values[6], dtype=np.int8),
        )
        for name, values in output.items()
    }


def fit_standardizer(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = value.mean(axis=0).astype(np.float32)
    std = value.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_standardizer(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((value - mean) / std).astype(np.float32)


def set_deterministic(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def train_scalar(
    state: np.ndarray,
    language: np.ndarray,
    target: np.ndarray,
    epochs: int,
    seed: int,
    batch_size: int = 512,
) -> ScalarProgressMLP:
    set_deterministic(seed)
    model = ScalarProgressMLP(state.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(torch.from_numpy(state), torch.from_numpy(language), torch.from_numpy(target))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model.train()
    for _ in range(epochs):
        for batch_state, batch_language, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_state, batch_language)
            loss = torch.mean(torch.square(prediction - batch_target))
            loss.backward(); optimizer.step()
    return model.eval()


def train_pairwise(
    state: np.ndarray,
    language_by_row: np.ndarray,
    pairs: PairData,
    epochs: int,
    seed: int,
    batch_size: int = 512,
) -> AntisymmetricPairMLP:
    set_deterministic(seed)
    model = AntisymmetricPairMLP(state.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(
        torch.from_numpy(pairs.a), torch.from_numpy(pairs.b), torch.from_numpy(pairs.label)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    state_tensor = torch.from_numpy(state)
    language_tensor = torch.from_numpy(language_by_row)
    model.train()
    for _ in range(epochs):
        for index_a, index_b, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(state_tensor[index_a], state_tensor[index_b], language_tensor[index_a])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_target)
            loss.backward(); optimizer.step()
    return model.eval()


def predict_scalar(model: ScalarProgressMLP, state: np.ndarray, language: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    values = []
    with torch.inference_mode():
        for start in range(0, len(state), batch_size):
            stop = min(start + batch_size, len(state))
            values.append(model(torch.from_numpy(state[start:stop]), torch.from_numpy(language[start:stop])).numpy())
    return np.concatenate(values).astype(np.float64)


def predict_pairwise(
    model: AntisymmetricPairMLP,
    state: np.ndarray,
    language_by_row: np.ndarray,
    pairs: PairData,
    language_override: np.ndarray | None = None,
    batch_size: int = 1024,
) -> np.ndarray:
    values = []
    source = language_by_row if language_override is None else language_override
    with torch.inference_mode():
        for start in range(0, len(pairs.a), batch_size):
            stop = min(start + batch_size, len(pairs.a))
            ia, ib = pairs.a[start:stop], pairs.b[start:stop]
            values.append(model(
                torch.from_numpy(state[ia]), torch.from_numpy(state[ib]), torch.from_numpy(source[ia])
            ).numpy())
    return np.concatenate(values).astype(np.float64)


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_temperature(score_difference: np.ndarray, label: np.ndarray, grid: Iterable[float]) -> float:
    trials = []
    for scale in grid:
        probability = np.clip(sigmoid(score_difference * scale), 1e-7, 1 - 1e-7)
        loss = -np.mean(label * np.log(probability) + (1 - label) * np.log(1 - probability))
        trials.append((float(loss), float(scale)))
    return min(trials)[1]


def binary_metrics(probability: np.ndarray, label: np.ndarray, signed_gap: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    label = np.asarray(label, dtype=np.float64)
    prediction = probability >= 0.5
    positives, negatives = label == 1, label == 0
    ranks = rankdata(probability, method="average")
    auc = (ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) / (positives.sum() * negatives.sum())
    rho = spearmanr(probability - 0.5, signed_gap).statistic
    return {
        "accuracy": float(np.mean(prediction == label)),
        "auroc": float(auc),
        "log_loss": float(-np.mean(label * np.log(probability) + (1 - label) * np.log(1 - probability))),
        "brier": float(np.mean(np.square(probability - label))),
        "gap_spearman": float(rho),
    }


def task_demo_block_bootstrap(
    correct: np.ndarray,
    task_id: np.ndarray,
    demo_id: np.ndarray,
    demo_values: tuple[int, ...],
    draws: np.ndarray,
) -> np.ndarray:
    sums = np.zeros((10, len(demo_values)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for task in range(10):
        for slot, demo in enumerate(demo_values):
            mask = (task_id == task) & (demo_id == demo)
            sums[task, slot] = correct[mask].sum()
            counts[task, slot] = mask.sum()
    sampled_sum = np.zeros(len(draws), dtype=np.float64)
    sampled_count = np.zeros(len(draws), dtype=np.float64)
    for task in range(10):
        sampled_sum += sums[task][draws[:, task]].sum(axis=1)
        sampled_count += counts[task][draws[:, task]].sum(axis=1)
    return sampled_sum / sampled_count


def evaluate_probabilities(
    probability: np.ndarray,
    pairs: PairData,
    demo_values: tuple[int, ...],
    draws: np.ndarray,
) -> dict[str, Any]:
    local = np.isin(pairs.gap_bin, [GAP_NAMES.index(name) for name in LOCAL_GAPS])
    correct = ((probability >= 0.5) == pairs.label).astype(np.float64)
    boot = task_demo_block_bootstrap(correct[local], pairs.task_id[local], pairs.demo_id[local], demo_values, draws)
    result: dict[str, Any] = {
        "overall": binary_metrics(probability, pairs.label, pairs.signed_gap),
        "local_le8": binary_metrics(probability[local], pairs.label[local], pairs.signed_gap[local]),
        "local_accuracy_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "positive_task_count": int(sum(
            np.mean(correct[local & (pairs.task_id == task)]) > 0.5 for task in range(10)
        )),
        "per_task_local_accuracy": {
            str(task): float(np.mean(correct[local & (pairs.task_id == task)])) for task in range(10)
        },
        "per_gap": {},
    }
    for index, name in enumerate(GAP_NAMES):
        mask = pairs.gap_bin == index
        result["per_gap"][name] = binary_metrics(probability[mask], pairs.label[mask], pairs.signed_gap[mask])
    return result


def bootstrap_accuracy_difference(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    pairs: PairData,
    mask: np.ndarray,
    demo_values: tuple[int, ...],
    draws: np.ndarray,
) -> tuple[float, list[float]]:
    first = task_demo_block_bootstrap(correct_a[mask], pairs.task_id[mask], pairs.demo_id[mask], demo_values, draws)
    second = task_demo_block_bootstrap(correct_b[mask], pairs.task_id[mask], pairs.demo_id[mask], demo_values, draws)
    difference = first - second
    point = float(np.mean(correct_a[mask]) - np.mean(correct_b[mask]))
    return point, [float(np.quantile(difference, 0.025)), float(np.quantile(difference, 0.975))]
