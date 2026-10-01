"""Small frozen-feature regressors for the visible-area extraction audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


VISUAL_FORMS = ("V0_global", "V1_patch_mean")
CONDITIONS = ("E0_visual", "E1_visual_language")
MODEL_FORMS = ("M1_linear", "M2_mlp")


class AreaHead(nn.Module):
    """Bounded linear or two-hidden-layer area regressor."""

    def __init__(self, input_dim: int, form: str) -> None:
        super().__init__()
        if form == "M1_linear":
            self.network = nn.Linear(input_dim, 1)
        elif form == "M2_mlp":
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
        else:
            raise ValueError(form)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


@dataclass(frozen=True)
class FittedAreaHead:
    form: str
    mean: np.ndarray
    scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    state_dict: dict[str, np.ndarray]


def parameter_count(input_dim: int, form: str) -> int:
    return sum(item.numel() for item in AreaHead(input_dim, form).parameters())


def standardizer(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    mean = array.mean(axis=0)
    scale = array.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    return mean, scale


def train_head(
    x: np.ndarray,
    y: np.ndarray,
    form: str,
    epochs: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
) -> FittedAreaHead:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    mean, scale = standardizer(x)
    target_mean = np.asarray(np.mean(y), dtype=np.float32)
    target_scale = np.asarray(max(float(np.std(y)), 1.0e-8), dtype=np.float32)
    feature = torch.from_numpy(((np.asarray(x) - mean) / scale).astype(np.float32))
    target = torch.from_numpy(((np.asarray(y, dtype=np.float32) - target_mean) / target_scale).astype(np.float32))
    model = AreaHead(feature.shape[1], form)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean(torch.square(model(feature) - target))
        loss.backward()
        optimizer.step()
    state = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    return FittedAreaHead(form, mean.astype(np.float32), scale.astype(np.float32), target_mean, target_scale, state)


def predict_head(x: np.ndarray, fitted: FittedAreaHead) -> np.ndarray:
    feature = ((np.asarray(x, dtype=np.float32) - fitted.mean) / fitted.scale).astype(np.float32)
    model = AreaHead(feature.shape[1], fitted.form)
    model.load_state_dict({name: torch.from_numpy(value) for name, value in fitted.state_dict.items()})
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(feature)).cpu().numpy().astype(np.float64)
    output = np.clip(output * float(fitted.target_scale) + float(fitted.target_mean), 0.0, 1.0)
    if not np.isfinite(output).all() or np.any(output < 0.0) or np.any(output > 1.0):
        raise FloatingPointError("invalid visible-area prediction")
    return output


def fitted_to_arrays(prefix: str, fitted: FittedAreaHead) -> dict[str, np.ndarray]:
    result = {
        f"{prefix}__mean": fitted.mean,
        f"{prefix}__scale": fitted.scale,
        f"{prefix}__target_mean": fitted.target_mean,
        f"{prefix}__target_scale": fitted.target_scale,
        f"{prefix}__form": np.asarray(fitted.form),
    }
    for name, value in fitted.state_dict.items():
        result[f"{prefix}__state__{name}"] = value
    return result


def fitted_from_arrays(prefix: str, arrays: dict[str, np.ndarray]) -> FittedAreaHead:
    marker = f"{prefix}__state__"
    state = {name[len(marker) :]: np.asarray(value) for name, value in arrays.items() if name.startswith(marker)}
    return FittedAreaHead(
        str(np.asarray(arrays[f"{prefix}__form"]).item()),
        np.asarray(arrays[f"{prefix}__mean"], dtype=np.float32),
        np.asarray(arrays[f"{prefix}__scale"], dtype=np.float32),
        np.asarray(arrays[f"{prefix}__target_mean"], dtype=np.float32),
        np.asarray(arrays[f"{prefix}__target_scale"], dtype=np.float32),
        state,
    )


def image_features(
    global_features: np.ndarray,
    patch_features: np.ndarray,
    language: np.ndarray,
    task_id: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the four frozen natural-image/language inputs."""

    global_array = np.asarray(global_features, dtype=np.float32)
    patch_mean = np.asarray(patch_features, dtype=np.float32).mean(axis=-2)
    if global_array.shape != (40, 3, 8, 512) or patch_mean.shape != global_array.shape:
        raise ValueError("unexpected natural feature shape")
    flat_global = global_array.reshape(-1, 512)
    flat_patch = patch_mean.reshape(-1, 512)
    task = np.repeat(np.asarray(task_id, dtype=np.int64), 24)
    text = np.asarray(language, dtype=np.float32)[task]
    result = {
        "E0_visual__V0_global": flat_global,
        "E0_visual__V1_patch_mean": flat_patch,
        "E1_visual_language__V0_global": np.concatenate((flat_global, text), axis=1),
        "E1_visual_language__V1_patch_mean": np.concatenate((flat_patch, text), axis=1),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("nonfinite natural feature")
    return result


def pair_rows(state: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return signed A0-A1/A0-A2/A1-A2 differences in canonical order."""

    value = np.asarray(state, dtype=np.float64)
    if value.shape != (40, 3, 8):
        raise ValueError(value.shape)
    rows, branch, horizon, pair = [], [], [], []
    for branch_index in range(40):
        for pair_index, (left, right) in enumerate(((0, 1), (0, 2), (1, 2))):
            for horizon_index in range(8):
                rows.append(value[branch_index, left, horizon_index] - value[branch_index, right, horizon_index])
                branch.append(branch_index)
                horizon.append(horizon_index + 1)
                pair.append(pair_index)
    return np.asarray(rows), {
        "branch": np.asarray(branch, dtype=np.int64),
        "horizon": np.asarray(horizon, dtype=np.int64),
        "pair": np.asarray(pair, dtype=np.int64),
    }


def scalar_pair_summary(state: np.ndarray) -> np.ndarray:
    """Exact six-slot F1 summary used by the frozen decomposition audit."""

    delta, _ = pair_rows(state)
    absolute = np.abs(delta)
    result = np.column_stack((absolute, absolute, absolute, absolute, np.zeros_like(absolute), absolute))
    if result.shape != (960, 6):
        raise ValueError(result.shape)
    return result


def candidate_ordering(real: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Three-candidate rank and extremum agreement across branch/horizon blocks."""

    from scipy.stats import spearmanr

    truth, estimate = np.asarray(real), np.asarray(predicted)
    correlations, pair_hits, top_hits, bottom_hits = [], [], [], []
    for branch in range(40):
        for horizon in range(8):
            a, b = truth[branch, :, horizon], estimate[branch, :, horizon]
            rho = spearmanr(a, b).statistic
            correlations.append(0.0 if not np.isfinite(rho) else float(rho))
            hits = []
            for left, right in ((0, 1), (0, 2), (1, 2)):
                sign_a, sign_b = np.sign(a[left] - a[right]), np.sign(b[left] - b[right])
                hits.append(float(sign_a == sign_b))
            pair_hits.append(float(np.mean(hits)))
            top_hits.append(float(np.argmax(a) == np.argmax(b)))
            bottom_hits.append(float(np.argmin(a) == np.argmin(b)))
    return {
        "mean_three_way_spearman": float(np.mean(correlations)),
        "pairwise_ordering_accuracy": float(np.mean(pair_hits)),
        "top_agreement": float(np.mean(top_hits)),
        "bottom_agreement": float(np.mean(bottom_hits)),
    }
