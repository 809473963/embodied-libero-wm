"""Low-capacity utilities for the frozen spatial visual target design study.

The functions here operate on already-frozen natural-image CLIP features and
offline object masks.  They do not implement an action-conditioned dynamics
model and never expose simulator labels as deployment inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.probes.spatial_relevance import PCAFit, transform_pca


PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ("A0-A1", "A0-A2", "A1-A2")


def footprint_from_object_masks(masks: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Return union occupancy maps and exact visible-area fractions per image."""

    value = np.asarray(masks, dtype=bool)
    if value.shape != (40, 3, 8, 2, 128, 128):
        raise ValueError(f"unexpected per-object mask shape: {value.shape}")
    union = np.any(value, axis=3)
    area = union.mean(axis=(-2, -1), dtype=np.float64)
    edges = np.floor(np.linspace(0, 128, resolution + 1)).astype(np.int64)
    occupancy = np.empty(union.shape[:3] + (resolution, resolution), dtype=np.float64)
    for row in range(resolution):
        for col in range(resolution):
            cell = union[..., edges[row] : edges[row + 1], edges[col] : edges[col + 1]]
            occupancy[..., row, col] = cell.mean(axis=(-2, -1), dtype=np.float64)
    return occupancy, area


def metadata_from_derived(derived: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(derived[f"metadata__{name}"])
        for name in ("branch", "task", "init_state", "horizon", "pair")
    }


def gather_pair_values(values: np.ndarray, metadata: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Gather candidate-first/second per-image values in frozen 960-row order."""

    source = np.asarray(values)
    first = []
    second = []
    for branch, horizon, pair in zip(
        metadata["branch"], metadata["horizon"], metadata["pair"], strict=True
    ):
        left, right = PAIR_INDICES[int(pair)]
        first.append(source[int(branch), left, int(horizon) - 1])
        second.append(source[int(branch), right, int(horizon) - 1])
    return np.asarray(first), np.asarray(second)


def _centroid(map_value: np.ndarray) -> np.ndarray:
    array = np.asarray(map_value, dtype=np.float64)
    side_y, side_x = array.shape[-2:]
    yy, xx = np.meshgrid(
        (np.arange(side_y, dtype=np.float64) + 0.5) / side_y,
        (np.arange(side_x, dtype=np.float64) + 0.5) / side_x,
        indexing="ij",
    )
    total = np.sum(np.maximum(array, 0.0), axis=(-2, -1))
    safe = np.maximum(total, 1.0e-12)
    cx = np.sum(np.maximum(array, 0.0) * xx, axis=(-2, -1)) / safe
    cy = np.sum(np.maximum(array, 0.0) * yy, axis=(-2, -1)) / safe
    return np.column_stack((cx, cy))


def spatial_pair_summary(
    state: np.ndarray,
    area: np.ndarray,
    metadata: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Six fixed, scale-transparent summaries of a per-image spatial state."""

    first, second = gather_pair_values(state, metadata)
    area_first, area_second = gather_pair_values(area, metadata)
    difference = np.abs(first - second).reshape(first.shape[0], -1)
    ordered = np.sort(difference, axis=1)
    top_count = max(1, difference.shape[1] // 4)
    centroid_delta = np.linalg.norm(_centroid(first) - _centroid(second), axis=1)
    summary = np.column_stack(
        (
            np.sqrt(np.mean(np.square(difference), axis=1)),
            np.mean(difference, axis=1),
            np.max(difference, axis=1),
            np.mean(ordered[:, -top_count:], axis=1),
            centroid_delta,
            np.abs(area_first - area_second),
        )
    )
    distance = np.sqrt(np.mean(np.square(difference), axis=1) + np.square(area_first - area_second))
    return summary.astype(np.float64), distance.astype(np.float64)


def generic_pair_summary(state: np.ndarray, metadata: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Fixed six-value summary for a generic [H,W,R] per-image map."""

    first, second = gather_pair_values(state, metadata)
    difference = np.abs(first - second).reshape(first.shape[0], -1)
    ordered = np.sort(difference, axis=1)
    top_count = max(1, difference.shape[1] // 4)
    magnitude_first = np.linalg.norm(first, axis=-1) if first.ndim == 4 else first
    magnitude_second = np.linalg.norm(second, axis=-1) if second.ndim == 4 else second
    centroid_delta = np.linalg.norm(_centroid(np.abs(magnitude_first)) - _centroid(np.abs(magnitude_second)), axis=1)
    summary = np.column_stack(
        (
            np.sqrt(np.mean(np.square(difference), axis=1)),
            np.mean(difference, axis=1),
            np.max(difference, axis=1),
            np.mean(ordered[:, -top_count:], axis=1),
            centroid_delta,
            np.abs(np.mean(first, axis=tuple(range(1, first.ndim))) - np.mean(second, axis=tuple(range(1, second.ndim)))),
        )
    )
    return summary.astype(np.float64), summary[:, 0].astype(np.float64)


def pca_patch_state(patch: np.ndarray, fit: PCAFit) -> np.ndarray:
    source = np.asarray(patch, dtype=np.float64)
    flat = source.reshape(-1, source.shape[-1])
    scores = transform_pca(flat, fit)
    return scores.reshape(source.shape[:-1] + (scores.shape[-1],))


def patch_global_residual_state(patch: np.ndarray, global_feature: np.ndarray) -> np.ndarray:
    p = np.asarray(patch, dtype=np.float64)
    g = np.asarray(global_feature, dtype=np.float64)[..., None, :]
    cosine = np.sum(p * g, axis=-1) / np.maximum(
        np.linalg.norm(p, axis=-1) * np.linalg.norm(g, axis=-1), 1.0e-12
    )
    return np.maximum(1.0 - cosine, 0.0)[..., None]


def reshape_patch_grid(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values)
    if value.shape[-2] != 49:
        raise ValueError(f"expected 49 patches: {value.shape}")
    return value.reshape(value.shape[:-2] + (7, 7, value.shape[-1]))


class SpatialProxy(nn.Module):
    """Task-agnostic per-patch decoder plus a pooled visible-area head."""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = int(hidden)
        if self.hidden == 0:
            self.patch_in = nn.Linear(512, 1)
            self.patch_out = None
            self.area_head = nn.Linear(512, 1)
        else:
            self.patch_in = nn.Linear(512, self.hidden)
            self.patch_out = nn.Linear(self.hidden, 1)
            self.area_head = nn.Sequential(
                nn.Linear(512, self.hidden), nn.GELU(), nn.Linear(self.hidden, 1)
            )

    def forward(self, patch: torch.Tensor, resolution: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.patch_out is None:
            logits = self.patch_in(patch).squeeze(-1)
        else:
            logits = self.patch_out(F.gelu(self.patch_in(patch))).squeeze(-1)
        logits = logits.reshape(-1, 1, 7, 7)
        if resolution != 7:
            if resolution != 4:
                raise ValueError(f"unsupported frozen proxy resolution: {resolution}")
            # Fixed separable area weights avoid CUDA's nondeterministic
            # adaptive-pooling backward kernel while preserving exact cell
            # overlap under a 7-to-4 grid conversion.
            source_edges = np.arange(8, dtype=np.float32)
            target_edges = np.linspace(0.0, 7.0, 5, dtype=np.float32)
            weight = np.zeros((4, 7), dtype=np.float32)
            for target_index in range(4):
                for source_index in range(7):
                    overlap = max(
                        0.0,
                        min(target_edges[target_index + 1], source_edges[source_index + 1])
                        - max(target_edges[target_index], source_edges[source_index]),
                    )
                    weight[target_index, source_index] = overlap / (
                        target_edges[target_index + 1] - target_edges[target_index]
                    )
            operator = torch.as_tensor(weight, dtype=logits.dtype, device=logits.device)
            logits = torch.einsum("ai,bij,cj->bac", operator, logits.squeeze(1), operator)
            logits = logits[:, None]
        area_logit = self.area_head(patch.mean(dim=1)).squeeze(-1)
        return logits.squeeze(1), area_logit


@dataclass(frozen=True)
class ProxyFit:
    hidden: int
    resolution: int
    state: dict[str, np.ndarray]


def _proxy_tensors(
    patch: np.ndarray, occupancy: np.ndarray, area: np.ndarray, indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(np.asarray(patch)[indices], dtype=torch.float32),
        torch.as_tensor(np.asarray(occupancy)[indices], dtype=torch.float32),
        torch.as_tensor(np.asarray(area)[indices], dtype=torch.float32),
    )


def fit_proxy(
    patch: np.ndarray,
    occupancy: np.ndarray,
    area: np.ndarray,
    hidden: int,
    resolution: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str = "cpu",
) -> tuple[ProxyFit, dict[str, float]]:
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch_device = torch.device(device)
    model = SpatialProxy(hidden).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    patch_t = torch.as_tensor(np.asarray(patch), dtype=torch.float32, device=torch_device)
    occupancy_t = torch.as_tensor(np.asarray(occupancy), dtype=torch.float32, device=torch_device)
    area_t = torch.as_tensor(np.asarray(area), dtype=torch.float32, device=torch_device)
    positive = float(occupancy_t.sum())
    negative = float(occupancy_t.numel() - positive)
    pos_weight = torch.tensor(
        min(max(negative / max(positive, 1.0), 1.0), 20.0), device=torch_device
    )
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits, area_logit = model(patch_t, resolution)
        occupancy_loss = F.binary_cross_entropy_with_logits(logits, occupancy_t, pos_weight=pos_weight)
        area_prediction = torch.sigmoid(area_logit)
        area_loss = F.mse_loss(area_prediction, area_t)
        loss = occupancy_loss + 10.0 * area_loss
        loss.backward()
        optimizer.step()
    state = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    return ProxyFit(int(hidden), int(resolution), state), {
        "loss": float(loss.detach()),
        "occupancy_loss": float(occupancy_loss.detach()),
        "area_loss": float(area_loss.detach()),
        "pos_weight": float(pos_weight),
    }


def predict_proxy(patch: np.ndarray, fit: ProxyFit, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    torch_device = torch.device(device)
    model = SpatialProxy(fit.hidden).to(torch_device)
    model.load_state_dict(
        {name: torch.as_tensor(value, device=torch_device) for name, value in fit.state.items()}
    )
    model.eval()
    with torch.no_grad():
        logits, area_logit = model(
            torch.as_tensor(np.asarray(patch), dtype=torch.float32, device=torch_device),
            fit.resolution,
        )
        occupancy = torch.sigmoid(logits).cpu().numpy()
        area = torch.sigmoid(area_logit).cpu().numpy()
    return occupancy.astype(np.float64), area.astype(np.float64)


def proxy_to_arrays(prefix: str, fit: ProxyFit) -> dict[str, np.ndarray]:
    result = {
        f"{prefix}__hidden": np.asarray(fit.hidden, dtype=np.int64),
        f"{prefix}__resolution": np.asarray(fit.resolution, dtype=np.int64),
    }
    result.update({f"{prefix}__state__{name}": value for name, value in fit.state.items()})
    return result


def proxy_from_arrays(prefix: str, arrays: Any) -> ProxyFit:
    marker = f"{prefix}__state__"
    state = {name[len(marker) :]: np.asarray(arrays[name]) for name in arrays if name.startswith(marker)}
    return ProxyFit(
        int(np.asarray(arrays[f"{prefix}__hidden"])),
        int(np.asarray(arrays[f"{prefix}__resolution"])),
        state,
    )


def recovery_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    area_prediction: np.ndarray,
    area_target: np.ndarray,
    threshold: float,
    baseline_map: np.ndarray,
    baseline_area: float,
) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    binary_true = true > 0.0
    binary_pred = pred >= float(threshold)
    intersection = float(np.logical_and(binary_true, binary_pred).sum())
    union = float(np.logical_or(binary_true, binary_pred).sum())
    tp = intersection
    fn = float(np.logical_and(binary_true, ~binary_pred).sum())
    tn = float(np.logical_and(~binary_true, ~binary_pred).sum())
    fp = float(np.logical_and(~binary_true, binary_pred).sum())
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    mse = float(np.mean(np.square(pred - true)))
    baseline_mse = float(np.mean(np.square(np.asarray(baseline_map) - true)))
    area_mae = float(np.mean(np.abs(np.asarray(area_prediction) - np.asarray(area_target))))
    baseline_area_mae = float(np.mean(np.abs(float(baseline_area) - np.asarray(area_target))))
    return {
        "occupancy_bce": float(-np.mean(true * np.log(np.maximum(pred, 1.0e-8)) + (1.0 - true) * np.log(np.maximum(1.0 - pred, 1.0e-8)))),
        "occupancy_mse": mse,
        "occupancy_mse_improvement_over_constant": (baseline_mse - mse) / max(baseline_mse, 1.0e-12),
        "iou": intersection / max(union, 1.0),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "positive_cell_recall": recall,
        "area_mae": area_mae,
        "area_mae_improvement_over_constant": (baseline_area_mae - area_mae) / max(baseline_area_mae, 1.0e-12),
    }


def temporal_and_candidate_distance(
    per_image: np.ndarray,
    distance_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate-pair rows [960] and adjacent temporal rows [840]."""

    value = np.asarray(per_image, dtype=np.float64)

    def distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        if distance_kind == "cosine":
            numerator = np.sum(first * second, axis=-1)
            denominator = np.maximum(np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1), 1.0e-12)
            value = np.maximum(1.0 - numerator / denominator, 0.0)
            return np.asarray(np.mean(value), dtype=np.float64)
        axes = tuple(range(first.ndim))
        if distance_kind == "rmse":
            return np.sqrt(np.mean(np.square(first - second), axis=axes))
        raise ValueError(distance_kind)

    candidate = []
    for branch in range(40):
        for left, right in PAIR_INDICES:
            for horizon in range(8):
                candidate.append(distance(value[branch, left, horizon], value[branch, right, horizon]))
    temporal = []
    for branch in range(40):
        for candidate_index in range(3):
            for horizon in range(7):
                temporal.append(distance(value[branch, candidate_index, horizon], value[branch, candidate_index, horizon + 1]))
    return np.asarray(candidate, dtype=np.float64).reshape(-1), np.asarray(temporal, dtype=np.float64).reshape(-1)
