"""Frozen low-capacity decoders for the natural spatial extraction audit.

This module maps already-frozen natural CLIP patch features to compact
object-footprint teachers.  It is not an action-conditioned dynamics model;
offline masks are labels only and are never decoder inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CAPACITIES = ("C0-linear", "C1-width32", "C2-width128-64", "C3-local-context")
DESIGNS = ("B1-binary", "B2-binary-area", "B3-ordinal", "B4-binary-coarse-area")


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


class PatchDecoder(nn.Module):
    """One frozen-capacity patch decoder with an optional area output."""

    def __init__(self, capacity: str, occupancy_classes: int, area_kind: str):
        super().__init__()
        if capacity not in CAPACITIES:
            raise ValueError(capacity)
        if occupancy_classes not in (2, 3):
            raise ValueError(occupancy_classes)
        if area_kind not in ("none", "continuous", "coarse"):
            raise ValueError(area_kind)
        self.capacity = capacity
        self.occupancy_classes = int(occupancy_classes)
        self.area_kind = area_kind
        output = 1 if occupancy_classes == 2 else occupancy_classes
        if capacity == "C0-linear":
            self.patch = nn.Linear(512, output)
            area_input = 512
        elif capacity == "C1-width32":
            self.patch = nn.Sequential(nn.Linear(512, 32), nn.GELU(), nn.Linear(32, output))
            area_input = 512
        elif capacity == "C2-width128-64":
            self.patch = nn.Sequential(
                nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, output)
            )
            area_input = 512
        else:
            self.token_projection = nn.Linear(512, 32)
            self.depthwise = nn.Conv2d(32, 32, 3, padding=1, groups=32)
            self.pointwise = nn.Conv2d(32, 32, 1)
            self.patch_output = nn.Conv2d(32, output, 1)
            area_input = 32
        if area_kind == "none":
            self.area = None
        else:
            area_output = 1 if area_kind == "continuous" else 3
            hidden = 32 if capacity in ("C1-width32", "C3-local-context") else 64
            if capacity == "C0-linear":
                self.area = nn.Linear(area_input, area_output)
            else:
                self.area = nn.Sequential(nn.Linear(area_input, hidden), nn.GELU(), nn.Linear(hidden, area_output))

    def forward(self, patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.capacity == "C3-local-context":
            features = self.token_projection(patch).reshape(-1, 7, 7, 32).permute(0, 3, 1, 2)
            features = F.gelu(self.pointwise(F.gelu(self.depthwise(features))))
            occupancy = self.patch_output(features).permute(0, 2, 3, 1)
            pooled = features.mean(dim=(-2, -1))
        else:
            occupancy = self.patch(patch).reshape(patch.shape[0], 7, 7, -1)
            pooled = patch.mean(dim=1)
        if self.occupancy_classes == 2:
            occupancy = occupancy.squeeze(-1)
        area = None if self.area is None else self.area(pooled)
        if self.area_kind == "continuous":
            area = area.squeeze(-1)
        return occupancy, area


@dataclass(frozen=True)
class DecoderFit:
    capacity: str
    occupancy_classes: int
    area_kind: str
    epochs: int
    state: dict[str, np.ndarray]


def _class_weights(target: torch.Tensor, classes: int) -> torch.Tensor:
    counts = torch.stack([(target == index).sum() for index in range(classes)]).float()
    weights = counts.sum() / torch.clamp(counts * float(classes), min=1.0)
    return torch.clamp(weights, min=0.25, max=20.0)


def _loss(
    occupancy_logits: torch.Tensor,
    area_output: torch.Tensor | None,
    occupancy_target: torch.Tensor,
    area_target: torch.Tensor | None,
    occupancy_classes: int,
    area_kind: str,
    positive_weight: torch.Tensor | None,
    ordinal_weights: torch.Tensor | None,
    area_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if occupancy_classes == 2:
        occupancy_loss = F.binary_cross_entropy_with_logits(
            occupancy_logits, occupancy_target.float(), pos_weight=positive_weight
        )
    else:
        occupancy_loss = F.cross_entropy(
            occupancy_logits.reshape(-1, occupancy_classes), occupancy_target.reshape(-1).long(), weight=ordinal_weights
        )
    if area_kind == "none":
        area_loss = torch.zeros((), dtype=occupancy_loss.dtype, device=occupancy_loss.device)
        total = occupancy_loss
    elif area_kind == "continuous":
        area_loss = F.mse_loss(torch.sigmoid(area_output), area_target.float())
        total = occupancy_loss + 10.0 * area_loss
    else:
        area_loss = F.cross_entropy(area_output, area_target.long(), weight=area_weights)
        total = occupancy_loss + area_loss
    return total, {
        "total": float(total.detach()),
        "occupancy": float(occupancy_loss.detach()),
        "area": float(area_loss.detach()),
    }


def fit_decoder(
    patch: np.ndarray,
    occupancy: np.ndarray,
    area: np.ndarray | None,
    capacity: str,
    occupancy_classes: int,
    area_kind: str,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    valid_patch: np.ndarray | None = None,
    valid_occupancy: np.ndarray | None = None,
    valid_area: np.ndarray | None = None,
) -> tuple[DecoderFit, dict[str, Any]]:
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch_device = torch.device(device)
    model = PatchDecoder(capacity, occupancy_classes, area_kind).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    x = torch.as_tensor(np.asarray(patch), dtype=torch.float32, device=torch_device)
    y = torch.as_tensor(np.asarray(occupancy), device=torch_device)
    a = None if area is None else torch.as_tensor(np.asarray(area), device=torch_device)
    if occupancy_classes == 2:
        positive = float(y.sum())
        negative = float(y.numel() - positive)
        positive_weight = torch.tensor(min(max(negative / max(positive, 1.0), 1.0), 20.0), device=torch_device)
        ordinal_weights = None
    else:
        positive_weight = None
        ordinal_weights = _class_weights(y, occupancy_classes)
    area_weights = _class_weights(a, 3) if area_kind == "coarse" and a is not None else None
    validation = valid_patch is not None
    if validation:
        vx = torch.as_tensor(np.asarray(valid_patch), dtype=torch.float32, device=torch_device)
        vy = torch.as_tensor(np.asarray(valid_occupancy), device=torch_device)
        va = None if valid_area is None else torch.as_tensor(np.asarray(valid_area), device=torch_device)
    best_epoch = int(max_epochs)
    best_value = float("inf")
    best_state: dict[str, np.ndarray] | None = None
    last_record: dict[str, float] = {}
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        occupancy_logits, area_output = model(x)
        total, last_record = _loss(
            occupancy_logits, area_output, y, a, occupancy_classes, area_kind,
            positive_weight, ordinal_weights, area_weights,
        )
        total.backward()
        optimizer.step()
        if validation:
            model.eval()
            with torch.no_grad():
                vo, varea = model(vx)
                validation_loss, validation_record = _loss(
                    vo, varea, vy, va, occupancy_classes, area_kind,
                    positive_weight, ordinal_weights, area_weights,
                )
            value = float(validation_loss)
            if value < best_value - 1.0e-12:
                best_value = value
                best_epoch = epoch
                best_state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
                last_validation_record = validation_record
    if best_state is None:
        best_state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
        best_value = float(last_record["total"])
        last_validation_record = last_record
    return DecoderFit(capacity, occupancy_classes, area_kind, best_epoch, best_state), {
        "best_epoch": best_epoch,
        "best_validation_loss": best_value,
        "final_train_loss": last_record,
        "best_validation_components": last_validation_record,
        "parameter_count": parameter_count(model),
        "positive_weight": None if positive_weight is None else float(positive_weight),
        "ordinal_class_weights": None if ordinal_weights is None else ordinal_weights.detach().cpu().tolist(),
        "area_class_weights": None if area_weights is None else area_weights.detach().cpu().tolist(),
    }


def predict_decoder(patch: np.ndarray, fit: DecoderFit, device: str) -> tuple[np.ndarray, np.ndarray | None]:
    model = PatchDecoder(fit.capacity, fit.occupancy_classes, fit.area_kind).to(device)
    model.load_state_dict({name: torch.as_tensor(value, device=device) for name, value in fit.state.items()})
    model.eval()
    with torch.no_grad():
        occupancy, area = model(torch.as_tensor(np.asarray(patch), dtype=torch.float32, device=device))
        if fit.occupancy_classes == 2:
            occupancy_prediction = torch.sigmoid(occupancy).cpu().numpy()
        else:
            occupancy_prediction = torch.softmax(occupancy, dim=-1).cpu().numpy()
        if fit.area_kind == "none":
            area_prediction = None
        elif fit.area_kind == "continuous":
            area_prediction = torch.sigmoid(area).cpu().numpy()
        else:
            area_prediction = torch.softmax(area, dim=-1).cpu().numpy()
    return occupancy_prediction.astype(np.float64), None if area_prediction is None else area_prediction.astype(np.float64)


def decoder_to_arrays(prefix: str, fit: DecoderFit) -> dict[str, np.ndarray]:
    result = {
        f"{prefix}__capacity": np.asarray(CAPACITIES.index(fit.capacity), dtype=np.int64),
        f"{prefix}__occupancy_classes": np.asarray(fit.occupancy_classes, dtype=np.int64),
        f"{prefix}__area_kind": np.asarray(("none", "continuous", "coarse").index(fit.area_kind), dtype=np.int64),
        f"{prefix}__epochs": np.asarray(fit.epochs, dtype=np.int64),
    }
    result.update({f"{prefix}__state__{name}": value for name, value in fit.state.items()})
    return result


def decoder_from_arrays(prefix: str, arrays: Any) -> DecoderFit:
    marker = f"{prefix}__state__"
    state = {name[len(marker):]: np.asarray(arrays[name]) for name in arrays if name.startswith(marker)}
    return DecoderFit(
        CAPACITIES[int(np.asarray(arrays[f"{prefix}__capacity"]))],
        int(np.asarray(arrays[f"{prefix}__occupancy_classes"])),
        ("none", "continuous", "coarse")[int(np.asarray(arrays[f"{prefix}__area_kind"]))],
        int(np.asarray(arrays[f"{prefix}__epochs"])),
        state,
    )


def binary_metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    predicted = probability >= float(threshold)
    tp = float(np.logical_and(predicted, target).sum())
    tn = float(np.logical_and(~predicted, ~target).sum())
    fp = float(np.logical_and(predicted, ~target).sum())
    fn = float(np.logical_and(~predicted, target).sum())
    recall = tp / max(tp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    order = np.argsort(probability, kind="stable")
    ranks = np.empty(probability.size, dtype=np.float64)
    ranks[order] = np.arange(probability.size, dtype=np.float64) + 1.0
    unique, inverse, counts = np.unique(probability, return_inverse=True, return_counts=True)
    del unique
    sums = np.bincount(inverse, weights=ranks)
    ranks = sums[inverse] / counts[inverse]
    positives = float(target.sum())
    negatives = float((~target).sum())
    auroc = (float(ranks[target].sum()) - positives * (positives + 1.0) / 2.0) / max(positives * negatives, 1.0)
    return {
        "iou": tp / max(tp + fp + fn, 1.0),
        "dice_f1": 2.0 * tp / max(2.0 * tp + fp + fn, 1.0),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "positive_recall": recall,
        "positive_precision": precision,
        "auroc": auroc,
    }


def ordinal_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    predicted = np.argmax(probability, axis=-1).reshape(-1)
    recalls = []
    f1s = []
    for label in range(3):
        tp = float(np.logical_and(predicted == label, target == label).sum())
        fp = float(np.logical_and(predicted == label, target != label).sum())
        fn = float(np.logical_and(predicted != label, target == label).sum())
        recalls.append(tp / max(tp + fn, 1.0))
        f1s.append(2.0 * tp / max(2.0 * tp + fp + fn, 1.0))
    return {
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "ordinal_mae": float(np.mean(np.abs(predicted - target)) / 2.0),
    }


def area_metrics(prediction: np.ndarray, target: np.ndarray, baseline: float) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mae = float(np.mean(np.abs(prediction - target)))
    rmse = float(np.sqrt(np.mean(np.square(prediction - target))))
    baseline_mae = float(np.mean(np.abs(float(baseline) - target)))
    return {
        "mae": mae,
        "rmse": rmse,
        "constant_mae": baseline_mae,
        "mae_improvement_over_constant": (baseline_mae - mae) / max(baseline_mae, 1.0e-12),
    }
