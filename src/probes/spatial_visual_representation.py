"""Small frozen-CLIP spatial representation probes for Stage R.

The CLIP encoder is deliberately absent from this module.  Inputs are already
frozen global or patch projections; only the registered pooling and pairwise
probe parameters are trainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.probes.semantic_value_distillation import PairSet


FAMILIES = ("R0", "R1", "R2", "R3")


def fixed_patch_coordinates() -> np.ndarray:
    axis = 2.0 * (np.arange(7, dtype=np.float32) + 0.5) / 7.0 - 1.0
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx, yy), axis=-1).reshape(49, 2).astype(np.float32)


def set_deterministic(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def parameter_count(model: nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters() if value.requires_grad))


class SpatialRepresentationProbe(nn.Module):
    """One of the four preregistered representations plus one shared probe form."""

    def __init__(self, family: str) -> None:
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(f"unknown Stage-R family: {family}")
        self.family = family
        if family == "R0":
            representation_dim = 512
        else:
            self.key = nn.Linear(514, 32)
            self.value = nn.Linear(514, 32)
            if family == "R1":
                self.fixed_query = nn.Parameter(torch.empty(32))
                nn.init.normal_(self.fixed_query, mean=0.0, std=0.02)
                representation_dim = 67
            elif family == "R2":
                self.query = nn.Linear(512, 32)
                representation_dim = 67
            else:
                self.query = nn.Linear(512, 64)
                representation_dim = 136
        raw_dim = 3 * representation_dim + 512
        self.raw = nn.Sequential(nn.Linear(raw_dim, 64), nn.GELU(), nn.Linear(64, 1))

    @staticmethod
    def _attention_summary(
        value: torch.Tensor, attention: torch.Tensor, coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # value: [B,49,32], attention: [B,S,49]
        slots = torch.einsum("bsp,bpd->bsd", attention, value)
        centers = torch.einsum("bsp,pd->bsd", attention, coordinates)
        entropy = -(attention * torch.log(attention.clamp_min(1.0e-12))).sum(dim=-1)
        entropy = entropy / float(np.log(49.0))
        return slots, centers, entropy

    def represent(
        self,
        visual: torch.Tensor,
        language: torch.Tensor,
        coordinates: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.family == "R0":
            return visual, None
        if visual.ndim != 3 or visual.shape[1:] != (49, 512):
            raise ValueError("patch representation requires [B,49,512]")
        coordinate_batch = coordinates.unsqueeze(0).expand(len(visual), -1, -1)
        patch_input = torch.cat((visual, coordinate_batch), dim=-1)
        key, value = self.key(patch_input), self.value(patch_input)
        if self.family == "R1":
            query = self.fixed_query.view(1, 1, 32).expand(len(visual), 1, 32)
        elif self.family == "R2":
            query = self.query(language).unsqueeze(1)
        else:
            query = self.query(language).reshape(len(visual), 2, 32)
        attention = torch.softmax(torch.einsum("bsd,bpd->bsp", query, key) / np.sqrt(32.0), dim=-1)
        slots, centers, entropy = self._attention_summary(value, attention, coordinates)
        mean_value = value.mean(dim=1)
        if self.family in ("R1", "R2"):
            representation = torch.cat(
                (slots[:, 0], mean_value, centers[:, 0], entropy[:, :1]), dim=-1
            )
        else:
            first, second = slots[:, 0], slots[:, 1]
            first_center, second_center = centers[:, 0], centers[:, 1]
            representation = torch.cat(
                (
                    first,
                    second,
                    first - second,
                    first * second,
                    first_center,
                    second_center,
                    first_center - second_center,
                    entropy,
                ),
                dim=-1,
            )
        return representation, attention if return_attention else None

    def _raw(self, a: torch.Tensor, b: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        return self.raw(torch.cat((a, b, b - a, language), dim=-1)).squeeze(-1)

    def forward(
        self,
        visual_a: torch.Tensor,
        visual_b: torch.Tensor,
        language: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        representation_a, _ = self.represent(visual_a, language, coordinates)
        representation_b, _ = self.represent(visual_b, language, coordinates)
        return self._raw(representation_a, representation_b, language) - self._raw(
            representation_b, representation_a, language
        )


@dataclass(frozen=True)
class FeatureStatistics:
    global_mean: np.ndarray
    global_std: np.ndarray
    patch_mean: np.ndarray
    patch_std: np.ndarray
    language_mean: np.ndarray
    language_std: np.ndarray


def fit_feature_statistics(
    global_features: np.ndarray,
    patch_features: np.ndarray,
    language: np.ndarray,
    train_mask: np.ndarray,
) -> FeatureStatistics:
    global_train = np.asarray(global_features[train_mask], dtype=np.float64)
    global_mean, global_std = global_train.mean(axis=0), global_train.std(axis=0)
    patch_sum = np.zeros(512, dtype=np.float64)
    patch_square = np.zeros(512, dtype=np.float64)
    count = 0
    indices = np.flatnonzero(train_mask)
    for start in range(0, len(indices), 128):
        value = np.asarray(patch_features[indices[start : start + 128]], dtype=np.float64)
        patch_sum += value.sum(axis=(0, 1))
        patch_square += np.square(value).sum(axis=(0, 1))
        count += int(value.shape[0] * value.shape[1])
    patch_mean = patch_sum / count
    patch_std = np.sqrt(np.maximum(patch_square / count - np.square(patch_mean), 0.0))
    language64 = np.asarray(language, dtype=np.float64)
    language_mean, language_std = language64.mean(axis=0), language64.std(axis=0)
    for value in (global_std, patch_std, language_std):
        value[value < 1.0e-6] = 1.0
    return FeatureStatistics(
        global_mean.astype(np.float32), global_std.astype(np.float32),
        patch_mean.astype(np.float32), patch_std.astype(np.float32),
        language_mean.astype(np.float32), language_std.astype(np.float32),
    )


def standardized_global(value: np.ndarray, statistics: FeatureStatistics) -> np.ndarray:
    return ((np.asarray(value, dtype=np.float32) - statistics.global_mean) / statistics.global_std).astype(np.float32)


def standardized_language(value: np.ndarray, statistics: FeatureStatistics) -> np.ndarray:
    return ((np.asarray(value, dtype=np.float32) - statistics.language_mean) / statistics.language_std).astype(np.float32)


def _batch_visual(
    family: str,
    index: torch.Tensor,
    global_tensor: torch.Tensor,
    patch_tensor: torch.Tensor,
    patch_mean: torch.Tensor,
    patch_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if family == "R0":
        return global_tensor[index].to(device, non_blocking=True)
    value = patch_tensor[index].to(device, dtype=torch.float32, non_blocking=True)
    return (value - patch_mean) / patch_std


def train_probe(
    family: str,
    global_features: np.ndarray,
    patch_features: np.ndarray,
    language: np.ndarray,
    pairs: PairSet,
    statistics: FeatureStatistics,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> SpatialRepresentationProbe:
    set_deterministic(seed)
    model = SpatialRepresentationProbe(family).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    target = ((pairs.label.astype(np.float32) + 1.0) / 2.0).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(pairs.a), torch.from_numpy(pairs.b),
        torch.from_numpy(pairs.language_id), torch.from_numpy(target),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator,
                        num_workers=0, pin_memory=device.type == "cuda")
    global_tensor = torch.from_numpy(standardized_global(global_features, statistics))
    patch_tensor = torch.from_numpy(np.asarray(patch_features))
    language_tensor = torch.from_numpy(standardized_language(language, statistics))
    patch_mean = torch.from_numpy(statistics.patch_mean).to(device).view(1, 1, 512)
    patch_std = torch.from_numpy(statistics.patch_std).to(device).view(1, 1, 512)
    coordinates = torch.from_numpy(fixed_patch_coordinates()).to(device)
    model.train()
    for _ in range(epochs):
        for a, b, language_id, y in loader:
            optimizer.zero_grad(set_to_none=True)
            va = _batch_visual(family, a, global_tensor, patch_tensor, patch_mean, patch_std, device)
            vb = _batch_visual(family, b, global_tensor, patch_tensor, patch_mean, patch_std, device)
            lang = language_tensor[language_id].to(device, non_blocking=True)
            logits = model(va, vb, lang, coordinates)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y.to(device))
            loss.backward()
            optimizer.step()
    return model.eval().cpu()


@torch.inference_mode()
def predict_logits(
    model: SpatialRepresentationProbe,
    global_features: np.ndarray,
    patch_features: np.ndarray,
    language: np.ndarray,
    pairs: PairSet,
    statistics: FeatureStatistics,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    family = model.family
    model = model.to(device).eval()
    global_tensor = torch.from_numpy(standardized_global(global_features, statistics))
    patch_tensor = torch.from_numpy(np.asarray(patch_features))
    language_tensor = torch.from_numpy(standardized_language(language, statistics))
    patch_mean = torch.from_numpy(statistics.patch_mean).to(device).view(1, 1, 512)
    patch_std = torch.from_numpy(statistics.patch_std).to(device).view(1, 1, 512)
    coordinates = torch.from_numpy(fixed_patch_coordinates()).to(device)
    output: list[np.ndarray] = []
    for start in range(0, len(pairs.a), batch_size):
        stop = min(start + batch_size, len(pairs.a))
        a = torch.from_numpy(pairs.a[start:stop])
        b = torch.from_numpy(pairs.b[start:stop])
        language_id = torch.from_numpy(pairs.language_id[start:stop])
        va = _batch_visual(family, a, global_tensor, patch_tensor, patch_mean, patch_std, device)
        vb = _batch_visual(family, b, global_tensor, patch_tensor, patch_mean, patch_std, device)
        lang = language_tensor[language_id].to(device, non_blocking=True)
        output.append(model(va, vb, lang, coordinates).float().cpu().numpy())
    model.cpu()
    return np.concatenate(output).astype(np.float64)


@torch.inference_mode()
def attention_arrays(
    model: SpatialRepresentationProbe,
    patch_features: np.ndarray,
    language: np.ndarray,
    frame: np.ndarray,
    language_id: np.ndarray,
    statistics: FeatureStatistics,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    if model.family == "R0":
        return np.empty((len(frame), 0, 49), dtype=np.float32)
    model = model.to(device).eval()
    patch_tensor = torch.from_numpy(np.asarray(patch_features))
    language_tensor = torch.from_numpy(standardized_language(language, statistics))
    patch_mean = torch.from_numpy(statistics.patch_mean).to(device).view(1, 1, 512)
    patch_std = torch.from_numpy(statistics.patch_std).to(device).view(1, 1, 512)
    coordinates = torch.from_numpy(fixed_patch_coordinates()).to(device)
    output: list[np.ndarray] = []
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        index = torch.from_numpy(np.asarray(frame[start:stop], dtype=np.int64))
        language_index = torch.from_numpy(np.asarray(language_id[start:stop], dtype=np.int64))
        patch = patch_tensor[index].to(device, dtype=torch.float32, non_blocking=True)
        patch = (patch - patch_mean) / patch_std
        lang = language_tensor[language_index].to(device, non_blocking=True)
        _, attention = model.represent(patch, lang, coordinates, return_attention=True)
        assert attention is not None
        output.append(attention.float().cpu().numpy())
    model.cpu()
    return np.concatenate(output).astype(np.float32)


def attention_diagnostics(
    original: np.ndarray,
    alternate: np.ndarray,
    coordinates: np.ndarray,
) -> dict[str, Any]:
    if original.shape != alternate.shape or original.ndim != 3:
        raise ValueError("attention diagnostic shape mismatch")
    entropy = lambda x: -(x * np.log(np.maximum(x, 1.0e-12))).sum(axis=-1) / np.log(49.0)
    original_center = np.einsum("nsp,pd->nsd", original, coordinates)
    alternate_center = np.einsum("nsp,pd->nsd", alternate, coordinates)
    total_variation = 0.5 * np.abs(alternate - original).sum(axis=-1)
    center_change = np.linalg.norm(alternate_center - original_center, axis=-1)
    return {
        "rows": int(len(original)),
        "slots": int(original.shape[1]),
        "original_entropy_mean": entropy(original).mean(axis=0).astype(float).tolist(),
        "alternate_entropy_mean": entropy(alternate).mean(axis=0).astype(float).tolist(),
        "language_swap_total_variation_mean": total_variation.mean(axis=0).astype(float).tolist(),
        "language_swap_center_change_mean": center_change.mean(axis=0).astype(float).tolist(),
        "original_center_mean": original_center.mean(axis=0).astype(float).tolist(),
        "alternate_center_mean": alternate_center.mean(axis=0).astype(float).tolist(),
    }
