"""Bounded readers and diagnostic probes for fixed CLIP spatial layers.

The frozen visual backbone is intentionally absent.  Every tensor accepted by
this module is an offline cache from one preregistered extraction point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.probes.semantic_value_distillation import PairSet, rank_auc


LAYERS = ("L0", "L3", "L6", "L9", "L12-pre", "L12-postLN", "L12-proj")
READERS = ("P0", "P1")


def fixed_patch_coordinates(grid_size: int = 7) -> np.ndarray:
    grid_size = int(grid_size)
    axis = 2.0 * (np.arange(grid_size, dtype=np.float32) + 0.5) / grid_size - 1.0
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx, yy), axis=-1).reshape(grid_size * grid_size, 2).astype(np.float32)


def coordinates_for_token_count(token_count: int) -> np.ndarray:
    grid_size = int(round(np.sqrt(int(token_count))))
    if grid_size * grid_size != int(token_count):
        raise ValueError("spatial token count is not a square grid")
    return fixed_patch_coordinates(grid_size)


def set_deterministic(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def parameter_count(model: nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters() if value.requires_grad))


@dataclass(frozen=True)
class LayerStatistics:
    token_mean: np.ndarray
    token_std: np.ndarray
    language_mean: np.ndarray
    language_std: np.ndarray


def fit_layer_statistics(
    tokens: np.ndarray, language: np.ndarray, train_mask: np.ndarray, chunk: int = 128
) -> LayerStatistics:
    indices = np.flatnonzero(train_mask)
    dimension = int(tokens.shape[-1])
    total = np.zeros(dimension, dtype=np.float64)
    square = np.zeros(dimension, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), chunk):
        value = np.asarray(tokens[indices[start : start + chunk]], dtype=np.float64)
        total += value.sum(axis=(0, 1))
        square += np.square(value).sum(axis=(0, 1))
        count += int(value.shape[0] * value.shape[1])
    mean = total / count
    std = np.sqrt(np.maximum(square / count - np.square(mean), 0.0))
    std[std < 1.0e-6] = 1.0
    lang = np.asarray(language, dtype=np.float64)
    language_mean = lang.mean(axis=0)
    language_std = lang.std(axis=0)
    language_std[language_std < 1.0e-6] = 1.0
    return LayerStatistics(
        mean.astype(np.float32), std.astype(np.float32),
        language_mean.astype(np.float32), language_std.astype(np.float32),
    )


def standardized_language(language: np.ndarray, statistics: LayerStatistics) -> np.ndarray:
    return ((np.asarray(language, dtype=np.float32) - statistics.language_mean)
            / statistics.language_std).astype(np.float32)


class P1StateEncoder(nn.Module):
    """One language-conditioned spatial slot with a fixed 51-D output."""

    def __init__(self, token_dim: int, language_conditioned: bool = True) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.language_conditioned = bool(language_conditioned)
        self.key = nn.Linear(self.token_dim + 2, 24)
        self.value = nn.Linear(self.token_dim + 2, 24)
        if self.language_conditioned:
            self.query = nn.Linear(512, 24)
        else:
            self.fixed_query = nn.Parameter(torch.empty(24))
            nn.init.normal_(self.fixed_query, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        language: torch.Tensor,
        coordinates: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if tokens.ndim != 3 or tokens.shape[2] != self.token_dim:
            raise ValueError("unexpected spatial-token tensor")
        if coordinates.shape != (tokens.shape[1], 2):
            raise ValueError("coordinate grid does not match spatial tokens")
        coordinate_batch = coordinates.unsqueeze(0).expand(len(tokens), -1, -1)
        patch = torch.cat((tokens, coordinate_batch), dim=-1)
        key, value = self.key(patch), self.value(patch)
        if self.language_conditioned:
            query = self.query(language)
        else:
            query = self.fixed_query.view(1, 24).expand(len(tokens), 24)
        attention = torch.softmax(
            torch.einsum("bd,bpd->bp", query, key) / np.sqrt(24.0), dim=-1
        )
        slot = torch.einsum("bp,bpd->bd", attention, value)
        mean_value = value.mean(dim=1)
        center = torch.einsum("bp,pd->bd", attention, coordinates)
        entropy = -(attention * torch.log(attention.clamp_min(1.0e-12))).sum(dim=-1, keepdim=True)
        entropy = entropy / float(np.log(tokens.shape[1]))
        state = torch.cat((slot, mean_value, center, entropy), dim=-1)
        return state, attention if return_attention else None


class InternalLayerPairReader(nn.Module):
    """P0/P1 state reader followed by one shared antisymmetric pair head."""

    def __init__(self, token_dim: int, family: str) -> None:
        super().__init__()
        if family not in READERS:
            raise ValueError(f"unknown reader: {family}")
        self.family = family
        self.token_dim = int(token_dim)
        self.encoder = P1StateEncoder(token_dim, language_conditioned=(family == "P1"))
        self.raw = nn.Sequential(nn.Linear(3 * 51 + 512, 64), nn.GELU(), nn.Linear(64, 1))

    def _raw(self, a: torch.Tensor, b: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        return self.raw(torch.cat((a, b, b - a, language), dim=-1)).squeeze(-1)

    def forward(
        self,
        tokens_a: torch.Tensor,
        tokens_b: torch.Tensor,
        language: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        a, _ = self.encoder(tokens_a, language, coordinates)
        b, _ = self.encoder(tokens_b, language, coordinates)
        return self._raw(a, b, language) - self._raw(b, a, language)


class ReferentChangeProbe(nn.Module):
    """Permutation-invariant same-scene referent-change diagnostic."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.encoder = P1StateEncoder(token_dim, True)
        self.head = nn.Linear(102, 1)

    def forward(
        self, tokens: torch.Tensor, language_a: torch.Tensor,
        language_b: torch.Tensor, coordinates: torch.Tensor,
    ) -> torch.Tensor:
        a, _ = self.encoder(tokens, language_a, coordinates)
        b, _ = self.encoder(tokens, language_b, coordinates)
        return self.head(torch.cat((torch.abs(a - b), a * b), dim=-1)).squeeze(-1)


class M2OrderingProbe(nn.Module):
    """Scalar state score used only to diagnose accessible M2 ordering."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.encoder = P1StateEncoder(token_dim, True)
        self.score = nn.Linear(51, 1)

    def state_score(
        self, tokens: torch.Tensor, language: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        state, _ = self.encoder(tokens, language, coordinates)
        return self.score(state).squeeze(-1)

    def forward(
        self, tokens_a: torch.Tensor, tokens_b: torch.Tensor,
        language: torch.Tensor, coordinates: torch.Tensor,
    ) -> torch.Tensor:
        return self.state_score(tokens_b, language, coordinates) - self.state_score(
            tokens_a, language, coordinates
        )


class GoalBooleanProbe(nn.Module):
    """Linear official-goal diagnostic over one P1 state representation."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.encoder = P1StateEncoder(token_dim, True)
        self.head = nn.Linear(51, 1)

    def forward(
        self, tokens: torch.Tensor, language: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        state, _ = self.encoder(tokens, language, coordinates)
        return self.head(state).squeeze(-1)


def _loader(*arrays: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    dataset = TensorDataset(*(torch.from_numpy(np.asarray(value)) for value in arrays))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def _token_batch(
    tokens: np.ndarray | torch.Tensor, index: torch.Tensor, statistics: LayerStatistics,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(tokens, torch.Tensor):
        value = tokens[index.to(tokens.device, non_blocking=True)]
    else:
        value = torch.from_numpy(np.asarray(tokens[index.numpy()], dtype=np.float32)).to(
            device, non_blocking=True
        )
    mean = torch.from_numpy(statistics.token_mean).to(device).view(1, 1, -1)
    std = torch.from_numpy(statistics.token_std).to(device).view(1, 1, -1)
    return (value - mean) / std


def train_pair_reader(
    token_dim: int, family: str, tokens: np.ndarray, language: np.ndarray,
    pairs: PairSet, statistics: LayerStatistics, epochs: int, batch_size: int,
    seed: int, device: torch.device,
) -> InternalLayerPairReader:
    set_deterministic(seed)
    model = InternalLayerPairReader(token_dim, family).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    target = ((pairs.label.astype(np.float32) + 1.0) / 2.0).astype(np.float32)
    loader = _loader(pairs.a, pairs.b, pairs.language_id, target,
                     batch_size=batch_size, seed=seed)
    language_tensor = torch.from_numpy(standardized_language(language, statistics))
    coordinates = torch.from_numpy(coordinates_for_token_count(tokens.shape[1])).to(device)
    model.train()
    for _ in range(epochs):
        for a, b, language_id, y in loader:
            optimizer.zero_grad(set_to_none=True)
            va = _token_batch(tokens, a, statistics, device)
            vb = _token_batch(tokens, b, statistics, device)
            lang = language_tensor[language_id].to(device, non_blocking=True)
            logits = model(va, vb, lang, coordinates)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y.to(device))
            loss.backward()
            optimizer.step()
    return model.eval().cpu()


@torch.inference_mode()
def predict_pair_logits(
    model: InternalLayerPairReader, tokens: np.ndarray, language: np.ndarray,
    pairs: PairSet, statistics: LayerStatistics, device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model = model.to(device).eval()
    language_tensor = torch.from_numpy(standardized_language(language, statistics))
    coordinates = torch.from_numpy(coordinates_for_token_count(tokens.shape[1])).to(device)
    output: list[np.ndarray] = []
    for start in range(0, len(pairs.a), batch_size):
        stop = min(start + batch_size, len(pairs.a))
        a = torch.from_numpy(pairs.a[start:stop])
        b = torch.from_numpy(pairs.b[start:stop])
        lang_id = torch.from_numpy(pairs.language_id[start:stop])
        va = _token_batch(tokens, a, statistics, device)
        vb = _token_batch(tokens, b, statistics, device)
        lang = language_tensor[lang_id].to(device, non_blocking=True)
        output.append(model(va, vb, lang, coordinates).float().cpu().numpy())
    model.cpu()
    return np.concatenate(output).astype(np.float64)


@torch.inference_mode()
def attention_for(
    model: InternalLayerPairReader, tokens: np.ndarray, language: np.ndarray,
    frame: np.ndarray, language_id: np.ndarray, statistics: LayerStatistics,
    device: torch.device, batch_size: int = 1024,
) -> np.ndarray:
    model = model.to(device).eval()
    language_tensor = torch.from_numpy(standardized_language(language, statistics))
    coordinates = torch.from_numpy(coordinates_for_token_count(tokens.shape[1])).to(device)
    output: list[np.ndarray] = []
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        index = torch.from_numpy(frame[start:stop].astype(np.int64))
        language_index = torch.from_numpy(language_id[start:stop].astype(np.int64))
        visual = _token_batch(tokens, index, statistics, device)
        lang = language_tensor[language_index].to(device, non_blocking=True)
        _, attention = model.encoder(visual, lang, coordinates, return_attention=True)
        assert attention is not None
        output.append(attention.float().cpu().numpy())
    model.cpu()
    return np.concatenate(output).astype(np.float32)


def attention_diagnostics(original: np.ndarray, alternate: np.ndarray) -> dict[str, float]:
    coordinates = coordinates_for_token_count(original.shape[1])
    original_center = np.einsum("bp,pd->bd", original, coordinates)
    alternate_center = np.einsum("bp,pd->bd", alternate, coordinates)
    total_variation = 0.5 * np.abs(original - alternate).sum(axis=-1)
    displacement = np.linalg.norm(original_center - alternate_center, axis=-1)
    return {
        "mean_total_variation": float(total_variation.mean()),
        "median_total_variation": float(np.median(total_variation)),
        "mean_center_displacement": float(displacement.mean()),
        "median_center_displacement": float(np.median(displacement)),
    }


def train_referent_probe(
    token_dim: int, tokens: np.ndarray, language: np.ndarray,
    frame: np.ndarray, language_a: np.ndarray, language_b: np.ndarray,
    label: np.ndarray, statistics: LayerStatistics, epochs: int, batch_size: int,
    seed: int, device: torch.device,
) -> ReferentChangeProbe:
    set_deterministic(seed)
    model = ReferentChangeProbe(token_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    loader = _loader(frame, language_a, language_b, label.astype(np.float32),
                     batch_size=batch_size, seed=seed)
    language_tensor = torch.from_numpy(standardized_language(language, statistics))
    coordinates = torch.from_numpy(coordinates_for_token_count(tokens.shape[1])).to(device)
    for _ in range(epochs):
        model.train()
        for index, la, lb, y in loader:
            optimizer.zero_grad(set_to_none=True)
            visual = _token_batch(tokens, index, statistics, device)
            logits = model(visual, language_tensor[la].to(device),
                           language_tensor[lb].to(device), coordinates)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y.to(device))
            loss.backward(); optimizer.step()
    return model.eval().cpu()


@torch.inference_mode()
def predict_referent_logits(
    model: ReferentChangeProbe, tokens: np.ndarray, language: np.ndarray,
    frame: np.ndarray, language_a: np.ndarray, language_b: np.ndarray,
    statistics: LayerStatistics, device: torch.device, batch_size: int = 1024,
) -> np.ndarray:
    model = model.to(device).eval()
    language_tensor = torch.from_numpy(standardized_language(language, statistics))
    coordinates = torch.from_numpy(coordinates_for_token_count(tokens.shape[1])).to(device)
    output=[]
    for start in range(0, len(frame), batch_size):
        stop=min(start+batch_size,len(frame)); index=torch.from_numpy(frame[start:stop])
        visual=_token_batch(tokens,index,statistics,device)
        la=torch.from_numpy(language_a[start:stop]); lb=torch.from_numpy(language_b[start:stop])
        output.append(model(visual,language_tensor[la].to(device),language_tensor[lb].to(device),coordinates).cpu().numpy())
    model.cpu(); return np.concatenate(output).astype(np.float64)


def train_m2_probe(
    token_dim: int, tokens: np.ndarray, language: np.ndarray, pairs: PairSet,
    statistics: LayerStatistics, epochs: int, batch_size: int, seed: int,
    device: torch.device,
) -> M2OrderingProbe:
    set_deterministic(seed); model=M2OrderingProbe(token_dim).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1.0e-3,weight_decay=1.0e-4)
    target=((pairs.label.astype(np.float32)+1.0)/2.0).astype(np.float32)
    loader=_loader(pairs.a,pairs.b,pairs.language_id,target,batch_size=batch_size,seed=seed)
    language_tensor=torch.from_numpy(standardized_language(language,statistics)); coordinates=torch.from_numpy(coordinates_for_token_count(tokens.shape[1])).to(device)
    for _ in range(epochs):
        model.train()
        for a,b,l,y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits=model(_token_batch(tokens,a,statistics,device),_token_batch(tokens,b,statistics,device),language_tensor[l].to(device),coordinates)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,y.to(device)); loss.backward(); optimizer.step()
    return model.eval().cpu()


@torch.inference_mode()
def predict_m2_logits(
    model: M2OrderingProbe, tokens: np.ndarray, language: np.ndarray,
    pairs: PairSet, statistics: LayerStatistics, device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model=model.to(device).eval(); language_tensor=torch.from_numpy(standardized_language(language,statistics)); coordinates=torch.from_numpy(coordinates_for_token_count(tokens.shape[1])).to(device); output=[]
    for start in range(0,len(pairs.a),batch_size):
        stop=min(start+batch_size,len(pairs.a)); a=torch.from_numpy(pairs.a[start:stop]); b=torch.from_numpy(pairs.b[start:stop]); l=torch.from_numpy(pairs.language_id[start:stop])
        output.append(model(_token_batch(tokens,a,statistics,device),_token_batch(tokens,b,statistics,device),language_tensor[l].to(device),coordinates).cpu().numpy())
    model.cpu(); return np.concatenate(output).astype(np.float64)


def train_goal_probe(
    token_dim: int, tokens: np.ndarray, language: np.ndarray, frame: np.ndarray,
    language_id: np.ndarray, label: np.ndarray, statistics: LayerStatistics,
    epochs: int, batch_size: int, seed: int, device: torch.device,
) -> GoalBooleanProbe:
    set_deterministic(seed); model=GoalBooleanProbe(token_dim).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1.0e-3,weight_decay=1.0e-4)
    positives=max(float(label.sum()),1.0); negative=max(float(len(label)-label.sum()),1.0)
    pos_weight=torch.tensor(negative/positives,device=device)
    loader=_loader(frame,language_id,label.astype(np.float32),batch_size=batch_size,seed=seed)
    language_tensor=torch.from_numpy(standardized_language(language,statistics)); coordinates=torch.from_numpy(coordinates_for_token_count(tokens.shape[1])).to(device)
    for _ in range(epochs):
        model.train()
        for index,l,y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits=model(_token_batch(tokens,index,statistics,device),language_tensor[l].to(device),coordinates)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,y.to(device),pos_weight=pos_weight); loss.backward(); optimizer.step()
    return model.eval().cpu()


@torch.inference_mode()
def predict_goal_logits(
    model: GoalBooleanProbe, tokens: np.ndarray, language: np.ndarray,
    frame: np.ndarray, language_id: np.ndarray, statistics: LayerStatistics,
    device: torch.device, batch_size: int = 1024,
) -> np.ndarray:
    model=model.to(device).eval(); language_tensor=torch.from_numpy(standardized_language(language,statistics)); coordinates=torch.from_numpy(coordinates_for_token_count(tokens.shape[1])).to(device); output=[]
    for start in range(0,len(frame),batch_size):
        stop=min(start+batch_size,len(frame)); index=torch.from_numpy(frame[start:stop]); l=torch.from_numpy(language_id[start:stop])
        output.append(model(_token_batch(tokens,index,statistics,device),language_tensor[l].to(device),coordinates).cpu().numpy())
    model.cpu(); return np.concatenate(output).astype(np.float64)


def binary_metrics(logits: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    label=np.asarray(label,dtype=np.bool_); prediction=np.asarray(logits)>0
    positive_accuracy=float(prediction[label].mean()) if label.any() else None
    negative_accuracy=float((~prediction[~label]).mean()) if (~label).any() else None
    balanced=None if positive_accuracy is None or negative_accuracy is None else 0.5*(positive_accuracy+negative_accuracy)
    return {
        "rows": int(len(label)), "positive_fraction": float(label.mean()),
        "accuracy": float((prediction==label).mean()),
        "balanced_accuracy": balanced,
        "auroc": float(rank_auc(np.asarray(logits,dtype=np.float64),label)),
    }
