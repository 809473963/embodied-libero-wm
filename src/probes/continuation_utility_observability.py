"""Stage2A feature contracts and root-level decision analysis.

No simulator or dataset is opened on import. Outcome metadata has no path into
the readout's four explicitly named inputs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .candidate_utility_oracle import candidate_subsets, decision_summary, sha256

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'results/continuation_utility_observability'
PROTOCOL = ROOT / 'configs/continuation_utility_observability_protocol.json'
ENTITIES = ('akita_black_bowl_1', 'akita_black_bowl_2', 'plate_1', 'cookies_1',
            'glazed_rim_porcelain_ramekin_1', 'main_table', 'wooden_cabinet_1', 'flat_stove_1')
IDS = candidate_subsets()['N32']
STATE_DIM = 476
MAX_FRAMES = 17
INPUT_KEYS = frozenset(('sequence', 'lengths', 'language', 'budget'))
RAW_KEYS = frozenset(f'{phase}_{field}' for phase in ('prefix', 'branch')
                     for field in ('rgb', 'proprio', 'state', 'actions', 'time'))


def sync(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    with tmp.open('w') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    sync(path.parent)


def write_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    sync(path.parent)


def validate_protocol():
    protocol = json.loads(PROTOCOL.read_text())
    for path, expected in protocol['frozen_files'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f'frozen input/source changed: {path}')
    return protocol, sha256(PROTOCOL)


def validate_raw(arrays):
    if set(arrays) != RAW_KEYS:
        raise ValueError(f'raw feature whitelist violation: {set(arrays) ^ RAW_KEYS}')
    for phase in ('prefix', 'branch'):
        leading = arrays[f'{phase}_time'].shape
        for field, tail in (('rgb', (128, 128, 3)), ('proprio', (12,)),
                            ('state', (STATE_DIM,)), ('actions', (7,))):
            value = arrays[f'{phase}_{field}']
            if value.shape != leading + tail or not np.isfinite(value).all():
                raise ValueError(f'invalid {phase}_{field}')
    if arrays['branch_time'].shape != (32, 4):
        raise ValueError('all fixed N32 candidates and four H8 samples required')


def fit_normalization(sequence, lengths, fit_rows):
    """Fit over valid timesteps of fit roots only; never padding or development."""
    values = sequence[fit_rows]
    valid = np.arange(values.shape[-2])[None, None, :] < lengths[fit_rows, :, None]
    frames = values[valid]
    if not len(frames) or not np.isfinite(frames).all():
        raise ValueError('normalization needs finite fit frames')
    return frames.mean(0), np.maximum(frames.std(0), 1e-4)


def normalize(sequence, lengths, mean, std):
    x = (sequence - mean) / std
    valid = np.arange(x.shape[-2])[None, None, :] < lengths[:, :, None]
    return np.where(valid[..., None], x, 0).astype(np.float32)


def check_inputs(inputs):
    if set(inputs) != INPUT_KEYS:
        raise ValueError(f'model feature whitelist violation: {set(inputs) ^ INPUT_KEYS}')
    n = len(inputs['lengths'])
    if inputs['sequence'].shape[0] != n or inputs['language'].shape != (n, 512) or inputs['budget'].shape != (n, 1):
        raise ValueError('model input shape mismatch')
    if np.any(inputs['lengths'] < 1) or np.any(inputs['lengths'] > MAX_FRAMES):
        raise ValueError('invalid sequence lengths')


def select(scores, threshold=0., votes=None):
    p = np.asarray(scores, float)
    if p.ndim != 2 or p.shape[1] != 32 or not np.isfinite(p).all():
        raise ValueError('finite root x N32 scores required')
    delta = p - p[:, :1]
    allowed = delta > threshold
    if votes is not None:
        if np.asarray(votes).shape != p.shape:
            raise ValueError('vote shape mismatch')
        allowed &= votes >= 2
    allowed[:, 0] = True
    return np.argmax(np.where(allowed, p, -np.inf), axis=1)


def decisions(y, local_choice):
    full = np.zeros((len(y), 64), bool)
    full[:, IDS] = y
    return decision_summary(full, np.asarray(IDS)[local_choice])


def choose_threshold(scores, y, grid, votes=None):
    choices = []
    for threshold in grid:
        m = decisions(y, select(scores, threshold, votes))
        choices.append({'threshold': threshold, 'metrics': m})
    safe = [x for x in choices if x['metrics']['harm_roots'] == 0]
    if not safe:
        raise ValueError('threshold grid must include A0-only fallback')
    chosen = max(safe, key=lambda x: (x['metrics']['net_success_gain_roots'],
                                    -x['metrics']['override_roots'], x['threshold']))
    return chosen['threshold'], choices


def make_readout(dimension):
    import torch
    from torch import nn

    class Readout(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(nn.Linear(dimension, 64), nn.LayerNorm(64), nn.GELU())
            self.memory = nn.GRU(64, 64, batch_first=True)
            self.language = nn.Sequential(nn.Linear(512, 64), nn.GELU())
            self.head = nn.Sequential(nn.Linear(129, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))

        def forward(self, **inputs):
            if set(inputs) != INPUT_KEYS:
                raise ValueError('model feature whitelist violation')
            x = self.input(inputs['sequence'])
            packed = nn.utils.rnn.pack_padded_sequence(x, inputs['lengths'].cpu(),
                                                     batch_first=True, enforce_sorted=False)
            _, h = self.memory(packed)
            z = torch.cat((h[-1], self.language(inputs['language']), inputs['budget']), dim=-1)
            return self.head(z).squeeze(-1)

    return Readout()


def loss_function(logits, labels, ranking_weight=.5):
    import torch
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    difference = logits[:, :, None] - logits[:, None, :]
    mask = (labels[:, :, None] == 1) & (labels[:, None, :] == 0)
    counts = mask.sum((1, 2))
    mixed = counts > 0
    rank = (F.softplus(-difference) * mask).sum((1, 2)) / counts.clamp_min(1)
    ranking = rank[mixed].mean() if mixed.any() else logits.sum() * 0
    return bce + ranking_weight * ranking, bce, ranking
