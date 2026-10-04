"""Pure decision analysis and selective init loading for the utility oracle.

No environment, HDF5, or model is opened on import.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


class _SelectionComplete(Exception):
    pass


def selective_init_states(path: Path, allowed: tuple[int, ...] = (31, 32, 33, 34)) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Extract only admitted vectors from the pinned numpy pickle container.

    Unlike torch.load(), this never constructs non-admitted numpy arrays. Earlier
    list records are placeholders; parsing stops after the last admitted BUILD.
    A matrix container transports one byte payload, but only admitted row slices
    are decoded as numbers. Transport and framing differ from array access.
    """
    if allowed != (31, 32, 33, 34):
        raise ValueError('only init31--34 are admitted by this experiment')
    arrays: dict[int, np.ndarray] = {}
    cursor = [-1]
    storage_format = ['numpy-list']

    class ArrayPlaceholder:
        def __init__(self, index: int):
            self.index = index

        def __setstate__(self, state):
            if self.index == 0 and len(state[1]) == 2:
                # Pinned LIBERO files may contain one C-contiguous matrix,
                # rather than a list of individual vectors. Decode the byte
                # storage, then materialize only explicitly admitted row slices.
                version, shape, dtype, fortran, raw = state
                if fortran or not isinstance(raw, bytes) or dtype.hasobject or shape[0] <= max(allowed):
                    raise ValueError('unsupported matrix init serialization')
                width = int(shape[1])
                for row in allowed:
                    value = np.frombuffer(raw, dtype=dtype, count=width,
                                          offset=row*width*dtype.itemsize).copy()
                    if not np.isfinite(value).all():
                        raise ValueError('invalid admitted simulator init state')
                    arrays[row] = value
                storage_format[0] = 'one C-contiguous matrix; admitted row slices only'
                raise _SelectionComplete()
            if self.index in allowed:
                value = np.empty(0, dtype=np.uint8)
                value.__setstate__(state)
                if value.ndim != 1 or not np.isfinite(value).all():
                    raise ValueError('invalid admitted simulator init state')
                arrays[self.index] = value.copy()
            if self.index == max(allowed):
                raise _SelectionComplete()

    def reconstruct(*args):
        cursor[0] += 1
        if cursor[0] > max(allowed):
            raise RuntimeError('reader passed the admitted init boundary')
        return ArrayPlaceholder(cursor[0])

    def selective_encode(value, encoding='latin1'):
        return codecs.encode(value, encoding) if cursor[0] in allowed or cursor[0] == 0 else b''

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == '_reconstruct':
                return reconstruct
            if module == 'numpy' and name == 'ndarray':
                return np.ndarray
            if module == 'numpy' and name == 'dtype':
                return np.dtype
            if module == '_codecs' and name == 'encode':
                return selective_encode
            raise ValueError(f'unsupported global in trusted init serialization: {module}.{name}')

    with zipfile.ZipFile(path) as archive:
        entries = [x for x in archive.namelist() if x.endswith('/data.pkl')]
        if len(entries) != 1:
            raise ValueError('expected one numpy-list pickle in pinned init archive')
        with archive.open(entries[0]) as stream:
            try:
                RestrictedUnpickler(stream).load()
            except _SelectionComplete:
                pass
    if set(arrays) != set(allowed):
        raise ValueError('init archive did not yield exactly the admitted records')
    return arrays, {
        'path': str(path), 'numeric_init_indices_accessed': list(allowed),
        'last_ndarray_record_parsed': cursor[0], 'storage_format': storage_format[0],
        'forbidden_init_indices_materialized': [],
        'nonadmitted_prior_records': 'placeholder only; no numeric ndarray reconstruction',
        'later_records': 'list: stopped after init34 BUILD; matrix: no nonadmitted row numerical decoding',
    }


def candidate_subsets() -> dict[str, list[int]]:
    return {f'N{n}': [0, *range(1, 1+c), *range(49, 49+g)]
            for n, c, g in ((8, 4, 3), (16, 10, 5), (32, 22, 9), (64, 48, 15))}


def t1_select(goal: np.ndarray, margin: np.ndarray, epsilon: float, indices: list[int] | None = None) -> int:
    goal, margin = np.asarray(goal, bool), np.asarray(margin, float)
    if goal.shape != margin.shape or goal.ndim != 1 or not np.isfinite(margin).all():
        raise ValueError('complete finite one-dimensional T1 coordinates required')
    ids = np.asarray(indices if indices is not None else range(len(goal)), int)
    if len(ids) == 0 or len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= len(goal)):
        raise ValueError('invalid candidate subset')
    eligible = ids[goal[ids] == goal[ids].max()]
    best = float(margin[eligible].max())
    ties = eligible[best - margin[eligible] <= epsilon]
    return 0 if 0 in ties else int(ties.min())


def identity(reference: dict[str, np.ndarray], actual: dict[str, np.ndarray], tolerance: float = 1e-10) -> dict[str, Any]:
    checks = {}
    for key in ('agentview_image', 'robot0_eef_pos', 'robot0_gripper_qpos', 'robot0_joint_pos'):
        checks[key] = bool(np.array_equal(reference[key], actual[key]))
    same_shape = reference['sim_state'].shape == actual['sim_state'].shape
    error = float(np.max(abs(reference['sim_state'] - actual['sim_state']))) if same_shape else None
    return {'observable_exact': checks, 'state_max_abs': error,
            'pass': bool(all(checks.values()) and same_shape and error <= tolerance)}


def decision_summary(success: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    y = np.asarray(success, bool)
    if y.ndim != 2 or y.shape[1] != 64 or selected.shape != (len(y),):
        raise ValueError('expected root x 64 matrix and one selected candidate per root')
    a0 = y[:, 0]
    oracle = y.any(axis=1)
    chosen = y[np.arange(len(y)), selected]
    rescue = ~a0 & oracle
    improve, harm = ~a0 & chosen, a0 & ~chosen
    override = selected != 0
    n = len(y)
    return {'roots': n, 'a0_successes': int(a0.sum()), 'oracle_successes': int(oracle.sum()),
            'selected_successes': int(chosen.sum()), 'rescue_roots': int(rescue.sum()),
            'a0_failure_roots': int((~a0).sum()), 'captured_rescue_roots': int(improve.sum()),
            'harm_roots': int(harm.sum()), 'override_roots': int(override.sum()),
            'net_success_gain_roots': int(improve.sum()-harm.sum()),
            'net_success_gain_rate': float((chosen.astype(float)-a0).mean()) if n else None,
            'mean_regret': float((oracle.astype(float)-chosen).mean()) if n else None,
            'capture_fraction': float(improve.sum()/rescue.sum()) if rescue.any() else None,
            'override_precision': float(improve[override].mean()) if override.any() else None,
            'override_harm_rate': float(harm[override].mean()) if override.any() else None,
            'rescue_rate_conditional_a0_failure': float(rescue.sum()/(~a0).sum()) if (~a0).any() else None}
