#!/usr/bin/env python3
"""Post-run artifact/label checks and checkpoint inference, without fitting or sim."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from src.probes.candidate_utility_oracle import sha256
from src.probes.continuation_utility_observability import (
    DIRECTORY, IDS, MAX_FRAMES, fit_normalization, make_readout, normalize,
    validate_protocol, validate_raw, write_json,
)


def choice(p, threshold=0., votes=None):
    selected = []
    for index, row in enumerate(p):
        allowed = [0] + [j for j in range(1, 32) if row[j] - row[0] > threshold
                         and (votes is None or votes[index, j] >= 2)]
        selected.append(max(allowed, key=lambda j: (row[j], -j)))
    return np.asarray(selected)


def assert_decision(y, selected, expected):
    actual = y[np.arange(len(y)), selected]
    a0 = y[:, 0]
    rescue = ~a0 & actual
    harm = a0 & ~actual
    values = {'roots': len(y), 'selected_successes': int(actual.sum()),
              'captured_rescue_roots': int(rescue.sum()), 'harm_roots': int(harm.sum()),
              'net_success_gain_roots': int(rescue.sum() - harm.sum()),
              'override_roots': int((selected != 0).sum())}
    assert all(expected[k] == v for k, v in values.items()), (values, expected)
    regret = float((y.any(1).astype(float) - actual).mean())
    assert abs(regret - expected['mean_regret']) <= 1e-12


def main():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    protocol, digest = validate_protocol()
    labels_path = DIRECTORY / 'labels.json'
    feature_manifest_path = DIRECTORY / 'feature_manifest.json'
    labels = json.loads(labels_path.read_text())
    features = json.loads(feature_manifest_path.read_text())
    training = json.loads((DIRECTORY / 'training_manifest.json').read_text())
    metrics = json.loads((DIRECTORY / 'metrics.json').read_text())
    manifest = json.loads((DIRECTORY / 'manifest.json').read_text())
    runtime = json.loads((DIRECTORY / 'runtime.json').read_text())
    for value in (labels, features, training, metrics, manifest, runtime):
        assert value['protocol_sha256'] == digest
    assert runtime['status'] == 'COMPLETE' and runtime['phase'] == 'finished'
    assert manifest['complete'] and len(manifest['completed_roots']) == 40
    assert manifest['simulation_steps_current_run'] == 71680
    assert features['complete'] and training['complete'] and training['fit_count'] == 6
    registered = json.loads((ROOT / 'configs/candidate_utility_oracle_roots.json').read_text())['roots']
    rows = labels['rows']
    assert [r['root'] for r in rows] == registered
    feature_records = {r['root_id']: r for r in features['roots']}
    cached = []
    checks = []
    for row in rows:
        root = row['root']
        key = root['root_id']
        meta = json.loads((DIRECTORY / 'roots' / f'{key}.json').read_text())
        source_path = ROOT / 'results/candidate_utility_oracle/roots' / f'{key}.json'
        source = json.loads(source_path.read_text())
        raw_path = DIRECTORY / 'arrays' / f'{key}.npz'
        old_path = ROOT / 'results/candidate_utility_oracle/arrays' / f'{key}.npz'
        feature_path = DIRECTORY / 'features' / f'{key}.npz'
        assert meta['complete'] and meta['root'] == root and meta['protocol_sha256'] == digest
        assert sha256(raw_path) == meta['arrays_sha256']
        assert sha256(source_path) == meta['source_root_sha256']
        assert sha256(old_path) == source['arrays_sha256']
        assert meta['candidate_ids'] == IDS == [g['candidate_id'] for g in meta['guards']]
        assert all(g['pass'] for g in meta['guards'])
        assert row['eligible'] == (source['reference']['first_success_step'] is None)
        assert row['split'] == ('fit' if root['init_index'] < 34 else 'development')
        assert row['y'] == [source['rows'][j]['success'] for j in IDS]
        assert row['success_by_h8'] == [source['rows'][j]['first_success_step'] is not None
               and source['rows'][j]['first_success_step'] <= root['prefix_steps'] + 8 for j in IDS]
        with np.load(raw_path, allow_pickle=False) as a:
            raw = {k: a[k] for k in a.files}
        validate_raw(raw)
        with np.load(old_path, allow_pickle=False) as old:
            assert np.array_equal(raw['branch_rgb'][:, -1], old['h8_rgb'][IDS])
            assert np.array_equal(raw['prefix_rgb'][-1], old['root_rgb'])
            for j, identifier in enumerate(IDS):
                means = old['candidates'][identifier].reshape(4, 2, 7).mean(1)
                assert np.array_equal(raw['branch_actions'][j], means)
        assert raw['prefix_time'].tolist() == list(range(0, root['prefix_steps'] + 1, 8))
        assert np.array_equal(raw['branch_time'], np.tile(root['prefix_steps'] + np.array([2, 4, 6, 8]), (32, 1)))
        assert np.array_equal(raw['branch_state'][..., 184:196], raw['branch_proprio'])
        assert sha256(feature_path) == feature_records[key]['features_sha256']
        assert feature_records[key]['raw_sha256'] == meta['arrays_sha256']
        with np.load(feature_path, allow_pickle=False) as a:
            assert set(a.files) == {'G_sequence', 'N_sequence', 'lengths', 'language', 'budget'}
            data = {k: a[k] for k in a.files}
        length = len(raw['prefix_time'])
        assert np.all(data['lengths'] == length + 4)
        for condition, dimension in (('G', 484), ('N', 532)):
            assert data[f'{condition}_sequence'].shape == (32, MAX_FRAMES, dimension)
            assert not data[f'{condition}_sequence'][:, length + 4:].any()
            sequence = data[f'{condition}_sequence']
            assert np.array_equal(sequence[:, length:length + 4, -8:-1], raw['branch_actions'])
            assert np.array_equal(sequence[:, length:length + 4, -1], (raw['branch_time'] / 300).astype(np.float32))
        assert np.array_equal(data['G_sequence'][:, length:length + 4, :476], raw['branch_state'])
        assert np.array_equal(data['N_sequence'][:, length:length + 4, 512:524], raw['branch_proprio'])
        assert np.all(data['budget'] == np.float32((300 - root['prefix_steps'] - 8) / 300))
        cached.append(data)
        checks.append({'root_id': key, 'branches': 32, 'artifact_hashes': True,
                       'reused_labels': True, 'h8_rgb_exact': True, 'feature_timing': True})
    y = np.asarray([r['y'] for r in rows], bool)
    fit = np.asarray([r['eligible'] and r['split'] == 'fit' for r in rows])
    dev = np.asarray([r['eligible'] and r['split'] == 'development' for r in rows])
    assert fit.sum() == 29 and dev.sum() == 9 and not np.any(fit & dev)
    lengths = np.stack([r['lengths'] for r in cached])
    readout_checks = []
    for condition in ('G', 'N'):
        sequence = np.stack([r[f'{condition}_sequence'] for r in cached])
        mean, std = fit_normalization(sequence, lengths, fit)
        with np.load(DIRECTORY / f'{condition}_normalization.npz', allow_pickle=False) as a:
            assert np.array_equal(a['mean'], mean) and np.array_equal(a['std'], std)
        x = normalize(sequence, lengths, mean, std)
        inputs = {'sequence': torch.from_numpy(x.reshape(-1, 17, x.shape[-1])).cuda(),
                  'lengths': torch.from_numpy(lengths.reshape(-1)).cuda(),
                  'language': torch.from_numpy(np.stack([r['language'] for r in cached]).reshape(-1, 512)).cuda(),
                  'budget': torch.from_numpy(np.stack([r['budget'] for r in cached]).reshape(-1, 1)).cuda()}
        predictions = []
        for seed in protocol['training']['seeds']:
            stem = DIRECTORY / 'readouts' / f'{condition}_seed{seed}'
            metadata = json.loads(stem.with_suffix('.json').read_text())
            assert metadata['protocol_sha256'] == digest
            for path, key in ((labels_path, 'labels_sha256'), (feature_manifest_path, 'feature_manifest_sha256'),
                              (stem.with_suffix('.pt'), 'checkpoint_sha256'), (stem.with_suffix('.npz'), 'prediction_sha256')):
                assert sha256(path) == metadata[key]
            model = make_readout(x.shape[-1]).cuda().eval()
            state = torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=True)
            assert state['condition'] == condition and state['seed'] == seed and state['protocol_sha256'] == digest
            model.load_state_dict(state['state_dict'])
            with torch.inference_mode():
                actual = model(**inputs).sigmoid().reshape(40, 32).cpu().numpy()
            with np.load(stem.with_suffix('.npz'), allow_pickle=False) as a:
                p = a['scores']
            error = float(np.max(np.abs(actual - p)))
            assert error <= 1e-6, (condition, seed, error)
            published = next(s for s in metrics['conditions'][condition]['seeds'] if s['seed'] == seed)
            assert_decision(y[fit], choice(p[fit]), published['fit_threshold0'])
            assert_decision(y[dev], choice(p[dev]), published['development_threshold0'])
            predictions.append(p)
            readout_checks.append({'condition': condition, 'seed': seed, 'checkpoint_inference_maxabs': error,
                                   'fit_and_development_decisions_recomputed': True})
            del model
        ensemble = np.mean(predictions, axis=0)
        votes = np.sum(np.asarray(predictions) > np.asarray(predictions)[:, :, :1], axis=0)
        condition_result = metrics['conditions'][condition]
        assert_decision(y[dev], choice(ensemble[dev], 0., votes[dev]), condition_result['development_threshold0'])
        assert_decision(y[dev], choice(ensemble[dev], condition_result['threshold'], votes[dev]), condition_result['development'])
        brier = float(((ensemble[dev] - y[dev]) ** 2).mean())
        assert abs(brier - condition_result['calibration_development']['brier']) < 1e-8
    result = {'protocol_sha256': digest, 'pass': True, 'root_count': 40, 'branch_count': 1280,
              'readout_count': 6, 'simulation_steps': 0, 'new_fits': 0, 'root_checks': checks,
              'readout_checks': readout_checks, 'normalization_recomputed_fit_only': True,
              'decisions_recomputed': True, 'wall_seconds': time.monotonic() - started,
              'verdict': metrics['verdict']}
    write_json(DIRECTORY / 'final_verification.json', result)
    print(json.dumps({k: result[k] for k in ('pass', 'root_count', 'branch_count', 'readout_count', 'wall_seconds', 'verdict')}))


if __name__ == '__main__':
    main()
