#!/usr/bin/env python3
"""Cache one frozen CLIP checkpoint and fit exactly G/N x three seeds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from src.probes.candidate_utility_oracle import sha256
from src.probes.continuation_utility_observability import (
    DIRECTORY, IDS, MAX_FRAMES, check_inputs, fit_normalization, loss_function,
    make_readout, normalize, sync, validate_protocol, validate_raw, write_json, write_npz,
)


def cache(protocol, digest):
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(1)
    model_spec = protocol['natural_encoder']
    snapshot = snapshot_download(model_spec['model_id'], revision=model_spec['revision'],
                                 cache_dir=ROOT / '.cache/huggingface/hub', local_files_only=True)
    processor = CLIPProcessor.from_pretrained(snapshot, local_files_only=True)
    clip = CLIPModel.from_pretrained(snapshot, local_files_only=True).cuda().eval()
    for parameter in clip.parameters():
        parameter.requires_grad_(False)

    def encode(images):
        batches = []
        for start in range(0, len(images), 32):
            inputs = processor(images=[Image.fromarray(x) for x in images[start:start + 32]], return_tensors='pt')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                z = clip.get_image_features(pixel_values=inputs['pixel_values'].cuda())
                z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            batches.append(z.float().cpu().numpy())
        return np.concatenate(batches)

    roots = json.loads((ROOT / 'configs/candidate_utility_oracle_roots.json').read_text())['roots']
    split = json.loads((ROOT / 'configs/continuation_utility_observability_split.json').read_text())
    label_rows, inputs_manifest = [], []
    started = time.monotonic()
    image_count = 0
    for root in roots:
        key = root['root_id']
        meta_path = DIRECTORY / 'roots' / f'{key}.json'
        meta = json.loads(meta_path.read_text())
        raw_path = DIRECTORY / 'arrays' / f'{key}.npz'
        source_path = ROOT / 'results/candidate_utility_oracle/roots' / f'{key}.json'
        if (not meta['complete'] or meta['protocol_sha256'] != digest or meta['candidate_ids'] != IDS
                or not all(x['pass'] for x in meta['guards']) or len(meta['guards']) != 32
                or sha256(raw_path) != meta['arrays_sha256'] or sha256(source_path) != meta['source_root_sha256']):
            raise ValueError('feature cache requires matching validated raw replay')
        source = json.loads(source_path.read_text())
        feature_path = DIRECTORY / 'features' / f'{key}.npz'
        feature_meta = feature_path.with_suffix('.json')
        if feature_meta.exists():
            record = json.loads(feature_meta.read_text())
            if (record['protocol_sha256'] != digest or record['raw_sha256'] != meta['arrays_sha256']
                    or sha256(feature_path) != record['features_sha256']):
                raise ValueError('feature cache resume hash mismatch')
        else:
            with np.load(raw_path, allow_pickle=False) as a:
                raw = {k: a[k] for k in a.files}
            validate_raw(raw)
            prefix_length = len(raw['prefix_time'])
            prefix_z = encode(raw['prefix_rgb'])
            branch_z = encode(raw['branch_rgb'].reshape(-1, 128, 128, 3)).reshape(32, 4, 512)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                tokens = processor(text=[protocol['instructions'][str(root['task_id'])]], padding=True, return_tensors='pt')
                text = clip.get_text_features(**{k: v.cuda() for k, v in tokens.items()})
                text = (text / text.norm(dim=-1, keepdim=True).clamp_min(1e-12)).float().cpu().numpy()[0]
            feature = {'lengths': np.full(32, prefix_length + 4, np.int64),
                       'language': np.repeat(text[None], 32, axis=0),
                       'budget': np.full((32, 1), (300 - root['prefix_steps'] - 8) / 300, np.float32)}
            for condition in ('G', 'N'):
                if condition == 'G':
                    prefix_base, branch_base = raw['prefix_state'], raw['branch_state']
                else:
                    prefix_base = np.concatenate((prefix_z, raw['prefix_proprio']), axis=-1)
                    branch_base = np.concatenate((branch_z, raw['branch_proprio']), axis=-1)
                prefix = np.concatenate((prefix_base, raw['prefix_actions'], raw['prefix_time'][:, None] / 300), axis=-1)
                branch = np.concatenate((branch_base, raw['branch_actions'], raw['branch_time'][..., None] / 300), axis=-1)
                sequence = np.zeros((32, MAX_FRAMES, prefix.shape[-1]), np.float32)
                sequence[:, :prefix_length] = prefix
                sequence[:, prefix_length:prefix_length + 4] = branch
                feature[f'{condition}_sequence'] = sequence
            write_npz(feature_path, **feature)
            record = {'protocol_sha256': digest, 'raw_sha256': meta['arrays_sha256'],
                      'features_sha256': sha256(feature_path), 'image_frames': prefix_length + 128,
                      'model_inputs': ['sequence', 'lengths', 'language', 'budget'],
                      'outcome_fields_in_features': [], 'natural_privileged_fields': [],
                      'prefix_length': prefix_length, 'dimensions': {'G': 484, 'N': 532}}
            write_json(feature_meta, record)
        image_count += record['image_frames']
        # Outcome labels and identifiers are a separate artifact, never forwarded.
        rows = [source['rows'][i] for i in IDS]
        with np.load(raw_path, allow_pickle=False) as raw_for_strata:
            target_plate_contact = raw_for_strata['branch_state'][:, -1, 106] > 0
        label_rows.append({'root': root, 'split': 'fit' if key in split['fit_roots'] else 'development',
                           'eligible': source['reference']['first_success_step'] is None,
                           'y': [bool(x['success']) for x in rows],
                           'success_by_h8': [x['first_success_step'] is not None and x['first_success_step'] <= root['prefix_steps'] + 8 for x in rows],
                           'h8_on': [x['h8']['on'] for x in rows], 'h8_m2': [x['h8']['m2'] for x in rows],
                           'h8_grasp': [bool(x['h8']['grasp']) for x in rows],
                           'h8_target_plate_contact': target_plate_contact.tolist()})
        inputs_manifest.append({'root_id': key, **record})
        print(json.dumps({'feature_root_complete': key, 'completed': len(label_rows), 'total': 40}), flush=True)
    write_json(DIRECTORY / 'labels.json', {'protocol_sha256': digest, 'rows': label_rows,
                                         'feature_usage': 'labels/strata only; excluded from model forward'})
    manifest_path = DIRECTORY / 'feature_manifest.json'
    manifest = {'protocol_sha256': digest, 'complete': True, 'roots': inputs_manifest,
                'image_frames': image_count, 'wall_seconds_current_run': time.monotonic() - started,
                'encoder': model_spec}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous[k] != manifest[k] for k in ('protocol_sha256', 'complete', 'roots', 'image_frames', 'encoder')):
            raise ValueError('completed feature manifest identity changed')
        # Keep its digest stable so completed/partial fits can safely resume.
    else:
        write_json(manifest_path, manifest)


def train(protocol, digest):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    labels_path = DIRECTORY / 'labels.json'
    labels = json.loads(labels_path.read_text())
    if labels['protocol_sha256'] != digest:
        raise ValueError('labels protocol mismatch')
    manifest_path = DIRECTORY / 'feature_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest['complete'] or manifest['protocol_sha256'] != digest:
        raise ValueError('complete matching cache required')
    rows = labels['rows']
    by_id = {r['root_id']: r for r in manifest['roots']}
    cached = []
    for row in rows:
        path = DIRECTORY / 'features' / f"{row['root']['root_id']}.npz"
        if sha256(path) != by_id[row['root']['root_id']]['features_sha256']:
            raise ValueError('training cache hash mismatch')
        with np.load(path, allow_pickle=False) as a:
            expected = {'G_sequence', 'N_sequence', 'lengths', 'language', 'budget'}
            if set(a.files) != expected:
                raise ValueError('feature whitelist violation')
            cached.append({k: a[k] for k in a.files})
    fit_rows = np.asarray([r['eligible'] and r['split'] == 'fit' for r in rows])
    y = np.asarray([r['y'] for r in rows], np.float32)
    lengths = np.stack([r['lengths'] for r in cached])
    language = np.stack([r['language'] for r in cached])
    budget = np.stack([r['budget'] for r in cached])
    settings = protocol['training']
    job_manifest = []
    for condition in ('G', 'N'):
        sequence = np.stack([r[f'{condition}_sequence'] for r in cached])
        mean, std = fit_normalization(sequence, lengths, fit_rows)
        sequence = normalize(sequence, lengths, mean, std)
        write_npz(DIRECTORY / f'{condition}_normalization.npz', mean=mean, std=std)
        batch = {'sequence': sequence.reshape(-1, MAX_FRAMES, sequence.shape[-1]),
                 'lengths': lengths.reshape(-1), 'language': language.reshape(-1, 512), 'budget': budget.reshape(-1, 1)}
        check_inputs(batch)
        inputs = {k: torch.from_numpy(v).cuda() for k, v in batch.items()}
        training_inputs = {k: torch.from_numpy(np.asarray(v).reshape((40, 32) + v.shape[1:])[fit_rows].reshape((-1,) + v.shape[1:])).cuda()
                           for k, v in batch.items()}
        targets = torch.from_numpy(y[fit_rows]).cuda()
        for seed in settings['seeds']:
            stem = f'{condition}_seed{seed}'
            metrics_path = DIRECTORY / 'readouts' / f'{stem}.json'
            prediction_path = metrics_path.with_suffix('.npz')
            checkpoint = metrics_path.with_suffix('.pt')
            if metrics_path.exists():
                m = json.loads(metrics_path.read_text())
                if (m['protocol_sha256'] != digest or m['labels_sha256'] != sha256(labels_path)
                        or m['feature_manifest_sha256'] != sha256(manifest_path)
                        or m['prediction_sha256'] != sha256(prediction_path) or m['checkpoint_sha256'] != sha256(checkpoint)):
                    raise ValueError('trained readout resume hash mismatch')
                job_manifest.append(m)
                continue
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = make_readout(sequence.shape[-1]).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
            started = time.monotonic()
            trace = []
            model.train()
            for epoch in range(settings['epochs']):
                optimizer.zero_grad(set_to_none=True)
                logits = model(**training_inputs).reshape(-1, 32)
                loss, bce, ranking = loss_function(logits, targets, settings['ranking_weight'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                if epoch % 25 == 0 or epoch == settings['epochs'] - 1:
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite training loss')
                    trace.append({'epoch': epoch + 1, 'loss': float(loss), 'bce': float(bce), 'ranking': float(ranking)})
                    print(json.dumps({'fit': stem, **trace[-1]}), flush=True)
            model.eval()
            with torch.inference_mode():
                scores = model(**inputs).sigmoid().reshape(40, 32).cpu().numpy()
                for _ in range(5):
                    model(**{k: v[:32] for k, v in inputs.items()})
                torch.cuda.synchronize()
                inference_start = time.monotonic()
                for _ in range(30):
                    model(**{k: v[:32] for k, v in inputs.items()})
                torch.cuda.synchronize()
                latency = (time.monotonic() - inference_start) * 1000 / 30
            write_npz(prediction_path, scores=scores)
            tmp = checkpoint.with_suffix('.pt.part')
            torch.save({'state_dict': model.state_dict(), 'dimension': sequence.shape[-1],
                        'condition': condition, 'seed': seed, 'protocol_sha256': digest}, tmp)
            sync(tmp)
            tmp.replace(checkpoint)
            sync(checkpoint.parent)
            m = {'condition': condition, 'seed': seed, 'protocol_sha256': digest,
                 'labels_sha256': sha256(labels_path), 'feature_manifest_sha256': sha256(manifest_path),
                 'prediction_sha256': sha256(prediction_path), 'checkpoint_sha256': sha256(checkpoint),
                 'parameters': sum(p.numel() for p in model.parameters()), 'epochs': settings['epochs'],
                 'checkpoint_selection': 'last fixed epoch; no development checkpoint selection',
                 'fit_roots': int(fit_rows.sum()), 'fit_branches': int(fit_rows.sum() * 32),
                 'trace': trace, 'wall_seconds': time.monotonic() - started,
                 'latency_ms_per_N32_root_readout_only': latency}
            write_json(metrics_path, m)
            job_manifest.append(m)
            print(json.dumps({'fit_complete': stem, 'completed': len(job_manifest), 'total': 6}), flush=True)
            del model, optimizer
    write_json(DIRECTORY / 'training_manifest.json', {'protocol_sha256': digest, 'complete': True,
               'fits': job_manifest, 'fit_count': len(job_manifest), 'encoder_finetuned': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('cache', 'train'), required=True)
    args = parser.parse_args()
    protocol, digest = validate_protocol()
    (cache if args.phase == 'cache' else train)(protocol, digest)


if __name__ == '__main__':
    main()
