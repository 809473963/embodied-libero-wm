#!/usr/bin/env python3
"""Build an offline explanation from existing Stage1/2A artifacts only."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.probes.candidate_utility_oracle import sha256, t1_select


def main():
    directory = ROOT / 'results/continuation_utility_observability'
    read = lambda path: json.loads(path.read_text())
    metrics = read(directory / 'metrics.json')
    first = read(ROOT / 'results/candidate_utility_oracle/metrics.json')
    labels = read(directory / 'labels.json')['rows']
    protocol = read(ROOT / 'configs/continuation_utility_observability_protocol.json')
    ids = protocol['scope']['candidate_ids']
    seeds = protocol['training']['seeds']
    predictions = {}
    sources = [directory / 'metrics.json', directory / 'labels.json',
               ROOT / 'results/candidate_utility_oracle/metrics.json',
               ROOT / 'configs/continuation_utility_observability_protocol.json']
    for condition in ('G', 'N'):
        predictions[condition] = {}
        for seed in seeds:
            path = directory / 'readouts' / f'{condition}_seed{seed}.npz'
            with np.load(path, allow_pickle=False) as a:
                predictions[condition][seed] = a['scores']
            sources.append(path)

    def png(array):
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format='PNG', optimize=True)
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')

    roots = []
    for index, row in enumerate(labels):
        if not row['eligible'] or row['split'] != 'development':
            continue
        key = row['root']['root_id']
        source_path = ROOT / 'results/candidate_utility_oracle/roots' / f'{key}.json'
        source = read(source_path)
        path = ROOT / 'results/candidate_utility_oracle/arrays' / f'{key}.npz'
        assert sha256(path) == source['arrays_sha256']
        sources.extend((source_path, path))
        with np.load(path, allow_pickle=False) as a:
            images = [png(a['h8_rgb'][j]) for j in ids]
            initial = png(a['root_rgb'])
        roots.append({**row, 'instruction': protocol['instructions'][str(row['root']['task_id'])],
                      't1': t1_select(np.asarray(row['h8_on']), np.asarray(row['h8_m2']), protocol['epsilon_goal']),
                      'scores': {c: {str(s): predictions[c][s][index].tolist() for s in seeds} for c in ('G', 'N')},
                      'forms': [source['candidate_forms'][j] for j in ids],
                      'first_success': [source['rows'][j]['first_success_step'] for j in ids],
                      'root_image': initial, 'images': images})
    assert len(roots) == 9
    data = {'roots': roots, 'ids': ids, 'seeds': seeds, 'thresholds': protocol['selection']['threshold_grid'],
            'metrics': metrics, 'stage1': first['primary'], 'accounting': metrics['accounting'],
            'registry': [{'root': r['root'], 'split': r['split'], 'eligible': r['eligible']} for r in labels],
            'protocol_sha256': sha256(ROOT / 'configs/continuation_utility_observability_protocol.json'),
            'sources': [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)} for p in sources]}
    template = (ROOT / 'docs/assets/bottleneck_explainer.template.html').read_text()
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    assert template.count('__EMBEDDED_DATA__') == 1
    output = ROOT / 'docs/project_bottleneck_explainer.html'
    output.write_text(template.replace('__EMBEDDED_DATA__', payload))
    print(json.dumps({'html': str(output), 'bytes': output.stat().st_size, 'roots': len(roots),
                      'images': len(roots) * 33, 'new_training_or_simulation': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
