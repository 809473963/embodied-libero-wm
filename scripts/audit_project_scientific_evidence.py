"""Read-only reanalysis of existing calibration evidence; no simulator or training.

This script opens only explicit calibration NPZs and compact result/config JSONs.
It does not open HDF5, encoder features, checkpoints, or development sidecars.
The language-pair contingency check is descriptive, not a selected new model.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.probes.semantic_value_distillation import make_cross_demo_controls


def main() -> None:
    inputs = {}

    def record(relative):
        path = ROOT / relative
        inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read_json(relative):
        return json.loads(record(relative).read_text())

    record('src/probes/semantic_value_distillation.py')
    with np.load(record('results/t1_semantic_teacher/calibration_sidecar.npz'), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    assert set(arrays['demo_id'].tolist()) == {31, 32, 33, 34}
    data = SimpleNamespace(**arrays, margin=arrays['selected_margin_values'])
    epsilon = read_json('configs/t1_teacher_frozen_definition.json')['epsilon_goal']
    output = {
        'date': '2026-10-04',
        'scope': 'post-hoc audit of archived calibration outputs, not a new confirmatory experiment',
        'access': {'hdf5_opened': False, 'simulator_executed': False,
                   'model_trained': False, 'development_sidecar_opened': False,
                   'raw_demo_ids': [31, 32, 33, 34]},
        'sample_counts': {}, 'cross_controls': {}, 'metric_recomputation': {},
    }
    for demos, name in [([31, 32, 33], 'fit'), ([34], 'selection'), ([31, 32, 33, 34], 'calibration')]:
        mask = np.isin(data.demo_id, demos)
        output['sample_counts'][name] = {
            'frames': int(mask.sum()), 'episodes': len(np.unique(data.episode_id[mask])),
            'tasks': len(np.unique(data.task_id[mask])),
            'note': 'episode count is not a verified count of unique independent scene layouts',
        }
    output['metric_tolerances'] = {
        'epsilon_M2': epsilon, 'epsilon_equivalent_single_active_slack_m': epsilon * 0.03,
        'cross_clear_margin_M2': 0.05, 'cross_clear_equivalent_single_active_slack_m': 0.05 * 0.03,
    }
    controls = make_cross_demo_controls(data, epsilon, 16, 0.05)
    for name, pairs in controls.items():
        keep = pairs.demo_id == 34
        a, b, lang = pairs.a[keep], pairs.b[keep], pairs.language_id[keep]
        ref_a = data.ground_referent[data.episode_id[a], lang]
        ref_b = data.ground_referent[data.episode_id[b], lang]
        ga, gb = data.goal[a, ref_a], data.goal[b, ref_b]
        delta = data.margin[b, ref_b] - data.margin[a, ref_a]
        correct = np.where(ga != gb, np.where(gb, 1, -1),
                           np.where(delta > epsilon, 1, np.where(delta < -epsilon, -1, 0)))
        unique = set((min(int(x), int(y)), max(int(x), int(y)), int(l)) for x, y, l in zip(a, b, lang))
        output['cross_controls'][name] = {
            'rows': len(a), 'unique_unordered_pairs': len(unique),
            'task_row_counts': {str(t): int((pairs.task_id[keep] == t).sum()) for t in range(10)},
            'endpoint_demo_ids': sorted(set(data.demo_id[a].tolist() + data.demo_id[b].tolist())),
            'two_unseen_endpoints_rows': int(((data.demo_id[a] == 34) & (data.demo_id[b] == 34)).sum()),
            'separate_endpoint_referent_label_mismatches': int((correct != pairs.label[keep]).sum()),
            'absolute_margin_difference_quantiles': np.quantile(abs(delta), [0, .25, .5, .75, 1]).tolist(),
            'same_goal_rows': int((ga == gb).sum()),
        }
    # Reconstruct the archived referent probe's eight repeated frames per language pair.
    rows = []
    for episode in np.unique(data.episode_id):
        ids = np.flatnonzero(data.episode_id == episode)
        languages = np.flatnonzero(data.ground_status[episode] == 1)
        frames = ids[np.unique(np.linspace(0, len(ids)-1, 8, dtype=np.int64))]
        for frame in frames:
            for left, la in enumerate(languages):
                for lb in languages[left+1:]:
                    rows.append((int(data.demo_id[frame]), int(data.task_id[frame]), int(la), int(lb),
                                 int(data.ground_referent[episode, la] != data.ground_referent[episode, lb])))
    contingency = defaultdict(list)
    for demo, task, la, lb, label in rows:
        if demo in (31, 32, 33):
            contingency[(la, lb)].append(label)
    fallback = int(np.mean([r[-1] for r in rows if r[0] in (31,32,33)]) > .5)
    val = [r for r in rows if r[0] == 34]
    predictions = [int(np.mean(contingency[(la, lb)]) > .5) if (la, lb) in contingency else fallback
                   for _, _, la, lb, label in val]
    output['referent_language_pair_contingency'] = {
        'inputs': 'unordered instruction identities only; no pixels, task ID, or state input',
        'rule': 'demo31--33 majority label per language pair; tied vote -> false; unseen -> global majority',
        'selection_rows': len(val),
        'unique_selection_episode_language_pairs': len(set((r[1], r[2], r[3]) for r in val)),
        'selection_accuracy': float(np.mean(np.asarray(predictions) == np.asarray([r[-1] for r in val]))),
        'unseen_language_pair_rows': sum((r[2], r[3]) not in contingency for r in val),
        'interpretation': 'post-hoc negative control only; does not prove the visual probes ignore images',
    }
    local = data.pair_gap[data.pair_demo_id == 34] <= 8
    for stage, result_path, sidecar_path in [
        ('I', 'results/clip_internal_layer_calibration.json', 'results/clip_internal_layer/calibration_sidecar.npz'),
        ('E', 'results/alternate_encoder_calibration.json', 'results/alternate_frozen_encoder/calibration_sidecar.npz'),
    ]:
        result = read_json(result_path)
        families = result['layers'] if stage == 'I' else result['families']
        recomputed = {}
        with np.load(record(sidecar_path), allow_pickle=False) as s:
            non_tie = local & (s['ordinary_label'] != 0)
            direction = s['cf_alternate_label'].astype(float) - s['cf_original_label'].astype(float)
            changed = direction != 0
            for family, summary in families.items():
                prefix = family + '_P1' if stage == 'I' else family
                summary = summary['P1'] if stage == 'I' else summary
                acc = float(np.mean(np.sign(s[prefix+'_ordinary_probability'][non_tie]-.5) == s['ordinary_label'][non_tie]))
                agreement = float(np.mean(((s[prefix+'_cf_alternate_probability']-s[prefix+'_cf_original_probability'])*direction > 0)[changed]))
                assert np.isclose(acc, summary['ordinary_local_non_tie']['accuracy'])
                assert np.isclose(agreement, summary['language_counterfactual']['teacher_change_agreement'])
                probe_acc = float(np.mean((s[family+'_M2_cross_logits'] > 0) == (s['M2_cross_label'] == 1)))
                recomputed[family] = {'ordinary': acc, 'teacher_change': agreement, 'M2_cross': probe_acc}
        output['metric_recomputation'][stage] = recomputed
    candidate = read_json('results/expanded_candidate_contract_restart/diagnosis.json')['families']['C48_G15']
    output['candidate_archived_summary'] = {k: candidate[k] for k in [
        'sample_count', 'relative_mse_gain', 'relative_mse_gain_ci95', 'dimension_oracle_relative_gain',
        'winner_category_fraction', 'diversity', 'default_raw_action_mse', 'oracle_raw_action_mse']}
    for stage, relative in [('I', 'results/clip_internal_layer/access_ledger.json'), ('E', 'results/alternate_frozen_encoder/access_ledger.json')]:
        ledger = read_json(relative)
        output['access'][stage] = {k: ledger[k] for k in ['opened_demo_ids', 'forbidden_numeric_access', 'forbidden_metadata_access']}
    output['inputs_sha256'] = inputs
    output['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = ROOT / 'results/full_project_scientific_strategic_audit/evidence.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
