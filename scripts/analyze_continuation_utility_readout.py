#!/usr/bin/env python3
"""Decision metrics on exposed pilot roots; no independent confirmation claim."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.probes.candidate_utility_oracle import sha256, t1_select
from src.probes.continuation_utility_observability import (
    DIRECTORY, IDS, choose_threshold, decisions, select, validate_protocol, write_json,
)


def calibration(scores, y, mask=None):
    p, target = scores.reshape(-1), y.reshape(-1)
    if mask is not None:
        p, target = p[mask.reshape(-1)], target[mask.reshape(-1)]
    if not len(p):
        return {'branches': 0}
    bins = []
    for lower, upper in zip(np.arange(0, 1, .2), np.arange(.2, 1.2, .2)):
        inside = (p >= lower) & (p < upper if upper < 1 else p <= 1)
        bins.append({'lower': float(lower), 'upper': float(upper), 'count': int(inside.sum()),
                     'predicted': float(p[inside].mean()) if inside.any() else None,
                     'observed': float(target[inside].mean()) if inside.any() else None})
    ece = sum(b['count'] * abs(b['predicted'] - b['observed']) for b in bins if b['count']) / len(p)
    pair_auc = []
    for index, (pr, yr) in enumerate(zip(scores, y)):
        if mask is not None:
            pr, yr = pr[mask[index]], yr[mask[index]]
        pos, neg = pr[yr], pr[~yr]
        if len(pos) and len(neg):
            pair_auc.append(float(((pos[:, None] > neg[None, :]) + .5 * (pos[:, None] == neg[None, :])).mean()))
    return {'branches': len(p), 'accuracy': float(((p >= .5) == target).mean()),
            'brier': float(((p - target) ** 2).mean()), 'ece_5bin': float(ece), 'bins': bins,
            'mixed_root_count': len(pair_auc), 'within_root_auc_macro': float(np.mean(pair_auc)) if pair_auc else None,
            'uncertainty_note': 'branch counts describe calibration only, not independent samples'}


def main():
    protocol, digest = validate_protocol()
    labels = json.loads((DIRECTORY / 'labels.json').read_text())
    training = json.loads((DIRECTORY / 'training_manifest.json').read_text())
    if labels['protocol_sha256'] != digest or training['protocol_sha256'] != digest or not training['complete'] or training['fit_count'] != 6:
        raise ValueError('six completed matching readouts required')
    rows = labels['rows']
    y = np.asarray([r['y'] for r in rows], bool)
    eligible = np.asarray([r['eligible'] for r in rows])
    fit = eligible & np.asarray([r['split'] == 'fit' for r in rows])
    dev = eligible & np.asarray([r['split'] == 'development' for r in rows])
    tasks = np.asarray([r['root']['task_id'] for r in rows])
    prefixes = np.asarray([r['root']['prefix_steps'] for r in rows])
    past_success = np.asarray([r['success_by_h8'] for r in rows], bool)
    grasp = np.asarray([r['h8_grasp'][0] for r in rows], bool)
    contact = np.asarray([r['h8_target_plate_contact'][0] for r in rows], bool)
    # The stored arrays already have local N32 ordering.
    t1 = np.asarray([t1_select(np.asarray(r['h8_on']), np.asarray(r['h8_m2']), protocol['epsilon_goal']) for r in rows])
    a0 = np.zeros(40, int)
    binarized = np.full(40, IDS.index(49), int)
    baselines = {split: {name: decisions(y[mask], choice[mask]) for name, choice in
                        (('A0', a0), ('T1_N32', t1), ('c0_binarized', binarized), ('constant_scores', a0))}
                 for split, mask in (('fit', fit), ('development', dev), ('all_exposed', eligible))}
    # One fixed ridge baseline on branch-mean labels. Its scores are identical
    # across candidates within a root, so its legal selector always retains A0.
    language, budget = [], []
    for row in rows:
        with np.load(DIRECTORY / 'features' / f"{row['root']['root_id']}.npz", allow_pickle=False) as a:
            language.append(a['language'][0])
            budget.append(a['budget'][0])
    x = np.concatenate((np.asarray(language), np.asarray(budget), np.ones((40, 1))), axis=1)
    coefficient = np.linalg.solve(x[fit].T @ x[fit] + np.eye(x.shape[1]), x[fit].T @ y[fit].mean(1))
    language_scores = np.repeat(np.clip(x @ coefficient, 0, 1)[:, None], 32, axis=1)
    for name, mask in (('fit', fit), ('development', dev)):
        baselines[name]['language_budget_only'] = {'decision': decisions(y[mask], a0[mask]),
                                                  'calibration': calibration(language_scores[mask], y[mask])}
    conditions = {}
    selected_by_condition = {}
    for condition in ('G', 'N'):
        predictions, seed_metrics = [], []
        for seed in protocol['training']['seeds']:
            path = DIRECTORY / 'readouts' / f'{condition}_seed{seed}.json'
            metadata = json.loads(path.read_text())
            prediction_path = path.with_suffix('.npz')
            if metadata['protocol_sha256'] != digest or sha256(prediction_path) != metadata['prediction_sha256']:
                raise ValueError('prediction hash/protocol mismatch')
            with np.load(prediction_path, allow_pickle=False) as a:
                p = a['scores']
            if p.shape != (40, 32) or not np.isfinite(p).all():
                raise ValueError('invalid predictions')
            predictions.append(p)
            seed_metrics.append({'seed': seed, 'parameters': metadata['parameters'],
                                 'readout_latency_ms_N32': metadata['latency_ms_per_N32_root_readout_only'],
                                 'development_threshold0': decisions(y[dev], select(p[dev])),
                                 'development_calibration': calibration(p[dev], y[dev]),
                                 'fit_threshold0': decisions(y[fit], select(p[fit]))})
        scores = np.mean(predictions, axis=0)
        votes = np.sum(np.asarray(predictions) - np.asarray(predictions)[:, :, :1] > 0, axis=0)
        threshold, threshold_grid = choose_threshold(scores[dev], y[dev], protocol['selection']['threshold_grid'], votes[dev])
        selected = select(scores, threshold, votes)
        raw_selected = select(scores, 0., votes)
        selected_by_condition[condition] = selected
        result = {'seeds': seed_metrics, 'threshold': threshold, 'threshold_grid_development': threshold_grid,
                  'selection_bias': 'threshold chosen and reported on the same exposed development roots; screening only',
                  'development': decisions(y[dev], selected[dev]),
                  'development_threshold0': decisions(y[dev], raw_selected[dev]),
                  'fit': decisions(y[fit], selected[fit]), 'all_exposed': decisions(y[eligible], selected[eligible]),
                  'calibration_development': calibration(scores[dev], y[dev]),
                  'calibration_development_not_successful_by_h8': calibration(scores[dev], y[dev], ~past_success[dev]),
                  'calibration_fit': calibration(scores[fit], y[fit]),
                  'per_task_development': {str(t): decisions(y[dev & (tasks == t)], selected[dev & (tasks == t)]) for t in range(10)},
                  'per_prefix_development': {str(t): decisions(y[dev & (prefixes == t)], selected[dev & (prefixes == t)]) for t in (0, 32, 64, 96)},
                  'per_A0_h8_grasp_development': {str(flag): decisions(y[dev & (grasp == flag)], selected[dev & (grasp == flag)]) for flag in (False, True)},
                  'per_A0_h8_target_plate_contact_development': {str(flag): decisions(y[dev & (contact == flag)], selected[dev & (contact == flag)]) for flag in (False, True)}}
        rng = np.random.default_rng(protocol['selection']['shuffle_seed'])
        shuffles = []
        for _ in range(100):
            shuffled = np.stack([p[rng.permutation(32)] for p in scores[dev]])
            shuffles.append(decisions(y[dev], select(shuffled, threshold)))
        result['within_root_score_shuffle'] = {
            'repetitions': 100, 'votes': 'not applied to shuffled scores',
            'net_gain_roots_quantiles': np.quantile([m['net_success_gain_roots'] for m in shuffles], [.05, .5, .95]).tolist(),
            'captured_rescue_roots_mean': float(np.mean([m['captured_rescue_roots'] for m in shuffles])),
            'harm_roots_mean': float(np.mean([m['harm_roots'] for m in shuffles])),
            'meaning': 'randomization diagnostic, not a confidence interval'}
        m, reference = result['development'], baselines['development']['T1_N32']
        result['screen_positive'] = bool(m['net_success_gain_roots'] > 0 and m['harm_roots'] == 0 and m['mean_regret'] < reference['mean_regret'])
        conditions[condition] = result
    g, n = (conditions[c]['screen_positive'] for c in ('G', 'N'))
    if g and n:
        verdict, next_step = 'G_AND_N_DEVELOPMENT_SCREEN_POSITIVE', '注册新scene，随后设计2B direct consequence；本轮不自动执行'
    elif g:
        verdict, next_step = 'G_POSITIVE_N_OBSERVATION_GAP', '先选一个身份/几何/历史的受控输入诊断，不继续扫描encoder'
    elif n:
        verdict, next_step = 'N_POSITIVE_G_REFERENCE_INSUFFICIENT', '保留自然模型筛查结果；检查G表示与读出，不能称privileged输入为严格上限'
    else:
        verdict, next_step = 'NO_USEFUL_DEVELOPMENT_READOUT', '先检查scene覆盖、读出、状态与分支目标；暂不启动2B或WM'
    per_root = []
    for index, row in enumerate(rows):
        per_root.append({'root': row['root'], 'split': row['split'], 'eligible': row['eligible'],
                         'a0_y': bool(y[index, 0]), 'oracle_y': bool(y[index].any()),
                         'T1': {'candidate_id': IDS[t1[index]], 'y': bool(y[index, t1[index]])},
                         **{c: {'candidate_id': IDS[selected_by_condition[c][index]],
                                'y': bool(y[index, selected_by_condition[c][index]])} for c in ('G', 'N')}})
    metrics = {'protocol_sha256': digest, 'verdict': verdict, 'next_step': next_step,
               'accounting': {'registered_roots': 40, 'eligible_fit': int(fit.sum()), 'eligible_development': int(dev.sum()),
                              'ceiling_excluded': int((~eligible).sum()), 'candidates': 32,
                              'eligible_branches': int(eligible.sum() * 32), 'success_by_h8_branches': int(past_success[eligible].sum())},
               'baselines': baselines, 'conditions': conditions, 'per_root': per_root,
               'independent_confirmation': False, 'deployment_ranking_authorized': False,
               'world_model_executed': False, 'learned_controller_executed': False}
    write_json(DIRECTORY / 'metrics.json', metrics)
    write_json(DIRECTORY / 'decision.json', {k: metrics[k] for k in ('protocol_sha256', 'verdict', 'next_step',
               'independent_confirmation', 'deployment_ranking_authorized', 'world_model_executed', 'learned_controller_executed')})
    report(metrics, training)
    print(json.dumps({'analysis_complete': True, 'verdict': verdict}), flush=True)


def report(m, training):
    a = m['accounting']
    dev = m['baselines']['development']
    lines = ['# 阶段2A：真实H8后果的continuation utility读出审计', '',
             f"结果：**`{m['verdict']}`**。这是已暴露pilot的工程筛查，不是独立确认或可部署候选排名。", '',
             f"协议SHA-256：`{m['protocol_sha256']}`。40个root、固定N32已补录；有效fit={a['eligible_fit']}、development={a['eligible_development']}；排除{a['ceiling_excluded']}个此前成功root。", '',
             '所有输入止于真实H8；标签复用阶段1的固定C0至总300步续行结果。本轮没有重跑续行。每条分支已通过root RGB/proprio/sim-state、H8 RGB与旧几何/接触诊断的一致性检查。', '',
             'G使用8个物体/fixture的世界坐标pose、velocity、接触，机器人状态及qpos/qvel历史；N使用initial/prefix/H8图像的固定CLIP特征、12D proprio与动作历史。两组均有语言、时间和剩余预算。G包含persistent referent角色，N不接收物体身份或privileged字段。', '',
             '| 开发集方法 | 成功root | 救回 | harm | override | 净gain | regret |',
             '|---|---:|---:|---:|---:|---:|---:|']
    methods = [('A0', dev['A0']), ('T1 N32', dev['T1_N32']), ('c0_binarized', dev['c0_binarized'])]
    for c in ('G', 'N'):
        methods.extend([(c + ' ensemble threshold0', m['conditions'][c]['development_threshold0']),
                        (c + ' ensemble dev-selected', m['conditions'][c]['development'])])
    for name, result in methods:
        lines.append(f"| {name} | {result['selected_successes']}/{result['roots']} | {result['captured_rescue_roots']} | {result['harm_roots']} | {result['override_roots']} | {result['net_success_gain_roots']:+d} | {result['mean_regret']:.4f} |")
    lines += ['', '## 模型与seed差异', '',
              '每组固定3个seed；固定最后epoch，不在开发集选checkpoint。fit root上的BCE加同root正负候选ranking loss。阈值在同一开发集选择后报告，存在开发集选择偏差；通过筛查只支持继续收集新scene。', '',
              '| 输入 | seed | 参数 | 开发救回 | harm | 净gain | Brier | N32读出ms |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in ('G', 'N'):
        for seed in m['conditions'][c]['seeds']:
            d = seed['development_threshold0']
            lines.append(f"| {c} | {seed['seed']} | {seed['parameters']} | {d['captured_rescue_roots']} | {d['harm_roots']} | {d['net_success_gain_roots']:+d} | {seed['development_calibration']['brier']:.4f} | {seed['readout_latency_ms_N32']:.3f} |")
    lines += ['', '延迟只计缓存特征之后的单个小读出器；三seed ensemble还需三次读出，未包含图像编码、仿真或部署通信。', '',
              f"有效root的{a['eligible_branches']}条N32分支中，{a['success_by_h8_branches']}条已在H8前命中success。完整指标另报告排除这些分支后的classification/calibration；首次成功时间只用于分层，不作为特征。", '',
              '## 开发root逐项结果', '',
              '| root | A0 | Oracle | T1(candidate/Y) | G(candidate/Y) | N(candidate/Y) |',
              '|---|---:|---:|---|---|---|']
    for r in m['per_root']:
        if r['split'] == 'development' and r['eligible']:
            lines.append(f"| {r['root']['root_id']} | {int(r['a0_y'])} | {int(r['oracle_y'])} | {r['T1']['candidate_id']}/{int(r['T1']['y'])} | {r['G']['candidate_id']}/{int(r['G']['y'])} | {r['N']['candidate_id']}/{int(r['N']['y'])} |")
    lines += ['', '## 解释与下一动作', '', m['next_step'] + '。', '',
              'privileged输入加有限小读出器不是数学信息上限。失败只约束当前状态表示、模型、数据和标签；不能推出视觉没有信息。自然输入较好也不能证明不需要身份或接触。', '',
              '全部40个root在项目历史中已经暴露。init31–33只用于fit，init34只用于开发；所有候选与历史跟随root分区，归一化仅在有效fit root估计。branch数量不作为独立样本数；本轮开发每任务至多一个scene，不报告伪精确的分支bootstrap区间。', '',
              '真实H8是执行后输入，不能直接用这些分数在root选择动作。未启动2B、独立新scene、WM、reranker或learned controller；旧阶段授权文件未改动。', '',
              '未打开HDF5。官方init容器只将31–34行构造为数值数组；未启动demo36–39评估，未访问demo40–49数组或组元数据。', '',
              '[完整指标](../results/continuation_utility_observability/metrics.json)、[运行状态](../results/continuation_utility_observability/runtime.json)、[preflight](../results/continuation_utility_observability/preflight.json)、[训练记录](../results/continuation_utility_observability/training_manifest.json)、[协议](../configs/continuation_utility_observability_protocol.json)。', '']
    path = ROOT / 'docs/continuation_utility_observability_audit.md'
    tmp = path.with_suffix('.md.part')
    tmp.write_text('\n'.join(lines))
    tmp.replace(path)


if __name__ == '__main__':
    main()
