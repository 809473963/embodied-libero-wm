#!/usr/bin/env python3
"""Post-hoc descriptive review; no fitting, simulation, or confirmation access."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.probes.relational_dynamics_pilot import score, normalization
from scripts.evaluate_relational_autorun import paired

OUT = ROOT / 'results/relational_autorun'

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def macro(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['root']].append(row['error_mm'])
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}

def main():
    validation = read(OUT / 'validation.json')
    assert digest(ROOT / 'docs/relational_autorun_audit.md') == validation['report_sha256']
    assert digest(OUT / 'evaluation.json') == validation['evaluation_sha256']
    frozen = read(ROOT / 'configs/relational_autorun_protocol.json')['frozen_files']
    for path, expected in frozen.items():
        assert digest(ROOT / path) == expected, path
    e = read(OUT / 'evaluation.json')
    c = read(OUT / 'coverage.json')
    meta = read(OUT / 'models/samples.json')
    with np.load(OUT / 'models/dataset.npz', allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    with np.load(OUT / 'models/normalization.npz', allow_pickle=False) as f:
        for k, v in normalization(d['x'], d['actions'], d['delta'], d['fit'], d['weights']).items():
            np.testing.assert_array_equal(v, f[k])
    predictions = {}
    checks = 0
    for name, m in e['models'].items():
        if name == 'unchanged':
            delta = np.zeros_like(d['delta'])
            prob = np.where(d['current'] >= 0, d['current'], np.nan)
        else:
            with np.load(OUT / f'models/{name}_predictions.npz', allow_pickle=False) as f:
                delta, prob = f['delta'], f['probability']
        predictions[name] = prob
        for split, mask in [('old_fit', d['fit']), ('new_development', ~d['fit'])]:
            assert score(delta, prob, d, meta, mask) == m[split]
            checks += 1
        assert paired(delta, d, meta, ~d['fit']) == m['new_pairs']
        checks += 1
    a = e['models']['action_ensemble']
    b = e['models']['unchanged']
    ap = macro(a['new_pairs']['rows']); bp = macro(b['new_pairs']['rows'])
    roots = sorted(ap)
    loso = {r: float(np.mean([bp[k]-ap[k] for k in roots if k != r])) for r in roots}
    variants = {}
    for v in range(1, 6):
        variants[v] = {name: float(np.mean(list(macro([r for r in e['models'][name]['new_pairs']['rows'] if r['variant'] == v]).values()))) for name in ['unchanged', 'action_ensemble']}
    # Physical release subset: the object is gripped at the root, not merely waiting for a quiet-window label.
    cov = {(r['root'], r['anchor'], r['variant']): r for r in c['rows']}
    held = np.array([not d['fit'][i] and cov[(m['root'],m['anchor'],m['variant'])]['current_grip'] for i,m in enumerate(meta)])
    known = held & (d['binary'][:, 8] >= 0)
    y = d['binary'][known, 8] == 1
    pred = predictions['action_ensemble'][known, 8] >= .5
    held_counts = dict(n=int(known.sum()), positives=int(y.sum()), tp=int((y & pred).sum()), fp=int((~y & pred).sum()), fn=int((y & ~pred).sum()))
    summary = dict(date='2026-10-08', type='post_hoc_descriptive_review', metric_groups_recomputed=checks,
        frozen_files_verified=len(frozen), source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [OUT/'evaluation.json',OUT/'coverage.json',OUT/'models/dataset.npz',OUT/'models/samples.json']},
        per_scene_pair_error_mm={r:dict(unchanged=bp[r],action=ap[r]) for r in roots},
        leave_one_scene_out_pair_improvement_mm=loso, by_variant_pair_error_mm=variants,
        held_at_root_release_prediction=held_counts, confirmation_executed=False)
    (OUT/'scientific_review.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines = ['# 新布局 H24 关系动力学：科学复核', '', '日期：2026-10-08。', '',
        '**结论：动作条件关系预测已获得跨开发布局的支持，但尚不足以据此执行可靠的放置决策。** 本次是保存结果的事后描述性复核，不改变原 gate，不拟合新模型。', '',
        '完整链自动完成20个场景的C0轨迹、132条正式分支、6次训练和指标复算；另有3条分支技术重复。总控制步数21,792，约48分钟。训练只用旧13个场景的228条H24分支，新数据只作开发评估。', '',
        '## 动作后果确实有可学习信号', '',
        '| 方法 | 目标—盘 mm | 目标—夹爪 mm | 候选差分 mm |', '|---|---:|---:|---:|']
    for name in ['unchanged','no_action_ensemble','action_ensemble']:
        m=e['models'][name]; s=m['new_development']
        lines.append(f"| {name} | {s['root_macro_target_plate_mm']:.3f} | {s['root_macro_target_gripper_mm']:.3f} | {m['new_pairs']['root_macro_error_mm']:.3f} |")
    lines += ['', '三个固定训练种子均通过原继续门槛。以上为场景宏平均；候选差分衡量不同动作造成的位置变化差别，不是成功率或动作价值。', '',
        f"逐一移除一个新场景后，候选差分相对不变基线的平均改善仍为 {min(loso.values()):.3f}–{max(loso.values()):.3f} mm。这排除了全部平均优势仅来自单个场景的解释，但不证明所有场景都改善。", '',
        '| 场景 | 不变差分误差 mm | 动作差分误差 mm |', '|---|---:|---:|']
    lines += [f'| {r} | {bp[r]:.3f} | {ap[r]:.3f} |' for r in roots]
    lines += ['', '## 仍然妨碍决策的错误', '',
        f"真实近静止（目标—盘变化≤1 mm）的30条分支中，不变误差 {b['new_development']['unchanged_target_plate_mm']:.3f} mm，动作模型误差 {a['new_development']['unchanged_target_plate_mm']:.3f} mm。模型会凭空预测移动；不能仅凭平均精度授权 override。这些数是分支平均，不是独立场景平均。", '',
        '| 目标事件的发生预测 | TP | FP | FN | Precision | Recall |', '|---|---:|---:|---:|---:|---:|']
    for key in ['on','attachment','detached','lift','quiet_on','release']:
        v=a['new_development']['binary']['target/'+key]['onset']
        lines.append(f"| {key} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']:.1%} | {v['recall']:.1%} |")
    lines += ['', 'On发生预测18次，仅6次正确；较好的位移预测尚未转化为可靠终态关系判断。release标签25次发生包含已经释放后等待稳定窗口成立，不能都算作成功执行释放。', '',
        f"只看起点仍有夹爪接触且释放标签已知的分支：n={held_counts['n']}，真实正例={held_counts['positives']}，TP={held_counts['tp']}、FP={held_counts['fp']}、FN={held_counts['fn']}。实际持有→释放仍只有6条、3个场景；同起点不同方案不是独立场景。", '',
        '## 泛化边界', '',
        '20个新开发场景只有9个产生符合预先定义的关系起点，共22个起点；其余11个被如实保留为覆盖缺失。当前结论针对这9个可达场景，不是全20个场景或全部任务。布局与审计过的旧40个布局非重复，但同任务、同物体，且未证明与C0全部历史训练场景不同。', '',
        '模型使用522维丰富特权历史，包括真实几何、接触与机器人状态。自然RGB/proprio虽已记录，尚未证明模型能从它们恢复足够的关系状态。也未证明新的对象类别、语言指代或任务关系可以泛化。', '',
        '## 下一步选择', '',
        '优先做固定数据上的最小关系状态消融：将丰富特权状态压缩为两碗相对盘/夹爪的几何、运动、接触和因果历史，与完整状态、无动作和不变基线比较；同时单独报告近静止误动、On误报和真正释放子集。先确定需要从视觉恢复哪些变量，再投入自然感知训练。', '',
        '该消融必须另立有限协议，旧H24拟合、新开发场景评估；这些新场景已是开发集，不能重新称为盲验证。不调整旧模型或旧标签以追认通过，不自动打开40个confirmation种子，不接学习式控制器。若压缩丢失动作差分，先定位必要状态；若压缩保留信号，再做自然观测替代。', '',
        f'验证：本次独立复算{checks}组保存预测指标、fit-only归一化及{len(frozen)}个冻结文件哈希；原报告与evaluation哈希一致。未重跑仿真或全仓库测试。', '',
        '复现：`env -u PYTHONPATH .venv/bin/python scripts/review_relational_autorun.py`。机器可读证据：[scientific_review.json](../results/relational_autorun/scientific_review.json)。']
    (ROOT/'docs/relational_autorun_scientific_review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
