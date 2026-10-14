#!/usr/bin/env python3
"""Readable full-object tables; no fitting or model selection."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.probes.candidate_utility_oracle import sha256,atomic_json

def main():
 out=ROOT/'results/unified_object_state_cv'
 assert json.loads((out/'validation.json').read_text())['status']=='PASS'
 r=json.loads((out/'evaluation.json').read_text());m=r['metrics']
 nodes=json.loads((ROOT/'results/unified_object_state_audit_v3/evaluation.json').read_text())['nodes']
 families=['mean_position','independent_ensemble','shared_ensemble'];a=m[families[2]];b=m[families[1]]
 lines=['# 多对象状态学习：首轮匹配结果解读','','日期：2026-10-09。旧13场景、4,843帧、3折与3个训练seed。不是独立新场景确认。','',
 f"共享模型关键关系均值 {a['key_mean_mm']:.2f} mm，独立关系模型 {b['key_mean_mm']:.2f} mm；每帧最坏关键关系分别为 {a['worst_key_mm']:.2f}/{b['worst_key_mm']:.2f} mm。",
 f"预定逐seed决策：{r['seed_gate']}；优先共享结构的工程线{'通过' if r['shared_prioritized'] else '未通过'}。",'',
 '## 当前瓶颈判断','', '本轮共享结构没有通过优先采用的工程线，3个seed均未通过。其关键边均值和最差关键边都略差于独立关系模型，尽管三角闭合按构造接近零。当前证据支持继续定位感知误差来源，不支持立即接事件模型或换掉原方案。','', '误差集中于第二只碗：共享位置161.22mm、独立恢复146.94mm；第一只碗分别39.96/34.87mm。第二只碗参与的多条关系因此同时受损。plate/cookies/ramekin的共享误差19.56/13.37/10.70mm，也未优于无图像均值14.03/13.13/10.42mm，不能认为其他对象都已经学好。','', '两种模型的关键关系误差均低于固定位置基线138.70mm，说明自然输入模型有可利用信号；但本轮没有proprio-only对照，不能把全部改善归为视觉。小数据覆盖、身份交换、可见性与读出能力仍未隔离。','', '下一项应先用现有OOF预测做逐场景错误可视核查：检查第二只碗是否在视野中、是否交换了两碗身份、场景位置是否超出fit覆盖，再决定定位/身份监督或数据补充。不得在同一结果上不断改loss寻找过线。旧CV与之前9场景bridge的误差来自不同划分和任务范围，不能直接当作模型退步幅度。','', '## 这次实际学了什么','','估计两个碗、盘、cookies、ramekin、桌子、柜子、炉子的world body原点；夹爪使用合法机器人测量。共享模型先给每个对象一个位置，再计算关系；基线独立预测同样36条关系。两者用完全相同关系损失，不额外给共享模型位置loss。无图像均值基线仅用训练场景平均位置。','',
 '共享模型的关系必然闭合；这不是精度证明。独立模型的“节点位置”只由其object—grip边恢复，其他关系可能与该位置不一致。','',
 '## 全部节点位置误差（scene-macro，mm）','','| 节点 | 无图像均值 | 独立关系恢复 | 共享状态 |','|---|---:|---:|---:|']
 for i,n in enumerate(nodes):lines.append('| '+n+' | '+' | '.join(f"{m[f]['node_mm'][i]:.2f}" for f in families)+' |')
 lines+=['','## 关系分组','','| 分组 | 无图像均值 | 独立关系 | 共享状态 |','|---|---:|---:|---:|']
 for g in ['object','robot','context']:lines.append('| '+g+' | '+' | '.join(f"{m[f]['groups_mm'][g]:.2f}" for f in families)+' |')
 lines+=['','object为含前5对象的环境边；robot为8条对象/参照—夹爪边；context为桌/柜/炉彼此3条边。训练各组等权；关键15边仅包括前5对象彼此及各自到夹爪。','',
 '## 全36条关系','','| 关系 | 无图像均值 | 独立关系 | 共享状态 |','|---|---:|---:|---:|']
 for k,(i,j) in enumerate(r['edges']):lines.append('| '+nodes[i]+' — '+nodes[j]+' | '+' | '.join(f"{m[f]['edge_mm'][k]:.2f}" for f in families)+' |')
 lines+=['','## 尚不能推出的结论','','本轮只做位置结构对照；没有姿态、速度、抓取/接触、语言身份或事件学习。当前像素与初始像素不是完整历史跟踪。不能由位置误差降低直接推导稳定抓取/抬起/放置预测可靠，也不能推出其他任务泛化。','',
 '若共享结构未过工程线，保留其一致性优势但不宣布替换原模型；先读逐对象误差、均值基线与场景分布，定位是物体未看准还是拟合/输入不足。不要立刻重训事件模型或开启controller。若通过，仍需遮挡/接触/移动分层和GT→预测状态的事件误差传递对照。','',
 '监督审计已确认8环境节点位置/姿态/速度记录可得，36条位置差标签闭合。合法FK最大位置误差0.558mm、旋转0.047°，在预定预算内；其余5实体可见mask尚未生成。该结果只在本批数据成立。','',
 '证据：[完整结果](../results/unified_object_state_cv/evaluation.json)、[独立复算](../results/unified_object_state_cv/validation.json)、[冻结协议](unified_object_state_cv_protocol.md)、[监督审计](unified_object_state_audit.md)。']
 p=ROOT/'docs/unified_object_state_cv_interpretation.md';p.write_text('\n'.join(lines)+'\n')
 atomic_json(out/'report_manifest.json',dict(source_sha256=sha256(out/'evaluation.json'),report_sha256=sha256(p),script_sha256=sha256(Path(__file__))))
 print(p)
if __name__=='__main__':main()
