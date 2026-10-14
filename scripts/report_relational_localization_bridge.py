#!/usr/bin/env python3
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.train_relational_localization_cv import normalization
from src.probes.relational_dynamics_pilot import score
from scripts.evaluate_relational_autorun import paired
from src.probes.candidate_utility_oracle import atomic_json,sha256
P=ROOT/'results/relational_localization_bridge'
def main():
 cfg=json.loads((ROOT/'configs/relational_localization_bridge_protocol.json').read_text())
 for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
 with np.load(ROOT/'results/relational_localization_cv/dataset.npz',allow_pickle=False) as f:old={k:f[k] for k in f.files}
 roots=np.array([m['root'] for m in json.loads((ROOT/'results/relational_localization_cv/metadata.json').read_text())['rows']]);w=np.array([1/(roots==r).sum() for r in roots],np.float32)
 with np.load(P/'normalization.npz',allow_pickle=False) as f:
  for k,v in normalization(old,np.ones(len(roots),bool),w).items():np.testing.assert_array_equal(f[k],v)
 with np.load(ROOT/'results/relational_natural_state/dataset.npz',allow_pickle=False) as f:n={k:f[k] for k in f.files}
 nr=np.array(json.loads((ROOT/'results/relational_natural_state/inventory.json').read_text())['roots']);dev=~n['fit']
 with np.load(ROOT/'results/relational_autorun/models/dataset.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
 meta=json.loads((ROOT/'results/relational_autorun/models/samples.json').read_text());mask=~d['fit'];ev=json.loads((P/'evaluation.json').read_text())['models'];checks=0;hashes={}
 for name,m in ev.items():
  with np.load(P/f'{name}_prediction.npz',allow_pickle=False) as f:pred=f['state']
  err=np.linalg.norm((pred[:,:12]-n['target'][:,:12]).reshape(-1,4,3),axis=2)*1000;scene={r:err[(nr==r)&dev].mean(0).tolist() for r in sorted(set(nr[dev]))};assert scene==m['perception_per_scene_mm'];checks+=1
  with np.load(P/f'{name}_bridge.npz',allow_pickle=False) as f:
   np.testing.assert_array_equal(f['endpoint_delta'],f['delta']+pred[n['history'][:,-1],:12]-d['x'][:,-1,104:116])
   assert score(f['delta'],f['probability'],d,meta,mask)==m['delta'];assert score(f['endpoint_delta'],f['probability'],d,meta,mask)==m['absolute_endpoint'];assert paired(f['delta'],d,meta,mask)==m['pairs'];checks+=3
  for suffix in ['prediction','bridge']:hashes[f'{name}_{suffix}.npz']=sha256(P/f'{name}_{suffix}.npz')
 for family in ['direct','localized']:
  ps=[]
  for seed in [11,23,47]:
   with np.load(P/f'{family}_seed{seed}_prediction.npz',allow_pickle=False) as f:ps.append(f['state'])
  with np.load(P/f'{family}_ensemble_prediction.npz',allow_pickle=False) as f:np.testing.assert_array_equal(np.mean(ps,0),f['state'])
 def values(m):return [m['absolute_endpoint']['root_macro_target_plate_mm'],m['absolute_endpoint']['root_macro_target_gripper_mm'],m['pairs']['root_macro_error_mm']]
 seedchecks=[]
 for seed in [11,23,47]:
  a=ev[f'localized_seed{seed}'];b=ev[f'direct_seed{seed}'];aa=np.array(values(a));bb=np.array(values(b));per=np.array(a['perception_scene_macro_mm'][:2]);base=np.array(b['perception_scene_macro_mm'][:2])
  seedchecks.append(dict(seed=seed,perception_pass=bool(np.all(per<=.9*base)),bridge_pass=bool(np.all(aa[:2]<=.9*bb[:2]) and aa[2]<=1.1*bb[2])))
 verdict='LOCALIZATION_DEVELOPMENT_BRIDGE_SUPPORTED' if sum(c['perception_pass'] and c['bridge_pass'] for c in seedchecks)>=2 else 'LOCALIZATION_DEVELOPMENT_BRIDGE_NOT_ESTABLISHED'
 lines=['# 定位辅助：全旧场景拟合与固定动力学开发评估','', '日期：2026-10-09。**'+verdict+'**。','', '固定上一轮架构、监督权重和40轮训练，旧13场景4,843帧拟合；原9开发场景2,812帧仅评估，动力学后果评估为132分支。两方法×3种子=6次训练，无epoch/seed选择。确认集未打开。','',
 '| 方法 | 当前目标—盘 mm | 当前目标—夹爪 mm | 未来目标—盘 mm | 未来目标—夹爪 mm | 候选差分 mm |','|---|---:|---:|---:|---:|---:|']
 for name in ['direct_ensemble','localized_ensemble']:
  m=ev[name];lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in m['perception_scene_macro_mm'][:2]+values(m))+' |')
 lines+=['', '当前误差来自去重观测；未来误差来自132条分支且包含起点感知偏差。二者样本分布不同。固定三模型geometry_history动力学均不重训，所有预测关系与夹爪速度均由自然输入恢复。','',
 '继续线：至少2/3种子两项当前关系与两项绝对未来关系误差均比direct下降10%，候选差分不退化超过10%。这是开发资源线，不是控制效用证明。', '', '`'+json.dumps(seedchecks)+'`。','',
 '| 方法 | 静止分支绝对未来误差 mm | On发生 TP/FP/FN | release标签发生 TP/FP/FN |','|---|---:|---:|---:|']
 for name in ['direct_ensemble','localized_ensemble']:
  m=ev[name]['absolute_endpoint'];counts=lambda k:'/'.join(str(m['binary']['target/'+k]['onset'][s]) for s in ['tp','fp','fn'])
  lines.append(f"| {name} | {m['unchanged_target_plate_mm']:.3f} | {counts('on')} | {counts('release')} |")
 lines+=['', 'release标签不全是真正持有后释放；稀少的真实事件覆盖不因模型训练增加。没有运行controller、candidate ranking或任务成功实验。', '',
 '此9场景开发集已多次用于诊断，不能称为独立确认。与三折结果比较时须考虑拟合场景数和评估群体不同。本轮不得用于事后选择新辅助权重或最佳checkpoint。', '',
 ('本轮支持将定位辅助作为后续候选，但仍需审查绝对精度与事件误报，不能直接部署。' if verdict.endswith('_SUPPORTED') else '三折辅助收益未满足本轮匹配开发桥的联合要求。保留全部结果，不追加训练追求过线；下一判断应定位哪些关系或场景退化，而非继续随机调loss。'), '',
 f'验证：{checks}组指标从保存预测重算，两个ensemble精确复算，归一化仅旧fit、原输入数组与定位数据对齐验证通过。','', '状态：[runtime.json](../results/relational_localization_bridge/runtime.json)。机器指标：[evaluation.json](../results/relational_localization_bridge/evaluation.json)。']
 report=ROOT/'docs/relational_localization_bridge_audit.md';report.write_text('\n'.join(lines)+'\n')
 atomic_json(P/'decision.json',dict(verdict=verdict,seed_checks=seedchecks,confirmation_opened=False))
 atomic_json(P/'validation.json',dict(status='PASS',metric_groups_recomputed=checks,prediction_sha256=hashes,report_sha256=sha256(report)))
 print(verdict,flush=True)
if __name__=='__main__':main()
