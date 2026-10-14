#!/usr/bin/env python3
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.candidate_utility_oracle import atomic_json,sha256
from scripts.train_relational_localization_cv import normalization
P=ROOT/'results/relational_localization_cv'
def main():
 cfg=json.loads((ROOT/'configs/relational_localization_cv_protocol.json').read_text())
 for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
 with np.load(P/'dataset.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
 collection=json.loads((P/'collection_runtime.json').read_text());assert collection['dataset_sha256']==sha256(P/'dataset.npz')
 meta=json.loads((P/'metadata.json').read_text());roots=np.array([m['root'] for m in meta['rows']]);groups=sorted(set(roots));w=np.array([1/(roots==r).sum() for r in roots],np.float32)
 for fold in range(3):
  fit=d['fold']!=fold
  with np.load(P/f'fold{fold}_normalization.npz',allow_pickle=False) as f:
   for k,v in normalization(d,fit,w).items():np.testing.assert_array_equal(v,f[k])
 out={};hashes={}
 for family in ['direct','localized']:
  predictions=[]
  for seed in [11,23,47]:
   name=f'{family}_seed{seed}';p=np.full_like(d['target'],np.nan);seen=np.zeros(len(p),int)
   for fold in range(3):
    with np.load(P/f'fold{fold}_{name}_prediction.npz',allow_pickle=False) as f:
     ii=f['indices'];assert (d['fold'][ii]==fold).all();p[ii]=f['state'];seen[ii]+=1
   assert (seen==1).all()
   with np.load(P/f'oof_{name}.npz',allow_pickle=False) as f:np.testing.assert_array_equal(p,f['state'])
   predictions.append(p)
  predictions.append(np.mean(predictions,0))
  for suffix,pred in zip(['seed11','seed23','seed47','ensemble'],predictions):
   name=f'{family}_{suffix}';path=P/f'oof_{name}.npz'
   with np.load(path,allow_pickle=False) as f:np.testing.assert_array_equal(pred,f['state'])
   err=np.linalg.norm((pred[:,:12]-d['target'][:,:12]).reshape(-1,4,3),axis=2)*1000
   per={r:err[roots==r].mean(0).tolist() for r in groups};out[name]=dict(scene_macro_mm=np.mean(list(per.values()),0).tolist(),per_scene_mm=per);hashes[path.name]=sha256(path)
 passes=[]
 for seed in [11,23,47]:
  a=np.array(out[f'localized_seed{seed}']['scene_macro_mm']);b=np.array(out[f'direct_seed{seed}']['scene_macro_mm']);passes.append(dict(seed=seed,pass_primary=bool(np.all(a[:2]<=.9*b[:2]) and np.all(a[2:]<=1.1*b[2:]))))
 verdict='LOCALIZATION_AUXILIARY_SIGNAL_SUPPORTED' if sum(c['pass_primary'] for c in passes)>=2 else 'LOCALIZATION_AUXILIARY_GAIN_NOT_ESTABLISHED'
 lines=['# 定位辅助监督：旧场景三折对照','', '日期：2026-10-08。**'+verdict+'**。','',
 f"13个旧场景、{len(roots)}条去重观测，全部RGB逐位重建一致后才生成定位监督。按场景排序轮转分为3折；每帧仅使用不含其场景的模型预测。每组3种子、每折40轮，共18次拟合。",'',
 '| 方法 | 目标—盘 mm | 目标—夹爪 mm | 另一碗—盘 mm | 另一碗—夹爪 mm |','|---|---:|---:|---:|---:|']
 for family in ['direct','localized']:
  lines.append('| '+family+' | '+' | '.join(f'{v:.3f}' for v in out[family+'_ensemble']['scene_macro_mm'])+' |')
 lines+=['','两组同网络、同种子、同数据顺序、同训练预算；区别仅为localized额外使用投影/深度与可见mask监督。它是定位辅助的关系回归，不是已经实现显式3D重建管线。','',
 '固定继续线：至少2/3种子两项目标关系误差均下降10%，另一碗两项均不退化超过10%。这是投入规则，不是显著性证明，也不保证事件或控制收益。', '', '`'+json.dumps(passes)+'`。','',
 '## 解释限制','', '没有打开9个新开发场景或40个确认种子；旧13场景已用于研究设计，三折是内部开发诊断，不能称为独立新场景确认。每场景所有分支、时刻和初始图像都跟随同一折。归一化及mask类别权重仅从该折训练场景计算。', '',
 'mask是可见区域的8×8面积比例；点监督是物理body原点投影与相机深度，二者不混为可见质心。物理身份槽位固定，未验证语言指代。没有使用GT像素标签作为推理输入。','',
 '当前结果只检验辅助定位监督能否改善关系恢复；没有重训动力学或将旧训练场景上的动力学结果当成泛化评估。若通过，也应先冻结唯一方案再做独立确认；未通过则检查可见定位是否学到、深度和几何误差分布，不能继续随机增大辅助loss。', '',
 '验证：18组分折预测覆盖与场景隔离、8组OOF合并/ensemble、3组fit-only归一化复算，原始source哈希及数据集哈希一致。','',
 '机器指标：[evaluation.json](../results/relational_localization_cv/evaluation.json)。状态：[pipeline.json](../results/relational_localization_cv/pipeline.json)。']
 report=ROOT/'docs/relational_localization_cv_audit.md';report.write_text('\n'.join(lines)+'\n')
 atomic_json(P/'evaluation.json',dict(verdict=verdict,models=out,seed_checks=passes))
 atomic_json(P/'validation.json',dict(status='PASS',oof_models_verified=8,fold_predictions_verified=18,normalizations_verified=3,prediction_sha256=hashes,report_sha256=sha256(report)))
 print(verdict,flush=True)
if __name__=='__main__':main()
