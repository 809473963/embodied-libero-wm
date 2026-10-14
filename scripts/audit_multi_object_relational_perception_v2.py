#!/usr/bin/env python3
"""Inventory all available object relations and graph consistency; no fitting."""
import sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.probes.candidate_utility_oracle import atomic_json,sha256
from src.probes.relational_graph_audit_v2 import to_world,closure,reconcile,macro_error
OUT=ROOT/'results/multi_object_relational_perception_v2'

def main():
 cfg=json.loads((ROOT/'configs/multi_object_relational_perception_v2_protocol.json').read_text())
 for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
 OUT.mkdir(exist_ok=True)
 if (OUT/'runtime.json').exists():raise RuntimeError('Existing audit retained')
 rt=dict(status='RUNNING',started_epoch=time.time(),phase='multi_object_audit',training=False,confirmation_opened=False);atomic_json(OUT/'runtime.json',rt)
 try:
  with np.load(ROOT/'results/relational_natural_state/dataset.npz',allow_pickle=False) as f:n={k:f[k] for k in f.files}
  with np.load(ROOT/'results/relational_localization_bridge/localized_ensemble_prediction.npz',allow_pickle=False) as f:p=f['state']
  with np.load(ROOT/'results/relational_geometry_diagnosis/auxiliary_object_predictions.npz',allow_pickle=False) as f:aux={k:f[k] for k in f.files}
  meta=json.loads((ROOT/'results/relational_autorun/models/samples.json').read_text());anchors={}
  for i,m in enumerate(meta):
   if m['split']=='new_development':anchors.setdefault((m['root'],m['anchor']),i)
  ix=np.array(list(anchors.values()));frames=n['history'][ix,-1];np.testing.assert_array_equal(frames,aux['frame_indices']);roots=np.array([meta[i]['root'] for i in ix])
  records=json.loads((ROOT/'results/relational_autorun/branches/manifest.json').read_text())['records'];recordmap={(r['root'],r['anchor'],r['variant']):r for r in records}
  positions=[];grips=[];rotations=[];state_rows=[];sources={}
  for i in ix:
   m=meta[i];r=recordmap[(m['root'],m['anchor'],m['variant'])];path=ROOT/r['path'];assert sha256(path)==r['sha256'];sources[r['path']]=r['sha256'];t=m['anchor']
   with np.load(path,allow_pickle=False) as f:
    positions.append(f['pose'][t,:,:3]);grips.append(f['grip_pos'][t]);rotations.append(f['grip_rot'][t]);state_rows.append(dict(root=m['root'],anchor=t,entity_names=r['geometry']['entities'],object_contact_pairs=np.argwhere(np.triu(f['contacts'][t],1)).tolist(),gripper_contacts=np.flatnonzero(f['gripper_contact'][t]).tolist(),on=f['on'][t].tolist()))
  positions=np.array(positions);grips=np.array(grips);rotations=np.array(rotations);np.testing.assert_allclose(positions[:,:3],aux['true_world'],atol=1e-12)
  truth=to_world(n['target'][frames,:12],rotations);world=to_world(p[frames,:12],rotations);fixed=reconcile(world)
  np.testing.assert_allclose(closure(truth),0,atol=1e-7);np.testing.assert_allclose(closure(fixed),0,atol=1e-10)
  rawpairs={'bowl1_plate':world[:,0],'bowl1_gripper':world[:,1],'bowl2_plate':world[:,2],'bowl2_gripper':world[:,3],
            'bowl1_bowl2_via_plate':world[:,0]-world[:,2],'bowl1_bowl2_via_gripper':world[:,1]-world[:,3],
            'gripper_plate_via_bowl1':world[:,0]-world[:,1],'gripper_plate_via_bowl2':world[:,2]-world[:,3]}
  targets={'bowl1_plate':positions[:,0]-positions[:,2],'bowl1_gripper':positions[:,0]-grips,'bowl2_plate':positions[:,1]-positions[:,2],'bowl2_gripper':positions[:,1]-grips,
           'bowl1_bowl2_via_plate':positions[:,0]-positions[:,1],'bowl1_bowl2_via_gripper':positions[:,0]-positions[:,1],
           'gripper_plate_via_bowl1':grips-positions[:,2],'gripper_plate_via_bowl2':grips-positions[:,2]}
  pairs={k:macro_error(v,targets[k],roots) for k,v in rawpairs.items()}
  fixedpairs={k:macro_error(fixed[:,i],targets[k],roots) for i,k in enumerate(['bowl1_plate','bowl1_gripper','bowl2_plate','bowl2_gripper'])}
  a=aux['predicted_world'];shared={}
  for i,j,label in [(0,2,'bowl1_plate'),(1,2,'bowl2_plate'),(0,1,'bowl1_bowl2')]:shared[label]=macro_error(a[:,i]-a[:,j],positions[:,i]-positions[:,j],roots)
  objects={name:macro_error(a[:,i],positions[:,i],roots) for i,name in enumerate(['bowl1','bowl2','plate'])}
  residual=macro_error(closure(world),np.zeros((len(world),3)),roots)
  worst=np.max(np.stack([np.linalg.norm(world[:,i]-truth[:,i],axis=1)*1000 for i in range(4)],1),1)
  # Descriptive distribution only, no post-hoc pass threshold.
  distribution=dict(scene_macro_worst_relation_mm=float(np.mean([worst[roots==r].mean() for r in sorted(set(roots))])),anchor_worst_relation_percentiles_mm={str(q):float(np.percentile(worst,q)) for q in [50,90,100]})
  bodyeef=macro_error(grips,n['obs'][frames,:3],roots)
  result=dict(anchors=len(ix),scenes=len(set(roots)),direct_relations=pairs,oracle_rotation_consistency_projection=fixedpairs,closure=residual,shared_auxiliary_relations=shared,auxiliary_objects=objects,
              worst_relation_distribution=distribution,eef_to_grip_site_offset=bodyeef,source_sha256=sources,current_contact_inventory=state_rows,
              caveats=['world conversion and consistency projection use true gripper rotation for evaluation only','shared auxiliary positions are not main relation outputs','three auxiliary object positions do not cover eight recorded entities','contacts, object orientation, velocities and identity grounding are not validated as current perceptual outputs'],confirmation_opened=False,training=False)
  atomic_json(OUT/'evaluation.json',result)
  np.savez_compressed(OUT/'audit_arrays.npz',direct_world=world,true_world_relations=truth,reconciled_world=fixed,auxiliary_positions=a,true_positions=positions,grip_positions=grips,grip_rotations=rotations,roots=roots,frames=frames)
  lines=['# 多物体关系感知：现有模型的完整范围审计','', '日期：2026-10-09。无训练、无仿真、确认集封存。9个已暴露开发场景、22个不同起点。','',
  '**评估范围扩展为两碗、盘、夹爪的全部六类无向位置关系，并明确完整场景状态尚未覆盖。** 推断得到的同一关系保留两条路径，不作为两份独立证据。', '',
  '## 已有直接关系输出与可推导关系','', '| 关系/推导路径 | scene宏平均误差 mm |','|---|---:|']
  for k,v in pairs.items():lines.append(f"| {k} | {v['scene_macro_mm']:.3f} |")
  lines+=['', '夹爪局部向量用真实grip旋转矩阵转为world，仅用于评估；没有假装自然模型已经估计姿态。gripper_plate的方向为夹爪减盘。碗间与夹爪—盘关系由已有输出相减得到，不是单独训练的预测。', '',
          '## 共享对象位置：一致性不等于准确','', '| 对象/关系 | 辅助位置头误差 mm |','|---|---:|']
  for k,v in objects.items():lines.append(f"| 对象 {k} | {v['scene_macro_mm']:.3f} |")
  for k,v in shared.items():lines.append(f"| 关系 {k} | {v['scene_macro_mm']:.3f} |")
  lines+=['','三个对象共享位置计算的三角关系按构造闭合，但位置误差仍可能很大；不能给“闭合为0”当成学会物理的加分。这些位置来自辅助uv/depth头，不是主关系输出的隐藏中间层。没有将真实夹爪位置混入共享对象预测来冒充四对象自然感知。', '',
          '## 直接关系是否描述同一个场景','',
          f"两种路径得到的碗1—碗2向量差（闭合残差）为 {residual['scene_macro_mm']:.3f} mm，scene宏平均。这也是分别经两碗推导夹爪—盘位置的不一致量，不应重复算两个独立缺陷。",'',
          '| 关系 | 原预测 mm | 约束投影后 mm |','|---|---:|---:|']
  for k,v in fixedpairs.items():lines.append(f"| {k} | {pairs[k]['scene_macro_mm']:.3f} | {v['scene_macro_mm']:.3f} |")
  lines+=['', '约束投影是在真实旋转辅助下将4个向量等权投影到闭合约束；它按数学构造不能增大到一致真值的总平方误差，不是学习实验或可部署修复，也不保证每个关系误差都降低。', '',
          f"每起点四个直接关系中最坏误差的scene宏平均为 {distribution['scene_macro_worst_relation_mm']:.3f} mm。仅看目标—盘会隐藏其他关系的错误；分位数另存JSON，来自起点不是独立scene置信区间。", '',
          '## 当前状态覆盖缺口','',
          '| 状态/对象 | 当前能力边界 |','|---|---|',
          '| 两碗、盘位置 | 有辅助位置头；准确性如上，不等于主关系输出 |',
          '| 夹爪位置/姿态 | 合法proprio有EEF位置及关节量；EEF不是自动等于grip site；本轮世界系转换仍用真值姿态 |',
          '| 其他物体/支撑与障碍 | 原始数据记录cookies、ramekin、table、cabinet、stove，但现有感知头不预测这些节点 |',
          '| 物体姿态/速度 | 有真值记录，无对应自然感知评估；夹爪速度是不同量 |',
          '| 接触、抓取、支撑 | 原始记录有接触和官方On；当前感知输出没有完整接触/支撑图，未来事件头不能冒充当前感知 |',
          '| 持久身份与语言 | 固定物理槽位监督，没有独立语言指代或遮挡身份验证 |',
          '| 不确定性 | 未校准，3seed均值不等于可靠置信度 |', '',
          f"在本批起点，真实grip site与合法proprio EEF位置的平均距离为 {bodyeef['scene_macro_mm']:.3f} mm；这只是坐标契约检查，不是模型误差。", '',
          '## 下一阶段范围','',
          '以统一节点状态与多类边关系为目标，首先在旧数据内部建立覆盖全部相关节点的监督和观测契约；与当前独立关系头做匹配对照。共享状态的几何一致性必须与逐对象、逐关系、最坏关系误差同时报告。按当前接触/遮挡/身份条件分层，不能用总体平均遮住失效对象。', '',
          '不能把“所有对象都要看”变成所有静态背景和全部有向pair等权堆叠：应显式区分被操作对象、参照、夹爪、支撑与潜在障碍，保留每类覆盖。接触语义不由距离自动替代；准确当前状态也不是可靠事件预测的充分条件。', '',
          '本轮只评估已有输出；未虚构尚未训练的节点能力。下一模型设计需要在旧场景内验证，不能据本报告直接部署或打开confirmation。']
  report=ROOT/'docs/multi_object_relational_perception_audit.md';report.write_text('\n'.join(lines)+'\n')
  # Recompute scalar metrics from the durable array file, not a new simulation.
  with np.load(OUT/'audit_arrays.npz',allow_pickle=False) as f:
   for i,k in enumerate(['bowl1_plate','bowl1_gripper','bowl2_plate','bowl2_gripper']):
    np.testing.assert_allclose(f['true_world_relations'][:,i],targets[k],atol=1e-7)
    assert macro_error(f['direct_world'][:,i],targets[k],f['roots'])==pairs[k]
   assert macro_error(closure(f['direct_world']),np.zeros((len(world),3)),f['roots'])==residual
  atomic_json(OUT/'validation.json',dict(status='PASS',direct_groups_recomputed=4,closure_recomputed=True,true_relations_close=True,report_sha256=sha256(report),arrays_sha256=sha256(OUT/'audit_arrays.npz')))
  rt.update(status='COMPLETE',phase='verified_and_reported',finished_epoch=time.time(),report=str(report.relative_to(ROOT)));atomic_json(OUT/'runtime.json',rt)
 except Exception:
  rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'runtime.json',rt);raise
if __name__=='__main__':main()
