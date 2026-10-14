#!/usr/bin/env python3
"""Inventory old-scene node supervision and test robot FK from legal proprio only."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.libero'))
import numpy as np
from src.probes.candidate_utility_oracle import atomic_json,sha256
OUT=ROOT/'results/unified_object_state_audit_v2'

def angle_degrees(a,b):
    relative=np.swapaxes(a,-1,-2)@b
    return np.degrees(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1)))

def main():
 cfg=json.loads((ROOT/'configs/unified_object_state_audit_v2_protocol.json').read_text())
 for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
 OUT.mkdir(exist_ok=True)
 if (OUT/'runtime.json').exists():raise RuntimeError('Existing audit retained')
 rt=dict(status='RUNNING',phase='supervision_inventory',frames=0,started_epoch=time.time(),training=False,confirmation_opened=False);atomic_json(OUT/'runtime.json',rt);env=None
 try:
  metadata=json.loads((ROOT/'results/relational_localization_cv/metadata.json').read_text());rows=metadata['rows'];assert len(rows)==4843
  with np.load(ROOT/'results/relational_localization_cv/dataset.npz',allow_pickle=False) as f:obs=f['obs'];fold=f['fold']
  manifest=json.loads((ROOT/'results/relational_event_h24/manifest.json').read_text())['records'];records={r['dense']['arrays']:r for r in manifest}
  names=manifest[0]['dense']['geometry']['entities'];assert len(names)==8
  poses=[];contacts=[];grip=[];rot=[];pads=[];external=[];ons=[];sources={};oldpath=None;a=None
  for i,row in enumerate(rows):
   path=row['source'];t=row['step'];rec=records[path];assert rec['dense']['geometry']['entities']==names
   if path!=oldpath:
    assert sha256(ROOT/path)==rec['dense']['sha256'];sources[path]=rec['dense']['sha256']
    with np.load(ROOT/path,allow_pickle=False) as f:a={k:f[k] for k in ['pose','contacts','grip_pos','grip_rot','pad_contact','non_gripper_contact','on','natural_step','proprio']}
    oldpath=path
   j=int(np.flatnonzero(a['natural_step']==t)[0]);np.testing.assert_array_equal(obs[i,:12],a['proprio'][j].astype(obs.dtype))
   poses.append(a['pose'][t]);contacts.append(a['contacts'][t]);grip.append(a['grip_pos'][t]);rot.append(a['grip_rot'][t]);pads.append(a['pad_contact'][t]);external.append(a['non_gripper_contact'][t]);ons.append(a['on'][t])
  poses=np.array(poses);contacts=np.array(contacts);grip=np.array(grip);rot=np.array(rot);pads=np.array(pads);external=np.array(external);ons=np.array(ons);roots=np.array([r['root'] for r in rows])
  assert np.isfinite(poses).all();np.testing.assert_allclose(np.linalg.norm(poses[:,:,3:7],axis=2),1,atol=1e-6);np.testing.assert_array_equal(contacts,contacts.transpose(0,2,1));np.testing.assert_allclose(rot.transpose(0,2,1)@rot,np.broadcast_to(np.eye(3),rot.shape),atol=1e-6)
  eef_offset=np.linalg.norm(obs[:,:3]-grip,axis=1)*1000
  edges=[(i,j) for i in range(9) for j in range(i+1,9)]
  xyz=np.concatenate([poses[:,:,:3],grip[:,None]],axis=1);relations=np.stack([xyz[:,i]-xyz[:,j] for i,j in edges],1)
  # Robot FK inputs contain only the 7 measured joint positions and 2 gripper qpos, plus fixed model calibration.
  from scripts.evaluate_c0_all import make_env
  contracts=json.loads((ROOT/'configs/task_grounding_contract.json').read_text())['tasks'];fkpos=np.zeros_like(grip);fkrot=np.zeros_like(rot);base_records=[];active=None
  rt['phase']='legal_proprio_fk';atomic_json(OUT/'runtime.json',rt)
  for i,row in enumerate(rows):
   root=row['root']
   if root!=active:
    if env:env.close()
    task=int(root[4:6]);env=make_env(ROOT/contracts[task]['bddl_path']);env.seed(30000+100*task+int(root[-2:])-30);env.env.reset();active=root
    robot=env.env.robots[0];sid=env.sim.model.site_name2id(robot.gripper.important_sites['grip_site']);base=env.sim.model.body_name2id(robot.robot_model.root_body)
    base_records.append(dict(root=root,position=env.sim.data.body_xpos[base].tolist(),rotation=env.sim.data.body_xmat[base].reshape(3,3).tolist(),joint_qpos_indices=list(robot._ref_joint_pos_indexes),gripper_qpos_indices=list(robot._ref_gripper_joint_pos_indexes)))
    assert len(robot._ref_joint_pos_indexes)==7 and len(robot._ref_gripper_joint_pos_indexes)==2
   env.sim.data.qpos[robot._ref_joint_pos_indexes]=obs[i,5:12];env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes]=obs[i,3:5];env.sim.forward()
   fkpos[i]=env.sim.data.site_xpos[sid];fkrot[i]=env.sim.data.site_xmat[sid].reshape(3,3)
   if i%200==0:rt['frames']=i;atomic_json(OUT/'runtime.json',rt)
  env.close();env=None
  for b in base_records[1:]:
   np.testing.assert_allclose(b['position'],base_records[0]['position'],atol=1e-12,rtol=0);np.testing.assert_allclose(b['rotation'],base_records[0]['rotation'],atol=1e-12,rtol=0)
  poserr=np.linalg.norm(fkpos-grip,axis=1)*1000;roterr=angle_degrees(fkrot,rot)
  stats=lambda x:dict(mean=float(np.mean(x)),median=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(np.max(x)))
  fk=dict(position_error_mm=stats(poserr),rotation_error_deg=stats(roterr),recorded_eef_vs_grip_mm=stats(eef_offset),constant_robot_base=True,inputs='proprio joint/gripper positions and fixed robot model only; no object pose, qvel or privileged grip rotation',within_predeclared_budget=bool(poserr.max()<=1 and roterr.max()<=.5),per_scene={r:dict(position_mm=stats(poserr[roots==r]),rotation_deg=stats(roterr[roots==r])) for r in sorted(set(roots))})
  inventory={}
  for k,name in enumerate(names):
   displacements=[]
   for r in sorted(set(roots)):
    ii=np.flatnonzero(roots==r);assert rows[int(ii[0])]['step']==0
    displacements.append(float(np.linalg.norm(poses[ii,k,:3]-poses[ii[0],k,:3],axis=1).max()*1000))
   inventory[name]=dict(frames=len(rows),scenes=len(set(roots)),max_displacement_from_initial_mm=max(displacements),scenes_displacement_over_1mm=sum(x>1 for x in displacements),has_pose_quaternion_velocity=True,gripper_pad_contact_frames=int(pads[:,:,k].any(1).sum()),non_gripper_contact_frames=int(external[:,k].sum()),visible_mask_available_in_existing_dataset=k<3)
  counts=np.sum(contacts,axis=0).tolist()
  np.savez_compressed(OUT/'node_supervision.npz',positions=xyz,object_pose_velocity=poses,contacts=contacts,pad_contact=pads,non_gripper_contact=external,on=ons,grip_rotation=rot,fk_grip_position=fkpos,fk_grip_rotation=fkrot,relations=relations,edges=np.array(edges),roots=roots,fold=fold)
  result=dict(nodes=names+['grip_frame'],inventory=inventory,edge_count=len(edges),contact_counts=counts,fk=fk,robot_calibration=base_records,source_sha256=sources,frame_alignment='same row order as relational_localization_cv/dataset.npz',support_semantics='contacts are observed contacts, not directed support or load-bearing; On only defined for two bowls and plate',training=False,confirmation_opened=False)
  atomic_json(OUT/'evaluation.json',result)
  lines=['# 统一对象状态监督与合法机器人坐标审计','', '日期：2026-10-09。仅旧13场景4,843条去重观测；无模型训练、控制动作或确认访问。','',
   '## 节点监督覆盖','', '| 节点 | 有记录的位置/姿态/速度 | 超过1mm位移的场景数 | 最大初始位移 mm | 现成可见mask |','|---|---|---:|---:|---|']
  for name,v in inventory.items():lines.append(f"| {name} | 是 | {v['scenes_displacement_over_1mm']}/13 | {v['max_displacement_from_initial_mm']:.3f} | {'有' if v['visible_mask_available_in_existing_dataset'] else '尚未生成'} |")
  lines+=['','位移统计覆盖同场景多个候选后果，是相对初始位置的观察范围，不是单条连续轨迹的运动距离。静态/近静态节点应单独报告，不能稀释移动对象误差。', '',
   '已经生成8个环境实体加grip frame共9节点的统一world位置及36个无向几何边标签。对象body原点不是可见质心；contact是对称接触记录，不等价于支撑、有载荷或稳定抓取。官方On仅对两碗与盘已有定义。', '',
   '## 机器人合法输入能提供什么','',
   f"现有合法proprio的EEF位置与记录grip site距离：{fk['recorded_eef_vs_grip_mm']} mm。", '',
   f"只使用合法7关节位置、2夹爪关节位置与固定机器人模型，FK位置误差：{fk['position_error_mm']} mm；旋转误差：{fk['rotation_error_deg']} degrees。", '',
   f"跨13场景robot base校准一致。预声明1 mm/0.5°最大误差预算：{fk['within_predeclared_budget']}。这是有限机器人frame工程检查，不是学习能力或全分布保证。", '',
   'proprio关节值与记录site派生位姿可能存在最后一次积分的时序差异；本轮没有使用真值qvel回退FK，也不声称精确复原。位置优先直接使用合法EEF测量；旋转若有残差必须记录，不能偷偷换成真值旋转。', '',
   '## 第一项匹配训练的范围','',
   '可开展world位置层面的旧场景三折对照：共享模型预测8对象位置并用合法grip位置构成节点；关系基线预测同一36条几何边。两者使用相同自然观测、相同scene split和预算。全部36边按移动对象/参照/机器人/背景分组评价，不将静态背景易预测成绩混为主要提升。', '',
   '这一步不需要姿态旋转作为推理输入；当前未通过任何自然姿态、速度、接触、身份或不确定性学习评估。其余5个实体mask尚未生成，不应伪称完整视觉定位监督已齐备。物体自由度和接触语义需用于后续解释，不自动推出自然可观测性。', '',
   '当前交付是标签与坐标契约，不是新共享模型已训练完成。原9开发场景及40确认seed仍封存于本轮之外。']
  report=ROOT/'docs/unified_object_state_audit.md';report.write_text('\n'.join(lines)+'\n')
  with np.load(OUT/'node_supervision.npz',allow_pickle=False) as f:
   np.testing.assert_allclose(f['relations'],np.stack([f['positions'][:,i]-f['positions'][:,j] for i,j in f['edges']],1),atol=0,rtol=0)
   for i in range(9):
    for j in range(i+1,9):
     for k in range(j+1,9):np.testing.assert_allclose(f['relations'][:,edges.index((i,j))]+f['relations'][:,edges.index((j,k))],f['relations'][:,edges.index((i,k))],atol=1e-12,rtol=0)
  atomic_json(OUT/'validation.json',dict(status='PASS',frames=len(rows),nodes=9,edges=36,triangle_closures_checked=84,dataset_sha256=sha256(OUT/'node_supervision.npz'),report_sha256=sha256(report)))
  rt.update(status='COMPLETE',frames=len(rows),phase='verified_and_reported',finished_epoch=time.time(),report=str(report.relative_to(ROOT)));atomic_json(OUT/'runtime.json',rt)
 except Exception:
  rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'runtime.json',rt);raise
 finally:
  if env:env.close()
if __name__=='__main__':main()
