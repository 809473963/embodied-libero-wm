#!/usr/bin/env python3
"""Bounded geometry, event and auxiliary-object diagnostics. No training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_relational_natural_state as bridge
from scripts.train_relational_localization_cv import Model
from src.probes.relational_dynamics_pilot import score,prf
from scripts.evaluate_relational_autorun import paired
from src.probes.candidate_utility_oracle import atomic_json,sha256
OUT=ROOT/'results/relational_geometry_diagnosis'
SOURCE=ROOT/'results/relational_localization_bridge'

def macro(values,roots):
    return np.mean([values[roots==r].mean(0) for r in sorted(set(roots))],0).tolist()

def world_from_head(head,matrix):
    uv=head[...,:6].reshape(-1,3,2).astype(np.float64).copy();depth=head[...,6:].astype(np.float64)
    uv[...,1]=127-uv[...,1]
    h=np.concatenate([uv*depth[...,None],depth[...,None],np.ones(depth.shape+(1,))],-1)
    return (h@np.linalg.inv(matrix).T)[...,:3]

def main():
    cfg=json.loads((ROOT/'configs/relational_geometry_diagnosis_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    OUT.mkdir(exist_ok=True)
    if (OUT/'runtime.json').exists():raise RuntimeError('Existing diagnostic retained')
    rt=dict(status='RUNNING',phase='diagnosis',started_epoch=time.time(),training=False,confirmation_opened=False);atomic_json(OUT/'runtime.json',rt)
    try:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with np.load(ROOT/'results/relational_natural_state/dataset.npz',allow_pickle=False) as f:n={k:f[k] for k in f.files}
        roots=np.array(json.loads((ROOT/'results/relational_natural_state/inventory.json').read_text())['roots'])
        with np.load(SOURCE/'localized_ensemble_prediction.npz',allow_pickle=False) as f:pred=f['state']
        with np.load(ROOT/'results/relational_autorun/models/dataset.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
        meta=json.loads((ROOT/'results/relational_autorun/models/samples.json').read_text())
        coverage=json.loads((ROOT/'results/relational_autorun/coverage.json').read_text())['rows'];cov={(r['root'],r['anchor'],r['variant']):r for r in coverage}
        anchors={}
        for i,m in enumerate(meta):
            if not d['fit'][i]:anchors.setdefault((m['root'],m['anchor']),i)
        ix=np.array(list(anchors.values()));frame=n['history'][ix,-1];aroots=np.array([meta[i]['root'] for i in ix])
        phases=[]
        for i in ix:
            m=meta[i];c=cov[(m['root'],m['anchor'],m['variant'])]
            phases.append('held_detached' if c['current_events'][0][1]==1 else 'on_without_grip' if d['current'][i,3]==1 and not c['current_grip'] else 'contact_not_detached' if c['current_grip'] else 'no_grip_no_on')
        phases=np.array(phases);error=(pred[frame,:12]-n['target'][frame,:12]).reshape(-1,4,3)*1000
        phase_rows={}
        for phase in sorted(set(phases)):
            mask=phases==phase
            phase_rows[phase]=dict(anchors=int(mask.sum()),scenes=len(set(aroots[mask])),target_plate_axis_mae_mm=macro(abs(error[mask,0]),aroots[mask]),target_plate_axis_bias_mm=macro(error[mask,0],aroots[mask]),target_plate_norm_mm=macro(np.linalg.norm(error[mask,0],axis=1),aroots[mask]))
        axis=dict(anchors=len(ix),scenes=len(set(aroots)),target_plate_axis_mae_mm=macro(abs(error[:,0]),aroots),target_plate_axis_bias_mm=macro(error[:,0],aroots),target_gripper_local_axis_mae_mm=macro(abs(error[:,1]),aroots),phases=phase_rows)
        variants={'natural':pred.copy()}
        for name,cols in [('fix_target_plate_xy',[0,1]),('fix_target_plate_z',[2]),('fix_target_plate',list(range(3))),('fix_target_gripper',list(range(3,6))),('fix_all_relations',list(range(12))),('fix_all_state',list(range(18)))]:
            v=pred.copy();v[:,cols]=n['target'][:,cols];variants[name]=v
        bridge.OUT=OUT;metrics={};event_subsets={};checks=0
        held=np.array([not d['fit'][i] and cov[(m['root'],m['anchor'],m['variant'])]['current_events'][0][1]==1 for i,m in enumerate(meta)])
        for name,v in variants.items():
            metrics[name]=bridge.evaluate(name,v,n,roots)
            with np.load(OUT/f'{name}_bridge.npz',allow_pickle=False) as f:
                assert score(f['delta'],f['probability'],d,meta,~d['fit'])==metrics[name]['delta']
                assert score(f['endpoint_delta'],f['probability'],d,meta,~d['fit'])==metrics[name]['absolute_endpoint']
                assert paired(f['delta'],d,meta,~d['fit'])==metrics[name]['pairs'];checks+=3
                valid=held & (d['binary'][:,8]>=0);event_subsets[name]=dict(n=int(valid.sum()),scenes=len({meta[i]['root'] for i in np.flatnonzero(valid)}),release=prf(d['binary'][valid,8]==1,f['probability'][valid,8]>=.5))
        with np.load(OUT/'natural_bridge.npz',allow_pickle=False) as a,np.load(SOURCE/'localized_ensemble_bridge.npz',allow_pickle=False) as b:
            for key in ['delta','probability','endpoint_delta']:np.testing.assert_allclose(a[key],b[key],atol=1e-6,rtol=1e-5)
        with np.load(OUT/'fix_all_state_bridge.npz',allow_pickle=False) as a,np.load(ROOT/'results/relational_minimal_state/geometry_history/action_ensemble_predictions.npz',allow_pickle=False) as b:
            for key in ['delta','probability']:np.testing.assert_allclose(a[key],b[key],atol=1e-6,rtol=1e-5)
        # Auxiliary head is a separate diagnostic path, NOT the direct relation head's hidden decomposition.
        with np.load(SOURCE/'normalization.npz',allow_pickle=False) as f:st={k:f[k] for k in f.files}
        images=np.concatenate([n['initial_images'][n['initial_ids'][frame]],n['images'][frame]],-1)
        im=torch.tensor(images.transpose(0,3,1,2).copy(),device='cuda',dtype=torch.float32)/255
        obs=torch.tensor((n['obs'][frame]-st['obs_mean'])/st['obs_std'],device='cuda');heads=[]
        for seed in [11,23,47]:
            m=Model().cuda();ck=torch.load(SOURCE/f'localized_seed{seed}.pt',map_location='cpu',weights_only=True);m.load_state_dict(ck['state_dict']);m.eval()
            with torch.no_grad():heads.append(m(im,obs)[1].cpu().numpy()*st['points_std']+st['points_mean'])
        head=np.mean(heads,0);camera=json.loads((ROOT/'results/camera_projection_contract/camera_inventory.json').read_text())['camera_reference'];matrix=np.array(camera['world_to_raw_pixel'])
        pos=world_from_head(head,matrix);truth=[];sources={}
        records=json.loads((ROOT/'results/relational_autorun/branches/manifest.json').read_text())['records'];recordmap={(r['root'],r['anchor'],r['variant']):r for r in records}
        for i in ix:
            m=meta[i];r=recordmap[(m['root'],m['anchor'],m['variant'])];path=ROOT/r['path'];assert sha256(path)==r['sha256'];sources[r['path']]=r['sha256']
            with np.load(path,allow_pickle=False) as f:truth.append(f['pose'][m['anchor'],:3,:3])
        truth=np.array(truth);actual=truth[:,0]-truth[:,2]
        relative=pos[:,0]-pos[:,2];repair_target=truth[:,0]-pos[:,2];repair_plate=pos[:,0]-truth[:,2]
        head_results=dict(path='auxiliary_uv_depth_head_only',nonpositive_depth_predictions=int((head[:,6:]<=0).sum()),object_position_error_mm=macro(np.linalg.norm(pos-truth,axis=2)*1000,aroots),relative_target_plate_mm=macro(np.linalg.norm(relative-actual,axis=1)*1000,aroots),true_target_only_mm=macro(np.linalg.norm(repair_target-actual,axis=1)*1000,aroots),true_plate_only_mm=macro(np.linalg.norm(repair_plate-actual,axis=1)*1000,aroots))
        # Projection inversion sanity with true coordinates, distinct from any prediction accuracy.
        from src.probes.camera_projection_contract import official_float_projection
        uv,depth=official_float_projection(truth,matrix);uv[...,1]=127-uv[...,1];truehead=np.concatenate([uv.reshape(-1,6),depth],1)
        np.testing.assert_allclose(world_from_head(truehead,matrix),truth,atol=1e-9,rtol=0)
        np.savez_compressed(OUT/'auxiliary_object_predictions.npz',head=head,predicted_world=pos,true_world=truth,frame_indices=frame)
        result=dict(axis=axis,models=metrics,held_release=event_subsets,auxiliary_objects=head_results,source_sha256=sources,training=False,confirmation_opened=False)
        atomic_json(OUT/'evaluation.json',result)
        def vals(m):return [m['absolute_endpoint']['root_macro_target_plate_mm'],m['absolute_endpoint']['root_macro_target_gripper_mm'],m['pairs']['root_macro_error_mm']]
        lines=['# 几何误差分解与事件修复诊断','', '日期：2026-10-09。无新训练、无新仿真、confirmation封存。全部分析为已暴露开发场景的描述性诊断。','',
            '## 方向与当前状态','',f"只保留22个不同root起点（9场景），不将每个起点的6条候选重复计数。目标—盘世界X/Y/Z的scene宏平均绝对误差为：{axis['target_plate_axis_mae_mm']} mm；带符号偏差为：{axis['target_plate_axis_bias_mm']} mm。",'',
            '| 当前关系分层 | 起点/场景 | X/Y/Z绝对误差 mm | 目标—盘误差 mm |','|---|---:|---|---:|']
        for name,s in phase_rows.items():lines.append(f"| {name} | {s['anchors']}/{s['scenes']} | "+'/'.join(f'{v:.2f}' for v in s['target_plate_axis_mae_mm'])+f" | {s['target_plate_norm_mm']:.2f} |")
        lines+=['','held_detached按v1持续持有证据；on_without_grip只表示当前On且无夹爪接触，不保证稳定放置。其余两组是接触状态，不冒称完整操作阶段。分层场景数量不同，不能作阶段因果比较。','',
            '## 修复几何能否修复后果和事件','', '| 输入修复 | 未来目标—盘 mm | 未来目标—夹爪 mm | 差分 mm | On发生 TP/FP/FN | release发生 TP/FP/FN |','|---|---:|---:|---:|---:|---:|']
        for name,m in metrics.items():
            b=m['absolute_endpoint']['binary'];counts=lambda k:'/'.join(str(b['target/'+k]['onset'][q]) for q in ['tp','fp','fn'])
            lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in vals(m))+f" | {counts('on')} | {counts('release')} |")
        lines+=['','修复使用每个历史时刻的真实量，是不可部署的诊断干预；部分通道修复可能制造相互不一致的关系，不能当真实可达物理状态。绝对未来误差同时包含起点位置修复收益，所以必须同时看事件与候选差分，不能把位置降幅直接解释为动力学改进。', '', '| 输入 | 持有起点release TP/FP/FN |','|---|---:|']
        for name,s in event_subsets.items():lines.append('| '+name+' | '+'/'.join(str(s['release'][q]) for q in ['tp','fp','fn'])+' |')
        lines+=['','真实持有→释放正例仍只有6分支、3场景；release全量发生包含稳定窗口形成。几何真实输入后仍有误报，并不单独证明必须增加接触输入：数据覆盖、动力学目标与读出也可能造成错误。', '',
            '## 分开看碗与盘：只在辅助定位头路径中成立','',
            f"通过相机逆投影，辅助头的碗1/碗2/盘位置误差为 {head_results['object_position_error_mm']} mm。该路径目标—盘误差 {head_results['relative_target_plate_mm']:.3f} mm；只修复目标碗后 {head_results['true_target_only_mm']:.3f} mm；只修复盘后 {head_results['true_plate_only_mm']:.3f} mm。非正预测深度数为 {head_results['nonpositive_depth_predictions']}。", '',
            '主网络直接输出相对关系，没有可唯一分解的目标位置和盘位置。上述辅助头不是主输出的中间状态，不能用其真值替换结果断言主输出误差一定来自某一物体。相机反投影已用真实坐标进行闭合验证；预测是否准确另行评价。', '',
            f'验证：{checks}组保存桥接指标复算；自然输入及全部真实状态与原保存结果通过相同容差复现；投影逆变换验证通过；文件哈希冻结。没有因诊断结果选择新模型或更改阈值。']
        report=ROOT/'docs/relational_geometry_diagnosis.md';report.write_text('\n'.join(lines)+'\n')
        atomic_json(OUT/'validation.json',dict(status='PASS',bridge_metric_groups_recomputed=checks,natural_and_oracle_reproduced=True,inverse_projection_verified=True,report_sha256=sha256(report)))
        rt.update(status='COMPLETE',phase='verified_and_reported',finished_epoch=time.time(),report=str(report.relative_to(ROOT)));atomic_json(OUT/'runtime.json',rt)
    except Exception:
        rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'runtime.json',rt);raise
if __name__=='__main__':main()
