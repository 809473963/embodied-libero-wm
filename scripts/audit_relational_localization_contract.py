#!/usr/bin/env python3
"""Old-fit-only static render / projection supervision audit. No control steps."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.libero'))
import numpy as np
from src.probes.candidate_utility_oracle import atomic_json,sha256
from src.probes.camera_projection_contract import independent_projection,official_float_projection,point_to_mask_distance
OUT=ROOT/'results/relational_localization_contract'

def main():
    cfg=json.loads((ROOT/'configs/relational_localization_contract_protocol.json').read_text())
    for p,h in cfg['frozen_files'].items():assert sha256(ROOT/p)==h,p
    OUT.mkdir(exist_ok=True)
    if (OUT/'runtime.json').exists():raise RuntimeError('Existing attempt retained')
    rt=dict(status='RUNNING',phase='static_render_check',completed_scenes=0,maximum_scenes=13,env_step_calls=0,started_epoch=time.time());atomic_json(OUT/'runtime.json',rt)
    try:
        from scripts.evaluate_c0_all import make_env
        from scripts.audit_camera_projection_contract import camera_record
        from scripts.inventory_visual_target_masks import render_rgb,render_segmentation
        from PIL import Image,ImageDraw
        camera=json.loads((ROOT/'results/camera_projection_contract/camera_inventory.json').read_text())['camera_reference']
        contracts=json.loads((ROOT/'configs/task_grounding_contract.json').read_text())['tasks']
        rows=[];tiles=[];uvs=[];xyzs=[];sources=[]
        for r in cfg['cases']:
            root=r['root_id'];path=ROOT/r['dense']['arrays'];assert sha256(path)==r['dense']['sha256']
            with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in ['qpos','qvel','pose','grip_pos','rgb','natural_step','proprio']}
            task=int(root[4:6]);np.random.seed(20261008+task);env=make_env(ROOT/contracts[task]['bddl_path'])
            try:
                for t in (0,r['anchor']):
                    sim=env.sim;sim.data.qpos[:]=a['qpos'][t];sim.data.qvel[:]=a['qvel'][t];sim.forward()
                    cam=camera_record(env.env,'agentview');difference=float(np.max(abs(np.array(cam['world_to_raw_pixel'])-np.array(camera['world_to_raw_pixel']))))
                    assert difference<1e-10,'Camera changed; inherited calibration invalid'
                    j=int(np.flatnonzero(a['natural_step']==t)[0]);stored=a['rgb'][j];rendered=render_rgb(env.env);seg=render_segmentation(env.env)
                    bodyids=[sim.model.body_name2id(b) for b in r['dense']['geometry']['bodies'][:3]]
                    restored=sim.data.body_xpos[bodyids].copy();truth=a['pose'][t,:3,:3]
                    grip=env.env.robots[0].gripper.important_sites['grip_site'];sid=sim.model.site_name2id(grip)
                    points=np.r_[truth,a['grip_pos'][t][None]]
                    raw,depth=official_float_projection(points,np.array(cam['world_to_raw_pixel']))
                    _,uv,dep2=independent_projection(points,np.array(cam['camera_position_world']),np.array(cam['camera_rotation_world']),cam['fovy_degrees'])
                    expected=raw.copy();expected[:,1]=127-raw[:,1];np.testing.assert_allclose(uv,expected,atol=1e-10);np.testing.assert_allclose(depth,dep2,atol=1e-12)
                    assert (depth>0).all()
                    uv64=(uv-.5)/2;recovered128=uv64*2+.5;raw2=recovered128.copy();raw2[:,1]=127-raw2[:,1]
                    pix=np.c_[raw2*depth[:,None],depth,np.ones(len(depth))]
                    back=pix@np.linalg.inv(np.array(cam['world_to_raw_pixel'])).T
                    roundtrip=float(np.max(abs(back[:,:3]-points)));assert roundtrip<1e-9
                    restored_grip=sim.data.site_xpos[sid].copy()
                    _,gripuv,_=independent_projection(restored_grip,np.array(cam['camera_position_world']),np.array(cam['camera_rotation_world']),cam['fovy_degrees'])
                    distance=point_to_mask_distance(gripuv,(seg[...,0]==6)&(seg[...,1]==sid))
                    row=dict(root=root,step=t,source=r['dense']['arrays'],source_sha256=r['dense']['sha256'],camera_max_difference=difference,
                        body_position_restore_max_m=float(np.max(abs(restored-truth))),grip_position_restore_max_m=float(np.max(abs(restored_grip-a['grip_pos'][t]))),
                        rgb_bitwise_equal=bool(np.array_equal(stored,rendered)),rgb_mae=float(abs(stored.astype(float)-rendered).mean()),rgb_changed_pixel_fraction=float(np.any(stored!=rendered,axis=2).mean()),
                        restored_grip_site_mask_distance_px=distance,projection_roundtrip_max_m=roundtrip,uv128=uv.tolist(),uv64=uv64.tolist(),in_frame=((uv[:,0]>=0)&(uv[:,0]<128)&(uv[:,1]>=0)&(uv[:,1]<128)).tolist())
                    rows.append(row);uvs.append(uv);xyzs.append(points);sources.append([root,t])
                    # Mark projected body origins; these are not asserted visible object centers.
                    tile=Image.fromarray(stored).resize((384,384),Image.Resampling.NEAREST);draw=ImageDraw.Draw(tile)
                    for k,(u,v) in enumerate(uv):
                        color=['red','cyan','yellow','lime'][k];x,y=u*3,v*3;draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2);draw.text((x+5,y),str(k),fill=color)
                    draw.rectangle((0,0,384,18),fill='black');draw.text((3,3),f'{root} t={t}',fill='white');tiles.append(tile)
                rt['completed_scenes']+=1;atomic_json(OUT/'runtime.json',rt);print(root,flush=True)
            finally:env.close()
        canvas=Image.new('RGB',(384*4,384*((len(tiles)+3)//4)),color='black')
        for i,tile in enumerate(tiles):canvas.paste(tile,((i%4)*384,(i//4)*384))
        canvas.save(OUT/'projection_overlays.png')
        np.savez_compressed(OUT/'projected_points.npz',uv128=np.array(uvs),xyz=np.array(xyzs))
        visible=[r['restored_grip_site_mask_distance_px'] for r in rows if r['restored_grip_site_mask_distance_px'] is not None]
        raster=bool(visible) and np.median(visible)<=1 and np.quantile(visible,.95)<=1.5 and max(visible)<=2
        exact=all(r['rgb_bitwise_equal'] for r in rows)
        verdict='STATIC_LOCALIZATION_RENDER_CONTRACT_VALIDATED' if exact and raster else 'STATIC_RECONSTRUCTION_NOT_EXACT'
        atomic_json(OUT/'evaluation.json',dict(verdict=verdict,rows=rows,exact_frames=sum(r['rgb_bitwise_equal'] for r in rows),total_frames=len(rows),visible_grip_landmarks=len(visible),raster_gate=raster,confirmation_opened=False,env_step_calls=0,training=False))
        lines=['# 关系定位监督：静态重建与投影契约审计','', '日期：2026-10-08。**'+verdict+'**。', '',
            '只使用旧fit的13个场景，每场景按已有manifest顺序取首条分支的初始帧和root帧，共26帧。没有访问新开发或确认数据；没有执行控制动作或训练模型。', '',
            '将保存qpos/qvel恢复到同任务环境并forward，只检验静态成像，不声称恢复controller/contact求解器历史。相机矩阵逐场景与已验证agentview契约比对；矩阵投影与独立几何公式互验，3D→128px→64px→3D闭合检查。', '',
            f'重建RGB逐位相同：{sum(r["rgb_bitwise_equal"] for r in rows)}/{len(rows)}。可见grip site紧致标记：{len(visible)}帧；原1/1.5/2像素median/p95/max标准：{raster}。', '',
            f'最大body位置重建差：{max(r["body_position_restore_max_m"] for r in rows)*1000:.6f} mm；最大grip位置重建差：{max(r["grip_position_restore_max_m"] for r in rows)*1000:.6f} mm。', '',
            '## 坐标与标签含义','', '保持原RGB方向，不翻图；只转换投影v为127−v_raw。2×2平均缩放后的像素中心为(u−0.5)/2，不是简单u/2。圆圈表示两个碗、盘的body原点及grip site，编号0/1/2/3；body原点不是可见区域质心，也不保证无遮挡。', '',
            '投影正确不代表图像中可观察深度，更不证明body原点可直接通过分割质心学到。未来监督应分开投影原点、可见mask与深度/相对几何，不能混用标签。', '',
            ('该样本上的静态RGB与投影契约得到支持，允许准备旧场景内部的对象定位监督对照；仍须报告遮挡/可见性和物体身份。' if verdict.endswith('VALIDATED') else '静态重建未满足原图精确一致，不能将重建分割直接贴回原RGB作为像素真值。先定位forward/渲染状态差异；本轮不据此训练定位模型或宣称视觉失败。'), '',
            '投影示意：[projection_overlays.png](../results/relational_localization_contract/projection_overlays.png)。完整每帧记录：[evaluation.json](../results/relational_localization_contract/evaluation.json)。']
        report=ROOT/'docs/relational_localization_contract_audit.md';report.write_text('\n'.join(lines)+'\n')
        rt.update(status='COMPLETE',phase='audited',verdict=verdict,finished_epoch=time.time(),report=str(report.relative_to(ROOT)));atomic_json(OUT/'runtime.json',rt)
    except Exception:
        rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'runtime.json',rt);raise
if __name__=='__main__':main()
